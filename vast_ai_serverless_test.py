"""Client and test for the pocket-tts-egy Vast serverless endpoint (plain HTTPS, stdlib only).

    python vast_ai_serverless_test.py            # VAST_API_KEY from the project .env
    python vast_ai_serverless_test.py --text "إزيك عامل إيه" --out-dir out

Use the client from your own code:

    from vast_ai_serverless_test import TTSVastClient
    client = TTSVastClient()                        # fetches (or reuses) the endpoint token
    wav = client.synthesize("إزيك عامل إيه")        # -> WAV bytes (24 kHz, 16-bit mono)
    open("out.wav", "wb").write(wav)

Every call goes through Vast's three steps:

  1. Token: GET https://console.vast.ai/api/v0/endptjobs/?client_id=me (account key) gives
     the endpoint's own key. It expires (a fixed date, about a week out), so the client keeps
     it in a temp file, reuses it while valid, and fetches a new one when Vast rejects it.
  2. Worker: POST https://run.vast.ai/route/ gives a worker URL and a one-time signed
     auth_data. Done for every request: workers change and signatures are single use.
  3. Request: POST <worker url>/generate with {"auth_data": ..., "payload": {"text": ...}}.
     The worker's TLS certificate is signed by Vast's CA (jvastai_root.cer).
"""

import argparse
import base64
import json
import os
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave

CONSOLE = "https://console.vast.ai"
ROUTER = "https://run.vast.ai"
VAST_CA_URL = f"{CONSOLE}/static/jvastai_root.cer"


def load_dotenv(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Read KEY=VALUE lines from the project's .env (git-ignored) into os.environ.
    Variables already set in the environment win."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)


class TokenRejected(Exception):
    """Vast refused the endpoint token: fetch a new one and retry."""


class TTSVastClient:
    ENDPOINT = "pocket-tts-egy"

    def __init__(self, account_key=None, token_file=None, route_wait=180.0, timeout=300.0):
        load_dotenv()
        self.account_key = account_key or os.environ.get("VAST_API_KEY")
        if not self.account_key:
            raise RuntimeError("VAST_API_KEY not found: add it to this project's .env or the environment")
        self.token_file = token_file or os.path.join(
            tempfile.gettempdir(), f"vast_endpoint_token_{self.ENDPOINT}.json")
        self.route_wait = route_wait
        self.timeout = timeout
        self.ssl_context = self._vast_ssl_context()
        self.token = None
        self.last_worker = None
        self.last_timing = {}
        self.fetch_token()

    # ------------------------------------------------------------ step 1: token

    def fetch_token(self, force=False):
        """Return the endpoint token: the saved one while it is valid, else a new one.

        force=True skips the saved token (used after Vast rejected it).
        """
        if not force:
            saved = self._read_saved_token()
            if saved:
                self.token = saved
                return self.token
        status, body = self._http_json(
            "GET", f"{CONSOLE}/api/v0/endptjobs/?client_id=me",
            headers={"Authorization": f"Bearer {self.account_key}"})
        if status != 200:
            raise RuntimeError(f"fetching the endpoint token failed: HTTP {status}: {body}")
        for e in body["results"]:
            if e["endpoint_name"] == self.ENDPOINT:
                self.token = e["api_key"]
                self._save_token(self.token)
                return self.token
        raise RuntimeError(f"no endpoint named {self.ENDPOINT!r} on this account")

    @staticmethod
    def token_expiry(token):
        """Expiry (unix time) written inside the token, or None if it cannot be read."""
        try:
            outer = json.loads(base64.b64decode(token + "=" * (-len(token) % 4)))
            return float(json.loads(outer["msg"])["expiration"])
        except Exception:
            return None

    def _read_saved_token(self):
        try:
            with open(self.token_file, encoding="utf-8") as f:
                token = json.load(f)["token"]
        except (OSError, ValueError, KeyError):
            return None
        expiry = self.token_expiry(token)
        if expiry is not None and expiry - time.time() < 300:   # expired or about to
            return None
        return token

    def _save_token(self, token):
        # Owner-only file: the token lets anyone call the endpoint.
        fd = os.open(self.token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"endpoint": self.ENDPOINT, "token": token,
                       "expires": self.token_expiry(token), "saved_at": time.time()}, f)

    # ------------------------------------------------------------ step 2: worker

    def _route(self, cost):
        """Ask the router for a worker, polling until one is ready."""
        body = {"endpoint": self.ENDPOINT, "api_key": self.token, "cost": cost}
        deadline = time.time() + self.route_wait
        while True:
            status, answer = self._http_json(
                "POST", f"{ROUTER}/route/", body,
                headers={"Authorization": f"Bearer {self.token}"})
            if status == 200 and isinstance(answer, dict) and answer.get("url"):
                return answer
            text = json.dumps(answer, ensure_ascii=False) if not isinstance(answer, str) else answer
            if status in (401, 403) or "unauthorized" in text.lower():
                raise TokenRejected(text)
            if status != 200 or "error_msg" in text or time.time() > deadline:
                raise RuntimeError(f"route failed: HTTP {status}: {text}")
            body["request_idx"] = answer.get("request_idx", 0)   # keep our place in the queue
            time.sleep(1)

    # ------------------------------------------------------------ step 3: request

    def request(self, path, payload, cost):
        """Route and send one request; on a rejected token fetch a new one and retry once.

        Returns (HTTP status, body bytes). Timings land in self.last_timing.
        """
        for attempt in (1, 2):
            try:
                auth = self._route(cost)
            except TokenRejected:
                if attempt == 2:
                    raise
                self.fetch_token(force=True)
                continue
            req = urllib.request.Request(
                auth["url"] + path, method="POST",
                data=json.dumps({"auth_data": auth, "payload": payload}).encode(),
                headers={"Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as r:
                    first = r.read(4096)
                    ttfb = time.time() - t0
                    status, data = r.status, first + r.read()
            except urllib.error.HTTPError as e:
                status, data, ttfb = e.code, e.read(), time.time() - t0
            if status == 401 and attempt == 1:     # worker refused the signature
                self.fetch_token(force=True)
                continue
            self.last_worker = auth["url"]
            self.last_timing = {"first_byte_s": ttfb, "total_s": time.time() - t0}
            return status, data

    def synthesize(self, text, voice_url=None, stream=False):
        """Text in, WAV bytes out.

        stream=False uses /generate (the complete file); stream=True uses /generate/stream,
        where the server starts sending audio while it is still synthesizing.
        voice_url: a built-in voice name or an http(s)/hf:// voice file; default voice if None.
        """
        payload = {"text": text}
        if voice_url:
            payload["voice_url"] = voice_url
        path = "/generate/stream" if stream else "/generate"
        status, data = self.request(path, payload, cost=max(len(text), 1))   # cost: characters
        if status != 200:
            raise RuntimeError(f"{path}: HTTP {status}: {data.decode(errors='replace')}")
        return data

    # ------------------------------------------------------------ HTTP helpers

    @staticmethod
    def _http_json(method, url, body=None, headers=None, timeout=30):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")

    @staticmethod
    def _vast_ssl_context():
        """TLS context trusting Vast's CA, which signs the workers' certificates."""
        path = os.path.join(tempfile.gettempdir(), "jvastai_root.cer")
        if not os.path.exists(path):
            with urllib.request.urlopen(VAST_CA_URL, timeout=10) as r:
                pem = r.read()
            with open(path, "wb") as f:
                f.write(pem)
        return ssl.create_default_context(cafile=path)


# ---------------------------------------------------------------- tests

DEFAULT_TEXTS = [
    "إزيك عامل إيه النهارده؟",
    "صباح الخير يا جماعة، الجو حلو قوي النهارده، تعالوا نخرج نتمشى شوية.",
]


def wav_seconds(path):
    """Duration of a WAV file. Streamed files carry placeholder sizes, so count the bytes."""
    with wave.open(path) as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
    return (os.path.getsize(path) - 44) / (rate * width * channels), rate


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--text", action="append", help="text to speak (repeatable)")
    ap.add_argument("--voice-url", help="built-in voice name or http(s)/hf:// voice URL")
    ap.add_argument("--out-dir", default="vast_tts_test_output", help="where to save the WAVs")
    args = ap.parse_args()
    texts = args.text or DEFAULT_TEXTS
    os.makedirs(args.out_dir, exist_ok=True)

    failures = 0

    def run(name, fn):
        nonlocal failures
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")

    client = TTSVastClient()
    expiry = TTSVastClient.token_expiry(client.token)
    print(f"endpoint: {client.ENDPOINT}\ntoken saved in {client.token_file}, expires "
          f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(expiry)) if expiry else 'unknown'}")

    def test_synthesize(stream, tag):
        def test():
            for i, text in enumerate(texts, 1):
                audio = client.synthesize(text, voice_url=args.voice_url, stream=stream)
                assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE", \
                    f"not a WAV file (starts with {audio[:16]!r})"
                out = os.path.join(args.out_dir, f"{tag}_{i}.wav")
                with open(out, "wb") as f:
                    f.write(audio)
                secs, rate = wav_seconds(out)
                assert secs > 0.3, f"audio too short: {secs:.2f} s"
                t = client.last_timing
                print(f"      {t['total_s']:5.2f} s total, first byte {t['first_byte_s']:5.2f} s, "
                      f"{secs:5.2f} s of audio at {rate} Hz "
                      f"({secs / t['total_s']:4.1f}x real time)  -> {out}")
        return test

    def test_bad_request():
        status, body = client.request("/generate", {"text": "   "}, cost=1)
        assert status in (400, 422), f"empty text accepted: HTTP {status}"
        print(f"      HTTP {status} for empty text (expected an error)")

    def test_token_refresh():
        # Plant a rejected token: the next call must replace it and still succeed.
        good = client.token
        client.token = "invalid-token"
        client._save_token(client.token)
        audio = client.synthesize(texts[0])
        assert audio[:4] == b"RIFF", "no audio after the token refresh"
        assert client.token == good, "the client did not fetch a new token"
        with open(client.token_file, encoding="utf-8") as f:
            assert json.load(f)["token"] == good, "the new token was not saved"
        print("      a rejected token was replaced, saved, and the request succeeded")

    run("synthesize()  POST /generate (complete WAV)", test_synthesize(False, "generate"))
    run("synthesize(stream=True)  POST /generate/stream", test_synthesize(True, "stream"))
    run("empty text is rejected", test_bad_request)
    run("rejected token is refreshed and saved", test_token_refresh)

    print(f"\n{'all tests passed' if not failures else f'{failures} test(s) failed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
