import json
import os
import urllib.request

ENDPOINT = os.getenv(
    "LLM_LAUNCHER_BENCH_URL",
    "http://127.0.0.1:8421/v1/chat/completions",
)
payload = {
    "model": "Darwin-36B-Opus.Q4_K_M",
    "messages": [{"role": "user", "content": "Diga em uma frase: o que é Python?"}],
    "max_tokens": 80,
    "temperature": 0.0,
    "stream": False,
}
req = urllib.request.Request(
    ENDPOINT,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=60) as r:
    body = r.read().decode("utf-8")
print(body)
