import os
import httpx
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY", "").strip()

if not groq_key or groq_key.startswith("gsk_your"):
    print("❌ ERROR: GROQ_API_KEY in .env is missing or still set to placeholder!")
    exit(1)

print(f"Checking Groq Key (starts with {groq_key[:8]}...)...")

headers = {
    "Authorization": f"Bearer {groq_key}",
    "Content-Type": "application/json"
}

try:
    res = httpx.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10.0)
    if res.status_code == 200:
        models = [m["id"] for m in res.json().get("data", []) if "whisper" not in m["id"]]
        print("\n✅ Groq API Key is VALID! Available models for your account:")
        for m in sorted(models):
            print(f"  • {m}")
        
        # Test a direct completion
        test_model = models[0]
        payload = {
            "model": test_model,
            "messages": [{"role": "user", "content": "Say 'Groq is ready' in 3 words."}],
            "max_tokens": 10
        }
        test_res = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=10.0)
        print(f"\nTest inference with '{test_model}': {test_res.json()['choices'][0]['message']['content'].strip()}")
    else:
        print(f"❌ API Error {res.status_code}: {res.text}")
except Exception as e:
    print(f"❌ Connection error: {e}")