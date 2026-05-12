import ollama, requests, subprocess, sys

client = ollama.Client(host='http://localhost:11434')

# Test 1: raw HTTP chat
print("[RAW HTTP] /api/chat")
r = requests.post('http://localhost:11434/api/chat', json={
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": False,
    "options": {"num_predict": 20}
})
print(f"  Status: {r.status_code}")
print(f"  Body: {r.text[:500]}")

# Test 2: raw HTTP generate
print("\n[RAW HTTP] /api/generate")
r2 = requests.post('http://localhost:11434/api/generate', json={
    "model": "qwen3:4b",
    "prompt": "What is 2+2? Answer with just the number.",
    "stream": False,
    "options": {"num_predict": 20}
})
print(f"  Status: {r2.status_code}")
print(f"  Body: {r2.text[:500]}")

# Test 3: try qwen2.5 instead
print("\n[ALT MODEL] qwen2.5:7b-instruct-q4_K_M")
r3 = requests.post('http://localhost:11434/api/generate', json={
    "model": "qwen2.5:7b-instruct-q4_K_M",
    "prompt": "What is 2+2?",
    "stream": False,
    "options": {"num_predict": 20}
})
print(f"  Status: {r3.status_code}")
print(f"  Body: {r3.text[:300]}")

# Test 4: ollama version
print("\n[VERSION]")
result = subprocess.run([sys.executable, "-m", "pip", "show", "ollama"], capture_output=True, text=True)
print(result.stdout)

result2 = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
print("Ollama binary:", result2.stdout.strip())