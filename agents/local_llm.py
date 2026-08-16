"""Talk to a LOCAL model via Ollama — no API key, offline, free.

Runs on your CPU (verified 100% CPU on this box; the AMD iGPU is not a usable
compute target). The 16-core Zen 5 makes small models snappy.

    uv run python agents/local_llm.py
"""

from openai import OpenAI

# Ollama exposes an OpenAI-compatible API on localhost:11434, so the standard
# OpenAI SDK talks to it directly -- the same code works against the real
# OpenAI API by swapping base_url + api_key.
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

resp = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "In one sentence: what is an AI agent?"},
    ],
)
print(resp.choices[0].message.content)
