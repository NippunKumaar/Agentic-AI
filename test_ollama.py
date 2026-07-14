from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"   # Dummy key; Ollama ignores it
)

response = client.chat.completions.create(
    model="qwen3:4b",
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ]
)

print(response.choices[0].message.content)