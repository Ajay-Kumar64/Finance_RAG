import os
from google import genai

# The client automatically looks for the GEMINI_API_KEY environment variable
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ✅ 1. List available models (Correct way)
print("Available Models:")
for m in client.models.list():
    print(f"- {m.name}")

# ✅ 2. Generate text (Correct way)
response = client.models.generate_content(
    model="gemini-3-flash-preview",  # Added '-preview'
    contents="Explain RBI in one sentence."
)

print("\nLLM Output:\n", response.text)