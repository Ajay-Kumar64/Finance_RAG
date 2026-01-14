from llama_cpp import Llama

llm = Llama(
    model_path="artifacts/llama_model/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
)

out = llm(
    "Q: What is rbi?\nA:",
    max_tokens=100,
)

print(out["choices"][0]["text"])
