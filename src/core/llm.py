from src.core.model_loader import get_llm

def generate_reply(prompt: str, max_tokens: int = 256):
    llm = get_llm()

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>"]
    )

    return output["choices"][0]["text"].strip()