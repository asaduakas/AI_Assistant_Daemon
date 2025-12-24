def build_prompt(
        context_chunks: list[str],
        question = str
) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
You are an AI assistant helping with a Python codebase.

You are given relevant code snippets from the project.
Answer the question using ONLY the information in the snippets.
If the answer is not contained in the snippets, say you don't know.

Code snippets:
{context}

Question:
{question}

Answer:
""".strip()

    return prompt