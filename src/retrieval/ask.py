from src.retrieval.retrieve import retrieve_chunks
from src.retrieval.prompting import build_prompt
from src.config import settings
from src.core.model_loader import get_embedding_model

def answer_question(question: str) -> str:
    chunks = retrieve_chunks(
        query=question,
        project_id= settings.PROJECT_ID,
    )

    if not chunks:
        return "I couldn't find relevant code for this question. Sawwy! :("
    
    prompt = build_prompt(
        context_chunks=chunks,
        question=question
    )

    llm = get_embedding_model()

    response = llm(
        prompt,
        max_tokens=512,
        temperature=0.1
    )

    return response["choices"][0]["text"].strip()