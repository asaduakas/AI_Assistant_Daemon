from fastapi import APIRouter
from pydantic import BaseModel

from app.services.llm_service import generate_reply

router = APIRouter(prefix="/llm")

class Prompt(BaseModel):
    prompt: str
    max_tokens: int = 256

@router.post("/ask")
async def ask_llm(data: Prompt):
    response = generate_reply(data.prompt, data.max_tokens)
    return {"response": response}