# backend/main.py

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import get_rag_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    message: Optional[str] = None


@app.get("/")
def health_check():
    return {"status": "NTPC RAG Chatbot is running"}


@app.post("/chat")
async def chat(request: ChatRequest):
    user_question = request.question or request.query or request.message

    if not user_question or not user_question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = await get_rag_answer(user_question)

        # If rag.py already returns correct JSON, send it directly
        if isinstance(result, dict):
            return result

        # If rag.py returns only text, convert it into frontend format
        return {
            "response": str(result),
            "sources": [],
            "followups": [
                "Latest Annual Report",
                "Revenue highlights",
                "Capacity addition",
            ],
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong.")