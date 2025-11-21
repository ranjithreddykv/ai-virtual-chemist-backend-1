# backend/routes/tutor_ai.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from groq import Groq

from app.utils.ai_tutor_prompt import build_ai_tutor_prompt
import os
# same Groq style as your mechanism generator
client = Groq(api_key=os.getenv("GROK_API_KEY"))

router = APIRouter()

# Request schema from React Tutor Widget
class TutorRequest(BaseModel):
    question: str
    pageContext: Optional[Dict[str, Any]] = None
    chemContext: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None


@router.post("/ai-tutor")
async def ai_tutor_chat(req: TutorRequest):
    """
    Main AI Tutor route.
    This route produces context-aware chemistry tutoring answers for:
    - Forward prediction
    - Mechanism pages
    - Retrosynthesis
    - Yield
    - General conceptual questions

    It uses the exact same tone/discipline as your mechanism explainer.
    """

    # Build the combined chemistry prompt
    final_prompt = build_ai_tutor_prompt(
        question=req.question,
        page_context=req.pageContext,
        chem_context=req.chemContext,
        history=req.history,
    )

    # Send to Groq LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # same as your mechanism file
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return {"answer": answer}
