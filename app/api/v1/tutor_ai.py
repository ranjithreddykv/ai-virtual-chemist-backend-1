# backend/routes/tutor_ai.py

import os
import json
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi.responses import FileResponse

from groq import Groq
import edge_tts

from app.utils.ai_tutor_prompt import build_ai_tutor_prompt


# -------------------------------------------------------------
# INIT CLIENTS
# -------------------------------------------------------------
groq_client = Groq(api_key=os.getenv("GROK_API_KEY"))

router = APIRouter()


# -------------------------------------------------------------
# REQUEST MODEL
# -------------------------------------------------------------
class TutorRequest(BaseModel):
    question: str
    pageContext: Optional[Dict[str, Any]] = None
    chemContext: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None


# -------------------------------------------------------------
# MAIN LLM + TTS ROUTE
# -------------------------------------------------------------
@router.post("/ai-tutor")
async def ai_tutor_chat(req: TutorRequest):
    """
    Produces:
        - text_answer (Markdown chemistry explanation)
        - voice_explanation (longer, spoken-style explanation)
        - audio_url (Edge-TTS generated MP3)
    """

    # ------------------ Build Prompt ------------------
    final_prompt = build_ai_tutor_prompt(
        question=req.question,
        page_context=req.pageContext,
        chem_context=req.chemContext,
        history=req.history,
    )

    # ------------------ Groq LLM Call ------------------
    llm_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.15,  # more controlled for chemistry
    )

    raw = llm_response.choices[0].message.content.strip()

    # ------------------ Parse JSON ------------------
    # We expect:
    # { "text_answer": "...", "voice_explanation": "..." }

    try:
        result = json.loads(raw)
    except Exception:
        # fallback: model didn't follow JSON perfectly
        result = {
            "text_answer": raw,
            "voice_explanation": (
                "Let me explain this verbally. "
                "Here is a clearer and more detailed explanation of the concept: "
                + raw
            ),
        }

    text_answer = result.get("text_answer", raw)
    spoken_text = result.get("voice_explanation", text_answer)

    # ------------------ Generate TTS ------------------
    audio_dir = "tts_audio"
    os.makedirs(audio_dir, exist_ok=True)

    # cache by hash so same answer doesn't regenerate
    file_id = f"{abs(hash(spoken_text))}.mp3"
    file_path = os.path.join(audio_dir, file_id)

    if not os.path.exists(file_path):
        # Generate high-quality neural voice
        tts = edge_tts.Communicate(
            text=spoken_text,
            voice="en-US-JennyNeural",   # Change voice here if needed
            rate="+0%",
        )
        await tts.save(file_path)

    # ------------------ Construct audio URL ------------------
    audio_url = f"http://localhost:8000/api/v1/ai-tutor/audio/{file_id}"

    # ------------------ Return result ------------------
    return {
        "answer": text_answer,       # Markdown explanation
        "audio_url": audio_url,      # MP3 file
        "voice_explanation": spoken_text,  # optional, for debugging
    }


# -------------------------------------------------------------
# SERVE AUDIO FILE
# -------------------------------------------------------------
@router.get("/ai-tutor/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join("tts_audio", filename)
    return FileResponse(file_path, media_type="audio/mpeg")
