import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

# ---- CORS (allow all origins) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Configure Gemini ----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")   # fast & free tier

class ImageRequest(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(request: ImageRequest):
    try:
        # Decode the base64 image to bytes
        image_data = base64.b64decode(request.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # Send to Gemini
    prompt = request.question
    try:
        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": image_data}   # Gemini will auto-detect format
        ])
        answer = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    return {"answer": answer}
