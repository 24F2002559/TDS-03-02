import base64
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face free Llava 1.5 endpoint (no authentication needed)
HF_API_URL = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"

class ImageRequest(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(req: ImageRequest):
    # Clean base64: remove data URI prefix if present
    b64 = req.image_base64
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        base64.b64decode(b64)   # just to validate
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # Send to Hugging Face
    payload = {
        "inputs": {
            "image": b64,          # raw base64 string (without prefix)
            "text": req.question
        }
    }

    try:
        resp = requests.post(HF_API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Response format: [{"generated_text": "answer"}]
        answer = data[0]["generated_text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HF error: {str(e)}")

    return {"answer": answer}
