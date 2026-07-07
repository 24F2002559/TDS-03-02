import os
import base64
import requests
import time
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

HF_API_URL = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"
HF_TOKEN = os.environ.get("HF_TOKEN")

class ImageRequest(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(req: ImageRequest):
    # Clean base64
    b64 = req.image_base64
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {
        "inputs": {
            "image": b64,
            "text": req.question
        },
        "parameters": {"max_new_tokens": 100}   # limit output length
    }

    max_retries = 2
    for attempt in range(max_retries):
        try:
            resp = requests.post(HF_API_URL, json=payload, headers=headers, timeout=60)
            # Check if model is loading
            if resp.status_code == 503:
                # The response is likely {"error": "Model is loading", "estimated_time": 20}
                data = resp.json()
                wait = data.get("estimated_time", 10)
                print(f"Model loading, waiting {wait} seconds...")
                time.sleep(wait)
                continue   # retry

            resp.raise_for_status()
            data = resp.json()

            # Llava output is usually [{"generated_text": "answer"}]
            if isinstance(data, list) and len(data) > 0:
                answer = data[0]["generated_text"].strip()
                # Remove the question part if it's echoed (some models repeat the prompt)
                if req.question in answer:
                    answer = answer.replace(req.question, "").strip()
            elif isinstance(data, dict) and "generated_text" in data:
                answer = data["generated_text"].strip()
            else:
                raise HTTPException(status_code=500, detail="Unexpected HF response format")

            return {"answer": answer}

        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"HF request failed: {str(e)}")

    raise HTTPException(status_code=503, detail="Model is taking too long to load, try again later")
