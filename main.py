from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import json
import re
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

def get_live_groq_model():
    """Groq se directly current active models ki list check karta hai."""
    preferred = [
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192"
    ]
    try:
        available_models = [m.id for m in client.models.list().data]
        for p in preferred:
            if p in available_models:
                return p
        return available_models[0]
    except Exception as e:
        print("[TrustLens] Model lookup error:", e)
        return "openai/gpt-oss-20b"

def clean_title(raw: str) -> str:
    cleaned = re.sub(r'[^\w\s\-\.\/]', ' ', raw)
    parts = re.split(r'[,|:\-\(\)\[\]]', cleaned)
    core = parts[0].strip()
    return core[:60] if len(core) >= 4 else cleaned[:60].strip()

@app.get("/analyze")
def analyze(item: str = ""):
    product_name = clean_title(item)
    active_model = get_live_groq_model()
    print(f"\n[TrustLens] Calling Real AI with model ({active_model}) for: {product_name}")

    prompt = f"""
    You are an expert product reviewer. 
    Analyze the real-world reputation, owner satisfaction, and defects for:
    "{product_name}"

    CRITICAL RULES:
    - If this is skincare/lotion/cream: evaluate hydration, stickiness, absorption speed, fragrance, and pump bottle dispenser issues.
    - If this is shoes/clothing: evaluate sizing accuracy, sole wear, stitching, and long-term comfort.
    - If this is electronics: evaluate battery life, thermals, build quality, and software/hardware bugs.
    - DO NOT return generic tech text for non-tech items.

    Return ONLY a valid JSON object matching this schema:
    {{
      "score": <realistic integer from 50 to 98>,
      "consensus": "<1 short sentence summarizing what real owners think>",
      "pros": "<1 specific standout benefit praised by actual users>",
      "flaws": "<1 specific complaint, defect, or wear issue reported by buyers>"
    }}
    Do not add markdown formatting or backticks. Return raw JSON only.
    """

    try:
        response = client.chat.completions.create(
            model=active_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        result["score"] = int(result.get("score", 82))
        print(f"[TrustLens] REAL AI OUTPUT: {result}")
        return result

    except Exception as e:
        print(f"[TrustLens ERROR]: {e}")
        raise HTTPException(status_code=500, detail=str(e))