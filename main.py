import os
import json
import urllib.parse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI(title="TrustLens Multi-Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

@app.get("/")
def home():
    return {"status": "TrustLens Multi-Platform Backend is active 24/7"}

@app.get("/analyze")
def analyze(item: str = "", current_price: str = "Unknown"):
    if not item.strip():
        raise HTTPException(status_code=400, detail="Product title cannot be empty")

    if not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")

    prompt = f"""
You are an expert e-commerce data analyst and pricing intelligence officer.
Analyze this product:
Product Title: "{item}"
Current Listed Price: "{current_price}"

Task:
1. Authenticity & Consensus: Evaluate verified buyer consensus, chronic defects, and quality.
2. Price Timing & Premium Detection: Estimate typical pricing cycles (festive sale vs regular vs overpriced premium). Is now a good time to buy, or should the user wait for a sale?

Return ONLY a valid JSON object matching this schema:
{{
  "score": <integer from 50 to 98 representing authenticity/quality>,
  "buy_timing": "<BUY NOW | FAIR PRICE | WAIT FOR SALE | OVERPRICED>",
  "timing_rationale": "<1 brief sentence explaining whether the current price is a deal, premium, or typical>",
  "typical_price_range": "<e.g. ₹1,299 - ₹1,899 or $15 - $25>",
  "consensus": "<1 short sentence summarizing what real owners think>",
  "pros": "<1 key standout benefit praised by actual users>",
  "flaws": "<1 key defect, sizing quirk, or common complaint>"
}}
Do not add markdown backticks. Return raw JSON only.
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content.strip())

        # Build clean deep-search queries for Reddit, YouTube, and X
        clean_name = " ".join(item.split()[:5])  # Keep first 5 words for accurate search
        encoded_query = urllib.parse.quote(clean_name)

        data["social_links"] = {
            "reddit": f"https://www.reddit.com/search/?q={encoded_query}+review",
            "youtube": f"https://www.youtube.com/results?search_query={encoded_query}+real+review",
            "twitter": f"https://twitter.com/search?q={encoded_query}+review&f=live"
        }

        return data

    except Exception as e:
        print(f"[TrustLens Error]: {e}")
        raise HTTPException(status_code=500, detail=str(e))