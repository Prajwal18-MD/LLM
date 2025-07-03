# app.py
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from typing import Dict, List
from collections import Counter

app = FastAPI()

# Enable CORS for all origins (safe on free tier; lock down later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ——— Pipelines loaded once at startup ———
_translators: Dict[str, pipeline] = {}
def get_translator(src: str, tgt: str):
    key = f"{src}-{tgt}"
    if key not in _translators:
        model_id = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        try:
            _translators[key] = pipeline("translation", model=model_id)
        except Exception as e:
            raise HTTPException(
                400, detail=f"Could not load translator {src}→{tgt}: {e}"
            )
    return _translators[key]

summarizer  = pipeline("summarization", model="facebook/bart-large-cnn")

# ——— Endpoints ———

@app.get("/")
async def home():
    return {"message": "Personalized LLM API is running 🚀"}

@app.post("/translate")
async def translate(payload: dict):
    """
    {
      "text": "...",
      "source_lang": "en",
      "target_lang": "de"
    }
    """
    text = payload.get("text","").strip()
    src  = payload.get("source_lang","").lower().strip()
    tgt  = payload.get("target_lang","").lower().strip()
    if not all([text, src, tgt]):
        raise HTTPException(400, detail="'text', 'source_lang' & 'target_lang' are required")
    translator = get_translator(src, tgt)
    out = translator(text, max_length=512)
    translation = out[0].get("translation_text") or out[0].get("generated_text")
    return {"source_lang": src, "target_lang": tgt, "translation": translation}


@app.post("/summarize")
async def summarize(payload: dict):
    """
    {
      "text": "Long English text..."
    }
    """
    text = payload.get("text","").strip()
    if not text:
        raise HTTPException(400, detail="'text' is required")
    out = summarizer(text, max_length=60, min_length=20, do_sample=False)
    return {"summary": out[0]["summary_text"]}

nlp = spacy.load("en_core_web_sm")

@app.post("/hashtags")
async def hashtags(payload: dict):
    """
    {
      "text": "Some English text..."
    }
    Returns 5 most frequent noun/proper-noun lemmas prefixed with '#'.
    """
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(400, detail="'text' is required")

    doc = nlp(text)
    # Collect noun and proper-noun lemmas
    lemmas: List[str] = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and token.is_alpha
    ]
    if not lemmas:
        return {"hashtags": ""}

    # Count and pick top 5
    top5 = [w for w, _ in Counter(lemmas).most_common(5)]
    # Build hashtags (camel‑case each multiword lemma)
    hashtags = []
    for lemma in top5:
        parts = lemma.split()
        tag = "#" + "".join(p.capitalize() for p in parts)
        hashtags.append(tag)

    return {"hashtags": " ".join(hashtags)}