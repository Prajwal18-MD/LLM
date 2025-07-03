import spacy
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from collections import Counter
from typing import List

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Quantization helper for t5-small
def load_quantized_t5():
    model_id = "t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model, tokenizer

# Load & quantize once
_t5_model, _t5_tok = load_quantized_t5()
summarizer = pipeline("summarization", model=_t5_model, tokenizer=_t5_tok)

# spaCy for hashtag extraction
nlp = spacy.load("en_core_web_sm")

@app.get("/")
async def home():
    return {"message": "LLM API (summarize & hashtags) is running 🚀"}

@app.post("/summarize")
async def summarize(payload: dict):
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(400, detail="'text' is required")
    out = summarizer(text, max_length=60, min_length=20, do_sample=False)
    return {"summary": out[0]["summary_text"]}

@app.post("/hashtags")
async def hashtags(payload: dict):
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(400, detail="'text' is required")

    doc = nlp(text)
    # noun/propn lemmas
    lemmas: List[str] = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN"} and token.is_alpha and not token.is_stop
    ]
    if not lemmas:
        return {"hashtags": ""}

    # top-5
    top5 = [w for w,_ in Counter(lemmas).most_common(5)]
    tags = ["#" + "".join(part.capitalize() for part in lemma.split()) for lemma in top5]
    return {"hashtags": " ".join(tags)}
