import spacy
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from typing import Dict, List
from collections import Counter

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ——— Helpers for quantized loading ———

def quantize_model(model_id: str):
    # 1. Load in full precision on CPU
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 2. Quantize all Linear layers to int8
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model, tokenizer

# ——— Pipelines ———

_translators: Dict[str, pipeline] = {}
def get_translator(src: str, tgt: str):
    key = f"{src}-{tgt}"
    if key not in _translators:
        # switch to the smaller ROMANCE bundle to cover most European languages:
        model_id = f"Helsinki-NLP/opus-mt-{src}-ROMANCE"
        try:
            model, tok = quantize_model(model_id)
            _translators[key] = pipeline("translation", model=model, tokenizer=tok)
        except Exception as e:
            raise HTTPException(400, detail=f"Could not load translator {src}→{tgt}: {e}")
    return _translators[key]

# Use t5-small (≈240 MB) instead of bart-large-cnn, then quantize
_t5_model, _t5_tok   = quantize_model("t5-small")
summarizer = pipeline("summarization", model=_t5_model, tokenizer=_t5_tok)

# spaCy pipeline for hashtags
nlp = spacy.load("en_core_web_sm")

# ——— Endpoints ———

@app.get("/")
async def home():
    return {"message": "Personalized LLM API is running 🚀"}

@app.post("/translate")
async def translate(payload: dict):
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
    text = payload.get("text","").strip()
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
    lemmas: List[str] = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and token.is_alpha
    ]
    if not lemmas:
        return {"hashtags": ""}

    top5 = [w for w, _ in Counter(lemmas).most_common(5)]
    tags = ["#" + "".join(p.capitalize() for p in lemma.split()) for lemma in top5]
    return {"hashtags": " ".join(tags)}
