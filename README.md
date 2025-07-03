# Local LLM API: Summarizer & Hashtag Extractor

This project is a lightweight, self‑hosted LLM API offering:
1. **English text summarization** (via a quantized `t5-small`)
2. **English hashtag extraction** (via spaCy noun/proper‑noun extraction)

It’s designed to run entirely on CPU, and it can be deployed on hosting Platforms.

---

## 🚀 Quick Start

1. Clone the repo

```bash
git clone https://github.com/Prajwal18-MD/LLM.git
cd LLM
```
2. Build & run locally with Docker

```bash
docker build -t llm_small .
docker run -p 8000:8000 llm_small
```
---

## 🤝 Contributing

1. Fork the repo and create a branch

2. In app.py, load or define your new pipeline

3. Add any new dependencies to requirements.txt.

4. rewrite the Dockerfile accordingly!

5. Update this README.md with docs and examples.

6. Submit a pull request. We’ll review and merge!

---

## 🌱 Extending the LLM

This is a local LLM setup. You can easily add or swap models:

* New summarizers: try facebook/bart-small-cnn or quantized variants.

* New extractors: use a T5 prompt for entity extraction, keyword generation, or sentiment analysis.

* Any→Any translation: load smaller MarianMT models (e.g. opus-mt-en-ROMANCE) with the same quantization helper.

Just follow the existing patterns:

1. Quantize your model with quantize_model().

2. Wrap it in a FastAPI endpoint.

3. Document usage here in the README.

---

## 📝 License

This project is licensed under the MIT License – see the LICENSE file for details.



