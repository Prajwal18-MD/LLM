# model_loader.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download

def main():
    # 1. Choose your model ID
    model_id = "google/flan-t5-small"

    print(f"Downloading model `{model_id}`…")
    # 2. Download model files to local cache
    snapshot_download(repo_id=model_id)

    print("Loading tokenizer and model into memory…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # 3. Test with a quick prompt
    prompt = "Translate to French: Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("✅ Model loaded and ran a test generation!")
    print("Test output:", result)

if __name__ == "__main__":
    main()
