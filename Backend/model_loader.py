from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_PATH = "../my_local_model"

def load_classifier():
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("🔄 Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    print("✅ Model loaded successfully")

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True
    )

    return classifier
