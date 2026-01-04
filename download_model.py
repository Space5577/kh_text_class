# download_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "kidkidmoon/xlm-r-khmer-news-classification"
SAVE_DIR = "./my_local_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print("✅ Model downloaded and saved locally")