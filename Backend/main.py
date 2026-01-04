from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import load_classifier

# -------------------------
# Create FastAPI app
# -------------------------
app = FastAPI(
    title="Khmer News Classification API",
    version="1.0"
)

# -------------------------
# Load model ONCE at startup
# -------------------------
classifier = load_classifier()

# -------------------------
# Request schema
# -------------------------
class TextRequest(BaseModel):
    text: str

# -------------------------
# Root endpoint (optional but useful)
# -------------------------
@app.get("/")
def root():
    return {
        "message": "Khmer News Classification API is running"
    }

# -------------------------
# Health check endpoint
# -------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "xlm-r-khmer-news-classification"
    }

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(request: TextRequest):
    result = classifier(request.text)

    return {
        "label": result[0]["label"],
        "score": float(result[0]["score"])
    }
