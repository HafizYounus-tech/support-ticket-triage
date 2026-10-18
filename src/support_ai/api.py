import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the trained model once, when the API starts
model = joblib.load("models/triage.joblib")

# Below this confidence, a human should check the routing.
# Chosen from the validation trade-off table in 01_explore.ipynb:
# 77.8% auto-routed at 96.2% accuracy.
CONFIDENCE_THRESHOLD = 0.6

app = FastAPI(title="Support ticket triage")


# Request body: what the client sends
class TicketIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


# Response body: what we send back
class Prediction(BaseModel):
    category: str
    confidence: float
    needs_review: bool


@app.post("/predict")
def predict(ticket: TicketIn) -> Prediction:
    # The model expects a list of texts, so wrap our one text in a list
    all_probs = model.predict_proba([ticket.text])

    # We sent one text, so take the first (and only) row of results
    probs = all_probs[0]

    # Find the position of the highest probability
    best_index = probs.argmax()

    # Look up which category sits at that position
    category = model.classes_[best_index]
    confidence = float(probs[best_index])

    return Prediction(
        category=category,
        confidence=round(confidence, 3),
        needs_review=confidence < CONFIDENCE_THRESHOLD,
    )