"""FastAPI endpoints for the ticket-routing demo."""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.model import classify_text, get_model

app = FastAPI(title="NLP Ticket Router", version="0.1.0")


class TicketRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/categories")
def categories():
    return {"categories": get_model().categories}


@app.post("/classify")
def classify(request: TicketRequest):
    if not request.text.strip():
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Message cannot be whitespace only")
    return classify_text(request.text)
