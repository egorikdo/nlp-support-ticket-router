from fastapi.testclient import TestClient
from app.api import app
from app.model import classify_text, evaluate

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_categories():
    result = client.get("/categories")
    assert result.status_code == 200
    assert len(result.json()["categories"]) == 5


def test_classify():
    response = client.post("/classify", json={"text": "Курьер задерживается с доставкой"})
    assert response.status_code == 200
    result = response.json()
    assert result["category"] in client.get("/categories").json()["categories"]
    assert 0 <= result["score"] <= 1


def test_blank_input():
    response = client.post("/classify", json={"text": "  "})
    assert response.status_code == 422


def test_eval():
    metrics = evaluate()
    assert metrics["test_size"] > 0
    assert 0 <= metrics["accuracy"] <= 1


def test_empty_text_direct():
    import pytest
    with pytest.raises(ValueError):
        classify_text(" ")
