import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_chat_endpoint_without_model(client):
    # Domyślnie, w środowisku testowym bez załadowanego modelu, API zwraca 500
    res = client.post('/api/chat', json={'message': 'test', 'session_id': 'test_123'})
    assert res.status_code == 500
    assert "error" in res.json
    assert res.json["error"] == "Model not ready."

def test_math_step_no_word(client):
    res = client.post('/api/math_step', json={'word': ''})
    assert res.status_code == 400
    assert "error" in res.json
    assert res.json["error"] == "Brak słowa lub modelu"

def test_train_status_format(client):
    res = client.get('/api/train/status')
    assert res.status_code == 200
    assert "is_training" in res.json
    assert "status" in res.json
    assert "epoch" in res.json["status"]

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
