import pytest
from fastapi.testclient import TestClient


from docqa import app


@pytest.fixture
def client():
    return TestClient(app)