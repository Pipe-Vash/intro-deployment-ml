from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

'''def test_null_prediction():
    response = client.post('/v1/prediction', json =
        {
        "opening_gross": 0,
        "screens": 0,
        "production_budget": 0,
        "title_year": 0,
        "aspect_ratio": 0,
        "duration": 0,
        "cast_total_facebook_likes": 0,
        "budget": 0,
        "imdb_score": 0
        })
    assert response.status_code==200
    assert response.json()['worldwide_gross'] == 0'''


def test_random_prediction():
    response = client.post('/v1/prediction', json =
        {
        "opening_gross": -0.7263681741853496,
        "screens": -1.0480550280002798,
        "production_budget": -1.1175815899923822,
        "title_year": 1.0642783300672123,
        "aspect_ratio": 0.7744997058770637,
        "duration": -1.8562183192466573,
        "cast_total_facebook_likes": -1.4954136408656198,
        "budget": -1.068243062268003,
        "imdb_score": -1.949377858583951
        })
    assert response.status_code==200
    assert response.json()['worldwide_gross'] != 0