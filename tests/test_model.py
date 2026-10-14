import pandas as pd
import pytest

from model import categorize_age, categorize_temperature, create_pipeline, predict_top_k


def test_temperature_boundaries_are_continuous():
    assert categorize_temperature(100.4) == "Rendah"
    assert categorize_temperature(100.5) == "Normal"
    assert categorize_temperature(102.5) == "Normal"
    assert categorize_temperature(102.55) == "Demam ringan"
    assert categorize_temperature(103.5) == "Demam ringan"
    assert categorize_temperature(103.6) == "Demam tinggi"


def test_age_category_depends_on_animal():
    assert categorize_age("sapi", 2) == "Anak"
    assert categorize_age("sapi", 7) == "Dewasa"
    assert categorize_age("kambing", 2) == "Muda"
    assert categorize_age("kambing", 4) == "Dewasa"
    with pytest.raises(ValueError):
        categorize_age("ayam", 2)


def test_pipeline_accepts_indonesian_raw_inputs_and_returns_probabilities():
    rows = pd.DataFrame([
        {"Hewan": "sapi", "Usia": 3, "Suhu": 103.1, "Gejala 1": "depresi", "Gejala 2": "benjolan tanpa rasa sakit", "Gejala 3": "kehilangan nafsu makan", "Penyakit": "pneumonia"},
        {"Hewan": "kerbau", "Usia": 13, "Suhu": 104.5, "Gejala 1": "benjolan tanpa rasa sakit", "Gejala 2": "kehilangan nafsu makan", "Gejala 3": "depresi", "Penyakit": "virus lumpy"},
        {"Hewan": "kambing", "Usia": 10, "Suhu": 100.4, "Gejala 1": "menggigil", "Gejala 2": "pembengkakan pada ekstremitas", "Gejala 3": "sesak napas", "Penyakit": "antraks"},
        {"Hewan": "domba", "Usia": 10, "Suhu": 100.3, "Gejala 1": "depresi", "Gejala 2": "pembengkakan pada tungkai", "Gejala 3": "pincang", "Penyakit": "blackleg"},
        {"Hewan": "kambing", "Usia": 7, "Suhu": 101.2, "Gejala 1": "kesulitan berjalan", "Gejala 2": "luka pada lidah", "Gejala 3": "kehilangan nafsu makan", "Penyakit": "penyakit mulut dan kuku"},
    ])
    model = create_pipeline(random_state=42, n_estimators=10)
    model.fit(rows.drop(columns="Penyakit"), rows["Penyakit"])
    result = predict_top_k(model, rows.drop(columns="Penyakit").iloc[[0]], k=3)
    assert len(result) == 3
    assert result[0][1] >= result[1][1] >= result[2][1]
    assert sum(prob for _, prob in result) <= 1.000001

    # Gejala boleh kurang dari tiga: kolom kosong tidak boleh membuat pipeline gagal.
    partial = rows.drop(columns="Penyakit").iloc[[0]].copy()
    partial["Gejala 2"] = None
    partial["Gejala 3"] = None
    result = predict_top_k(model, partial, k=3)
    assert len(result) == 3
    assert all(0.0 <= prob <= 1.0 for _, prob in result)
