from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from model import DISEASE_MAP

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_celsius_conversion_and_disease_information():
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    temperature = next(widget for widget in app.number_input if widget.label == "Suhu tubuh (°C)")
    temperature.set_value(39.0)
    for disease in DISEASE_MAP.values():
        with patch("model.predict_top_k", return_value=[(disease, 1.0)]) as predict:
            app.button[0].click().run()
        assert not app.exception
        assert not app.error
        assert predict.call_args.args[1]["Suhu"].iloc[0] == pytest.approx(102.2)
        assert any(disease.title() in expander.label for expander in app.expander)
        assert any(item.value == "**Upaya penanganan awal**" for item in app.markdown)
        assert not any("Informasi belum tersedia" in item.value for item in app.markdown)


def test_partial_symptoms_are_padded_with_none():
    app = AppTest.from_file(str(APP)).run()
    app.multiselect[0].set_value(["pincang", "menggigil"])
    with patch("model.predict_top_k", return_value=[("antraks", 1.0)]) as predict:
        app.button[0].click().run()
    assert not app.exception
    assert not app.error
    row = predict.call_args.args[1].iloc[0]
    assert row["Gejala 1"] == "pincang"
    assert row["Gejala 2"] == "menggigil"
    assert row["Gejala 3"] is None


def test_empty_symptoms_show_error():
    app = AppTest.from_file(str(APP)).run()
    app.multiselect[0].set_value([])
    app.button[0].click().run()
    assert app.error
