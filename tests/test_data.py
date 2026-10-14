from pathlib import Path

from model import load_dataset


def test_dataset_is_translated_to_indonesian():
    path = Path(__file__).parents[1] / "data" / "animal_disease_dataset.csv"
    df = load_dataset(path)
    assert list(df.columns) == ["Hewan", "Usia", "Suhu", "Gejala 1", "Gejala 2", "Gejala 3", "Penyakit"]
    assert set(df["Hewan"].unique()) == {"sapi", "kerbau", "domba", "kambing"}
    assert set(df["Penyakit"].unique()) == {"antraks", "blackleg", "penyakit mulut dan kuku", "virus lumpy", "pneumonia"}
    assert len(df) == 43778
