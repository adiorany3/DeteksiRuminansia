from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

COLUMN_MAP = {
    "Animal": "Hewan", "Age": "Usia", "Temperature": "Suhu",
    "Symptom 1": "Gejala 1", "Symptom 2": "Gejala 2", "Symptom 3": "Gejala 3",
    "Disease": "Penyakit",
}
ANIMAL_MAP = {"cow": "sapi", "buffalo": "kerbau", "sheep": "domba", "goat": "kambing"}
DISEASE_MAP = {
    "anthrax": "antraks", "blackleg": "blackleg",
    "foot and mouth": "penyakit mulut dan kuku", "lumpy virus": "virus lumpy",
    "pneumonia": "pneumonia",
}
SYMPTOM_MAP = {
    "blisters on gums": "lepuh pada gusi", "blisters on hooves": "lepuh pada kuku",
    "blisters on mouth": "lepuh pada mulut", "blisters on tongue": "lepuh pada lidah",
    "chest discomfort": "rasa tidak nyaman di dada", "chills": "menggigil",
    "crackling sound": "suara berderak", "depression": "depresi",
    "difficulty walking": "kesulitan berjalan", "fatigue": "kelelahan",
    "lameness": "pincang", "loss of appetite": "kehilangan nafsu makan",
    "painless lumps": "benjolan tanpa rasa sakit", "shortness of breath": "sesak napas",
    "sores on gums": "luka pada gusi", "sores on hooves": "luka pada kuku",
    "sores on mouth": "luka pada mulut", "sores on tongue": "luka pada lidah",
    "sweats": "berkeringat", "swelling in abdomen": "pembengkakan pada perut",
    "swelling in extremities": "pembengkakan pada ekstremitas",
    "swelling in limb": "pembengkakan pada tungkai",
    "swelling in muscle": "pembengkakan pada otot", "swelling in neck": "pembengkakan pada leher",
}
FEATURES = ["Hewan", "Usia", "Suhu", "Gejala 1", "Gejala 2", "Gejala 3"]
SYMPTOM_COLUMNS = ["Gejala 1", "Gejala 2", "Gejala 3"]
SYMPTOMS = sorted(SYMPTOM_MAP.values())


def categorize_temperature(value: float) -> str:
    value = float(value)
    if value < 100.5:
        return "Rendah"
    if value <= 102.5:
        return "Normal"
    if value <= 103.5:
        return "Demam ringan"
    return "Demam tinggi"


def categorize_age(animal: str, age: int) -> str:
    age = int(age)
    if animal in {"sapi", "kerbau"}:
        if age <= 2: return "Anak"
        if age <= 6: return "Muda"
        if age <= 10: return "Dewasa"
        return "Senior"
    if animal in {"domba", "kambing"}:
        if age <= 1: return "Anak"
        if age <= 3: return "Muda"
        if age <= 7: return "Dewasa"
        return "Senior"
    raise ValueError(f"Jenis hewan tidak dikenali: {animal}")


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["Kelompok Usia"] = [categorize_age(a, u) for a, u in zip(result["Hewan"], result["Usia"])]
    result["Kategori Suhu"] = result["Suhu"].map(categorize_temperature)
    # Multi-hot: gejala boleh 1-3; kolom kosong (None/NaN) berarti gejala tidak diisi.
    for symptom in SYMPTOMS:
        result[f"Ada: {symptom}"] = result[SYMPTOM_COLUMNS].eq(symptom).any(axis=1).astype(int)
    return result.drop(columns=["Usia", "Suhu", *SYMPTOM_COLUMNS])


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns=COLUMN_MAP)
    frame["Hewan"] = frame["Hewan"].map(ANIMAL_MAP)
    frame["Penyakit"] = frame["Penyakit"].map(DISEASE_MAP)
    for column in SYMPTOM_COLUMNS:
        frame[column] = frame[column].map(SYMPTOM_MAP)
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Terjemahan tidak lengkap pada kolom: {missing}")
    return frame


def create_pipeline(random_state: int = 42, n_estimators: int = 300) -> Pipeline:
    # min_samples_leaf=1 terukur lebih akurat (0.834 vs 0.831); plafon representasi 0.840.
    categorical = ["Hewan", "Kelompok Usia", "Kategori Suhu"]
    return Pipeline([
        ("fitur", FunctionTransformer(engineer_features, validate=False)),
        ("kategori", ColumnTransformer(
            [("onehot", OneHotEncoder(handle_unknown="ignore"), categorical)],
            remainder="passthrough",
        )),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced", random_state=random_state,
            n_jobs=-1, min_samples_leaf=1,
        )),
    ])


def predict_top_k(model: Pipeline, input_data: pd.DataFrame, k: int = 3) -> list[tuple[str, float]]:
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    return [(str(label), float(probability)) for label, probability in ranked[:k]]
