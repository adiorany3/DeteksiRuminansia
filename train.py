from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from model import engineer_features

from model import FEATURES, SYMPTOM_COLUMNS, create_pipeline, load_dataset

ROOT = Path(__file__).parent
DATASET = ROOT / "data" / "animal_disease_dataset.csv"
MODEL_FILE = ROOT / "model.joblib"


def augment_partial_symptoms(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Tambahkan versi 2-gejala dan 1-gejala agar model belajar masukan tidak lengkap."""
    partials = []
    for keep in (2, 1):
        partial = x.copy()
        for column in SYMPTOM_COLUMNS[keep:]:
            partial[column] = None
        partials.append(partial)
    return pd.concat([x, *partials], ignore_index=True), pd.concat([y] * 3, ignore_index=True)

def main() -> None:
    data = load_dataset(DATASET)
    groups = pd.util.hash_pandas_object(engineer_features(data[FEATURES]), index=False)
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(data, groups=groups))
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43).split(
        data.iloc[development], groups=groups.iloc[development],
    ))
    train, validation = development[train], development[validation]
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[validation])
    assert set(groups.iloc[development]).isdisjoint(groups.iloc[test])
    x_train, y_train = data.iloc[train][FEATURES], data.iloc[train]["Penyakit"]
    # Evaluasi pada data uji yang tidak pernah dilihat model.
    model = create_pipeline()
    model.fit(*augment_partial_symptoms(x_train, y_train))
    for name, indices in (("Validasi", validation), ("Uji", test)):
        for count in (1, 2, 3):
            inputs = data.iloc[indices][FEATURES].copy()
            inputs[SYMPTOM_COLUMNS[count:]] = None
            actual = data.iloc[indices]["Penyakit"]
            prediction = model.predict(inputs)
            print(f"{name}, {count} gejala: {accuracy_score(actual, prediction):.4f}")
            print(classification_report(actual, prediction, zero_division=0))
            for animal in inputs["Hewan"].unique():
                mask = inputs["Hewan"].eq(animal)
                print(animal, int(mask.sum()), accuracy_score(actual[mask], prediction[mask]))

    # Latih ulang pada seluruh data untuk artefak produksi (evaluasi sudah selesai).
    final_model = create_pipeline()
    final_model.fit(*augment_partial_symptoms(data[FEATURES], data["Penyakit"]))
    joblib.dump(final_model, MODEL_FILE, compress=3)
    print(f"Model tersimpan: {MODEL_FILE}")


if __name__ == "__main__":
    main()
