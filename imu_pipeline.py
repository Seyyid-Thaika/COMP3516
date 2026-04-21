from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
CSV_PATH = Path("activity_masked.csv")
DATA_DIR = Path("data_sources")
MODEL_PATH = Path("imu_model.joblib")
PREDICTION_CSV_PATH = Path("activity_imu.csv")
METRICS_PATH = Path("imu_metrics.json")

ID_TO_ACTIVITY = {
    1: "sit",
    2: "walk",
    3: "sleep",
    4: "falldown",
    5: "jump",
}


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_sample(filename: str, data_dir: Path = DATA_DIR) -> dict:
    sample_path = data_dir / filename
    with sample_path.open("rb") as f:
        return CompatUnpickler(f).load()


def extract_imu_frames(sample: dict) -> np.ndarray:
    return np.asarray(sample["modality_data"]["imu"][0]["frames"], dtype=np.float32)


def extract_imu_features(frames: np.ndarray) -> np.ndarray:
    """Turn a variable-length IMU sequence into a fixed-length feature vector."""
    frames = np.asarray(frames, dtype=np.float32)
    time_steps, axes, joints = frames.shape

    flat = frames.reshape(time_steps, axes * joints)
    temporal_diff = (
        np.diff(flat, axis=0)
        if time_steps > 1
        else np.zeros((1, axes * joints), dtype=np.float32)
    )
    joint_magnitude = np.linalg.norm(frames, axis=1)

    features: list[float] = []
    feature_blocks = [flat, np.abs(flat), temporal_diff, joint_magnitude]

    for block in feature_blocks:
        features.extend(np.mean(block, axis=0).ravel())
        features.extend(np.std(block, axis=0).ravel())
        features.extend(np.min(block, axis=0).ravel())
        features.extend(np.max(block, axis=0).ravel())
        features.extend(np.percentile(block, 25, axis=0).ravel())
        features.extend(np.percentile(block, 50, axis=0).ravel())
        features.extend(np.percentile(block, 75, axis=0).ravel())

    features.extend(np.mean(flat ** 2, axis=0).ravel())
    features.append(float(time_steps))
    diff_norm = np.linalg.norm(temporal_diff, axis=1)
    features.append(float(np.mean(diff_norm)))
    features.append(float(np.std(diff_norm)))

    # Frequency-domain summaries help separate rhythmic motions like walking/jumping.
    fft_magnitude = np.abs(np.fft.rfft(flat, axis=0))
    fft_magnitude = fft_magnitude[1:] if fft_magnitude.shape[0] > 1 else fft_magnitude
    if fft_magnitude.size:
        features.extend(np.mean(fft_magnitude, axis=0).ravel())
        features.extend(np.std(fft_magnitude, axis=0).ravel())
    else:
        features.extend(np.zeros(flat.shape[1] * 2, dtype=np.float32))

    return np.asarray(features, dtype=np.float32)


@dataclass
class DatasetBundle:
    metadata: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray | None
    user_ids: np.ndarray


def build_dataset(metadata: pd.DataFrame) -> DatasetBundle:
    feature_rows = []
    user_ids = []

    labels = (
        metadata["activity_id"].astype(int).to_numpy()
        if metadata["activity_id"].notna().all()
        else None
    )

    for filename in metadata["filename"]:
        sample = load_sample(filename)
        frames = extract_imu_frames(sample)
        feature_rows.append(extract_imu_features(frames))
        user_ids.append(int(sample["user_id"]))

    return DatasetBundle(
        metadata=metadata.copy(),
        features=np.vstack(feature_rows),
        labels=labels,
        user_ids=np.asarray(user_ids, dtype=np.int32),
    )


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=6000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def evaluate_group_cv(train_bundle: DatasetBundle) -> dict:
    model = build_model()
    gkf = GroupKFold(n_splits=5)

    y_true = train_bundle.labels
    assert y_true is not None
    y_pred = np.zeros_like(y_true)
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(
        gkf.split(train_bundle.features, y_true, train_bundle.user_ids), start=1
    ):
        model.fit(train_bundle.features[train_idx], y_true[train_idx])
        pred = model.predict(train_bundle.features[valid_idx])
        y_pred[valid_idx] = pred
        acc = accuracy_score(y_true[valid_idx], pred)
        fold_scores.append(acc)
        print(f"Fold {fold_idx} accuracy: {acc:.4f}")

    mean_accuracy = float(np.mean(fold_scores))
    print(f"Mean grouped CV accuracy: {mean_accuracy:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

    return {
        "fold_accuracies": fold_scores,
        "mean_grouped_accuracy": mean_accuracy,
        "classification_report": classification_report(
            y_true, y_pred, digits=4, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_full_model(train_bundle: DatasetBundle) -> Pipeline:
    model = build_model()
    assert train_bundle.labels is not None
    model.fit(train_bundle.features, train_bundle.labels)
    return model


def write_prediction_csv(
    full_df: pd.DataFrame, test_mask: pd.Series, predicted_ids: np.ndarray
) -> pd.DataFrame:
    output_df = full_df.copy()
    output_df.loc[test_mask, "activity_id"] = predicted_ids
    output_df.loc[test_mask, "activity"] = [
        ID_TO_ACTIVITY[int(label_id)] for label_id in predicted_ids
    ]
    output_df["activity_id"] = output_df["activity_id"].astype("Int64")
    output_df.to_csv(PREDICTION_CSV_PATH, index=False)
    return output_df


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    train_df = df.dropna(subset=["activity_id"]).copy()
    test_mask = df["activity_id"].isna()
    test_df = df.loc[test_mask].copy()

    print(f"Training rows: {len(train_df)}")
    print(f"Masked test rows: {len(test_df)}")

    train_bundle = build_dataset(train_df)
    test_bundle = build_dataset(test_df)

    metrics = evaluate_group_cv(train_bundle)

    model = train_full_model(train_bundle)
    predictions = model.predict(test_bundle.features)

    joblib.dump(model, MODEL_PATH)
    write_prediction_csv(df, test_mask, predictions)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved submission CSV to {PREDICTION_CSV_PATH}")


if __name__ == "__main__":
    main()
