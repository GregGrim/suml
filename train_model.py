# SUML Delivery project series
# By Hryhorii Hrymailo s27157
import json

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_NAME = "Iris_Training_Experiment"
MODEL_TAGS = {"version": "v1.0.0", "project": "suml", "dataset": "iris"}
ARTIFACTS_DIR = Path("artifacts")
MODEL_OUTPUT_PATH = Path("app") / "model.joblib"


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _train_evaluate_log(
    model,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_labels: list[str],
) -> Dict[str, Any]:
    """Train one model, log to MLflow, persist artifacts, and return metrics.

    Returns a dict with keys: accuracy, precision, recall, f1, roc_auc (optional),
    model, run_id, artifact_dir.
    """
    # Per-model artifacts dir
    model_art_dir = ARTIFACTS_DIR / model_name
    model_art_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"{model_name}_Iris") as run:
        mlflow.set_tags(MODEL_TAGS)
        mlflow.log_param("model_name", model_name)
        # Log all hyperparameters
        for p_name, p_value in getattr(model, "get_params", lambda: {})().items():
            try:
                mlflow.log_param(p_name, p_value)
            except Exception:
                mlflow.log_param(p_name, str(p_value))

        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("precision_macro", float(precision))
        mlflow.log_metric("recall_macro", float(recall))
        mlflow.log_metric("f1_macro", float(f1))

        # ROC-AUC if possible
        roc_auc = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                # Encode string labels to integers for roc_auc
                le = LabelEncoder().fit(y_train)
                y_test_int = le.transform(y_test)
                roc_auc = roc_auc_score(y_test_int, y_proba, multi_class="ovr", average="macro")
                mlflow.log_metric("roc_auc_macro_ovr", float(roc_auc))
            except Exception:
                pass

        # Artifacts: confusion matrix image
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        cm_img_path = model_art_dir / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(cm, class_labels, cm_img_path)
        mlflow.log_artifact(str(cm_img_path))

        # Classification report text
        report = classification_report(y_test, y_pred, target_names=class_labels)
        report_path = model_art_dir / f"{model_name}_classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(str(report_path))

        # Serialized model file (.joblib)
        model_file_path = model_art_dir / f"{model_name}.joblib"
        joblib.dump(model, model_file_path)
        mlflow.log_artifact(str(model_file_path))

        # MLflow logged model flavor
        try:
            mlflow.sklearn.log_model(model, name="model")
        except Exception:
            pass

        # Console summary per model
        print(f"[{model_name}] Accuracy: {accuracy:.4f}")
        print(f"[{model_name}] Precision (macro): {precision:.4f}")
        print(f"[{model_name}] Recall (macro): {recall:.4f}")
        print(f"[{model_name}] F1 (macro): {f1:.4f}")
        if roc_auc is not None:
            print(f"[{model_name}] ROC-AUC (macro ovr): {roc_auc:.4f}")

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": None if roc_auc is None else float(roc_auc),
            "model": model,
            "run_id": run.info.run_id,
            "artifact_dir": str(model_art_dir),
            "model_file_path": str(model_file_path),
        }


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X: np.ndarray = iris.data
    y_str: np.ndarray = iris.target_names[iris.target]
    class_labels = iris.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_str, test_size=0.2, random_state=42, stratify=y_str
    )

    # Prepare models (four models in the model zoo)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    # Set MLflow experiment once
    mlflow.set_experiment(EXPERIMENT_NAME)

    results: Dict[str, Dict[str, Any]] = {}
    best_name = None
    best_f1 = -1.0
    best_model = None
    best_run_id = None

    for name, model in models.items():
        res = _train_evaluate_log(
            model=model,
            model_name=name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            class_labels=class_labels,
        )
        results[name] = res
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_name = name
            best_model = res["model"]
            best_run_id = res.get("run_id")

    # Save best model metadata JSON for the app
    meta_output_path = Path("app") / "model_meta.json"
    meta_output_path.parent.mkdir(parents=True, exist_ok=True)
    if best_name is None:
        raise RuntimeError("No best model selected; training might have failed.")
    best_res = results[best_name]

    # Register the best model in MLflow Model Registry
    registered_model_name = "IrisModel"
    registered_model_version = None
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        try:
            mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            registered_model_version = mv.version
            print(f"Registered model '{registered_model_name}' as version {registered_model_version} from run {best_run_id}.")
        except Exception as e:
            print(f"WARNING: Failed to register model to MLflow Model Registry: {e}")
    else:
        print("WARNING: best_run_id is missing; cannot register model.")

    best_model_results = {
        "best_model": best_name,
        "metrics": {
            "accuracy": round(float(best_res.get("accuracy", 0.0)), 3),
            "f1_macro": round(float(best_res.get("f1", 0.0)), 3),
        },
        "mlflow_run_id": best_res.get("run_id"),
        "version": MODEL_TAGS.get("version", "v1.0.0"),
        "registered_model": {
            "name": registered_model_name,
            "version": registered_model_version,
        },
    }
    with open(meta_output_path, mode="w", encoding="utf-8") as file:
        json.dump(best_model_results, file, ensure_ascii=False, indent=2)

    # Save the best model for the app
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)

    print("Training for all models completed.")
    print("Summary (F1-macro):")
    for name, res in results.items():
        print(f" - {name}: {res['f1']:.4f}")
    print(f"Best model by F1-macro: {best_name} ({best_f1:.4f})")
    print(f"Best model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
