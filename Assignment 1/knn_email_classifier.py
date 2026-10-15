"""Train and evaluate KNN using only the first 1,000 CSV data rows.

Run: python knn_email_classifier.py
Dependencies: pandas, scikit-learn, joblib
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("Phishing_Email.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("knn_results"))
    args = parser.parse_args()

    # The row limit applies before cleaning; no later rows replace missing texts.
    data = pd.read_csv(args.csv, nrows=1000, keep_default_na=False)
    required = {"Email Text", "Email Type"}
    if not required.issubset(data.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")
    loaded_rows = len(data)
    data["source_row"] = range(1, loaded_rows + 1)
    data["Email Text"] = data["Email Text"].str.strip()
    usable = data["Email Text"].ne("") & data["Email Type"].ne("")
    data = data.loc[usable].copy()
    text = data["Email Text"]
    labels = data["Email Type"]

    # Keep identical texts in one partition to prevent train/test contamination.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, test_indices = next(splitter.split(text, labels, groups=text))
    train = data.iloc[train_indices]
    test = data.iloc[test_indices]
    assert set(train["Email Text"]).isdisjoint(test["Email Text"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000,
                                  ngram_range=(1, 2), sublinear_tf=True)),
        ("knn", KNeighborsClassifier(n_neighbors=5, metric="cosine",
                                    algorithm="brute", weights="distance")),
    ])


    # Fit vocabulary and IDF exclusively on training texts.
    # Fit and make predictions
    model.fit(train["Email Text"], train["Email Type"])
    predictions = model.predict(test["Email Text"])
    classes = model.classes_.tolist()
    accuracy = accuracy_score(test["Email Type"], predictions)
    report = classification_report(test["Email Type"], predictions,
                                   labels=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(test["Email Type"], predictions, labels=classes)
    summary = {
        "source_file": str(args.csv.resolve()),
        "row_limit": 1000,
        "loaded_rows": loaded_rows,
        "excluded_missing_rows": loaded_rows - len(data),
        "usable_rows": len(data),
        "training_rows": len(train),
        "test_rows": len(test),
        "class_counts": labels.value_counts().to_dict(),
        "split": "First fold of shuffled 5-fold StratifiedGroupKFold; identical texts grouped",
        "random_state": 42,
        "n_neighbors": 5,
        "metric": "cosine",
        "weights": "distance",
        "features": "TF-IDF word unigrams and bigrams, English stopwords removed, max 10000 features",
        "fitted_feature_count": len(model.named_steps["tfidf"].vocabulary_),
        "accuracy": accuracy,
        "majority_class_baseline_accuracy": float(
            test["Email Type"].eq(train["Email Type"].mode().iloc[0]).mean()),
        "classification_report": report,
        "confusion_matrix_labels": classes,
        "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
        "saved_model_training_rows": len(train),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    result = test[["source_row", "Email Type"]].copy()
    result = result.rename(columns={"Email Type": "actual_label"})
    result["predicted_label"] = predictions
    result["correct"] = result["actual_label"].eq(result["predicted_label"])
    result.sort_values("source_row").to_csv(args.output_dir / "test_predictions.csv", index=False)
    joblib.dump(model, args.output_dir / "knn_pipeline.joblib")

    print(f"Loaded {loaded_rows} rows; excluded {loaded_rows - len(data)} missing rows.")
    print(f"Training rows: {len(train)}; test rows: {len(test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(test["Email Type"], predictions, labels=classes,
                                digits=3, zero_division=0))
    print("Confusion matrix (rows=actual, columns=predicted):", classes)
    print(matrix)
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
