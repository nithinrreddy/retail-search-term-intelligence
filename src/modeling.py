"""Rules baseline -> Bag of Words -> TF-IDF -> Naive Bayes -> Logistic Regression."""

from pathlib import Path
from difflib import SequenceMatcher

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


def df_to_markdown_table(df: pd.DataFrame) -> str:
    # manual markdown table -- avoids needing the optional 'tabulate' package
    header = "| " + " | ".join(df.columns) + " |"
    separator = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, separator] + rows)

from text_cleaning import clean_text

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
LABELS = ["relevant", "uncertain", "unrelated"]
N_FOLDS = 5
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    labels = pd.read_csv(DATA_DIR / "search_term_labels.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    df = labels.merge(products, on="product_id", how="left")

    # model input is query + product context, not just the query -- relevance
    # only makes sense relative to the product
    df["combined_text"] = (
        df["search_term"] + " " + df["product_title"] + " " + df["category"]
    ).apply(clean_text)

    return df


def build_duplicate_groups(texts: list, threshold: float = 0.92) -> np.ndarray:
    # groups near-identical queries so GroupKFold keeps them in the same fold
    n = len(texts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if SequenceMatcher(None, texts[i], texts[j]).ratio() >= threshold:
                union(i, j)

    groups = np.array([find(i) for i in range(n)])
    n_groups = len(set(groups))
    if n_groups < n:
        print(f"  Found {n - n_groups} near-duplicate pair(s), grouped so they share a fold.")
    return groups


def rule_based_predict(search_term: str, product_title: str, category: str) -> str:
    query_words = set(clean_text(search_term).split())
    product_words = set(clean_text(f"{product_title} {category}").split())
    overlap = query_words & product_words

    if not overlap:
        return "unrelated"
    if len(overlap) >= 2:
        return "relevant"
    return "uncertain"


def evaluate(y_true, y_pred, y_proba=None, model_name="model"):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)

    roc_auc = np.nan
    if y_proba is not None:
        y_true_binarized = label_binarize(y_true, classes=LABELS)
        try:
            roc_auc = roc_auc_score(y_true_binarized, y_proba, average="macro", multi_class="ovr")
        except ValueError:
            pass

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    return {
        "model": model_name,
        "precision_macro": round(precision, 3),
        "recall_macro": round(recall, 3),
        "f1_macro": round(f1, 3),
        "mcc": round(mcc, 3),
        "roc_auc_macro": round(roc_auc, 3) if not np.isnan(roc_auc) else "n/a",
        "confusion_matrix": cm,
    }


def print_result(result: dict):
    print(f"\n{result['model']}")
    print(f"  precision (macro): {result['precision_macro']}")
    print(f"  recall    (macro): {result['recall_macro']}")
    print(f"  f1        (macro): {result['f1_macro']}")
    print(f"  MCC:               {result['mcc']}")
    print(f"  ROC-AUC   (macro): {result['roc_auc_macro']}")
    print(f"  confusion matrix (rows=true, cols=pred, order={LABELS}):")
    print(f"  {result['confusion_matrix']}")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    print(f"Loaded {len(df)} labeled rows.")
    print(df["relevance_label"].value_counts().to_string())

    label_encoder = LabelEncoder().fit(LABELS)
    y = df["relevance_label"].values
    texts = df["combined_text"].tolist()

    # duplicate check runs on search_term alone, not combined_text -- rows
    # from the same product share product_title/category, so comparing the
    # combined text groups whole products together and starves folds
    print("\nChecking for near-duplicate queries...")
    search_terms_only = df["search_term"].apply(clean_text).tolist()
    groups = build_duplicate_groups(search_terms_only)
    cv = GroupKFold(n_splits=N_FOLDS)

    results = []

    rule_preds = [
        rule_based_predict(row.search_term, row.product_title, row.category)
        for row in df.itertuples()
    ]
    results.append(evaluate(y, rule_preds, model_name="Rules baseline"))

    vectorizers = {
        "BoW": CountVectorizer(ngram_range=(1, 2)),
        "TF-IDF": TfidfVectorizer(ngram_range=(1, 2)),
    }
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    for vec_name, vectorizer in vectorizers.items():
        for clf_name, classifier in classifiers.items():
            pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
            model_name = f"{vec_name} + {clf_name}"

            y_pred = cross_val_predict(pipeline, texts, y, cv=cv, groups=groups)
            y_proba = cross_val_predict(pipeline, texts, y, cv=cv, groups=groups, method="predict_proba")

            # sklearn sorts classes alphabetically, reindex to our fixed order
            class_order = sorted(set(y))
            proba_df = pd.DataFrame(y_proba, columns=class_order)[LABELS].values

            results.append(evaluate(y, y_pred, proba_df, model_name=model_name))

    for r in results:
        print_result(r)

    comparison_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in results]
    ).sort_values("f1_macro", ascending=False)

    print("\nMODEL COMPARISON (sorted by macro F1)")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    with open(REPORTS_DIR / "model_comparison.md", "w") as f:
        f.write("# Model Comparison\n\n")
        f.write("5-fold cross-validation, grouped so near-duplicate queries never split across folds.\n\n")
        f.write(df_to_markdown_table(comparison_df))
        f.write("\n\n## Notes\n\n")
        f.write(
            "- Small synthetic dataset (108 rows). Numbers show relative ranking, "
            "not real-world performance.\n"
        )
        f.write(
            "- Rules baseline scores highest, but that's partly circular: the synthetic "
            "labels were written using similar word-overlap logic to the baseline. On real "
            "search terms with typos, synonyms, and unfamiliar brand names, that overlap "
            "logic breaks down, which is why the ML stages matter.\n"
        )
        f.write(
            "- TF-IDF + Naive Bayes collapsed to predicting the majority class ('relevant') "
            "for every row (MCC = 0.0). Bag of Words + Naive Bayes did better, since raw "
            "counts pair more naturally with Naive Bayes's assumptions than TF-IDF weights.\n"
        )
    print(f"\nSaved comparison table to {REPORTS_DIR / 'model_comparison.md'}")

    # exclude rules baseline from the save -- dashboard needs a trained model for LIME
    ml_results = comparison_df[comparison_df["model"] != "Rules baseline"]
    best_name = ml_results.iloc[0]["model"]
    print(f"\nBest ML model: {best_name}. Refitting on the full dataset.")

    vec_name, clf_name = [part.strip() for part in best_name.split("+")]
    best_pipeline = Pipeline([
        ("vectorizer", vectorizers[vec_name]),
        ("classifier", classifiers[clf_name]),
    ])
    best_pipeline.fit(texts, y)

    joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    print(f"Saved pipeline to {MODELS_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
