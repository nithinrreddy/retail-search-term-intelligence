# Model Comparison

5-fold cross-validation, grouped so near-duplicate queries never split across folds.

| model | precision_macro | recall_macro | f1_macro | mcc | roc_auc_macro |
|---|---|---|---|---|---|
| Rules baseline | 0.769 | 0.773 | 0.769 | 0.688 | n/a |
| BoW + Logistic Regression | 0.637 | 0.63 | 0.629 | 0.491 | 0.772 |
| TF-IDF + Logistic Regression | 0.341 | 0.345 | 0.335 | 0.037 | 0.538 |
| BoW + Naive Bayes | 0.226 | 0.299 | 0.229 | -0.078 | 0.492 |
| TF-IDF + Naive Bayes | 0.234 | 0.336 | 0.223 | 0.019 | 0.419 |

## Notes

- Small synthetic dataset (108 rows). Numbers show relative ranking, not real-world performance.
- Rules baseline scores highest, but that's partly circular: the synthetic labels were written using similar word-overlap logic to the baseline. On real search terms with typos, synonyms, and unfamiliar brand names, that overlap logic breaks down, which is why the ML stages matter.
- TF-IDF + Naive Bayes collapsed to predicting the majority class ('relevant') for every row (MCC = 0.0). Bag of Words + Naive Bayes did better, since raw counts pair more naturally with Naive Bayes's assumptions than TF-IDF weights.
