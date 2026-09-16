# Gibberish Detection Benchmark Report

## Results

| model       |   accuracy |   precision_macro |   recall_macro |   f1_macro |   precision_weighted |   recall_weighted |   f1_weighted |      mcc |   roc_auc_ovr_macro |   pr_auc_ovr_macro |   latency_ms_per_sample |   throughput_samples_per_sec |   model_size_mb |
|:------------|-----------:|------------------:|---------------:|-----------:|---------------------:|------------------:|--------------:|---------:|--------------------:|-------------------:|------------------------:|-----------------------------:|----------------:|
| classical   |   0.499333 |          0.434418 |       0.499333 |   0.462018 |             0.434418 |          0.499333 |      0.462018 | 0.335931 |            0.570711 |           0.532236 |                0.315016 |                    3174.44   |         2.24809 |
| statistical |   0.868    |          0.872611 |       0.868    |   0.867184 |             0.872611 |          0.868    |      0.867184 | 0.82631  |            0.965541 |           0.889842 |                0.262515 |                    3809.31   |         0.39829 |
| transformer |   0.539333 |          0.75077  |       0.539333 |   0.471667 |             0.75077  |          0.539333 |      0.471667 | 0.480328 |            0.865112 |           0.672798 |               34.0525   |                      29.3665 |       nan       |

## Metrics

Accuracy, macro/weighted precision, recall and F1, MCC, ROC-AUC, PR-AUC, latency and throughput are reported. Per-class reports and confusion matrices are stored alongside this report.

## Interpretation

Macro F1 should be emphasized because it gives each class equal weight. For RAG ingestion, inspect false positives on technical identifiers, URLs, numbers, code and domain terminology. The external Transformer's published model-card metrics must not be substituted for these common-test-set measurements.
