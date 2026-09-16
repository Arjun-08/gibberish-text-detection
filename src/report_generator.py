from pathlib import Path
import json,pandas as pd,matplotlib.pyplot as plt,seaborn as sns
from .common import RESULT_DIR,PLOT_DIR,LABELS
def main():
    print("\n===== REPORT GENERATION ====="); rows=[]
    for p in RESULT_DIR.glob("*.json"):
        o=json.loads(p.read_text(encoding="utf-8")); rows.append({"model":o["model"],**o["metrics"]})
        plt.figure(figsize=(7,6)); sns.heatmap(o["confusion_matrix"],annot=True,fmt="d",xticklabels=LABELS,yticklabels=LABELS)
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(o["model"]); plt.tight_layout(); plt.savefig(PLOT_DIR/f"{o['model']}_confusion_matrix.png",dpi=200); plt.close()
    df=pd.DataFrame(rows); df.to_csv(RESULT_DIR/"model_comparison.csv",index=False)
    md="# Gibberish Detection Benchmark Report\n\n## Results\n\n"+df.to_markdown(index=False)+"\n\n## Metrics\n\nAccuracy, macro/weighted precision, recall and F1, MCC, ROC-AUC, PR-AUC, latency and throughput are reported. Per-class reports and confusion matrices are stored alongside this report.\n\n## Interpretation\n\nMacro F1 should be emphasized because it gives each class equal weight. For RAG ingestion, inspect false positives on technical identifiers, URLs, numbers, code and domain terminology. The external Transformer's published model-card metrics must not be substituted for these common-test-set measurements.\n"
    (RESULT_DIR/"report.md").write_text(md,encoding="utf-8"); print("[REPORT] Saved comparison and report.")
if __name__=="__main__": main()
