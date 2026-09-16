from pathlib import Path
import time,joblib,numpy as np,pandas as pd
from .common import ARTIFACT_DIR,RESULT_DIR,LABEL_TO_ID,evaluate_predictions
from .transformer_model import TransformerGibberishDetector
from .transformer_labels import normalize_external_label

def main():
    print("\n===== UNIFIED COMMON TEST-SET EVALUATION =====")
    b=Path(__file__).resolve().parents[1]/"data"/"benchmark"; te=pd.read_csv(b/"test.csv"); texts=te.text.tolist(); y=te.label.map(LABEL_TO_ID).values
    print("[EVAL] Test samples:",len(te)); results=[]
    for name,path in [("statistical",ARTIFACT_DIR/"statistical_ngram.joblib"),("classical",ARTIFACT_DIR/"classical_tfidf_logreg.joblib")]:
        print(f"\n[EVAL] Loading {name}...")
        m=joblib.load(path); t=time.perf_counter(); p=m.predict_proba(texts); sec=time.perf_counter()-t; pred=p.argmax(1)
        size=path.stat().st_size/1024**2; results.append(evaluate_predictions(name,y,pred,p,sec/len(te)*1000,size))
    print("\n[EVAL] Loading Transformer...")
    m=TransformerGibberishDetector(); t=time.perf_counter(); p,raw=m.predict(texts); sec=time.perf_counter()-t
    aligned=np.zeros((len(te),4)); pred=[]
    for j in range(p.shape[1]):
        lab=normalize_external_label(m.model.config.id2label[j])
        if lab is None: raise RuntimeError(f"Unmapped Transformer label: {m.model.config.id2label[j]}")
        aligned[:,LABEL_TO_ID[lab]]=p[:,j]
    for i in raw:
        lab=normalize_external_label(m.model.config.id2label[int(i)])
        if lab is None: raise RuntimeError(f"Unmapped Transformer label: {m.model.config.id2label[int(i)]}")
        pred.append(LABEL_TO_ID[lab])
    results.append(evaluate_predictions("transformer",y,np.array(pred),aligned,sec/len(te)*1000))
    rows=[{"model":r["model"],**r["metrics"]} for r in results]; df=pd.DataFrame(rows); df.to_csv(RESULT_DIR/"model_comparison.csv",index=False)
    print("\n===== FINAL COMPARISON ====="); print(df.to_string(index=False))
if __name__=="__main__": main()
