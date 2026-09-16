from pathlib import Path
import time,joblib,pandas as pd
from .classical_model import ClassicalGibberishClassifier
from .common import ARTIFACT_DIR,LABEL_TO_ID,evaluate_predictions
def main():
    print("\n===== MODEL B: CLASSICAL ML =====")
    b=Path(__file__).resolve().parents[1]/"data"/"benchmark"; tr=pd.read_csv(b/"train.csv"); va=pd.read_csv(b/"val.csv")
    m=ClassicalGibberishClassifier().fit(tr.text.tolist(),tr.label.values)
    t=time.perf_counter(); p=m.predict_proba(va.text.tolist()); sec=time.perf_counter()-t; y=va.label.map(LABEL_TO_ID); pred=p.argmax(1); lat=sec/len(va)*1000
    evaluate_predictions("classical",y,pred,p,lat)
    joblib.dump(m,ARTIFACT_DIR/"classical_tfidf_logreg.joblib"); print("[CLASSICAL] Model saved.")
if __name__=="__main__": main()
