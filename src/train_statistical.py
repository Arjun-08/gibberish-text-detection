from pathlib import Path
import time,joblib,pandas as pd
from .common import ARTIFACT_DIR,LABEL_TO_ID,evaluate_predictions
from .statistical_model import CharacterNGramLM
def main():
    print("\n===== MODEL A: FROM-SCRATCH CHARACTER N-GRAM =====")
    b=Path(__file__).resolve().parents[1]/"data"/"benchmark"; tr=pd.read_csv(b/"train.csv"); va=pd.read_csv(b/"val.csv")
    m=CharacterNGramLM(); m.fit(tr.text,tr.label)
    t=time.perf_counter(); p=m.predict_proba(va.text); sec=time.perf_counter()-t
    y=va.label.map(LABEL_TO_ID); pred=p.argmax(1); lat=sec/len(va)*1000
    evaluate_predictions("statistical",y,pred,p,lat)
    joblib.dump(m,ARTIFACT_DIR/"statistical_ngram.joblib"); print("[STAT] Model saved.")
if __name__=="__main__": main()
