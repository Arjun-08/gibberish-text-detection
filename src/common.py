from pathlib import Path
import json, numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=ROOT/"data"; ARTIFACT_DIR=ROOT/"artifacts"; RESULT_DIR=ROOT/"results"; PLOT_DIR=RESULT_DIR/"plots"
for p in (DATA_DIR,ARTIFACT_DIR,RESULT_DIR,PLOT_DIR): p.mkdir(parents=True,exist_ok=True)
LABELS=["clean","noise","mild_gibberish","word_salad"]
LABEL_TO_ID={x:i for i,x in enumerate(LABELS)}

def save_json(obj,path):
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2,ensure_ascii=False)

def evaluate_predictions(name,y_true,y_pred,y_proba=None,latency_ms=None,model_size_mb=None):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    pw,rw,fw,_=precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    m={"accuracy":float(accuracy_score(y_true,y_pred)),"precision_macro":float(p),"recall_macro":float(r),"f1_macro":float(f),
       "precision_weighted":float(pw),"recall_weighted":float(rw),"f1_weighted":float(fw),"mcc":float(matthews_corrcoef(y_true,y_pred))}
    if y_proba is not None:
        try: m["roc_auc_ovr_macro"]=float(roc_auc_score(y_true,y_proba,multi_class="ovr",average="macro",labels=list(range(4))))
        except Exception as e: print("[METRICS] ROC-AUC unavailable:",e); m["roc_auc_ovr_macro"]=None
        try: m["pr_auc_ovr_macro"]=float(average_precision_score(np.eye(4)[y_true],y_proba,average="macro"))
        except Exception as e: print("[METRICS] PR-AUC unavailable:",e); m["pr_auc_ovr_macro"]=None
    if latency_ms is not None:
        m["latency_ms_per_sample"]=float(latency_ms); m["throughput_samples_per_sec"]=float(1000/latency_ms) if latency_ms>0 else None
    if model_size_mb is not None: m["model_size_mb"]=float(model_size_mb)
    cm=confusion_matrix(y_true,y_pred,labels=list(range(4)))
    out={"model":name,"metrics":m,"classification_report":classification_report(y_true,y_pred,labels=list(range(4)),target_names=LABELS,output_dict=True,zero_division=0),"confusion_matrix":cm.tolist(),"labels":LABELS}
    save_json(out,RESULT_DIR/f"{name}.json")
    print(f"\n===== {name.upper()} EVALUATION =====")
    for k,v in m.items(): print(f"{k:30s}: {v:.6f}" if isinstance(v,float) else f"{k:30s}: {v}")
    print("\n[METRICS] Per-class report:\n",classification_report(y_true,y_pred,labels=list(range(4)),target_names=LABELS,digits=4,zero_division=0))
    print("[METRICS] Confusion matrix:\n",cm)
    return out
