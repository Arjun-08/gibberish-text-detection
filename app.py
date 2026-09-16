from pathlib import Path
import time,joblib,numpy as np,pandas as pd,streamlit as st
from src.common import LABELS,LABEL_TO_ID
from src.transformer_model import TransformerGibberishDetector
from src.transformer_labels import normalize_external_label
ROOT=Path(__file__).resolve().parent; A=ROOT/"artifacts"; R=ROOT/"results"
st.set_page_config(page_title="Gibberish Detection Benchmark",layout="wide")
st.title("Gibberish Detection Benchmark")
@st.cache_resource
def stat(): print("[APP] Loading Model A"); return joblib.load(A/"statistical_ngram.joblib")
@st.cache_resource
def classical(): print("[APP] Loading Model B"); return joblib.load(A/"classical_tfidf_logreg.joblib")
@st.cache_resource
def transformer(): print("[APP] Loading Model C"); return TransformerGibberishDetector()
tab1,tab2,tab3=st.tabs(["Live Detection","Benchmark","RAG Quality Gate"])
with tab1:
    text=st.text_area("Enter text","The model achieved 92% accuracy on the validation dataset.",height=160)
    if st.button("Run all models",type="primary"):
        rows=[]
        m=stat(); t=time.perf_counter(); p=m.predict_proba([text])[0]; lat=(time.perf_counter()-t)*1000; i=p.argmax(); rows.append(["From-scratch n-gram",LABELS[i],p[i],lat,1-p[0]])
        m=classical(); t=time.perf_counter(); p=m.predict_proba([text])[0]; lat=(time.perf_counter()-t)*1000; i=p.argmax(); rows.append(["TF-IDF + Logistic Regression",LABELS[i],p[i],lat,1-p[0]])
        m=transformer(); t=time.perf_counter(); lab,score=m.predict_one(text); lat=(time.perf_counter()-t)*1000; rows.append(["Open-source Transformer",normalize_external_label(lab) or lab,score,lat,None])
        df=pd.DataFrame(rows,columns=["Model","Class","Confidence","Latency (ms)","Gibberish score"]); st.dataframe(df,use_container_width=True,hide_index=True)
        votes=sum(x[1] in {"noise","mild_gibberish","word_salad"} for x in rows)
        if votes>=2: st.error("Majority decision: suspicious / gibberish-like text.")
        elif votes==1: st.warning("Models disagree. Consider an alternate parser/OCR pass.")
        else: st.success("All models classify the text as clean.")
with tab2:
    p=R/"model_comparison.csv"
    if p.exists():
        df=pd.read_csv(p); st.dataframe(df,use_container_width=True,hide_index=True); st.bar_chart(df.set_index("model")["f1_macro"])
        for name in df.model:
            img=R/"plots"/f"{name}_confusion_matrix.png"
            if img.exists(): st.image(str(img),caption=f"{name} confusion matrix")
    else: st.info("Run python run_pipeline.py first.")
with tab3:
    st.write("Use this after PDF parsing/OCR. Suspicious extraction can be sent through another parser before chunking.")
    threshold=st.slider("Suspicion threshold",.50,.99,.80,.01); x=st.text_area("Extracted PDF text","The rnodel achleved 92% accuracy on HG001.",height=120)
    if st.button("Check extraction quality"):
        p=stat().predict_proba([x])[0]; score=1-p[LABEL_TO_ID["clean"]]; st.metric("Suspicious score",f"{score:.4f}")
        if score>=threshold: st.error("QUALITY GATE FAILED: retry extraction with another parser/OCR.")
        else: st.success("QUALITY GATE PASSED: continue to chunking.")
