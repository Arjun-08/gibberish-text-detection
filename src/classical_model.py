import re,numpy as np
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class ClassicalGibberishClassifier:
    def __init__(self):
        print("[CLASSICAL] Initializing TF-IDF features...")
        self.char=TfidfVectorizer(analyzer="char",ngram_range=(2,5),min_df=2,max_features=60000,sublinear_tf=True)
        self.word=TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True)
        self.scaler=StandardScaler()
        self.clf=LogisticRegression(max_iter=1000,class_weight="balanced")
    def hand(self,texts):
        rows=[]
        for x in texts:
            x=str(x); n=len(x); w=re.findall(r"\b\w+\b",x); a=sum(c.isalpha() for c in x); d=sum(c.isdigit() for c in x); s=sum(c.isspace() for c in x); sp=max(0,n-a-d-s)
            lens=[len(z) for z in w]
            rows.append([np.log1p(n),np.log1p(len(w)),a/max(1,n),d/max(1,n),sp/max(1,n),len(set(x.lower()))/max(1,n),np.mean(lens) if lens else 0,np.mean(np.array(lens)<=1) if lens else 0,1-len(set(z.lower() for z in w))/len(w) if w else 0])
        return np.asarray(rows,dtype="float32")
    def feats(self,texts,fit=False):
        if fit:
            c=self.char.fit_transform(texts); w=self.word.fit_transform(texts); h=self.scaler.fit_transform(self.hand(texts))
        else:
            c=self.char.transform(texts); w=self.word.transform(texts); h=self.scaler.transform(self.hand(texts))
        return hstack([c,w,csr_matrix(h)]).tocsr()
    def fit(self,texts,labels):
        x=self.feats(texts,True); print("[CLASSICAL] Feature matrix:",x.shape,"nonzero:",x.nnz); print("[CLASSICAL] Training Logistic Regression..."); self.clf.fit(x,labels); print("[CLASSICAL] Training complete."); return self
    def predict_proba(self,texts): return self.clf.predict_proba(self.feats(texts))
    def predict(self,texts): return self.clf.predict(self.feats(texts))
