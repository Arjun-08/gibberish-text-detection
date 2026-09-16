import math
from collections import Counter
import numpy as np
from .common import LABELS

class CharacterNGramLM:
    def __init__(self,n=3,alpha=.5):
        self.n=n; self.alpha=alpha; self.counts={l:Counter() for l in LABELS}; self.totals={l:0 for l in LABELS}; self.vocab=set()
    def grams(self,text):
        t=str(text).lower(); p="^"*(self.n-1)+t+"$"
        return [p[i:i+self.n] for i in range(len(p)-self.n+1)]
    def fit(self,texts,labels):
        print(f"[STAT] Training character {self.n}-gram models...")
        for x,l in zip(texts,labels):
            g=self.grams(x); self.counts[l].update(g); self.totals[l]+=len(g); self.vocab.update(g)
        print("[STAT] Vocabulary size:",len(self.vocab))
        for l in LABELS: print(f"[STAT] {l}: {self.totals[l]} n-grams")
        return self
    def score(self,text,label):
        g=self.grams(text); den=self.totals[label]+self.alpha*len(self.vocab)
        return sum(math.log((self.counts[label].get(x,0)+self.alpha)/den) for x in g)/max(1,len(g))
    def predict_proba_one(self,text):
        s=np.array([self.score(text,l) for l in LABELS]); s-=s.max(); p=np.exp(s); return p/p.sum()
    def predict_proba(self,texts):
        print(f"[STAT] Predicting {len(texts)} samples...")
        return np.vstack([self.predict_proba_one(x) for x in texts])
