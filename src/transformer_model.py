import numpy as np,torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
MODEL_ID="madhurjindal/autonlp-Gibberish-Detector-492513457"
class TransformerGibberishDetector:
    def __init__(self,model_id=MODEL_ID):
        print("[TRANSFORMER] Loading:",model_id); self.device="cuda" if torch.cuda.is_available() else "cpu"; print("[TRANSFORMER] Device:",self.device)
        self.tokenizer=AutoTokenizer.from_pretrained(model_id); self.model=AutoModelForSequenceClassification.from_pretrained(model_id); self.model.to(self.device).eval()
        print("[TRANSFORMER] Parameters:",sum(p.numel() for p in self.model.parameters())); print("[TRANSFORMER] Labels:",self.model.config.id2label)
    def predict(self,texts,batch_size=32):
        out=[]
        total=(len(texts)+batch_size-1)//batch_size
        for i in range(0,len(texts),batch_size):
            print(f"[TRANSFORMER] Batch {i//batch_size+1}/{total}")
            x=self.tokenizer(texts[i:i+batch_size],padding=True,truncation=True,max_length=256,return_tensors="pt")
            x={k:v.to(self.device) for k,v in x.items()}
            with torch.inference_mode(): out.append(torch.softmax(self.model(**x).logits,dim=-1).cpu().numpy())
        p=np.vstack(out); return p,p.argmax(1)
    def predict_one(self,text):
        p,y=self.predict([text],1); i=int(y[0]); return self.model.config.id2label[i],float(p[0,i])
