# SEG: calibration of baseline probabilities (leave-one-well-out) and sensitivity of the HMM/Viterbi gain to sample spacing.
# Needs facies_data.csv (SEG contest file) in the working directory, with an added ILD_log10 column (log10 of ILD).
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
RS=42
df=pd.read_csv('facies_data.csv'); df.columns=[c.strip() for c in df.columns]
F=['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS']
df=df.sort_values(['Well Name','Depth']).reset_index(drop=True)
X=df[F]; y=df['Facies'].values-1; g=df['Well Name'].values; K=9
def trans(sub):
    c=np.full((K,K),1.0); ini=np.full(K,1.0)
    for w,gg in sub.groupby('Well Name'):
        s=gg['Facies'].values-1; ini[s[0]]+=1
        for a,b in zip(s[:-1],s[1:]): c[a,b]+=1
    return c/c.sum(1,keepdims=True), ini/ini.sum()
def vit(E,T,I,eps=1e-12):
    n=len(E); lt=np.log(T+eps); ld=np.zeros((n,K)); bp=np.zeros((n,K),int); ld[0]=np.log(I+eps)+np.log(E[0]+eps)
    for t in range(1,n):
        s=ld[t-1][:,None]+lt; bp[t]=s.argmax(0); ld[t]=s.max(0)+np.log(E[t]+eps)
    p=np.zeros(n,int); p[-1]=ld[-1].argmax()
    for t in range(n-2,-1,-1): p[t]=bp[t+1,p[t+1]]
    return p
P=np.zeros((len(y),K)); res={1:[], 2:[], 3:[]}
for tr,te in LeaveOneGroupOut().split(X,y,g):
    sw=np.sqrt(compute_sample_weight('balanced',y[tr]))
    m=xgb.XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.1,random_state=RS,eval_metric='mlogloss'); m.fit(X.iloc[tr],y[tr],sample_weight=sw)
    E=m.predict_proba(X.iloc[te]); P[te]=E
    for st in (1,2,3):
        # subsample every st-th sample in test well AND training wells (coarser spacing); transition matrix at same stride
        idx=np.arange(0,len(te),st); sub=df.iloc[tr]; trs=sub[(sub.groupby('Well Name').cumcount()%st)==0]
        T,I=trans(trs); hp=vit(E[idx],T,I); yt=y[te][idx]; bp=E[idx].argmax(1)
        res[st].append((len(idx),(bp==yt).sum(),(hp==yt).sum()))
pred=P.argmax(1); conf=P.max(1); corr=(pred==y).astype(float)
b=np.linspace(0,1,11); ece=0
for i in range(10):
    mk=(conf>b[i])&(conf<=b[i+1])
    if mk.sum(): ece+=mk.mean()*abs(corr[mk].mean()-conf[mk].mean())
print(f"SEG LOWO calibration: accuracy {corr.mean():.4f}, mean confidence {conf.mean():.3f}, ECE {ece:.4f}, Brier {np.mean(np.sum((P-np.eye(K)[y])**2,1)):.4f}")
for st in (1,2,3):
    a=np.array(res[st]); n=a[:,0].sum(); print(f"stride {st} (spacing {0.5*st} ft): n={n}, baseline acc {a[:,1].sum()/n:.4f}, HMM acc {a[:,2].sum()/n:.4f}, gain {(a[:,2].sum()-a[:,1].sum())/n:+.4f}, wells improved {(a[:,2]>a[:,1]).sum()}/8")
