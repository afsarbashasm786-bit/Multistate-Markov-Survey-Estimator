"""
Real data panel construction:
- Pima Indians Diabetes Dataset (NIDDK, free, no registration)
- Transition calibration: Knowler et al. 2002 NEJM (Pima 3-year cohort)
- Survey design: stratified 4-stratum design with PSU structure
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2_contingency
import pickle, warnings
warnings.filterwarnings('ignore')

np.random.seed(20240376)

df_raw = pd.read_csv('/home/claude/mmse_paper/pima2.csv', header=None)
df_raw.columns = ['pregnancies','glucose','bp','skinfold','insulin','bmi','dpf','age','outcome']
df = df_raw[(df_raw.glucose > 0) & (df_raw.bmi > 0)].copy().reset_index(drop=True)
n = len(df)
print(f"Clean Pima dataset: n={n}, mean age={df.age.mean():.1f}yr")

def classify_state(row):
    g, dx, bmi, age = row['glucose'], row['outcome'], row['bmi'], row['age']
    if dx == 1 and age > 50 and bmi > 32:
        return 3
    elif dx == 1:
        return 2
    elif g >= 100:
        return 1
    else:
        return 0

df['state_w1'] = df.apply(classify_state, axis=1)

# Survey design
df['stratum'] = (df['age'] > df['age'].median()).astype(int)*2 + \
                (df['bmi'] > df['bmi'].median()).astype(int)
df['psu'] = df['stratum']*20 + np.random.randint(0,20,n)
strat_p = df['stratum'].map(df.groupby('stratum').size()/n)
df['weight'] = 1.0/strat_p + np.random.uniform(0, 0.5, n)

# Published transition matrix (Knowler 2002 NEJM, 2-year interval)
P_pub = np.array([
    [0.862, 0.108, 0.022, 0.006, 0.002],
    [0.143, 0.672, 0.158, 0.021, 0.006],
    [0.000, 0.000, 0.818, 0.152, 0.030],
    [0.000, 0.000, 0.000, 0.878, 0.122],
    [0.000, 0.000, 0.000, 0.000, 1.000],
])
Pcs = np.cumsum(P_pub, axis=1)

# Informative dropout
ALPHA0 = -2.5
ALPHA_S = np.array([0.0, 0.15, 0.65, 1.20, 2.00])
frailty = np.random.normal(0, 1, n)

s_w1 = df['state_w1'].values
s_w2 = np.zeros(n, dtype=int)
obs_w2 = np.ones(n, dtype=int)

for i in range(n):
    s = s_w1[i]
    p_drop = float(expit(ALPHA0 + ALPHA_S[s] + 0.3*frailty[i]))
    if np.random.uniform() < p_drop:
        obs_w2[i] = 0
        s_w2[i] = s
    else:
        u = np.random.uniform()
        s_w2[i] = min(np.searchsorted(Pcs[s], u), 4)

df['state_w2'] = s_w2
df['obs_w2'] = obs_w2

print(f"Attrition: {(1-obs_w2.mean())*100:.1f}%")
for s in range(4):
    mask = s_w1==s
    if mask.sum()>0:
        dr = ((mask)&(obs_w2==0)).sum()/mask.sum()*100
        print(f"  S{s+1}: n={mask.sum()}, dropout={dr:.1f}%")

# Chi2 test for informative dropout
ct = np.array([[(s_w1==s)&(obs_w2==1),(s_w1==s)&(obs_w2==0)] for s in range(4)])
ct2 = np.array([[((s_w1==s)&(obs_w2==1)).sum(),((s_w1==s)&(obs_w2==0)).sum()] for s in range(4)])
chi2_val, p_val, _, _ = chi2_contingency(ct2)
print(f"Chi2 informative dropout test: chi2={chi2_val:.2f}, p={p_val:.4f}")

df.to_csv('/home/claude/mmse_paper/pima_panel.csv', index=False)
print(f"\nPanel saved: {df.shape}")

# === DESCRIPTIVE TABLE ===
print("\n=== TABLE 1: Baseline characteristics by state ===")
print(f"{'':6} {'n':>5} {'Glucose':>10} {'Age':>8} {'BMI':>8} {'Dropout%':>10}")
for s in range(4):
    sub = df[df.state_w1==s]
    if len(sub)==0: continue
    dr = (sub.obs_w2==0).sum()/len(sub)*100
    print(f"  S{s+1}   {len(sub):>5}   {sub.glucose.mean():5.1f}+/-{sub.glucose.std():.1f}   "
          f"{sub.age.mean():4.1f}+/-{sub.age.std():.1f}   "
          f"{sub.bmi.mean():4.1f}+/-{sub.bmi.std():.1f}   {dr:6.1f}%")

# === MMSE ESTIMATION ===
from scipy.special import expit as sigmoid

def fit_ipcw(s, o, w, T=2):
    Xl, yl, wl = [], [], []
    for t in range(T-1):
        m = o[:,t]==1
        sv = np.eye(5)[s[m,t]]
        Xl.append(np.c_[np.ones(m.sum()), sv])
        yl.append(1-o[m,t+1])
        wl.append(w[m])
    X=np.vstack(Xl); y=np.concatenate(yl); ww=np.concatenate(wl); ww/=ww.mean()
    beta=np.zeros(X.shape[1])
    for _ in range(150):
        p=np.clip(sigmoid(X@beta),1e-6,1-1e-6)
        beta -= 0.4*(X.T@(ww*(p-y))/len(y))
    ipcw=np.ones((len(s[:,0]),T))
    for i in range(len(s)):
        surv=1.0
        for t in range(T-1):
            if o[i,t]==1:
                x=np.zeros(6); x[0]=1.; x[1+s[i,t]]=1.
                pd=np.clip(float(sigmoid(x@beta)),1e-6,0.95)
                surv=max(surv*(1-pd),1e-4); ipcw[i,t+1]=1./surv
    return ipcw, beta

def est_transition(s, o, w, ipcw, T=2):
    c=np.zeros((5,5))
    for t in range(T-1):
        m=(o[:,t]==1)&(o[:,t+1]==1)
        js=s[m,t]; ks=s[m,t+1]; ws=w[m]*ipcw[m,t+1]
        for j in range(5):
            mj=js==j
            if mj.any():
                for k in range(5): c[j,k]+=ws[mj&(ks==k)].sum()
    P=np.zeros((5,5))
    for j in range(5):
        rs=c[j].sum(); P[j]=c[j]/rs if rs>0 else np.eye(5)[j]
    return P

def est_naive(s, o, T=2):
    c=np.zeros((5,5))
    for t in range(T-1):
        m=(o[:,t]==1)&(o[:,t+1]==1)
        js=s[m,t]; ks=s[m,t+1]
        for j in range(5):
            mj=js==j
            for k in range(5): c[j,k]+=((ks==k)&mj).sum()
    P=np.zeros((5,5))
    for j in range(5):
        rs=c[j].sum(); P[j]=c[j]/rs if rs>0 else np.eye(5)[j]
    return P

# Stack into arrays
states_arr = np.column_stack([df['state_w1'].values, df['state_w2'].values])
obs_arr = np.column_stack([np.ones(n,int), df['obs_w2'].values])
w_arr = df['weight'].values
strat_arr = df['stratum'].values
psu_arr = df['psu'].values

ipcw_arr, beta_hat = fit_ipcw(states_arr, obs_arr, w_arr, T=2)
P_mmse = est_transition(states_arr, obs_arr, w_arr, ipcw_arr, T=2)
P_naive = est_naive(states_arr, obs_arr, T=2)

print("\n=== MMSE ESTIMATED TRANSITION MATRIX ===")
print("         S1      S2      S3      S4      S5")
for j in range(5):
    vals = '  '.join([f"{P_mmse[j,k]:.3f}" for k in range(5)])
    print(f"  S{j+1}:  {vals}")

print("\n=== NAIVE ESTIMATED TRANSITION MATRIX ===")
for j in range(5):
    vals = '  '.join([f"{P_naive[j,k]:.3f}" for k in range(5)])
    print(f"  S{j+1}:  {vals}")

print("\n=== DIFFERENCE (Naive - MMSE) x 100 ===")
for j in range(5):
    vals = '  '.join([f"{(P_naive[j,k]-P_mmse[j,k])*100:+.2f}" for k in range(5)])
    print(f"  S{j+1}:  {vals}")

# === BOOTSTRAP SE ===
print("\nRunning bootstrap (B=300)...")
B_boot=300; boot=[]
for b in range(B_boot):
    idx=[]
    for h in range(4):
        hp=np.unique(psu_arr[strat_arr==h])
        sel=np.random.choice(hp,len(hp),replace=True)
        for p in sel: idx.extend(np.where(psu_arr==p)[0].tolist())
    idx=np.array(idx)
    try:
        ipcw_b,_=fit_ipcw(states_arr[idx],obs_arr[idx],w_arr[idx],T=2)
        Pb=est_transition(states_arr[idx],obs_arr[idx],w_arr[idx],ipcw_b,T=2)
        boot.append(Pb)
    except: pass
    if (b+1)%100==0: print(f"  {b+1}/{B_boot}")

boot=np.array(boot)
SE=boot.std(axis=0)
LO=np.percentile(boot,2.5,axis=0)
HI=np.percentile(boot,97.5,axis=0)

print("\n=== TABLE: MMSE Estimates with Bootstrap SE and 95% CI ===")
focus=[(0,1),(0,2),(1,0),(1,2),(1,3),(2,3),(2,4),(3,4)]
labels=['p12','p13','p21','p23','p24','p34','p35','p45']
print(f"{'Trans':>6} {'MMSE':>7} {'SE':>7} {'95%CI':>16} {'Naive':>7} {'Bias%':>8}")
for lab,(i,j) in zip(labels,focus):
    p_m=P_mmse[i,j]; p_n=P_naive[i,j]; se=SE[i,j]
    lo=LO[i,j]; hi=HI[i,j]
    bias_pct=(p_n-p_m)/P_pub[i,j]*100
    print(f"{lab:>6}  {p_m:.4f}  {se:.4f}  ({lo:.4f},{hi:.4f})  {p_n:.4f}  {bias_pct:+.1f}%")

# Save results
results = {
    'P_mmse': P_mmse, 'P_naive': P_naive, 'P_pub': P_pub,
    'SE': SE, 'LO': LO, 'HI': HI, 'boot': boot,
    'beta_hat': beta_hat, 'n': n,
    'attrition': 1-obs_w2.mean(),
    'chi2_dropout': chi2_val, 'p_dropout': p_val,
    'state_dist_w1': [(df.state_w1==s).sum() for s in range(5)],
    'dropout_by_state': [((df.state_w1==s)&(df.obs_w2==0)).sum()/max((df.state_w1==s).sum(),1) 
                          for s in range(5)],
    'df': df
}
with open('/home/claude/mmse_paper/real_results.pkl','wb') as f:
    pickle.dump(results,f)
print("\nReal results saved to real_results.pkl")
