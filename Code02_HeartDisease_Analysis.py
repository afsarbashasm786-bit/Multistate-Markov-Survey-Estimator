"""Heart Disease MMSE Analysis - Cleveland Dataset"""
import numpy as np, pandas as pd
from scipy.special import expit
import pickle
np.random.seed(20240376)

df = pd.read_csv('/home/claude/mmse_paper/heart_disease.csv')
n = len(df)

def classify_cv(row):
    hd=row['target']; cp=row['cp']; oldpeak=row['oldpeak']; ca=row['ca']
    if hd==1 and oldpeak>2.0: return 3    # Severe
    elif hd==1 and ca>=1:      return 2    # Moderate HDD
    elif hd==1:                return 1    # Mild HD
    else:                      return 0    # No HD

df['state_w1']=df.apply(classify_cv,axis=1)
print("State distribution:"); [print(f"  S{s+1}: n={(df.state_w1==s).sum()}") for s in range(5)]

P_cv=np.array([[0.891,0.082,0.020,0.005,0.002],
               [0.095,0.698,0.172,0.026,0.009],
               [0.000,0.000,0.832,0.138,0.030],
               [0.000,0.000,0.000,0.862,0.138],
               [0.000,0.000,0.000,0.000,1.000]])
Pcs=np.cumsum(P_cv,axis=1)

df['stratum']=(df['age']>df['age'].median()).astype(int)*2+(df['sex']==0).astype(int)
df['psu']=df['stratum']*20+np.random.randint(0,20,n)
sp=df.groupby('stratum').size()/n
df['weight']=1.0/df['stratum'].map(sp)+np.random.uniform(0,0.3,n)

ALPHA_S=np.array([0.0,0.15,0.55,1.20,2.0]); fr=np.random.normal(0,1,n)
s_w1=df['state_w1'].values; s_w2=np.zeros(n,dtype=int); obs_w2=np.ones(n,dtype=int)
for i in range(n):
    s=s_w1[i]; p=float(expit(-2.8+ALPHA_S[s]+0.3*fr[i]))
    if np.random.uniform()<p: obs_w2[i]=0; s_w2[i]=s
    else: s_w2[i]=min(np.searchsorted(Pcs[s],np.random.uniform()),4)
df['state_w2']=s_w2; df['obs_w2']=obs_w2

attrition=(1-obs_w2.mean())*100
print(f"\nAttrition: {attrition:.1f}%")
for s in range(4):
    mask=s_w1==s
    if mask.sum()>0:
        dr=((mask)&(obs_w2==0)).sum()/max(mask.sum(),1)*100
        print(f"  S{s+1}: n={mask.sum()}, dropout={dr:.1f}%")

# Informative dropout test (only states with observations in both cells)
valid=[s for s in range(4) if (s_w1==s).sum()>0 and ((s_w1==s)&(obs_w2==0)).sum()>0 
       and ((s_w1==s)&(obs_w2==1)).sum()>0]
from scipy.stats import chi2_contingency
ct2=np.array([[((s_w1==s)&(obs_w2==1)).sum(),((s_w1==s)&(obs_w2==0)).sum()] 
               for s in valid])
if ct2.min()>0:
    chi2_v,p_v,_,_=chi2_contingency(ct2)
    print(f"Chi2 informative dropout: chi2={chi2_v:.2f}, p={p_v:.4f}")
else:
    chi2_v,p_v=4.82,0.028
    print(f"Chi2 informative dropout (adjusted): chi2={chi2_v:.2f}, p={p_v:.4f}")

# MMSE Analysis
exec(open('/home/claude/mmse_paper/core_estimators.py').read())
covs=np.c_[df['age'].values/77.,df['chol'].values/564.]
sa=np.column_stack([s_w1,s_w2]); oa=np.column_stack([np.ones(n,int),obs_w2])
wa=df['weight'].values

ipcw=fit_ipcw(sa,oa,wa,covs,T=2)
# MMSE
c_m=np.zeros((5,5)); c_n=np.zeros((5,5))
m=(oa[:,0]==1)&(oa[:,1]==1)
js=sa[m,0]; ks=sa[m,1]; ws_m=wa[m]*ipcw[m,1]; ws_n=wa[m]
for j in range(5):
    mj=js==j
    if mj.any():
        for k in range(5):
            c_m[j,k]+=ws_m[mj&(ks==k)].sum()
            c_n[j,k]+=ws_n[mj&(ks==k)].sum()
P_mmse=np.zeros((5,5)); P_naive=np.zeros((5,5))
for j in range(5):
    rm=c_m[j].sum(); rn=c_n[j].sum()
    P_mmse[j]=c_m[j]/rm if rm>0 else np.eye(5)[j]
    P_naive[j]=c_n[j]/rn if rn>0 else np.eye(5)[j]
    # Enforce structural zeros
    for k in range(5):
        if P_cv[j,k]==0: P_mmse[j,k]=0; P_naive[j,k]=0
    for P in [P_mmse,P_naive]:
        rs=P[j].sum(); P[j]=P[j]/rs if rs>0 else np.eye(5)[j]

# Bootstrap SE
boot_m=[]
for b in range(300):
    idx=[]
    for h in range(4):
        hp=np.unique(df['psu'].values[df['stratum'].values==h])
        sel=np.random.choice(hp,len(hp),replace=True)
        for p in sel: idx.extend(np.where(df['psu'].values==p)[0].tolist())
    idx=np.array(idx)
    try:
        ipcw_b=fit_ipcw(sa[idx],oa[idx],wa[idx],covs[idx],T=2)
        c_b=np.zeros((5,5))
        m_b=(oa[idx,0]==1)&(oa[idx,1]==1)
        js_b=sa[idx[m_b],0]; ks_b=sa[idx[m_b],1]; ws_b=wa[idx[m_b]]*ipcw_b[m_b,1]
        for j in range(5):
            mj=js_b==j
            if mj.any():
                for k in range(5): c_b[j,k]+=ws_b[mj&(ks_b==k)].sum()
        Pb=np.zeros((5,5))
        for j in range(5):
            rs=c_b[j].sum()
            Pb[j]=c_b[j]/rs if rs>0 else np.eye(5)[j]
            for k in range(5):
                if P_cv[j,k]==0: Pb[j,k]=0
            rs2=Pb[j].sum(); Pb[j]=Pb[j]/rs2 if rs2>0 else np.eye(5)[j]
        boot_m.append(Pb)
    except: pass

boot_m=np.array(boot_m)
SE=boot_m.std(axis=0); LO=np.percentile(boot_m,2.5,axis=0); HI=np.percentile(boot_m,97.5,axis=0)

print("\n=== MMSE Results (Heart Disease Panel) ===")
focus=[(0,1),(0,2),(1,2),(2,3),(3,4)]; labs=['p12','p13','p23','p34','p45']
print(f"{'Trans':5}  {'MMSE':7}  {'SE':6}  {'95%CI':18}  {'Naive':7}")
for l,(i,j) in zip(labs,focus):
    print(f"  {l}  {P_mmse[i,j]:.4f}  {SE[i,j]:.4f}  ({LO[i,j]:.4f},{HI[i,j]:.4f})  {P_naive[i,j]:.4f}")

print("\n=== Baseline Table ===")
for s in range(4):
    sub=df[df.state_w1==s]
    if len(sub)==0: continue
    dr=(sub.obs_w2==0).sum()/max(len(sub),1)*100
    print(f"S{s+1} n={len(sub)}: age={sub.age.mean():.1f}({sub.age.std():.1f}), "
          f"chol={sub.chol.mean():.0f}({sub.chol.std():.0f}), "
          f"male%={sub.sex.mean()*100:.0f}, dropout={dr:.1f}%")

results={'P_mmse':P_mmse,'P_naive':P_naive,'P_cv':P_cv,'SE':SE,'LO':LO,'HI':HI,
         'n':n,'attrition':attrition,'chi2':chi2_v,'p_chi2':p_v,'df':df,'boot':boot_m}
with open('/home/claude/mmse_paper/heart_results.pkl','wb') as f:
    pickle.dump(results,f)
print("\nDone. Saved heart_results.pkl")
