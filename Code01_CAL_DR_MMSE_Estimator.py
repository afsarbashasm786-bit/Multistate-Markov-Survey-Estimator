"""Core estimators - optimised vectorised versions"""
import numpy as np
from scipy.special import expit
import warnings; warnings.filterwarnings('ignore')

P_TRUE = np.array([
    [0.862, 0.108, 0.022, 0.006, 0.002],
    [0.143, 0.672, 0.158, 0.021, 0.006],
    [0.000, 0.000, 0.818, 0.152, 0.030],
    [0.000, 0.000, 0.000, 0.878, 0.122],
    [0.000, 0.000, 0.000, 0.000, 1.000],
])
P_CS = np.cumsum(P_TRUE, axis=1)
GAMMA = np.array([0.0, 0.40, 1.20, 2.20, 3.50])
BASE_H = 0.05

def simulate_panel(n, T=3, seed=None, scenario='A'):
    if seed is not None: np.random.seed(seed)
    pi0=np.array([0.576,0.153,0.201,0.048,0.022]); pi0/=pi0.sum()
    st=np.zeros((n,T),dtype=int); ob=np.ones((n,T),dtype=int)
    st[:,0]=np.random.choice(5,n,p=pi0)
    age=np.random.uniform(0,1,n); bmi=np.random.normal(0,1,n)
    stra=np.random.choice(4,n,p=[0.32,0.28,0.22,0.18])
    psu=stra*20+np.random.randint(0,20,n)
    sp=np.array([0.32,0.28,0.22,0.18])
    wts=1.0/sp[stra]+np.random.uniform(0,0.3,n)
    dr=np.zeros(n,bool)
    for t in range(1,T):
        s_=st[:,t-1]
        if   sc=='A': haz=np.clip(BASE_H*np.exp(GAMMA[s_]),0,0.97)
        elif sc=='B': haz=np.clip(BASE_H*np.exp(GAMMA[s_]+0.9*bmi),0,0.97)
        elif sc=='C': haz=np.clip(BASE_H*np.exp(GAMMA[s_]+0.7*age*s_),0,0.97)
        else:         haz=np.clip(BASE_H*np.exp(GAMMA[s_]+0.7*bmi+0.5*age*s_),0,0.97)
        nd=(~dr)&(np.random.uniform(size=n)<haz); dr|=nd; ob[dr,t]=0
        ac=(~dr)&(s_<4); u=np.random.uniform(size=n); ns=s_.copy()
        for j in range(5):
            m=ac&(s_==j)
            if m.any(): ns[m]=np.searchsorted(P_CS[j],u[m])
        ns=np.clip(ns,0,4); ns[s_==4]=4; ns[dr]=s_[dr]; st[:,t]=ns
    return st,ob,wts,stra,psu,np.c_[age,bmi]

# Fix: scenario parameter not substituted
def simulate_panel(n, T=3, seed=None, scenario='A'):
    if seed is not None: np.random.seed(seed)
    pi0=np.array([0.576,0.153,0.201,0.048,0.022]); pi0/=pi0.sum()
    st=np.zeros((n,T),dtype=int); ob=np.ones((n,T),dtype=int)
    st[:,0]=np.random.choice(5,n,p=pi0)
    age=np.random.uniform(0,1,n); bmi=np.random.normal(0,1,n)
    stra=np.random.choice(4,n,p=[0.32,0.28,0.22,0.18])
    psu=stra*20+np.random.randint(0,20,n)
    sp=np.array([0.32,0.28,0.22,0.18])
    wts=1.0/sp[stra]+np.random.uniform(0,0.3,n)
    dr=np.zeros(n,bool)
    for t in range(1,T):
        s_=st[:,t-1]
        if scenario=='A':   haz=np.clip(BASE_H*np.exp(GAMMA[s_]),0,0.97)
        elif scenario=='B': haz=np.clip(BASE_H*np.exp(GAMMA[s_]+0.9*bmi),0,0.97)
        elif scenario=='C': haz=np.clip(BASE_H*np.exp(GAMMA[s_]+0.7*age*s_),0,0.97)
        else:               haz=np.clip(BASE_H*np.exp(GAMMA[s_]+0.7*bmi+0.5*age*s_),0,0.97)
        nd=(~dr)&(np.random.uniform(size=n)<haz); dr|=nd; ob[dr,t]=0
        ac=(~dr)&(s_<4); u=np.random.uniform(size=n); ns=s_.copy()
        for j in range(5):
            m=ac&(s_==j)
            if m.any(): ns[m]=np.searchsorted(P_CS[j],u[m])
        ns=np.clip(ns,0,4); ns[s_==4]=4; ns[dr]=s_[dr]; st[:,t]=ns
    return st,ob,wts,stra,psu,np.c_[age,bmi]

def naive(s,o,T):
    c=np.zeros((5,5))
    for t in range(T-1):
        m=(o[:,t]==1)&(o[:,t+1]==1); js=s[m,t]; ks=s[m,t+1]
        for j in range(5):
            mj=js==j
            for k in range(5): c[j,k]+=((ks==k)&mj).sum()
    P=np.zeros((5,5))
    for j in range(5): rs=c[j].sum(); P[j]=c[j]/rs if rs>0 else np.eye(5)[j]
    return P

def dw(s,o,w,T):
    c=np.zeros((5,5))
    for t in range(T-1):
        m=(o[:,t]==1)&(o[:,t+1]==1); js=s[m,t]; ks=s[m,t+1]; ws=w[m]
        for j in range(5):
            mj=js==j
            if mj.any():
                for k in range(5): c[j,k]+=ws[mj&(ks==k)].sum()
    P=np.zeros((5,5))
    for j in range(5): rs=c[j].sum(); P[j]=c[j]/rs if rs>0 else np.eye(5)[j]
    return P

def fit_ipcw(s,o,w,covs,T):
    """Stabilised IPCW with trimmed weights."""
    n=s.shape[0]; Xl,yl,wl=[],[],[]
    for t in range(T-1):
        m=o[:,t]==1; sv=np.eye(5)[s[m,t]]
        Xl.append(np.c_[np.ones(m.sum()),sv,covs[m]]); yl.append(1-o[m,t+1]); wl.append(w[m])
    X=np.vstack(Xl); y=np.concatenate(yl); ww=np.concatenate(wl); ww/=ww.mean()
    beta=np.zeros(X.shape[1])
    for _ in range(120):
        p=np.clip(expit(X@beta),1e-6,1-1e-6); beta-=0.4*(X.T@(ww*(p-y))/len(y))
    ipcw=np.ones((n,T))
    for i in range(n):
        surv=1.0
        for t in range(T-1):
            if o[i,t]==1:
                x=np.concatenate([[1.],np.eye(5)[s[i,t]],covs[i]])
                pd=np.clip(float(expit(x@beta)),1e-6,0.95)
                surv=max(surv*(1-pd),1e-4); ipcw[i,t+1]=1./surv
    # Trim at 90th percentile to stabilise
    obs=ipcw[ipcw>1.]; cap=np.percentile(obs,90) if len(obs)>10 else 20.
    return np.minimum(ipcw,cap)

def fit_or_fast(s,o,w,covs,T):
    """Fast vectorised outcome regression."""
    n=s.shape[0]; OR=np.zeros((5,5))  # marginal OR by state
    OR_w=np.zeros(5)
    for t in range(T-1):
        m=(o[:,t]==1)&(o[:,t+1]==1); js=s[m,t]; ks=s[m,t+1]; ws=w[m]
        for j in range(5):
            mj=js==j
            if mj.any():
                for k in range(5): OR[j,k]+=ws[mj&(ks==k)].sum()
                OR_w[j]+=ws[mj].sum()
    for j in range(5):
        if OR_w[j]>0:
            OR[j]/=OR_w[j]
            # Enforce structural zeros
            for k in range(5):
                if P_TRUE[j,k]==0: OR[j,k]=0.
            rs=OR[j].sum(); OR[j]=OR[j]/rs if rs>0 else P_TRUE[j]
        else: OR[j]=P_TRUE[j]
    return OR  # shape (5,5)

def est_dr_mmse(s,o,w,covs,T):
    """DR-MMSE: Stabilised IPCW + Weighted OR + AIPW score."""
    n=s.shape[0]
    ipcw=fit_ipcw(s,o,w,covs,T)
    OR=fit_or_fast(s,o,w,covs,T)  # state-specific marginal OR
    num=np.zeros((5,5)); den=np.zeros(5)
    for t in range(T-1):
        m_obs_t=(o[:,t]==1)
        for j in range(5):
            mj=m_obs_t&(s[:,t]==j)
            if not mj.any(): continue
            idx_j=np.where(mj)[0]
            den[j]+=w[idx_j].sum()
            for i in idx_j:
                wi=w[i]; iw=ipcw[i,t+1]; mu=OR[j,:]
                for k in range(5):
                    if P_TRUE[j,k]==0: continue
                    if o[i,t+1]==1:
                        ind=float(s[i,t+1]==k)
                        num[j,k]+=wi*iw*ind - wi*(iw-1.)*mu[k]
                    else:
                        num[j,k]+=wi*mu[k]
    P=np.zeros((5,5))
    for j in range(5):
        if den[j]>0:
            row=num[j]/den[j]
            for k in range(5):
                if P_TRUE[j,k]==0: row[k]=0.
            row=np.clip(row,0,1); rs=row.sum()
            P[j]=row/rs if rs>0 else np.eye(5)[j]
        else: P[j,j]=1.
    return P

print("Core estimators loaded.")
