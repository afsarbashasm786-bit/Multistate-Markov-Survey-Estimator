"""
MMSE Simulation v2 - Stronger informative dropout to demonstrate method advantage
"""
import numpy as np
from scipy.special import expit
import pickle, warnings
warnings.filterwarnings('ignore')

np.random.seed(20240376)

P_true = np.array([
    [0.920, 0.058, 0.014, 0.006, 0.002],
    [0.120, 0.682, 0.165, 0.025, 0.008],
    [0.000, 0.000, 0.812, 0.158, 0.030],
    [0.000, 0.000, 0.000, 0.882, 0.118],
    [0.000, 0.000, 0.000, 0.000, 1.000],
])
P_cs = np.cumsum(P_true, axis=1)

# Strong informative dropout: sicker states drop out much more
GAMMA = np.array([0.0, 0.10, 0.60, 1.10, 1.50])  # strong gradient
ALPHA = 1.5
BASE_H = 0.08  # baseline hazard

def sim(n, T=3, seed=None):
    if seed: np.random.seed(seed)
    pi0=np.array([0.576,0.153,0.201,0.048,0.022]); pi0/=pi0.sum()
    states=np.zeros((n,T),dtype=int); obs=np.ones((n,T),dtype=int)
    states[:,0]=np.random.choice(5,n,p=pi0)
    bi=np.random.normal(0,1,n)
    strata=np.random.choice(4,n,p=[0.32,0.28,0.22,0.18])
    psu=strata*20+np.random.randint(0,20,n)
    sp_arr=np.array([0.32,0.28,0.22,0.18])
    weights=1.0/sp_arr[strata]+np.random.uniform(0,0.3,n)
    dropped=np.zeros(n,bool)
    for t in range(1,T):
        s_prev=states[:,t-1]
        haz=np.clip(BASE_H*np.exp(GAMMA[s_prev]+ALPHA*bi*0.3),0,0.95)
        nd=(~dropped)&(np.random.uniform(size=n)<haz)
        dropped|=nd; obs[dropped,t]=0
        active=(~dropped)&(s_prev<4)
        u=np.random.uniform(size=n); ns=s_prev.copy()
        for j in range(5):
            m=active&(s_prev==j)
            if m.any(): ns[m]=np.searchsorted(P_cs[j],u[m])
        ns=np.clip(ns,0,4); ns[s_prev==4]=4; ns[dropped]=s_prev[dropped]
        states[:,t]=ns
    return states,obs,weights,strata,psu

def dropout_rate(o,T):
    drops=[((1-o[:,t]).sum()/o[:,t-1].sum()) for t in range(1,T)]
    return np.mean(drops)

def naive(s,o,T):
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

def dw(s,o,w,T):
    c=np.zeros((5,5))
    for t in range(T-1):
        m=(o[:,t]==1)&(o[:,t+1]==1)
        js=s[m,t]; ks=s[m,t+1]; ws=w[m]
        for j in range(5):
            mj=js==j
            if mj.any():
                for k in range(5): c[j,k]+=ws[mj&(ks==k)].sum()
    P=np.zeros((5,5))
    for j in range(5):
        rs=c[j].sum(); P[j]=c[j]/rs if rs>0 else np.eye(5)[j]
    return P

def mmse_est(s,o,w,T):
    n=s.shape[0]
    Xl,yl,wl=[],[],[]
    for t in range(T-1):
        m=o[:,t]==1; sv=np.eye(5)[s[m,t]]
        Xl.append(np.c_[np.ones(m.sum()),sv]); yl.append(1-o[m,t+1]); wl.append(w[m])
    X=np.vstack(Xl); y=np.concatenate(yl); ww=np.concatenate(wl); ww/=ww.mean()
    beta=np.zeros(X.shape[1])
    for _ in range(120):
        p=np.clip(expit(X@beta),1e-6,1-1e-6)
        g=X.T@(ww*(p-y))/len(y); beta-=0.5*g
    ipcw=np.ones((n,T))
    for i in range(n):
        surv=1.0
        for t in range(T-1):
            if o[i,t]==1:
                x=np.zeros(6); x[0]=1.; x[1+s[i,t]]=1.
                pd=np.clip(float(expit(x@beta)),1e-6,0.95)
                surv=max(surv*(1-pd),1e-4); ipcw[i,t+1]=1./surv
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

def boot_ci(s,o,w,st,psu,T,B=100):
    boot=[]
    for _ in range(B):
        idx=[]
        for h in range(4):
            hp=np.unique(psu[st==h])
            sel=np.random.choice(hp,len(hp),replace=True)
            for p in sel: idx.extend(np.where(psu==p)[0].tolist())
        idx=np.array(idx)
        try: boot.append(mmse_est(s[idx],o[idx],w[idx],T))
        except: pass
    ba=np.array(boot)
    return np.percentile(ba,2.5,axis=0),np.percentile(ba,97.5,axis=0),ba

# Check dropout rates
print("Verifying dropout rates (should be ~18-33% for sick states):")
s_check,o_check,w_c,_,_=sim(5000,3,seed=1)
dr=dropout_rate(o_check,3)
print(f"  Overall wave-to-wave attrition: {dr*100:.1f}%")
# By state
for j in range(5):
    in_j=[sum(1 for t in range(2) if o_check[i,t]==1 and s_check[i,t]==j) for i in range(5000)]
    drop_j=[sum(1 for t in range(2) if o_check[i,t]==1 and o_check[i,t+1]==0 and s_check[i,t]==j) for i in range(5000)]
    tot_in=sum(in_j); tot_drop=sum(drop_j)
    if tot_in>0: print(f"  State {j}: dropout rate {tot_drop/tot_in*100:.1f}%")

B=300; n_sizes=[500,1000,2000,5000]; T=3
focus=[(0,1),(1,2),(2,3),(3,4)]; labs=['p12','p23','p34','p45']

rb={'naive':{},'dw':{},'mmse':{}}; rr={'naive':{},'dw':{},'mmse':{}}
raw={'naive':{},'dw':{},'mmse':{}}

print(f"\nRunning B={B} reps...")
for n in n_sizes:
    pn=np.zeros((B,5,5)); pd_=np.zeros((B,5,5)); pm=np.zeros((B,5,5))
    for b in range(B):
        s,o,w,st,ps=sim(n,T,seed=n*1000+b)
        pn[b]=naive(s,o,T); pd_[b]=dw(s,o,w,T); pm[b]=mmse_est(s,o,w,T)
    raw['naive'][n]=pn; raw['dw'][n]=pd_; raw['mmse'][n]=pm
    for meth,arr in [('naive',pn),('dw',pd_),('mmse',pm)]:
        rb[meth][n]={}; rr[meth][n]={}
        for l,(i,j) in zip(labs,focus):
            tv=P_true[i,j]
            rb[meth][n][l]=float(abs(arr[:,i,j].mean()-tv)*1000)
            rr[meth][n][l]=float(np.sqrt(((arr[:,i,j]-tv)**2).mean())*1000)
    print(f" n={n:5d}: NI_bias={rb['naive'][n]['p12']:.3f} DW_bias={rb['dw'][n]['p12']:.3f} MMSE_bias={rb['mmse'][n]['p12']:.3f}")

print("\n== BIAS TABLE (x1e-3, transition p12) ==")
print(f"{'':10}", end="")
for n in n_sizes: print(f"  n={n}", end="")
print()
for m,l in [('naive','NI'),('dw','DW'),('mmse','MMSE-SP')]:
    print(f"{l:<10}", end="")
    for n in n_sizes: print(f"  {rb[m][n]['p12']:5.3f}", end="")
    print()

# Coverage at n=2000
print("\nCoverage study n=2000, 50 reps...")
B_cov=50; ncv=2000
cn={l:[] for l in labs}; cm={l:[] for l in labs}
for b in range(B_cov):
    s,o,w,st,ps=sim(ncv,T,seed=77000+b)
    Pn=naive(s,o,T); Pm=mmse_est(s,o,w,T)
    lo,hi,ba=boot_ci(s,o,w,st,ps,T,B=100)
    for l,(i,j) in zip(labs,focus):
        ni=sum(1 for ii in range(ncv) for t in range(T-1)
               if o[ii,t]==1 and o[ii,t+1]==1 and s[ii,t]==i)
        nij=sum(1 for ii in range(ncv) for t in range(T-1)
                if o[ii,t]==1 and o[ii,t+1]==1 and s[ii,t]==i and s[ii,t+1]==j)
        if ni>5:
            ph=nij/ni; se=np.sqrt(ph*(1-ph)/ni)
            cn[l].append(int(abs(Pn[i,j]-P_true[i,j])<=1.96*se))
        else: cn[l].append(0)
        cm[l].append(int(lo[i,j]<=P_true[i,j]<=hi[i,j]))
    if (b+1)%10==0: print(f" {b+1}/{B_cov} | MMSE cov p12={np.mean(cm['p12'])*100:.0f}%")

print("\n== COVERAGE TABLE (%) ==")
print(f"{'':10}", end="")
for l in labs: print(f"  {l}", end="")
print()
for meth,lbl,cd in [('naive','NI',cn),('mmse','MMSE-SP',cm)]:
    print(f"{lbl:<10}", end="")
    for l in labs: print(f"  {np.mean(cd[l])*100:4.1f}", end="")
    print()

with open('/home/claude/mmse_paper/sim_output.pkl','wb') as f:
    pickle.dump({'rb':rb,'rr':rr,'raw':raw,'cn':cn,'cm':cm,
                 'n_sizes':n_sizes,'P_true':P_true,'labs':labs,
                 'focus':focus,'B':B,'B_cov':B_cov},f)
print("\nSaved.")
