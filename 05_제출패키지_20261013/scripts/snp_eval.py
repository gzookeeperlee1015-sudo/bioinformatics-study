import sys, glob
sys.path.insert(0,'C:/junny/bioinformaticsstudy/bioinfo대회/04_brain_imbalance/mri_screen_20261013')
import numpy as np, pandas as pd, mri_screen as ms
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
D=ms.load_all(glob.glob('mri/*')[0]); b=D['BASE']
R=pd.read_csv('snp_scan_r.csv'); K=np.load('snp_kept.npz',allow_pickle=True); names=list(K['names']); X=K['X']
idx={n:i for i,n in enumerate(names)}
import zipfile
z=zipfile.ZipFile(glob.glob('C:/junny/bioinformaticsstudy/bioinfo대회/01_대회원자료/유전체/*.zip')[0]); fam=[l.split()[0] for l in z.read('synthetic_kchip_chr1.fam').decode().splitlines()]
lab=b.reindex(pd.Index(fam)); tr=np.load('snp_split_tr.npy'); te=np.load('snp_split_te.npy')
Q=['T_DRINK','T_SMOKE','T_EXER','T_FMFHT1','T_FMFHT2','T_FMFDM1','T_FMFDM2','T_PSM','T_BMI','T_WAIST']
for c in Q: lab['q_'+c]=ms._clean(lab[c])
qc=['q_'+c for c in Q]
def lr(): return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegressionCV(Cs=8,cv=3,max_iter=3000))
def lead_snps(trait, maxn=15, minr=0.05, win=1_000_000):
    r=R['r_'+trait].values; order=np.argsort(-np.abs(r)); chosen=[]
    for i in order:
        if abs(r[i])<minr or len(chosen)>=maxn: break
        c,p,nm=R.chr.values[i],R.pos.values[i],R.snp.values[i]
        if nm not in idx: continue
        if all(not(c==R.chr.values[j] and abs(p-R.pos.values[j])<win) for j in chosen): chosen.append(i)
    return [R.snp.values[i] for i in chosen]
# MRI 참조: 학습 정상군의 60%로 참조, 나머지 + 학습 질환으로 분류기
trd=lab.iloc[tr]; rn,rest=ms.split_ref(trd.assign(lab=trd.lab),0); ref=ms.Ref().fit(rn,0)
res=[]
for trait,tname in [('any','전체 질환'),('htn','고혈압'),('dm','당뇨'),('lip','고지혈')]:
    pos_fn=(lambda d: d.lab=='disease') if trait=='any' else (lambda d: d[trait]==1)
    leads=lead_snps(trait); print(f'[{tname}] 대표 SNP {len(leads)}개:',leads[:8])
    ctr=pd.concat([rest, trd[pos_fn(trd)]]); yc=ctr.index.isin(trd[pos_fn(trd)].index).astype(int)
    ted=lab.iloc[te]; mk=((ted.lab=='normal')|pos_fn(ted)).values; ted=ted[mk]; yt=pos_fn(ted).astype(int).values
    Ftr,Fte=ref.transform(ctr),ref.transform(ted); mc=ms.colsets(Ftr)['age+z+clip'][2:]
    def snp_mat(d): 
        pos={n:i for i,n in enumerate(fam)}; ii=[pos[n] for n in d.index]; return np.stack([X[idx[s]][ii] for s in leads],axis=1)
    parts={'age':lambda F,d:F[['age','sex']].values,'mri':lambda F,d:F[mc].values,'q':lambda F,d:d[qc].values,'snp':lambda F,d:snp_mat(d)}
    combos={'나이·성별':['age'],'+SNP':['age','snp'],'+MRI':['age','mri'],'+MRI+SNP':['age','mri','snp'],'+설문·체격':['age','q'],'+설문·체격+SNP':['age','q','snp'],'+MRI+설문·체격':['age','mri','q'],'전부(MRI+설문·체격+SNP)':['age','mri','q','snp']}
    for cn,ps in combos.items():
        Xa=np.hstack([parts[p](Ftr,ctr) for p in ps]); Xb=np.hstack([parts[p](Fte,ted) for p in ps])
        m=lr().fit(Xa,yc); s=m.predict_proba(Xb)[:,1]
        rng=np.random.RandomState(0); bs=[]; 
        for _ in range(300):
            i=rng.choice(len(yt),len(yt)); bs.append(roc_auc_score(yt[i],s[i]))
        res.append(dict(target=tname,model=cn,n=len(yt),pos=int(yt.sum()),auc=roc_auc_score(yt,s),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),auc_age=ms.strat_auc(s,yt,ted.age.values)))
Rr=pd.DataFrame(res); Rr.to_csv('snp_eval.csv',index=False); pd.set_option('display.width',200)
for t,g in Rr.groupby('target',sort=False): print('\n',t); print(g[['model','auc','lo','hi','auc_age']].round(3).to_string(index=False))
