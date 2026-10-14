import sys, glob
sys.path.insert(0, 'C:/junny/bioinformaticsstudy/bioinfo대회/04_brain_imbalance/mri_screen_20261013')
import numpy as np, pandas as pd
import mri_screen as ms
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
mri = glob.glob('mri/*')[0]
D = ms.load_all(mri); base = D['BASE'].copy()
C = ms.CLIN
rep = base[['T_HTN','T_DM','T_LIP']]
has = (rep == 2).any(axis=1); unk = rep.isin([99999]).any(axis=1)|rep.isna().any(axis=1)
bad = ((ms._clean(base.T_SBP)>=140)|(ms._clean(base.T_DBP)>=90)|(ms._clean(base.T_GLU0)>=126)|(ms._clean(base.T_HBA1C)>=6.5)|(ms._clean(base.T_TCHL)>=240))
base['grp'] = np.where(has,'diagnosed',np.where(unk,'unk',np.where(bad,'undiagnosed','normal')))
print(base.grp.value_counts().to_dict())
# 설문/생활습관/가족력 + 체격
life = ['T_DRINK','T_SMOKE','T_EXER','T_FMFHT1','T_FMFHT2','T_FMFDM1','T_FMFDM2','T_PSM','T_BMI','T_WAIST']
for c in life: base['q_'+c] = ms._clean(base[c]) if c in base else np.nan
qcols = ['q_'+c for c in life if c in base]
lab = base[base.grp.isin(['normal','undiagnosed'])].copy(); y = (lab.grp=='undiagnosed').astype(int).values
print('n', len(lab), 'positives', y.sum())
def lr(): return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegressionCV(Cs=8, cv=3, max_iter=3000))
res = {k: np.zeros(len(lab)) for k in ['나이+성별','+MRI','+설문·가족력(체격 제외)','+체격(BMI·허리)','MRI+설문','MRI+설문+체격']}
q_nobody = [c for c in qcols if c not in ('q_T_BMI','q_T_WAIST')]; q_body = ['q_T_BMI','q_T_WAIST']
for f,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(lab,y)):
    trd, ted = lab.iloc[tr], lab.iloc[te]
    nm = trd[trd.grp=='normal']; rng=np.random.RandomState(f); idx=rng.permutation(len(nm)); k=int(.6*len(nm)); ref_nm, rest = nm.iloc[idx[:k]], nm.iloc[idx[k:]]
    ref = ms.Ref().fit(ref_nm.assign(lab='normal'), f)
    ctr = pd.concat([rest, trd[trd.grp=='undiagnosed']]); yc = (ctr.grp=='undiagnosed').astype(int).values
    Ftr, Fte = ref.transform(ctr), ref.transform(ted)
    cs = ms.colsets(Ftr); mri_cols = cs['age+z+clip'][2:]
    def X(F, d, parts):
        cols=[F[['age','sex']].values]
        if 'mri' in parts: cols.append(F[mri_cols].values)
        if 'q' in parts: cols.append(d[q_nobody].values)
        if 'b' in parts: cols.append(d[q_body].values)
        return np.hstack(cols)
    for name, parts in {'나이+성별':[], '+MRI':['mri'], '+설문·가족력(체격 제외)':['q'], '+체격(BMI·허리)':['q','b'], 'MRI+설문':['mri','q'], 'MRI+설문+체격':['mri','q','b']}.items():
        m = lr().fit(X(Ftr,ctr,parts), yc); res[name][te] = m.predict_proba(X(Fte,ted,parts))[:,1]
for n,s in res.items():
    r = ms.metrics(s, y, lab.age.values); print(f'{n:<26s} AUC {r["auc"]:.3f}  나이구간별 {r["auc_age"]:.3f}  AP {r["ap"]:.3f}')
