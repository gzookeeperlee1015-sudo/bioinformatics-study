"""모달리티별 기여 장부: 질환(진단) vs 정상 선별에서 모달리티를 하나씩 더했을 때 성능.
모달리티: 나이·성별 / MRI(정상참조 편차+클리핑지시자) / 설문·가족력·생활습관 / 체격(BMI·허리) / (SNP는 별도 단계)
평가: BASE 5폴드x2 반복 교차검증, 참조는 학습 정상군의 60%만으로 만들고 점수 학습은 나머지로.
"""
import sys, glob, itertools
sys.path.insert(0,'C:/junny/bioinformaticsstudy/bioinfo대회/04_brain_imbalance/mri_screen_20261013')
import numpy as np, pandas as pd, mri_screen as ms
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
D=ms.load_all(glob.glob('mri/*')[0]); base=D['BASE']
lab=base[base.lab.isin(['normal','disease'])].copy(); y=(lab.lab=='disease').astype(int).values
Q=['T_DRINK','T_SMOKE','T_EXER','T_FMFHT1','T_FMFHT2','T_FMFDM1','T_FMFDM2','T_PSM']; B=['T_BMI','T_WAIST']
for c in Q+B: lab['q_'+c]=ms._clean(lab[c])
q=['q_'+c for c in Q]; bd=['q_'+c for c in B]
def lr(): return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegressionCV(Cs=8,cv=3,max_iter=3000))
combos={'나이·성별':[], '+MRI':['mri'], '+설문·가족력':['q'], '+체격':['b'], 'MRI+설문':['mri','q'], 'MRI+체격':['mri','b'], '설문+체격':['q','b'], 'MRI+설문+체격':['mri','q','b']}
REPS=2; rows=[]
for rep in range(REPS):
    oof={k:np.zeros(len(lab)) for k in combos}
    for f,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=rep).split(lab,y)):
        trd,ted=lab.iloc[tr],lab.iloc[te]; rn,rest=ms.split_ref(trd,rep*10+f); ref=ms.Ref().fit(rn,rep*10+f)
        ctr=pd.concat([rest,trd[trd.lab=='disease']]); yc=(ctr.lab=='disease').astype(int).values
        Ftr,Fte=ref.transform(ctr),ref.transform(ted); mc=ms.colsets(Ftr)['age+z+clip'][2:]
        def X(F,d,parts):
            cs=[F[['age','sex']].values]
            if 'mri' in parts: cs.append(F[mc].values)
            if 'q' in parts: cs.append(d[q].values)
            if 'b' in parts: cs.append(d[bd].values)
            return np.hstack(cs)
        for k,p in combos.items(): oof[k][te]=lr().fit(X(Ftr,ctr,p),yc).predict_proba(X(Fte,ted,p))[:,1]
    for k in combos:
        r=ms.metrics(oof[k],y,lab.age.values); r.update(model=k,rep=rep); rows.append(r)
    print('rep',rep,flush=True)
R=pd.DataFrame(rows).groupby('model')[['auc','auc_age','ap']].mean().round(3).loc[list(combos)]
print(R); R.to_csv('ledger_any.csv')
