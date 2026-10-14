import glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
M=glob.glob('mri/*')[0]; C='C:/junny/bioinformaticsstudy/bioinfo대회/01_대회원자료/임상/'; H='BV_subctx_hippocampus'
m1=pd.read_csv(M+'/BrainVol_FOLLOW_01.csv').set_index('T00_ID'); m5=pd.read_csv(M+'/BrainVol_FOLLOW_05.csv').set_index('T00_ID')
c1=pd.read_csv(C+'FOLLOW_01_DATA.txt').set_index('T00_ID'); c5=pd.read_csv(C+'FOLLOW_05_DATA.txt').set_index('T00_ID')
d=pd.DataFrame({'h1':m1[H],'icv':m1.BV_icv,'age':c1.T01_AGE,'sex':c1.T00_SEX,'htn':(c1.T01_HTN==2)*1,'dm':(c1.T01_DM==2)*1,'lip':(c1.T01_LIP==2)*1})
d['stay']=d.index.isin(m5.index.intersection(c5.index))*1
print('잔류율',d.stay.mean().round(3))
d['dis']=((d.htn+d.dm+d.lip)>0)*1
print(d.groupby('dis').agg(n=('stay','size'),stay=('stay','mean'),h1_stay=('h1',lambda s:s[d.loc[s.index,'stay']==1].mean()),h1_drop=('h1',lambda s:s[d.loc[s.index,'stay']==0].mean())).round(3))
# 해마 크기 보정 후 질환의 h1 차이: 전체 1000명 vs 잔류 698명
for nm,sub in [('전체1000',d),('잔류698',d[d.stay==1])]:
    X=np.c_[sub.htn,sub.dm,sub.lip,sub.age-55,(sub.age-55)**2,sub.sex,sub.icv]; b=LinearRegression().fit(X,sub.h1).coef_
    print(nm,'기저 해마 차이 htn/dm/lip',b[:3].round(3))
X=np.c_[d.h1,d.age,d.sex,d.icv,d.htn,d.dm,d.lip]; Xs=(X-X.mean(0))/X.std(0)
lr=LogisticRegression(C=1.0,max_iter=2000).fit(Xs,d.stay); print('잔류 로지스틱 표준화 계수:',dict(zip(['h1','age','sex','icv','htn','dm','lip'],lr.coef_[0].round(3))))
# IPW 로 재가중한 '단순 감소량' 질환 효과
p=lr.predict_proba(Xs)[:,1]; w=1/p[d.stay==1]
s=d[d.stay==1].copy(); s['h5']=m5.loc[s.index,H]; s['dec']=s.h1-s.h5; s['yrs']=c5.loc[s.index,'T05_AGE']-s.age
Xc=np.c_[s.htn,s.dm,s.lip,s.age-55,(s.age-55)**2,s.sex,s.icv,s.yrs]
for nm,ww in [('가중 없음',None),('IPW(탈락 보정)',w)]:
    b=LinearRegression().fit(Xc,s.dec,sample_weight=ww).coef_[:3]; print(nm,'단순 감소량 질환 효과 htn/dm/lip',b.round(3))
