import sys, glob
sys.path.insert(0,'C:/junny/bioinformaticsstudy/bioinfo대회/04_brain_imbalance/mri_screen_20261013')
import numpy as np, pandas as pd, mri_screen as ms
P='C:/junny/bioinformaticsstudy/bioinfo대회/'
D=ms.load_all(glob.glob('mri/*')[0]); b=D['BASE']; H='BV_subctx_hippocampus'
def ols(y,X):
    X1=np.c_[np.ones(len(X)),X]; c=np.linalg.lstsq(X1,y,rcond=None)[0]; e=y-X1@c; XtXi=np.linalg.inv(X1.T@X1)
    se=np.sqrt(np.diag(XtXi@(X1.T@(X1*(e**2)[:,None]))@XtXi)); return c[1:],se[1:]
# 합성 BASE 횡단
age=b.age.values; sx=b.sex.values
c,se=ols(b[H].values,np.c_[age-55,sx,b.BV_icv.values]); print(f'합성 BASE 해마: 나이 {c[0]*10:+.3f} mL/10년, 성별(2 vs 1) {c[1]:+.3f}, ICV {c[2]*100:+.3f} mL/100mL')
c2,_=ols(b.BV_icv.values,np.c_[age-55,sx]); print(f'합성 ICV: 나이 {c2[0]*10:+.1f} mL/10년, 성별(2 vs 1) {c2[1]:+.1f} mL (평균 {b.BV_icv.mean():.0f})')
m=(age>=40)
for lo,hi in [(40,50),(50,60),(60,70),(70,90)]:
    s=b[(age>=lo)&(age<hi)]; print(f'  합성 {lo}-{hi}: n={len(s)}, 해마 평균 {s[H].mean():.2f}')
# DLBS 횡단
R=P+'03_DLBS_원자료/확장제공본/'
mri=pd.ExcelFile(R+'derivatives/brainsummary/Template_Structural_MRI.xlsx'); p=pd.read_csv(R+'participants.tsv',sep='\t'); p['S']=pd.to_numeric(p.participant_id.str.replace('sub-',''))
s=mri.parse('SubcorticalVolume-W1'); g=mri.parse('GlobalVariables-W1')
d=pd.DataFrame({'S':s['S#'],'h':(pd.to_numeric(s.LhHippocampusVol,errors='coerce')+pd.to_numeric(s.RhHippocampusVol,errors='coerce'))/1000}).merge(pd.DataFrame({'S':g['S#'],'icv':pd.to_numeric(g.TotalIntracranialVol,errors='coerce')/1000}),on='S').merge(p[['S','AgeMRI_W1','Sex']],on='S')
d['sex']=d.Sex.map({'m':1,'f':2}); d=d[d.AgeMRI_W1>=40].dropna()
c,se=ols(d.h.values,np.c_[d.AgeMRI_W1-55,d.sex,d.icv]); print(f'실제 DLBS 해마(n={len(d)}): 나이 {c[0]*10:+.3f} mL/10년, 성별 {c[1]:+.3f}, ICV {c[2]*100:+.3f}')
c2,_=ols(d.icv.values,np.c_[d.AgeMRI_W1-55,d.sex]); print(f'실제 DLBS ICV: 나이 {c2[0]*10:+.1f} mL/10년, 성별(여-남) {c2[1]:+.1f} mL (평균 {d.icv.mean():.0f})')
# 정상군 안에서 MRI 편차 종합점수와 측정 혈압의 관계 (생리 반영 여부)
nm=b[b.lab=='normal'].copy(); ref=ms.Ref().fit(nm.sample(frac=.5,random_state=0),0); rest=nm.drop(nm.sample(frac=.5,random_state=0).index)
F=ref.transform(rest); z=F[[c_ for c_ in F.columns if c_.startswith('z_')]]
comp=-z.mean(axis=1)   # 클수록 전반적 위축
for v in ['T_SBP','T_DBP','T_GLU0','T_BMI']:
    x=ms._clean(rest[v]); k=x.notna(); print(f'정상군 내 MRI 위축 종합점수 vs {v}: Spearman {pd.Series(comp[k]).corr(x[k],method="spearman"):+.3f} (n={int(k.sum())})')
dis=b[b.lab=='disease']; Fd=ref.transform(dis); print('질환군 평균 위축 종합점수', round(float((-Fd[[c_ for c_ in Fd.columns if c_.startswith("z_")]].mean(axis=1)).mean()),3),'| 정상군', round(float(comp.mean()),3))
