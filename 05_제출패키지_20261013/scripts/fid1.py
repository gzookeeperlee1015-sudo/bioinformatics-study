import numpy as np, pandas as pd, glob
P='C:/junny/bioinformaticsstudy/bioinfo대회/'
bands=[(40,50),(50,60),(60,70),(70,90)]
def summarize(name, age, pct):
    out=[]
    for lo,hi in bands:
        m=(age>=lo)&(age<hi)
        out.append(f"{lo}-{hi}: n={int(m.sum()):<4d} 중앙 {np.nanmedian(pct[m]):+.2f}%/년" if m.sum()>=10 else f"{lo}-{hi}: n={int(m.sum())}")
    print(f"{name:<22s}"+" | ".join(out))
# 합성 FOLLOW (해마 1차->5차)
M=glob.glob('mri/*')[0]; H='BV_subctx_hippocampus'
m1=pd.read_csv(M+'/BrainVol_FOLLOW_01.csv').set_index('T00_ID'); m5=pd.read_csv(M+'/BrainVol_FOLLOW_05.csv').set_index('T00_ID')
c1=pd.read_csv(P+'01_대회원자료/임상/FOLLOW_01_DATA.txt').set_index('T00_ID'); c5=pd.read_csv(P+'01_대회원자료/임상/FOLLOW_05_DATA.txt').set_index('T00_ID')
ids=m5.index.intersection(m1.index); yrs=(c5.T05_AGE-c1.T01_AGE).loc[ids]
pct=((m5.loc[ids,H]-m1.loc[ids,H])/m1.loc[ids,H]*100/yrs).values; age=c1.T01_AGE.loc[ids].values
print('합성(FOLLOW 1→5, 약 10.8년)'); summarize('합성 대회자료',age,pct)
print('  음수(증가) 비율', round(float((pct>0).mean()),3), '| 개인 변화율 SD', round(float(np.nanstd(pct)),2),'%/년')
# DLBS
R=P+'03_DLBS_원자료/확장제공본/'
mri=pd.ExcelFile(R+'derivatives/brainsummary/Template_Structural_MRI.xlsx'); p=pd.read_csv(R+'participants.tsv',sep='\t'); p['S']=pd.to_numeric(p.participant_id.str.replace('sub-',''))
def hip(w):
    s=mri.parse(f'SubcorticalVolume-W{w}'); return pd.DataFrame({'S':s['S#'],f'h{w}':(pd.to_numeric(s.LhHippocampusVol,errors='coerce')+pd.to_numeric(s.RhHippocampusVol,errors='coerce'))})
W=hip(1).merge(hip(2),on='S',how='left').merge(hip(3),on='S',how='left').merge(p[['S','AgeMRI_W1','MRIW1toW2','MRIW1toW3']],on='S')
for k,nm in [(2,'DLBS 1→2차(약4년)'),(3,'DLBS 1→3차(약9년)')]:
    d=W.dropna(subset=[f'h{k}','h1',f'MRIW1toW{k}']); d=d[d.AgeMRI_W1>=40]
    pct=((d[f'h{k}']-d.h1)/d.h1*100/d[f'MRIW1toW{k}']).values; print(nm); summarize('실제 DLBS',d.AgeMRI_W1.values,pct)
    print('  증가 비율',round(float((pct>0).mean()),3),'| SD',round(float(np.nanstd(pct)),2),'%/년')
# ADNI
a=pd.read_csv(P+'02_ADNI_원자료/ADNIMERGE.csv',low_memory=False)
a=a[a.Hippocampus.notna()&a.ICV.notna()]; a['date']=pd.to_datetime(a.EXAMDATE)
cn=a[a.DX_bl=='CN'].sort_values(['RID','date']); g=cn.groupby('RID')
first=g.first(); last=g.last(); yrs=(last.date-first.date).dt.days/365.25
ok=yrs>=1.5; pct=((last.Hippocampus-first.Hippocampus)/first.Hippocampus*100/yrs)[ok]
print('ADNI CN (>=1.5년 간격, 중앙 %.1f년)'%yrs[ok].median()); summarize('실제 ADNI(CN)',(first.AGE+0)[ok].values,pct.values)
print('  증가 비율',round(float((pct>0).mean()),3),'| SD',round(float(pct.std()),2),'%/년')
