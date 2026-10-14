"""BASE 10,000명 x 1,000,000 SNP 에서 질환 연관 스캔(학습 70%만 사용)."""
import sys, glob, zipfile, io, numpy as np, pandas as pd
sys.path.insert(0,'C:/junny/bioinformaticsstudy/bioinfo대회/04_brain_imbalance/mri_screen_20261013')
import mri_screen as ms
D=ms.load_all(glob.glob('mri/*')[0]); b=D['BASE']
z=zipfile.ZipFile(glob.glob('C:/junny/bioinformaticsstudy/bioinfo대회/01_대회원자료/유전체/*.zip')[0])
fam=[l.split()[0] for l in z.read('synthetic_kchip_chr1.fam').decode().splitlines()]; N=len(fam)
ids=pd.Index(fam); lab=b.reindex(ids)
rng=np.random.RandomState(0)
use=lab.lab.isin(['normal','disease']).values
order=np.where(use)[0]; rng.shuffle(order); ntr=int(.7*len(order)); tr=np.sort(order[:ntr]); te=np.sort(order[ntr:])
np.save('snp_split_tr.npy',tr); np.save('snp_split_te.npy',te)
traits={'any':(lab.lab=='disease').values.astype(float),'htn':lab.htn.values.astype(float),'dm':lab.dm.values.astype(float),'lip':lab.lip.values.astype(float)}
isn=(lab.lab=='normal').values
# 각 질환 vs 정상 비교용 마스크
masks={'any':isn|(lab.lab=='disease').values,'htn':isn|(lab.htn.values==1),'dm':isn|(lab.dm.values==1),'lip':isn|(lab.lip.values==1)}
trmask=np.zeros(N,bool); trmask[tr]=True
LUT=np.array([2,np.nan,1,0],dtype=np.float32)
res=[]; keep={}; nbytes=(N+3)//4
for chrom in range(1,23):
    bim=pd.read_csv(io.BytesIO(z.read(f'synthetic_kchip_chr{chrom}.bim')),sep='\t',header=None,names=['chr','snp','cm','pos','a1','a2'])
    raw=np.frombuffer(z.read(f'synthetic_kchip_chr{chrom}.bed'),dtype=np.uint8)[3:]
    ns=len(bim); G=raw.reshape(ns,nbytes)
    out=np.empty((ns,4),np.float32)
    CH=4000
    for s in range(0,ns,CH):
        blk=G[s:s+CH]; bits=np.stack([(blk>>(2*k))&3 for k in range(4)],axis=2).reshape(blk.shape[0],-1)[:,:N]
        X=LUT[bits]  # (snps, N), nan=결측
        mu=np.nanmean(X,axis=1,keepdims=True); X=np.where(np.isnan(X),mu,X)
        for j,t in enumerate(['any','htn','dm','lip']):
            m=trmask&masks[t]; y=traits[t][m]; Xm=X[:,m]
            Xc=Xm-Xm.mean(1,keepdims=True); yc=y-y.mean()
            r=(Xc@yc)/(np.sqrt((Xc**2).sum(1))*np.sqrt((yc**2).sum())+1e-12); out[s:s+CH,j]=r
        # 후보 저장: rs 번호가 붙은 SNP 또는 학습 |r| 큰 것
        sel=[i for i in range(blk.shape[0]) if (not bim.snp.iloc[s+i].startswith('bg_')) or np.abs(out[s+i]).max()>0.035]
        for i in sel: keep[bim.snp.iloc[s+i]]=X[i].astype(np.float32)
    d=bim.copy(); d[['r_any','r_htn','r_dm','r_lip']]=out; res.append(d); print('chr',chrom,ns,'kept so far',len(keep),flush=True)
R=pd.concat(res); R.to_csv('snp_scan_r.csv',index=False)
np.savez_compressed('snp_kept.npz',names=np.array(list(keep)),X=np.stack(list(keep.values())))
print('done')
