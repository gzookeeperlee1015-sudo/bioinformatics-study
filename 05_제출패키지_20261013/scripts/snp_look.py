import numpy as np, pandas as pd
R=pd.read_csv('snp_scan_r.csv'); R['rs']=~R.snp.str.startswith('bg_')
print('SNP 수',len(R),'| rs 번호가 붙은 SNP',int(R.rs.sum()))
tr=np.load('snp_split_tr.npy'); print('학습 n',len(tr))
for t in ['any','htn','dm','lip']:
    r=R['r_'+t].values; n_eff=4000  # 대략적 크기: 정상+해당질환 학습 인원, 아래에서 z는 r*sqrt(n) 근사
    z=np.abs(r)*np.sqrt(4300); thr=5.45
    print(f'[{t}] 최대 |r|={np.abs(r).max():.3f} (귀무 기대 최대 약 {4.9/np.sqrt(4300):.3f}) | 전장 유의 근사 |z|>5.45: {int((z>thr).sum())}개 | 그중 rs: {int(((z>thr)&R.rs.values).sum())}')
    top=R.reindex(np.argsort(-np.abs(r))[:6])[['snp','chr','pos',f'r_{t}']]; print(top.to_string(index=False))
print('\nrs SNP의 |r| 분포 vs bg SNP')
for t in ['any','htn','dm','lip']:
    a=np.abs(R.loc[R.rs,'r_'+t]); b=np.abs(R.loc[~R.rs,'r_'+t]); print(f'{t}: rs 평균|r| {a.mean():.4f} (n={len(a)}), bg 평균|r| {b.mean():.4f}, rs 상위1% {np.quantile(a,.99):.3f} bg 상위1% {np.quantile(b,.99):.3f}')
