import sys, glob, zipfile, numpy as np, pandas as pd, joblib
sys.path.insert(0,sys.argv[1]+'/scripts')
import mri_screen as ms
P=sys.argv[1]
D=ms.load_all(glob.glob('mri/*')[0]); b=D['BASE']
z=zipfile.ZipFile(glob.glob('C:/junny/bioinformaticsstudy/bioinfo대회/01_대회원자료/유전체/*.zip')[0]); fam=[l.split()[0] for l in z.read('synthetic_kchip_chr1.fam').decode().splitlines()]
K=np.load('snp_kept.npz',allow_pickle=True); idx={n:i for i,n in enumerate(K['names'])}; X=K['X']
M=joblib.load(P+'/model/multimodal.joblib'); te=np.load('snp_split_te.npy')[:12]
ids=[fam[i] for i in te]; d=b.loc[ids]
cols=['age','sex','BV_icv']+M['ref'].vols+ms.CLIPPED+M['Q']
out=d[cols].copy(); out['truth(참고용)']=d.lab.values
for h in M['leads']:
    for s in M['leads'][h]:
        if s not in out.columns: out[s]=X[idx[s]][te]
out.index.name='id'; out.to_csv(P+'/model/example_input_full.csv')
snp=[s for h in M['leads'] for s in M['leads'][h]]
out.drop(columns=[c for c in snp if c in out.columns]).to_csv(P+'/model/example_input_noSNP.csv')
mri_only=out[['age','sex','BV_icv']+M['ref'].vols+ms.CLIPPED]; mri_only.to_csv(P+'/model/example_input_MRIonly.csv')
print(out.shape)
