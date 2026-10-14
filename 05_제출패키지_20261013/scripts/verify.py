"""SNP·MRI 신호 재검증: 누수/오류 여부, 안정성, 효과 크기 현실성, 신호가 묶인 대상(라벨 vs 측정 수치)."""
import sys, glob, zipfile
sys.path.insert(0, 'C:/junny/bioinformaticsstudy/bioinfo대회/05_제출패키지_20261013/scripts')
import numpy as np, pandas as pd, mri_screen as ms
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

D = ms.load_all(glob.glob('mri/*')[0]); b = D['BASE']
z = zipfile.ZipFile(glob.glob('C:/junny/bioinformaticsstudy/bioinfo대회/01_대회원자료/유전체/*.zip')[0])
fam = [l.split()[0] for l in z.read('synthetic_kchip_chr1.fam').decode().splitlines()]
print('fam ID가 BASE 와 일치:', set(fam) == set(b.index), '| FOLLOW ID 포함:', any(f.startswith('K_FOLLOW') for f in fam))
R = pd.read_csv('snp_scan_r.csv'); K = np.load('snp_kept.npz', allow_pickle=True); names = list(K['names']); X = K['X']; idx = {n: i for i, n in enumerate(names)}
lab = b.reindex(pd.Index(fam)); N = len(fam)
meta = R.set_index('snp').loc[names]            # chr,pos for kept SNPs
chrv, posv = meta.chr.values, meta.pos.values

def scan(mask, y, chunk=6000):
    out = np.empty(X.shape[0], np.float32); ym = y - y.mean(); sy = np.sqrt((ym ** 2).sum())
    for s in range(0, X.shape[0], chunk):
        Xm = X[s:s + chunk][:, mask]; Xc = Xm - Xm.mean(1, keepdims=True)
        out[s:s + chunk] = (Xc @ ym) / (np.sqrt((Xc ** 2).sum(1)) * sy + 1e-12)
    return out
def leads_from(r, maxn=15, minr=0.0, win=1_000_000):
    ch = []
    for i in np.argsort(-np.abs(r)):
        if len(ch) >= maxn: break
        if all(not (chrv[i] == chrv[j] and abs(posv[i] - posv[j]) < win) for j in ch): ch.append(i)
    return ch
def lrm(): return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=8, cv=3, max_iter=3000))

traits = {'any': (lab.lab == 'disease').values, 'htn': (lab.htn == 1).values}
isn = (lab.lab == 'normal').values; lab_ok = lab.lab.isin(['normal', 'disease']).values
agesex = np.c_[lab.age.values, lab.sex.values]

def run_split(seed, trait, permute=False):
    rng = np.random.RandomState(seed); ids = np.where(lab_ok)[0]; rng.shuffle(ids); k = int(.7 * len(ids)); tr, te = np.sort(ids[:k]), np.sort(ids[k:])
    pos = traits[trait]; mem = isn | pos
    trm = np.zeros(N, bool); trm[tr] = True; trm &= mem
    y = pos[trm].astype(float)
    if permute: y = rng.permutation(y)
    r = scan(trm, y); L = leads_from(r)
    tem = np.zeros(N, bool); tem[te] = True; tem &= mem
    tr_i, te_i = np.where(trm)[0], np.where(tem)[0]
    ytr = pos[tr_i].astype(int) if not permute else y.astype(int); yte = pos[te_i].astype(int)
    G = lambda ii: np.c_[agesex[ii], X[L][:, ii].T]
    m0 = lrm().fit(agesex[tr_i], ytr); a0 = roc_auc_score(yte, m0.predict_proba(agesex[te_i])[:, 1])
    m1 = lrm().fit(G(tr_i), ytr); a1 = roc_auc_score(yte, m1.predict_proba(G(te_i))[:, 1])
    return a0, a1, float(np.abs(r[L]).max())

print('\n=== A. 라벨 섞기(음성 대조)와 B. 다른 분할 5개: 테스트 AUC (나이·성별 -> +SNP15개) ===')
for trait in ['htn', 'any']:
    for seed in [1, 2, 3, 4, 5]:
        a0, a1, mr = run_split(seed, trait); print(f'[{trait}] 분할 {seed}: {a0:.3f} -> {a1:.3f}  (학습 최대|r| {mr:.3f})')
    for seed in [101, 102]:
        a0, a1, mr = run_split(seed, trait, permute=True); print(f'[{trait}] 라벨 섞기 {seed}: {a0:.3f} -> {a1:.3f}  (최대|r| {mr:.3f})  <- 0.5 근처여야 정상')

print('\n=== C. 효과 크기 현실성: 대표 SNP 의 대립유전자당 오즈비 (정상 vs 고혈압, 전체 라벨 개체) ===')
mem = isn | (lab.htn == 1).values; y = (lab.htn == 1).values[mem].astype(int)
r = scan(mem, y.astype(float)); L = leads_from(r, 8)
for i in L:
    g = X[i][mem]; maf = g.mean() / 2
    m = LogisticRegression(penalty=None, max_iter=500).fit(np.c_[g, lab.age.values[mem], lab.sex.values[mem]], y)
    print(f'{names[i]:<20s} A1빈도 {maf:.2f}  오즈비/대립유전자 {np.exp(m.coef_[0][0]):.2f}  |r| {abs(r[i]):.3f}')
print('(참고: 실제 혈압 GWAS 대표 변이의 오즈비는 대체로 1.03~1.10 수준)')

print('\n=== D. 신호가 묶인 대상: 진단 라벨인가, 측정 수치인가 ===')
rng = np.random.RandomState(0); ids = np.where(lab_ok)[0]; rng.shuffle(ids); tr = np.sort(ids[:int(.7 * len(ids))])
trm = np.zeros(N, bool); trm[tr] = True; trm &= (isn | (lab.htn == 1).values)
Lh = leads_from(scan(trm, (lab.htn == 1).values[trm].astype(float)))
mh = lrm().fit(np.c_[X[Lh][:, trm].T], (lab.htn == 1).values[trm].astype(int))
snp_sc = mh.decision_function(np.c_[X[Lh].T])
trd = lab.iloc[tr]; rn, rest = ms.split_ref(trd, 0); ref = ms.Ref().fit(rn, 0)
Fz = ref.transform(lab.fillna({'age': lab.age.mean()}).assign(BV_icv=lab.BV_icv)); zc = Fz[[c for c in Fz.columns if c.startswith('z_')]]
atr = -zc.mean(axis=1).values   # 클수록 위축
sbp = ms._clean(lab.T_SBP).values; glu = ms._clean(lab.T_GLU0).values
grp = np.where(lab.htn.values == 1, 'HTN진단', np.where(isn, '정상', np.where(lab.lab.values == 'ambiguous', '미진단/모호', '기타')))
test_mask = np.ones(N, bool); test_mask[tr] = False   # 학습에 쓰지 않은 개체만
def sp(a, b_):
    ok = ~(np.isnan(a) | np.isnan(b_)); return pd.Series(a[ok]).corr(pd.Series(b_[ok]), method='spearman'), int(ok.sum())
for g in ['정상', '미진단/모호', 'HTN진단']:
    m = (grp == g) & test_mask
    c1, n1 = sp(snp_sc[m], sbp[m]); c2, n2 = sp(atr[m], sbp[m]); print(f'{g:<8s} n={n1:<5d} 수축기혈압과의 Spearman: SNP점수 {c1:+.3f} | MRI 위축점수 {c2:+.3f}')
for nm, v in [('SNP점수(고혈압)', snp_sc), ('MRI 위축점수', atr)]:
    m1 = (grp == 'HTN진단') & test_mask; m0 = ((grp == '정상') | (grp == '미진단/모호')) & test_mask
    yv = np.r_[np.ones(m1.sum()), np.zeros(m0.sum())]; s = np.r_[v[m1], v[m0]]
    print(f'{nm}: HTN진단 vs (정상+미진단) AUC {roc_auc_score(yv, s):.3f}')
