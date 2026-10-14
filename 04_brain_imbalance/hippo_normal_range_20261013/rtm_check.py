"""임상 신호가 평균 회귀(기저값 보정 인공물)인지 가려내는 점검.
hippo_design.py 의 자료 준비부를 그대로 재사용한다.
"""
import sys
src = open(sys.argv[3], encoding="utf-8").read().split("# ---------- 모델 ----------")[0]
sys.argv = ["x", sys.argv[1], sys.argv[2]]
exec(src)
from scipy.stats import rankdata, norm as _norm

def ols(y, X, names):
    X = np.asarray(X, float); y = np.asarray(y, float)
    X1 = np.c_[np.ones(len(X)), X]
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    e = y - X1 @ b
    XtXi = np.linalg.inv(X1.T @ X1)
    meat = X1.T @ (X1 * (e ** 2)[:, None]) * len(y) / (len(y) - X1.shape[1])
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return pd.DataFrame({"coef": b[1:], "se": se[1:]}, index=names)

def fmt(df, rows):
    return "  ".join(f"{r}: {df.loc[r,'coef']:+.3f}±{1.96*df.loc[r,'se']:.3f}" for r in rows)

# ---------- 1. 기저 해마의 지속적 집단 차이? ----------
print("== 1. 질환 자기보고에 따른 해마 크기 차이 (나이·성별·ICV 보정, mL, ±95%CI) ==")
bb = base.copy()
for k, c in [("htn", "T_HTN"), ("dm", "T_DM"), ("lip", "T_LIP")]: bb[k] = bb[c].map({1: 0, 2: 1})
bb = bb.dropna(subset=["htn", "dm", "lip"])
a_ = bb.T_AGE - 55
t1 = ols(bb[H], np.c_[bb.htn, bb.dm, bb.lip, a_, a_ ** 2, bb.T_SEX, bb.BV_icv], ["htn", "dm", "lip", "a", "a2", "sex", "icv"])
print(f"BASE(n={len(bb)}) 해마:", fmt(t1, ["htn", "dm", "lip"]))
dd = d.copy()
for k in ["htn", "dm", "lip"]: dd[k] = dd[k].map({1: 0, 2: 1})
dd["dis"] = ((dd.htn == 1) | (dd.dm == 1) | (dd.lip == 1)).astype(float)
cv = ["age", "age2", "sex", "icv"]
for nm_, y_ in [("FOLLOW1 h1", dd.h1), ("FOLLOW5 h5", dd.h5)]:
    t = ols(y_, dd[["htn", "dm", "lip"] + cv], ["htn", "dm", "lip"] + cv); print(nm_ + ":", fmt(t, ["htn", "dm", "lip"]))

# ---------- 2/3. 보정 방식별 질환 효과 (로드의 역설 + 측정오차 민감도) ----------
print("\n== 2·3. 위축(기대-실제 또는 h1-h5, 클수록 위축) 질환 효과, mL ==")
grp_mean = dd.groupby("dis").h1.transform("mean")
base_cov = ["htn", "dm", "lip", "age", "age2", "sex", "icv", "yrs"]
names = base_cov
def row(label, adj):
    Xc = dd[base_cov].copy()
    if adj is not None: Xc["adj"] = adj
    # 결과: h5 를 종속변수로 두고 계수 부호를 뒤집어 '위축'으로 해석
    t = ols(dd.h5 if adj is not None else (dd.h1 - dd.h5), Xc, list(Xc.columns))
    sign = -1 if adj is not None else 1
    return f"{label:<44s}" + "  ".join(f"{k}: {sign*t.loc[k,'coef']:+.3f}±{1.96*t.loc[k,'se']:.3f}" for k in ["htn", "dm", "lip"])
print(row("(a) 단순 감소량(h1 보정 없음)", None))
print(row("(b) h1 보정(ANCOVA, 우리가 쓴 P와 같은 방식)", dd.h1))
for lam in (.95, .9, .85, .8):
    print(row(f"(c) h1 오차 보정, 신뢰도 λ={lam}", grp_mean + lam * (dd.h1 - grp_mean)))
print(row("(d) 신뢰도 보정 극단 λ=0.6", grp_mean + .6 * (dd.h1 - grp_mean)))

# ---------- 4. 임상 증분(B 대 A)이 보정 가정에서도 유지되나 ----------
print("\n== 4. 조건부 위축 정답을 λ별로 다시 정의했을 때 임상 증분(Spearman, 정상범위 중앙 80%) ==")
lo, hi = np.quantile(zref_h, [.1, .9]); sub = np.where((d.zh.values >= lo) & (d.zh.values <= hi))[0]
A_cols = A; B_cols = A + CLIN
def ridge(): return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
n = len(d); RR = 10
pA = np.zeros((RR, n)); pB = np.zeros((RR, n)); anc_l = {lam: np.zeros((RR, n)) for lam in (1.0, .95, .9, .85, .8)}
dis = dd.dis.values; gm = dd.groupby("dis").h1.transform("mean").values
for rp in range(RR):
    for tr, te in KFold(5, shuffle=True, random_state=rp).split(d):
        for cols, store in [(A_cols, pA), (B_cols, pB)]:
            store[rp, te] = ridge().fit(d.iloc[tr][cols].values, d.h5.iloc[tr].values).predict(d.iloc[te][cols].values)
        for lam in anc_l:
            h1c = gm + lam * (d.h1.values - gm)
            Xa = np.c_[h1c, d[A[1:]].values]
            m_ = LinearRegression().fit(Xa[tr], d.h5.values[tr]); anc_l[lam][rp, te] = m_.predict(Xa[te])
pA, pB = pA.mean(0), pB.mean(0)
def sp(t, p): return np.corrcoef(rankdata(t), rankdata(p))[0, 1]
rngb = np.random.RandomState(1); boots = [rngb.choice(sub, len(sub)) for _ in range(1000)]
for lam in anc_l:
    anc = anc_l[lam].mean(0); truth_ = anc - d.h5.values; ra, rb = anc - pA, anc - pB
    diff = np.array([sp(truth_[b], rb[b]) - sp(truth_[b], ra[b]) for b in boots])
    print(f"λ={lam:<4}: A {sp(truth_[sub], ra[sub]):+.3f}  B {sp(truth_[sub], rb[sub]):+.3f}  증분 {sp(truth_[sub], rb[sub]) - sp(truth_[sub], ra[sub]):+.3f}  [{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]")
