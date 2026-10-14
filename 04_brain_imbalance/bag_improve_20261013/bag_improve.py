"""BAG 개선 비교 실험 (대회 합성자료 BASE/FOLLOW_01/FOLLOW_05).

모든 모델은 BASE '정상군' 일부로만 학습한다. 정상/질환 라벨은 모델 입력에 쓰지 않는다
(단, 상한 참조용 M5 지도학습 모델 제외 - 별도 표기).
"""
import glob, json, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.covariance import LedoitWolf
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
S = sys.argv[1]
OUT = sys.argv[2]
C = "C:/junny/bioinformaticsstudy/bioinfo대회/01_대회원자료/임상/"
M = glob.glob(S + "/mri/*")[0]
NSEED = 5

# ---------------- 데이터 ----------------
def clean(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s < 66666)

def label(df, p, sbp, dbp, glu, a1c, tch, htn, dm, lip, prior=None):
    rep = pd.concat([df[htn], df[dm], df[lip]], axis=1)
    has = (rep == 2).any(axis=1)
    if prior is not None:
        has = has | prior.reindex(df.index).fillna(False)
    unk = rep.isin([99999]).any(axis=1) | rep.isna().any(axis=1)
    bad = ((clean(df[sbp]) >= 140) | (clean(df[dbp]) >= 90) | (clean(df[glu]) >= 126)
           | (clean(df[a1c]) >= 6.5) | (clean(df[tch]) >= 240))
    lab = pd.Series("ambiguous", index=df.index)
    lab[(~has) & (~unk) & (~bad)] = "normal"
    lab[has] = "disease"
    return lab

b = pd.concat([pd.read_csv(C + f"BASE_DATA{i}.txt").set_index("T_ID") for i in (1, 2, 3)], axis=1)
b = b.loc[:, ~b.columns.duplicated()]
mb = pd.read_csv(M + "/BrainVol_BASE.csv").set_index("T_ID")
base = b.join(mb, how="inner")
base["age"] = base.T_AGE; base["sex"] = base.T_SEX
base["lab"] = label(base, "T_", "T_SBP", "T_DBP", "T_GLU0", "T_HBA1C", "T_TCHL", "T_HTN", "T_DM", "T_LIP")

def flagged(k):
    c = pd.read_csv(C + f"FOLLOW_{k:02d}_DATA.txt").set_index("T00_ID"); p = f"T{k:02d}_"
    return (c[p + "HTN"] == 2) | (c[p + "DM"] == 2) | (c[p + "LIP"] == 2)

def follow(k):
    prior = None
    for j in range(1, k):  # 2~5차 질환 표시는 '신규 진단'이므로 이전 차수와 누적한다
        fj = flagged(j)
        prior = fj if prior is None else (prior.reindex(fj.index).fillna(False) | fj).reindex(fj.index.union(prior.index)).fillna(False)
    c = pd.read_csv(C + f"FOLLOW_{k:02d}_DATA.txt").set_index("T00_ID")
    m = pd.read_csv(M + f"/BrainVol_FOLLOW_{k:02d}.csv").set_index("T00_ID")
    d = c.join(m, how="inner")
    p = f"T{k:02d}_"
    d["age"] = d[p + "AGE"]
    lab = label(d, p, p + "SBP", p + "DBP", p + "GLU0", p + "HBA1C", p + "TCHL", p + "HTN", p + "DM", p + "LIP", prior)
    d["lab"] = lab
    return d

f1, f5 = follow(1), follow(5)
f1["sex"] = f1["T00_SEX"]
f5["sex"] = f1["sex"].reindex(f5.index)
VOL_ALL = [c for c in mb.columns if c != "BV_icv"]
CLIPPED = ["BV_lateral_ventricle", "BV_ventricles", "BV_inf_lat_vent", "BV_lateral_ventricle_w_inf"]
VOL_NC = [c for c in VOL_ALL if c not in CLIPPED]

def feat(d, vols=None):
    vols = vols or VOL_ALL
    return np.c_[d[vols].values / d.BV_icv.values[:, None], d.BV_icv.values]

print("라벨 분포")
for n, d in [("BASE", base), ("F01", f1), ("F05", f5)]:
    print(n, d.lab.value_counts().to_dict(), "나이 범위", d.age.min(), d.age.max())
clip = {n: {v: round(float((d[v] == d[v].max()).mean()), 3) for v in CLIPPED[:2]} for n, d in
        [("BASE", base), ("F01", f1), ("F05", f5)]}
print("최소/최대값에 몰린 셀 수(클리핑 의심):", clip)

# ---------------- 도구 ----------------
BINS = [39, 44, 49, 54, 59, 64, 69, 74]
def age_bin(a): return np.digitize(a, BINS[1:], right=True)  # 0..6 , 7=75+

def fit_age_model(Xtr, ytr, seed):
    return make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .9], alphas=30, cv=5,
                                                        max_iter=5000, random_state=seed)).fit(Xtr, ytr)

def fit_bias(pred_cal, age_cal):
    a = age_cal - 55.0
    A = np.c_[np.ones(len(a)), a, a ** 2]
    coef = np.linalg.lstsq(A, pred_cal, rcond=None)[0]
    return lambda pred, age: pred - (np.c_[np.ones(len(age)), age - 55.0, (age - 55.0) ** 2] @ coef)

def strat_auc(score, y, age, edges=(0, 50, 60, 70, 200)):
    """연령 10년 구간별 AUC를 (양성×음성 쌍 수)로 가중 평균."""
    num = den = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (age >= lo) & (age < hi)
        n1, n0 = y[m].sum(), (1 - y[m]).sum()
        if n1 >= 10 and n0 >= 10:
            w = n1 * n0
            num += w * roc_auc_score(y[m], score[m]); den += w
    return num / den if den else np.nan

def evalset(score, d, name):
    y = (d.lab.values == "disease").astype(int)
    keep = d.lab.values != "ambiguous"
    s, y, age = score[keep], y[keep], d.age.values[keep]
    return dict(auc=roc_auc_score(y, s), auc_age=strat_auc(s, y, age),
                mean_norm=float(s[y == 0].mean()), mean_dis=float(s[y == 1].mean()),
                corr_age_norm=float(np.corrcoef(s[y == 0], age[y == 0])[0, 1]))

class Normative:
    def __init__(self, vols):
        self.vols = vols
    def design(self, d):
        a = d.age.values - 55.0
        return np.c_[a, a ** 2, d.sex.values, d.BV_icv.values]
    def fit(self, tr, cal):
        X = self.design(tr)
        Y = tr[self.vols].values
        self.lr = LinearRegression().fit(X, Y)
        self.sd = (Y - self.lr.predict(X)).std(0)
        Zc = self.z(cal)
        self.cov = LedoitWolf().fit(Zc)
        self.sign = np.sign([np.corrcoef(tr.age.values, Y[:, j])[0, 1] for j in range(Y.shape[1])])
        return self
    def z(self, d):
        return (d[self.vols].values - self.lr.predict(self.design(d))) / self.sd
    def maha(self, d):
        return self.cov.mahalanobis(self.z(d))
    def nai(self, d):
        return (self.z(d) * self.sign).mean(1)  # sign=corr(나이,체적) 부호 -> 위축(z<0)일 때 점수 +, 클수록 노화 방향

# ---------------- 반복 실험 ----------------
methods = ["M0 기존(보정 없음)", "M1 +편향보정", "M1c +클리핑변수 제거", "M2c 균형샘플링(+제거)",
           "M2rc 무작위 같은N(+제거)", "M4 3모델 앙상블(+제거)", "M3a 정상참조 노화지수", "M3b 정상참조 Mahalanobis", "M5 지도학습(상한 참조)"]
rows = []; mae_rows = []
for seed in range(NSEED):
    rng = np.random.RandomState(seed)
    nm = base[base.lab == "normal"]
    ds = base[base.lab == "disease"]
    idx = rng.permutation(len(nm))
    cut1, cut2 = int(.6 * len(nm)), int(.8 * len(nm))
    tr, cal, te_n = nm.iloc[idx[:cut1]], nm.iloc[idx[cut1:cut2]], nm.iloc[idx[cut2:]]
    di = rng.permutation(len(ds)); ds_tr, ds_te = ds.iloc[di[:len(ds) // 2]], ds.iloc[di[len(ds) // 2:]]
    sets = {"BASE-test": pd.concat([te_n, ds_te]), "FOLLOW1": f1, "FOLLOW5": f5}
    scores = {m: {} for m in methods}

    def age_model_scores(key, tset, vols, correct=True):
        mm = fit_age_model(feat(tset, vols), tset.age.values, seed)
        cc = fit_bias(mm.predict(feat(cal, vols)), cal.age.values)
        for sn, d in sets.items():
            p = mm.predict(feat(d, vols))
            scores[key][sn] = cc(p, d.age.values) if correct else p - d.age.values
        return mm
    mdl = age_model_scores(methods[0], tr, VOL_ALL, correct=False)
    age_model_scores(methods[1], tr, VOL_ALL)
    mdl_c = age_model_scores(methods[2], tr, VOL_NC)
    trw = tr[(tr.age >= 40) & (tr.age < 75)]
    bb = age_bin(trw.age.values)
    n_per = min((bb == k).sum() for k in range(7))
    tr_bal = trw.iloc[np.concatenate([rng.choice(np.where(bb == k)[0], n_per, replace=False) for k in range(7)])]
    tr_rnd = tr.iloc[rng.choice(len(tr), len(tr_bal), replace=False)]
    mdl_bal = age_model_scores(methods[3], tr_bal, VOL_NC)
    age_model_scores(methods[4], tr_rnd, VOL_NC)
    # M4: ElasticNet + SVR + HGB 보정 BAG 평균(가중치 고정 1/3)
    parts = [mdl_c,
             make_pipeline(StandardScaler(), SVR(C=10, epsilon=.1)).fit(feat(tr, VOL_NC), tr.age.values),
             HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, learning_rate=.05,
                                           l2_regularization=10, early_stopping=False, random_state=seed).fit(feat(tr, VOL_NC), tr.age.values)]
    cors = [fit_bias(m_.predict(feat(cal, VOL_NC)), cal.age.values) for m_ in parts]
    for sn, d in sets.items():
        scores[methods[5]][sn] = np.mean([c_(m_.predict(feat(d, VOL_NC)), d.age.values) for m_, c_ in zip(parts, cors)], axis=0)
    nv = Normative(VOL_NC).fit(tr, cal)
    for sn, d in sets.items():
        scores[methods[6]][sn] = nv.nai(d)
        scores[methods[7]][sn] = nv.maha(d)
    lt = pd.concat([tr, ds_tr])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(
        np.c_[nv.z(lt), lt.age.values], (lt.lab == "disease").astype(int))
    for sn, d in sets.items():
        scores[methods[8]][sn] = clf.decision_function(np.c_[nv.z(d), d.age.values])
    mae_rows.append(dict(seed=seed, n_train_full=len(tr), n_train_bal=len(tr_bal),
                         mae_all=float(np.abs(mdl.predict(feat(te_n)) - te_n.age.values).mean()),
                         mae_noclip=float(np.abs(mdl_c.predict(feat(te_n, VOL_NC)) - te_n.age.values).mean()),
                         mae_bal=float(np.abs(mdl_bal.predict(feat(te_n, VOL_NC)) - te_n.age.values).mean())))
    for m in methods:
        for sn, d in sets.items():
            r = evalset(scores[m][sn], d, sn); r.update(method=m, set=sn, seed=seed); rows.append(r)
    print("seed", seed, "done", flush=True)

R = pd.DataFrame(rows)
agg = R.groupby(["set", "method"])[["auc", "auc_age", "mean_norm", "mean_dis", "corr_age_norm"]].agg(["mean", "std"]).round(3)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(agg)
print(pd.DataFrame(mae_rows).mean().round(3))
R.to_csv(OUT + "/bag_compare_raw.csv", index=False)
agg.to_csv(OUT + "/bag_compare_summary.csv")

# ---------------- 신규 발병 예측 (F01 정상 -> F05 질환) ----------------
print("\n=== 신규 발병: F01 정상 → F05 질환 ===")
nm1 = f1[f1.lab == "normal"].copy()
later = pd.concat([flagged(k) for k in (2, 3, 4, 5)], axis=1)
later = later.reindex(nm1.index).fillna(False).any(axis=1)  # 2~5차 중 한 번이라도 신규 진단
has5 = nm1.index.isin(f5.index)
inc = nm1[has5].copy(); inc["y"] = later[has5].astype(int).values
print("n =", len(inc), "신규 질환", int(inc.y.sum()))
nm = base[base.lab == "normal"]
rng = np.random.RandomState(0); idx = rng.permutation(len(nm)); k = int(.75 * len(nm))
tr, cal = nm.iloc[idx[:k]], nm.iloc[idx[k:]]
mdl = fit_age_model(feat(tr, VOL_NC), tr.age.values, 0)
corr = fit_bias(mdl.predict(feat(cal, VOL_NC)), cal.age.values)
nv = Normative(VOL_NC).fit(tr, cal)
inc["bag"] = corr(mdl.predict(feat(inc, VOL_NC)), inc.age.values)
inc["nai"] = nv.nai(inc)
cl = ["age", "sex", "T01_SBP", "T01_DBP", "T01_BMI" if "T01_BMI" in inc else "T01_WEIGHT", "T01_GLU0", "T01_TCHL", "T01_HDL", "T01_TG", "T01_SMOKE", "T01_EXER"]
for c in cl:
    if c not in ("age", "sex"): inc[c] = clean(inc[c])
Xc = inc[cl].fillna(inc[cl].median())
y = inc.y.values
def cvauc(X, reps=20):
    out = []
    for r in range(reps):
        sk = StratifiedKFold(5, shuffle=True, random_state=r); p = np.zeros(len(y))
        for a, bq in sk.split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(X.iloc[a], y[a]); p[bq] = m.predict_proba(X.iloc[bq])[:, 1]
        out.append(roc_auc_score(y, p))
    return np.mean(out), np.std(out)
res = {}
res["임상만"] = cvauc(Xc)
res["BAG만(보정)"] = cvauc(inc[["bag"]])
res["임상+BAG"] = cvauc(pd.concat([Xc, inc[["bag"]]], axis=1))
res["임상+노화지수"] = cvauc(pd.concat([Xc, inc[["nai"]]], axis=1))
res["임상+BAG+노화지수"] = cvauc(pd.concat([Xc, inc[["bag", "nai"]]], axis=1))
for k_, v in res.items(): print(f"{k_}: AUC {v[0]:.3f} ± {v[1]:.3f}")
json.dump({"incident": {k_: list(v) for k_, v in res.items()}, "n": len(inc), "events": int(y.sum()),
           "clip_cells": clip, "mae": pd.DataFrame(mae_rows).mean().to_dict(), "n_labels": {n: d.lab.value_counts().to_dict() for n, d in [("BASE", base), ("F01", f1), ("F05", f5)]}}, open(OUT + "/bag_incident.json", "w"), ensure_ascii=False, indent=1)
