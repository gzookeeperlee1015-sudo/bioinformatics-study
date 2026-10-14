"""해마 정상 범위 하위집단에서 미래 해마 감소 예측: 해마 단독 기준 대비 증분 평가.

사전 고정한 설계(결과를 보기 전):
 - 정상 범위 참조 = BASE 정상군(나이, 나이^2, 성별, ICV 보정 해마 z), 중앙 60/80/90%
 - 학습은 FOLLOW 698명 전체, 평가는 정상 범위 하위집단
 - 정답 P(주): 조건부 위축 = H5 - ANCOVA 기대값(H1, 나이, 성별, ICV, 간격; 훈련 폴드에서 적합)
 - 정답 S(보조): 단순 감소량 H1-H5
 - 비교: A(해마 기준) vs 임상 / 영역편차 / 둘 다 / 전체영역 / 잡음 대조 / 비선형
"""
import glob, json, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")
S, OUT = sys.argv[1], sys.argv[2]
R = 10          # 5-fold 반복 횟수
NB = 1000       # 부트스트랩 횟수
C = "C:/junny/bioinformaticsstudy/bioinfo대회/01_대회원자료/임상/"
M = glob.glob(S + "/mri/*")[0]
H = "BV_subctx_hippocampus"
CLIPPED = ["BV_lateral_ventricle", "BV_ventricles", "BV_inf_lat_vent", "BV_lateral_ventricle_w_inf"]

def clean(s):
    s = pd.to_numeric(s, errors="coerce"); return s.where(s < 66666)

# ---------- BASE 정상군(참조) ----------
b = pd.concat([pd.read_csv(C + f"BASE_DATA{i}.txt").set_index("T_ID") for i in (1, 2, 3)], axis=1)
b = b.loc[:, ~b.columns.duplicated()]
base = b.join(pd.read_csv(M + "/BrainVol_BASE.csv").set_index("T_ID"), how="inner")
rep = base[["T_HTN", "T_DM", "T_LIP"]]
bad = ((clean(base.T_SBP) >= 140) | (clean(base.T_DBP) >= 90) | (clean(base.T_GLU0) >= 126)
       | (clean(base.T_HBA1C) >= 6.5) | (clean(base.T_TCHL) >= 240))
ref = base[(rep == 1).all(axis=1) & (~bad)]
print("참조 정상군(BASE) n =", len(ref))

ROI_ALL = [c for c in ref.columns if c.startswith("BV_") and c != "BV_icv" and c not in CLIPPED]
ROI_CUR = ["BV_ctx_entorhinal", "BV_ctx_parahippocampal", "BV_subctx_amygdala", "BV_ctx_fusiform",
           "BV_ctx_inferior_temporal", "BV_ctx_middle_temporal", "BV_ctx_posterior_cingulate",
           "BV_ctx_precuneus", "BV_subctx_thalamus"]

def design(age, sex, icv):
    a = np.asarray(age, float) - 55.0
    return np.c_[a, a ** 2, np.asarray(sex, float), np.asarray(icv, float)]

Xr = design(ref.T_AGE, ref.T_SEX, ref.BV_icv)
norm = LinearRegression().fit(Xr, ref[ROI_ALL].values)
sd = (ref[ROI_ALL].values - norm.predict(Xr)).std(0)
def zscores(age, sex, icv, vol):
    return pd.DataFrame((vol[ROI_ALL].values - norm.predict(design(age, sex, icv))) / sd, index=vol.index, columns=ROI_ALL)
zref_h = zscores(ref.T_AGE, ref.T_SEX, ref.BV_icv, ref)[H].values

# ---------- FOLLOW 698명 ----------
m1 = pd.read_csv(M + "/BrainVol_FOLLOW_01.csv").set_index("T00_ID"); m5 = pd.read_csv(M + "/BrainVol_FOLLOW_05.csv").set_index("T00_ID")
c1 = pd.read_csv(C + "FOLLOW_01_DATA.txt").set_index("T00_ID"); c5 = pd.read_csv(C + "FOLLOW_05_DATA.txt").set_index("T00_ID")
ids = m5.index.intersection(m1.index).intersection(c1.index).intersection(c5.index)
m1, m5, c1, c5 = m1.loc[ids], m5.loc[ids], c1.loc[ids], c5.loc[ids]
d = pd.DataFrame(index=ids)
d["h1"], d["h5"], d["icv"] = m1[H], m5[H], m1.BV_icv
d["age"], d["sex"] = c1.T01_AGE, c1.T00_SEX
d["age2"] = (d.age - 55.0) ** 2
d["yrs"] = c5.T05_AGE - c1.T01_AGE
Z = zscores(d.age, d.sex, d.icv, m1)
d["zh"] = Z[H]
for r_ in ROI_CUR: d["z_" + r_] = Z[r_]
for lob in ["temporal_lobe", "parietal_lobe", "frontal_lobe", "occipital_lobe", "cingulate"]:
    l, r_ = m1[f"BV_ctx_lh_{lob}"], m1[f"BV_ctx_rh_{lob}"]; d["asym_" + lob] = (l - r_) / (l + r_)
l, r_ = m1["BV_lh_lateral_ventricle_w_inf"], m1["BV_rh_lateral_ventricle_w_inf"]; d["asym_ventricle"] = (l - r_) / (l + r_)
ASYM = [c for c in d.columns if c.startswith("asym_")]
ZCUR = ["z_" + r_ for r_ in ROI_CUR]
for r_ in ROI_ALL: d["za_" + r_] = Z[r_]
ZALL = ["za_" + r_ for r_ in ROI_ALL]
h_ = clean(c1.T01_HEIGHT) / 100.0
d["bmi"] = clean(c1.T01_WEIGHT) / h_ ** 2
for k, col in [("sbp", "T01_SBP"), ("dbp", "T01_DBP"), ("a1c", "T01_HBA1C"), ("glu", "T01_GLU0"), ("tchl", "T01_TCHL"),
               ("hdl", "T01_HDL"), ("tg", "T01_TG"), ("smoke", "T01_SMOKE"), ("drink", "T01_DRINK"), ("exer", "T01_EXER"),
               ("htn", "T01_HTN"), ("dm", "T01_DM"), ("lip", "T01_LIP")]:
    d[k] = clean(c1[col])
edu = clean(c1.T01_EDU).astype("Int64").astype(str)
EDU = pd.get_dummies(edu, prefix="edu", dtype=float); d = d.join(EDU)
CLIN = ["bmi", "sbp", "dbp", "a1c", "glu", "tchl", "hdl", "tg", "smoke", "drink", "exer", "htn", "dm", "lip"] + list(EDU.columns)
rng0 = np.random.RandomState(7)
for i in range(10): d[f"noise{i}"] = rng0.randn(len(d))
NOISE = [f"noise{i}" for i in range(10)]
A = ["h1", "icv", "age", "age2", "sex", "yrs"]
_a = LinearRegression().fit(d[A[1:] + ["h1"]], d.h5); _r = _a.predict(d[A[1:] + ["h1"]]) - d.h5
for _k in ["htn", "dm"]: print("조건부 위축(기대-실제, mL) 평균:", _k, {int(v): round(float(_r[d[_k] == v].mean()), 3) for v in (1, 2)}, "n", d[_k].value_counts().to_dict())
print("N =", len(d), "| 정상범위(중앙80%) z 구간:", np.round(np.quantile(zref_h, [.1, .9]), 2))
dec = (d.h1 - d.h5).values

# ---------- 모델 ----------
def ridge(): return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
def enet(): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .9], alphas=25, cv=5, max_iter=5000))
def spline_ridge(cols):
    ct = ColumnTransformer([("sp", SplineTransformer(n_knots=4, degree=3), [cols.index("h1"), cols.index("age")])], remainder="passthrough")
    return make_pipeline(ct, StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
def hgb(): return HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=150, l2_regularization=10, early_stopping=False, random_state=0)
MODELS = {
    "A 해마기준(Ridge)": (A, ridge),
    "A' 해마기준+비선형(스플라인)": (A, lambda: spline_ridge(A)),
    "A+잡음10개(음성대조)": (A + NOISE, ridge),
    "B 기준+임상": (A + CLIN, ridge),
    "C 기준+영역편차+비대칭": (A + ZCUR + ASYM, ridge),
    "D 기준+임상+영역편차": (A + CLIN + ZCUR + ASYM, ridge),
    "E 기준+전체영역편차(ElasticNet)": (A + ZALL, enet),
    "D-GBM 비선형": (A + CLIN + ZCUR + ASYM, hgb),
}
n = len(d)
H5hat = {k: np.zeros((R, n)) for k in MODELS}; ANC = np.zeros((R, n))
for rep_ in range(R):
    for tr, te in KFold(5, shuffle=True, random_state=rep_).split(d):
        anc = LinearRegression().fit(d.iloc[tr][A[1:] + ["h1"]], d.h5.iloc[tr])
        ANC[rep_, te] = anc.predict(d.iloc[te][A[1:] + ["h1"]])
        for k, (cols, mk) in MODELS.items():
            mdl = mk().fit(d.iloc[tr][cols].values, d.h5.iloc[tr].values)
            H5hat[k][rep_, te] = mdl.predict(d.iloc[te][cols].values)
    print("repeat", rep_, "done", flush=True)

anc_m = ANC.mean(0)
truth = {"P 조건부 위축": anc_m - d.h5.values, "S 단순 감소량": dec}   # 클수록 위축이 큼
pred = {}
for k in MODELS:
    h5m = H5hat[k].mean(0)
    pred[k] = {"P 조건부 위축": anc_m - h5m, "S 단순 감소량": d.h1.values - h5m}

def hits_top(t, p, frac=.2):
    k_ = int(np.ceil(frac * len(t)))
    return len(set(np.argsort(-t)[:k_]) & set(np.argsort(-p)[:k_]))
def spear(t, p): return np.corrcoef(rankdata(t), rankdata(p))[0, 1]
def r2(t, p): return 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()

rows = []
rngb = np.random.RandomState(0)
for q in (.6, .8, .9, 1.0):
    lo, hi = (-np.inf, np.inf) if q == 1.0 else np.quantile(zref_h, [(1 - q) / 2, 1 - (1 - q) / 2])
    sub = np.where((d.zh.values >= lo) & (d.zh.values <= hi))[0]
    boots = [rngb.choice(sub, len(sub)) for _ in range(NB)]
    for tg in truth:
        t_ = truth[tg]
        base_h = np.array([hits_top(t_[bi], pred["A 해마기준(Ridge)"][tg][bi]) for bi in boots])
        base_s = np.array([spear(t_[bi], pred["A 해마기준(Ridge)"][tg][bi]) for bi in boots])
        for k in MODELS:
            p_ = pred[k][tg]
            bh = np.array([hits_top(t_[bi], p_[bi]) for bi in boots]); bs = np.array([spear(t_[bi], p_[bi]) for bi in boots])
            kk = int(np.ceil(.2 * len(sub)))
            rows.append(dict(q=q, target=tg, model=k, n=len(sub), n_sel=kk, hits=hits_top(t_[sub], p_[sub]),
                             precision=hits_top(t_[sub], p_[sub]) / kk, spearman=spear(t_[sub], p_[sub]), r2=r2(t_[sub], p_[sub]),
                             d_hits=float(np.mean(bh - base_h)), d_hits_lo=float(np.percentile(bh - base_h, 2.5)), d_hits_hi=float(np.percentile(bh - base_h, 97.5)),
                             d_sp=float(np.mean(bs - base_s)), d_sp_lo=float(np.percentile(bs - base_s, 2.5)), d_sp_hi=float(np.percentile(bs - base_s, 97.5))))
res = pd.DataFrame(rows)
res.to_csv(OUT + "/hippo_results.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
for tg in truth:
    print("\n=====", tg, "| 정상 범위 중앙 80% =====")
    print(res[(res.q == .8) & (res.target == tg)][["model", "n", "n_sel", "hits", "precision", "spearman", "r2", "d_sp", "d_sp_lo", "d_sp_hi", "d_hits", "d_hits_lo", "d_hits_hi"]].round(3).to_string(index=False))
for q in (.6, .9, 1.0):
    print(f"\n--- 민감도 q={q} (Spearman, 모델 A / D / E 비교) ---")
    print(res[(res.q == q) & res.model.str[0].isin(list("ADE"))][["target", "model", "n", "hits", "spearman", "d_sp", "d_sp_lo", "d_sp_hi"]].round(3).to_string(index=False))
print("\n랜덤 선별 기대 정탐(80%):", int(np.ceil(.2 * res[(res.q == .8)].n.iloc[0])) * .2)
json.dump(dict(n_ref=len(ref), n=len(d), reps=R, boot=NB), open(OUT + "/hippo_meta.json", "w"))

# ---------- 그림 ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for a_, tg in zip(ax, truth):
    s = res[(res.q == .8) & (res.target == tg)].reset_index(drop=True)
    y = np.arange(len(s))[::-1]
    a_.errorbar(s.d_sp, y, xerr=[s.d_sp - s.d_sp_lo, s.d_sp_hi - s.d_sp], fmt="o", color="#265974", capsize=3)
    a_.axvline(0, color="#999", ls="--"); a_.set_yticks(y); a_.set_yticklabels(s.model)
    a_.set_title(f"{tg}\n(A 대비 Spearman 증분, 95% 신뢰구간)"); a_.set_xlabel("Δ Spearman vs A")
plt.suptitle("해마 정상 범위 하위집단(중앙 80%): 해마 단독 기준 대비 추가 정보", y=1.02)
plt.tight_layout(); plt.savefig(OUT + "/hippo_increment.png", dpi=150, bbox_inches="tight")
