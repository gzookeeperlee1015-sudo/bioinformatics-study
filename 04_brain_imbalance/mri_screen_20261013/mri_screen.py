"""MRI 단독 대사질환 선별 모델 (정상 참조 편차 + 보정 BAG + 선별 점수).

사용:
  cv      BASE 개발 교차검증(모델 선택은 여기서만)
  fit     선택한 모델을 BASE 전체로 학습해 저장하고 FOLLOW 1·5차를 한 번 평가(동결 외부 평가)
  predict 저장 모델로 새 사람의 출력(BAG, 편차 지도, 선별 점수) 생성
정상/질환 정의(이 연구에서 정한 규칙): 자기보고 HTN/DM/LIP 없음 + SBP<140, DBP<90, 공복혈당<126,
HbA1c<6.5, 총콜레스테롤<240 이면 정상, 자기보고가 있으면 질환, 나머지는 모호군(학습·평가 제외).
"""
import argparse, glob, json, os, sys, warnings
import numpy as np, pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
CLIN = "C:/junny/bioinformaticsstudy/bioinfo대회/01_대회원자료/임상/"
CEIL = {"BV_lateral_ventricle": 24.0, "BV_ventricles": 34.0, "BV_inf_lat_vent": 5.0, "BV_lateral_ventricle_w_inf": 28.0}
CLIPPED = ["BV_lateral_ventricle", "BV_ventricles", "BV_inf_lat_vent", "BV_lateral_ventricle_w_inf"]  # 상한에 잘린 변수

# ---------------------------------------------------------------- 자료
def _clean(s):
    s = pd.to_numeric(s, errors="coerce"); return s.where(s < 66666)

def _label(df, sbp, dbp, glu, a1c, tch, htn, dm, lip, prior=None):
    rep = df[[htn, dm, lip]]
    has = (rep == 2).any(axis=1)
    if prior is not None: has = has | prior.reindex(df.index).fillna(False)
    unk = rep.isin([99999]).any(axis=1) | rep.isna().any(axis=1)
    bad = ((_clean(df[sbp]) >= 140) | (_clean(df[dbp]) >= 90) | (_clean(df[glu]) >= 126)
           | (_clean(df[a1c]) >= 6.5) | (_clean(df[tch]) >= 240))
    lab = pd.Series("ambiguous", index=df.index)
    lab[(~has) & (~unk) & (~bad)] = "normal"; lab[has] = "disease"
    return lab

def load_all(mri_dir):
    b = pd.concat([pd.read_csv(CLIN + f"BASE_DATA{i}.txt").set_index("T_ID") for i in (1, 2, 3)], axis=1)
    b = b.loc[:, ~b.columns.duplicated()]
    base = b.join(pd.read_csv(mri_dir + "/BrainVol_BASE.csv").set_index("T_ID"), how="inner")
    base["age"], base["sex"] = base.T_AGE, base.T_SEX
    base["lab"] = _label(base, "T_SBP", "T_DBP", "T_GLU0", "T_HBA1C", "T_TCHL", "T_HTN", "T_DM", "T_LIP")
    for k_, c_ in [("htn", "T_HTN"), ("dm", "T_DM"), ("lip", "T_LIP")]: base[k_] = (base[c_] == 2).astype(int)
    out = {"BASE": base}
    f1c = pd.read_csv(CLIN + "FOLLOW_01_DATA.txt").set_index("T00_ID")
    def flagged(k):
        c = pd.read_csv(CLIN + f"FOLLOW_{k:02d}_DATA.txt").set_index("T00_ID"); p = f"T{k:02d}_"
        return (c[p + "HTN"] == 2) | (c[p + "DM"] == 2) | (c[p + "LIP"] == 2)
    for k in (1, 5):
        c = pd.read_csv(CLIN + f"FOLLOW_{k:02d}_DATA.txt").set_index("T00_ID")
        m = pd.read_csv(mri_dir + f"/BrainVol_FOLLOW_{k:02d}.csv").set_index("T00_ID")
        d = c.join(m, how="inner"); p = f"T{k:02d}_"
        prior = None
        for j in range(1, k):  # 2~5차 질환 표시는 '신규 진단'이므로 이전 차수와 누적
            fj = flagged(j); prior = fj if prior is None else (prior.reindex(fj.index).fillna(False) | fj)
        d["age"] = d[p + "AGE"]; d["sex"] = f1c["T00_SEX"].reindex(d.index)
        d["lab"] = _label(d, p + "SBP", p + "DBP", p + "GLU0", p + "HBA1C", p + "TCHL", p + "HTN", p + "DM", p + "LIP", prior)
        out[f"FOLLOW{k}"] = d
    return out

# ---------------------------------------------------------------- 정상 참조 (뇌연령 BAG + 영역 편차)
EXT = {"on": False}
def _design(age, sex, icv):
    a = np.asarray(age, float) - 55.0; sx = np.asarray(sex, float); ic = np.asarray(icv, float)
    if EXT["on"]:   # 확장 정상 참조: 나이 3차항, 성별×나이, ICV×나이
        return np.c_[a, a ** 2, a ** 3, sx, sx * a, sx * a ** 2, ic, ic * a]
    return np.c_[a, a ** 2, sx, ic]

class Ref:
    """정상군만으로 학습하는 참조. 뇌연령 3모델 앙상블(보정) + 영역별 정상 기댓값."""
    def fit(self, nm, seed=0):
        self.vols = [c for c in nm.columns if c.startswith("BV_") and c != "BV_icv" and c not in CLIPPED]
        rng = np.random.RandomState(seed); idx = rng.permutation(len(nm)); k = int(.75 * len(nm))
        a, b = nm.iloc[idx[:k]], nm.iloc[idx[k:]]
        self.models = [
            make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .9], alphas=25, cv=5, max_iter=5000, random_state=seed)),
            make_pipeline(StandardScaler(), SVR(C=10, epsilon=.1)),
            HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, learning_rate=.05,
                                          l2_regularization=10, early_stopping=False, random_state=seed)]
        Xa, Xb = self._x(a), self._x(b)
        self.coefs = []
        for m in self.models:
            m.fit(Xa, a.age.values)
            ag = b.age.values - 55.0
            self.coefs.append(np.linalg.lstsq(np.c_[np.ones(len(ag)), ag, ag ** 2], m.predict(Xb), rcond=None)[0])
        X = _design(nm.age, nm.sex, nm.BV_icv)
        self.norm = LinearRegression().fit(X, nm[self.vols].values)
        self.sd = (nm[self.vols].values - self.norm.predict(X)).std(0)
        self.bag_ref_age = b.age.values; self.bag_ref = self._bag(b)
        return self
    def _x(self, d): return np.c_[d[self.vols].values / d.BV_icv.values[:, None], d.BV_icv.values]
    def _bag(self, d):
        X = self._x(d); ag = d.age.values - 55.0; A = np.c_[np.ones(len(ag)), ag, ag ** 2]
        return np.mean([m.predict(X) - A @ c for m, c in zip(self.models, self.coefs)], axis=0)
    def transform(self, d):
        out = pd.DataFrame(index=d.index)
        out["age"], out["sex"], out["icv"] = d.age.values, d.sex.values, d.BV_icv.values
        out["bag"] = self._bag(d)
        z = (d[self.vols].values - self.norm.predict(_design(d.age, d.sex, d.BV_icv))) / self.sd
        for i, v in enumerate(self.vols): out["z_" + v] = z[:, i]
        r = d[self.vols].values / d.BV_icv.values[:, None]
        for i, v in enumerate(self.vols): out["r_" + v] = r[:, i]
        for v in CLIPPED:   # 상한 변수: 값과 '상한에 걸림' 지시자
            out["k_" + v] = d[v].values / d.BV_icv.values; out["kf_" + v] = (d[v].values >= CEIL[v]).astype(float)
        return out

# ---------------------------------------------------------------- 선별 점수 후보
def _lr(): return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=8, cv=3, max_iter=3000))
def _hgb(): return HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=.05, l2_regularization=5, early_stopping=False, random_state=0)

class AgeBalancedLR:
    """나이 구간 안에서 질환/정상 비율을 맞추는 가중 로지스틱(나이가 아닌 MRI 정보를 쓰게 유도)."""
    def __init__(self, penalty="l2"): self.penalty = penalty
    def fit(self, X, y):
        self.sc = StandardScaler().fit(X); age = X[:, 0]; b = np.digitize(age, [45, 50, 55, 60, 65, 70])
        w = np.ones(len(y))
        for k in np.unique(b):
            m = b == k; p = y[m].mean(); p = min(max(p, 1e-3), 1 - 1e-3)
            w[m & (y == 1)] = .5 / p; w[m & (y == 0)] = .5 / (1 - p)
        kw = dict(Cs=8, cv=3, max_iter=3000)
        if self.penalty == "enet": kw.update(penalty="elasticnet", solver="saga", l1_ratios=[.5], max_iter=2000)
        self.m = LogisticRegressionCV(**kw).fit(self.sc.transform(X), y, sample_weight=w); return self
    def predict_proba(self, X): return self.m.predict_proba(self.sc.transform(X))

class Avg:
    def __init__(self, a, b): self.a, self.b = a, b
    def fit(self, X, y): self.a.fit(X, y); self.b.fit(X, y); return self
    def predict_proba(self, X): return (self.a.predict_proba(X) + self.b.predict_proba(X)) / 2

def colsets(F):
    z = [c for c in F.columns if c.startswith("z_")]; r = [c for c in F.columns if c.startswith("r_")]
    k = [c for c in F.columns if c.startswith("k_") or c.startswith("kf_")]
    return {"age": ["age", "sex"], "age+bag": ["age", "sex", "bag"], "age+z": ["age", "sex"] + z, "age+bag+z": ["age", "sex", "bag"] + z,
            "age+ratios": ["age", "sex", "icv"] + r, "age+z+clip": ["age", "sex"] + z + k, "all": ["age", "sex", "icv", "bag"] + z + r}

CANDIDATES = {  # 이름: (열집합, 분류기)
    "1 나이+성별(기준)": ("age", _lr), "2 +BAG": ("age+bag", _lr), "3 +영역편차z": ("age+z", _lr),
    "4 +BAG+z": ("age+bag+z", _lr), "5 +체적비(원값)": ("age+ratios", _lr),
    "6 GBM(all)": ("all", _hgb), "7 GBM(+BAG+z)": ("age+bag+z", _hgb),
    "8 z 나이균형 가중": ("age+z", lambda: AgeBalancedLR()), "9 z ElasticNet-LR": ("age+z", lambda: AgeBalancedLR("enet")),
    "B z+클리핑변수(지시자)": ("age+z+clip", _lr),
    "A LR(z)+GBM(+BAG+z) 평균": ("age+bag+z", lambda: Avg(_lr(), _hgb()))}

def fit_candidates(Ftr, y, names):
    cs = colsets(Ftr); out = {}
    for n in names:
        cols, mk = CANDIDATES[n]; out[n] = (cols, mk().fit(Ftr[cs[cols]].values, y))
    return out

def score(model, F):
    cols, m = model; X = F[colsets(F)[cols]].values
    return m.predict_proba(X)[:, 1]

def strat_auc(s, y, age, edges=(0, 50, 60, 70, 200)):
    num = den = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (age >= lo) & (age < hi); n1, n0 = y[m].sum(), (1 - y[m]).sum()
        if n1 >= 10 and n0 >= 10: w = n1 * n0; num += w * roc_auc_score(y[m], s[m]); den += w
    return num / den if den else np.nan

def metrics(s, y, age):
    return dict(auc=roc_auc_score(y, s), auc_age=strat_auc(s, y, age), ap=average_precision_score(y, s))

def split_ref(trd, seed):
    rng = np.random.RandomState(seed); nm = trd[trd.lab == "normal"]; idx = rng.permutation(len(nm)); k = int(.6 * len(nm))
    return nm.iloc[idx[:k]], nm.iloc[idx[k:]]   # (참조용 정상군, 선별 점수 학습용 정상군)

def cmd_cv(a):
    D = load_all(a.mri); base = D["BASE"]; lab = base[base.lab.isin(["normal", "disease"])]
    names = list(CANDIDATES) if not a.only else [n for n in CANDIDATES if n[0] in a.only]
    rows = []; oof_keep = {}
    for rep in range(a.reps):
        oof = {n: np.zeros(len(lab)) for n in names}
        for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(lab, lab.lab)):
            trd, ted = lab.iloc[tr], lab.iloc[te]
            ref_nm, rest_nm = split_ref(trd, rep * 10 + f)
            ref = Ref().fit(ref_nm, rep * 10 + f)
            clf_tr = pd.concat([rest_nm, trd[trd.lab == "disease"]])
            Ftr, Fte = ref.transform(clf_tr), ref.transform(ted)
            models = fit_candidates(Ftr, (clf_tr.lab == "disease").astype(int).values, names)
            for n in names: oof[n][te] = score(models[n], Fte)
        y = (lab.lab == "disease").astype(int).values
        for n in names:
            r = metrics(oof[n], y, lab.age.values); r.update(model=n, rep=rep); rows.append(r)
        oof_keep = oof
        print("rep", rep, "done", flush=True)
    R = pd.DataFrame(rows); pd.set_option("display.width", 200)
    print(R.groupby("model")[["auc", "auc_age", "ap"]].agg(["mean", "std"]).round(3))
    R.to_csv(a.out + "/cv_results.csv", index=False)

def cmd_heads(a):
    D = load_all(a.mri); base = D["BASE"]; lab = base[base.lab.isin(["normal", "disease"])]
    heads = ["htn", "dm", "lip"]; res = []
    for rep in range(a.reps):
        P = {h: np.zeros(len(lab)) for h in heads + ["any"]}
        for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(lab, lab.lab)):
            trd, ted = lab.iloc[tr], lab.iloc[te]
            ref_nm, rest_nm = split_ref(trd, rep * 10 + f); ref = Ref().fit(ref_nm, rep * 10 + f)
            Fte = ref.transform(ted); cols = colsets(Fte)["age+z+clip"]
            for h in heads + ["any"]:
                pos = trd[trd.lab == "disease"] if h == "any" else trd[(trd.lab == "disease") & (trd[h] == 1)]
                ctr = pd.concat([rest_nm, pos]); Ftr = ref.transform(ctr)
                m = _lr().fit(Ftr[cols].values, (ctr.index.isin(pos.index)).astype(int))
                P[h][te] = m.predict_proba(Fte[cols].values)[:, 1]
        y = (lab.lab == "disease").astype(int).values; age = lab.age.values
        comb = {"any 단일": P["any"], "질환별 최댓값": np.max([P[h] for h in heads], axis=0),
                "질환별 noisy-OR": 1 - np.prod([1 - P[h] for h in heads], axis=0),
                "any+질환별 평균": (P["any"] + np.max([P[h] for h in heads], axis=0)) / 2}
        for n, sc in comb.items():
            r = metrics(sc, y, age); r.update(kind="전체 질환 vs 정상", model=n, rep=rep); res.append(r)
        for h in heads:
            mk = ((lab.lab == "normal") | (lab[h] == 1)).values
            r = metrics(P[h][mk], (lab[h] == 1).astype(int).values[mk], age[mk]); r.update(kind=f"{h} vs 정상", model=f"{h} 전용", rep=rep); res.append(r)
            r = metrics(P["any"][mk], (lab[h] == 1).astype(int).values[mk], age[mk]); r.update(kind=f"{h} vs 정상", model="any 단일", rep=rep); res.append(r)
        print("rep", rep, "done", flush=True)
    R = pd.DataFrame(res); pd.set_option("display.width", 200)
    print(R.groupby(["kind", "model"])[["auc", "auc_age", "ap"]].mean().round(3))

FINAL_COLS = "age+z+clip"
HEADS = ["any", "htn", "dm", "lip"]

def _fit_heads(ref, rest_nm, dis, cols):
    out = {}
    for h in HEADS:
        pos = dis if h == "any" else dis[dis[h] == 1]
        ctr = pd.concat([rest_nm, pos]); F = ref.transform(ctr)
        out[h] = _lr().fit(F[colsets(F)[cols]].values, ctr.index.isin(pos.index).astype(int))
    return out

def _score_heads(ref, heads, d):
    F = ref.transform(d); cols = colsets(F)[FINAL_COLS]
    return pd.DataFrame({h: heads[h].predict_proba(F[cols].values)[:, 1] for h in HEADS}, index=d.index), F

def boot_ci(s, y, n=1000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        i = rng.choice(len(y), len(y))
        if y[i].min() != y[i].max(): v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])

def cmd_fit(a):
    D = load_all(a.mri); base = D["BASE"]; lab = base[base.lab.isin(["normal", "disease"])]
    # (1) 임계값: 5겹 OOF 에서 정상군의 90백분위(=특이도 90%)
    oof = {h: np.zeros(len(lab)) for h in HEADS}
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=123).split(lab, lab.lab)):
        trd, ted = lab.iloc[tr], lab.iloc[te]; rn, rest = split_ref(trd, 500 + f); r = Ref().fit(rn, 500 + f)
        hd = _fit_heads(r, rest, trd[trd.lab == "disease"], FINAL_COLS); sc, _ = _score_heads(r, hd, ted)
        for h in HEADS: oof[h][te] = sc[h].values
    normal_mask = (lab.lab == "normal").values
    thr = {h: float(np.quantile(oof[h][normal_mask], .90)) for h in HEADS}
    y = (lab.lab == "disease").astype(int).values
    dev = metrics(oof["any"], y, lab.age.values)
    # (2) 최종 모델: BASE 전체
    ref_nm, rest_nm = split_ref(lab, 0); ref = Ref().fit(ref_nm, 0)
    heads = _fit_heads(ref, rest_nm, lab[lab.lab == "disease"], FINAL_COLS)
    base_clf = _lr().fit(ref.transform(pd.concat([rest_nm, lab[lab.lab == "disease"]]))[["age", "sex"]].values,
                         pd.concat([rest_nm, lab[lab.lab == "disease"]]).lab.eq("disease").astype(int).values)
    joblib.dump(dict(ref=ref, heads=heads, thr=thr, cols=FINAL_COLS, base_clf=base_clf), a.out + "/model.joblib")
    contract = dict(required=["age", "sex(1=남,2=여)", "BV_icv"] + ref.vols + CLIPPED, note="열 이름은 대회 BrainVol 파일과 동일. 체적 단위 mL, 나이는 MRI 시점 나이.")
    json.dump(contract, open(a.out + "/input_contract.json", "w"), ensure_ascii=False, indent=1)
    # (3) 동결 외부 평가: FOLLOW 1·5차 (모델 선택에 사용하지 않음)
    rows = []
    for k in ("FOLLOW1", "FOLLOW5"):
        d = D[k]; d = d[d.lab.isin(["normal", "disease"])]; yk = (d.lab == "disease").astype(int).values
        sc, F = _score_heads(ref, heads, d); sb = base_clf.predict_proba(F[["age", "sex"]].values)[:, 1]
        m = metrics(sc["any"].values, yk, d.age.values); mb = metrics(sb, yk, d.age.values)
        lo, hi = boot_ci(sc["any"].values, yk)
        flag = sc["any"].values >= thr["any"]
        rows.append(dict(set=k, n=len(d), n_disease=int(yk.sum()), auc=m["auc"], auc_lo=lo, auc_hi=hi, auc_age=m["auc_age"], ap=m["ap"],
                         base_auc=mb["auc"], base_auc_age=mb["auc_age"], sens=float(flag[yk == 1].mean()), spec=float((~flag)[yk == 0].mean())))
    R = pd.DataFrame(rows); R.to_csv(a.out + "/external_eval.csv", index=False)
    json.dump(dict(dev_cv_auc=dev, thresholds=thr, n_train=len(lab)), open(a.out + "/fit_summary.json", "w"), indent=1)
    print("개발 OOF(any):", {k: round(v, 3) for k, v in dev.items()}); print("임계값(특이도90%):", {k: round(v, 3) for k, v in thr.items()})
    pd.set_option("display.width", 220); print(R.round(3).to_string(index=False))

def cmd_predict(a):
    M = joblib.load(a.model); ref, heads, thr = M["ref"], M["heads"], M["thr"]
    d = pd.read_csv(a.input); d.index = d["id"] if "id" in d.columns else d.index
    miss = [c for c in ["age", "sex", "BV_icv"] + ref.vols + CLIPPED if c not in d.columns]
    if miss: sys.exit("입력에 없는 열: " + ", ".join(miss))
    sc, F = _score_heads(ref, heads, d)
    out = pd.DataFrame(index=d.index)
    out["bag_years"] = F["bag"].round(2)
    pct = []
    for ag, bg in zip(d.age.values, F["bag"].values):
        w = np.abs(ref.bag_ref_age - ag) <= 5
        if w.sum() < 30: w = np.abs(ref.bag_ref_age - ag) <= 8
        pct.append(round(float((ref.bag_ref[w] < bg).mean() * 100), 1))
    out["bag_percentile_in_age"] = pct
    z = F[[c for c in F.columns if c.startswith("z_")]]
    top = []
    for _, r in z.iterrows():
        o = r.abs().sort_values(ascending=False).index[:3]; top.append("; ".join(f"{c[5:]}({r[c]:+.1f})" for c in o))
    out["top3_deviant_regions(z)"] = top
    out["score_any_disease"] = sc["any"].round(3); out["score_htn"] = sc["htn"].round(3)
    out["score_dm"] = sc["dm"].round(3); out["score_lip"] = sc["lip"].round(3)
    out["screen_flag(spec90)"] = (sc["any"] >= thr["any"]).astype(int)
    out.to_csv(a.output, index_label="id"); print("저장:", a.output, "| 선별 표시:", int(out["screen_flag(spec90)"].sum()), "/", len(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--mri"); ap.add_argument("--out", default=".")
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--only", default=""); ap.add_argument("--model"); ap.add_argument("--input"); ap.add_argument("--output"); ap.add_argument("--ext", action="store_true")
    a = ap.parse_args(); EXT["on"] = a.ext
    if a.cmd == "cv": cmd_cv(a)
    if a.cmd == "heads": cmd_heads(a)
    if a.cmd == "fit": cmd_fit(a)
    if a.cmd == "predict": cmd_predict(a)
