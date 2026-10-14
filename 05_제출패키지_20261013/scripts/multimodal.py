"""모달리티 결측 강건 멀티모달 질환 선별 모델 (MRI + 설문·체격 + SNP).

학습: BASE 라벨 개체 70%(SNP 대표 변이 선택과 모델 학습), 평가: 나머지 30%.
질환(전체/고혈압/당뇨/고지혈)마다 모달리티 조합 8가지를 따로 학습해 두고,
predict 때 입력에 있는 모달리티로 가능한 가장 풍부한 조합을 자동 선택한다.
의존: 같은 폴더의 mri_screen.py (정상 참조, 영역 편차).
"""
import argparse, glob, json, sys, zipfile, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
import mri_screen as ms
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Q = ["T_DRINK", "T_SMOKE", "T_EXER", "T_FMFHT1", "T_FMFHT2", "T_FMFDM1", "T_FMFDM2", "T_PSM", "T_BMI", "T_WAIST"]
HEADS = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
COMBOS = {"나이·성별": ("age",), "+SNP": ("age", "snp"), "+MRI": ("age", "mri"), "+MRI+SNP": ("age", "mri", "snp"),
          "+설문·체격": ("age", "q"), "+설문·체격+SNP": ("age", "q", "snp"), "+MRI+설문·체격": ("age", "mri", "q"),
          "전부": ("age", "mri", "q", "snp")}

def _lr(): return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegressionCV(Cs=8, cv=3, max_iter=3000))

def lead_snps(R, idx, trait, maxn=15, minr=0.05, win=1_000_000):
    r = R["r_" + trait].values; chosen = []
    for i in np.argsort(-np.abs(r)):
        if abs(r[i]) < minr or len(chosen) >= maxn: break
        if R.snp.values[i] not in idx: continue
        if all(not (R.chr.values[i] == R.chr.values[j] and abs(R.pos.values[i] - R.pos.values[j]) < win) for j in chosen): chosen.append(i)
    return [R.snp.values[i] for i in chosen]

def blocks(ref, d, snps, parts):
    F = ref.transform(d); mc = ms.colsets(F)["age+z+clip"][2:]
    out = {"age": F[["age", "sex"]].values, "mri": F[mc].values}
    if "q" in parts: out["q"] = np.column_stack([ms._clean(d[c]).values for c in Q])
    if snps is not None: out["snp"] = snps
    return np.hstack([out[p] for p in parts]), F

NAMES = {"age": "나이·성별", "mri": "MRI", "q": "설문·체격", "snp": "SNP"}

def evidence(model, Xa, parts, nq, nsnp):
    """점수(로짓)를 모달리티별 기여로 분해한다: 절편 + 나이·성별 + MRI + 설문·체격 + SNP = 로짓.
    양수는 질환 쪽, 음수는 정상 쪽으로 점수를 민 정도. 로지스틱 모델이 선형이라 정확히 더해진다."""
    imp, sc, lr = model.steps[0][1], model.steps[1][1], model.steps[2][1]
    Z = sc.transform(imp.transform(Xa)) * lr.coef_[0]; n0 = Xa.shape[1]
    size = {"age": 2, "q": nq, "snp": nsnp}; size["mri"] = n0 - sum(size[p] for p in parts if p != "mri")
    owner = np.concatenate([[p] * size[p] for p in parts])      # 원 열 -> 모달리티
    ind = getattr(imp, "indicator_", None); ind_cols = list(ind.features_) if ind is not None else []
    col_owner = np.concatenate([owner[:Z.shape[1] - len(ind_cols)], owner[ind_cols]]) if ind_cols else owner[:Z.shape[1]]
    res = []
    for row in Z:
        parts_ = {p: row[col_owner == p].sum() for p in parts}
        res.append("; ".join(f"{NAMES[p]}({v:+.2f})" for p, v in sorted(parts_.items(), key=lambda kv: -abs(kv[1]))) + f"; 절편({lr.intercept_[0]:+.2f})")
    return res

def cmd_fit(a):
    D = ms.load_all(a.mri); b = D["BASE"]
    R = pd.read_csv(a.scan); K = np.load(a.kept, allow_pickle=True); names = list(K["names"]); X = K["X"]; idx = {n: i for i, n in enumerate(names)}
    z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
    pos = {n: i for i, n in enumerate(fam)}
    lab = b.reindex(pd.Index(fam)); tr = np.load(a.tr); te = np.load(a.te)
    trd = lab.iloc[tr]; rn, rest = ms.split_ref(trd, 0); ref = ms.Ref().fit(rn, 0)
    models, leads, thr, rows = {}, {}, {}, []
    for h, hn in HEADS.items():
        posf = (lambda d: d.lab == "disease") if h == "any" else (lambda d: d[h] == 1)
        leads[h] = lead_snps(R, idx, h)
        sn = lambda d, L=leads[h]: np.stack([X[idx[s]][[pos[n] for n in d.index]] for s in L], axis=1)
        ctr = pd.concat([rest, trd[posf(trd)]]); yc = ctr.index.isin(trd[posf(trd)].index).astype(int)
        ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf(ted)).values]; yt = posf(ted).astype(int).values
        for cn, parts in COMBOS.items():
            Xa, _ = blocks(ref, ctr, sn(ctr), parts); Xb, Fb = blocks(ref, ted, sn(ted), parts)
            m = _lr().fit(Xa, yc); s = m.predict_proba(Xb)[:, 1]; models[(h, cn)] = m
            thr[(h, cn)] = float(np.quantile(s[yt == 0], .90))
            bs = [roc_auc_score(yt[i], s[i]) for i in (np.random.RandomState(k).choice(len(yt), len(yt)) for k in range(300))]
            flag = s >= thr[(h, cn)]
            rows.append(dict(target=hn, model=cn, n=len(yt), pos=int(yt.sum()), auc=roc_auc_score(yt, s), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             auc_age=ms.strat_auc(s, yt, ted.age.values), sens_at_spec90=float(flag[yt == 1].mean())))
        print(h, "done", flush=True)
    joblib.dump(dict(ref=ref, models=models, leads=leads, thr=thr, combos=COMBOS, Q=Q), a.out + "/multimodal.joblib")
    pd.DataFrame(rows).to_csv(a.out + "/multimodal_test_eval.csv", index=False)
    pd.set_option("display.width", 220); print(pd.DataFrame(rows).round(3).to_string(index=False))

def cmd_predict(a):
    M = joblib.load(a.model); ref = M["ref"]; d = pd.read_csv(a.input); d.index = d["id"] if "id" in d.columns else d.index
    has_mri = all(c in d.columns for c in ["age", "sex", "BV_icv"] + ref.vols + ms.CLIPPED)
    has_q = all(c in d.columns for c in M["Q"])
    out = pd.DataFrame(index=d.index)
    if has_mri:
        F = ref.transform(d); out["bag_years"] = F["bag"].round(2)
        z = F[[c for c in F.columns if c.startswith("z_")]]
        out["top3_deviant_regions(z)"] = ["; ".join(f"{c[5:]}({r[c]:+.1f})" for c in r.abs().sort_values(ascending=False).index[:3]) for _, r in z.iterrows()]
    used = {}
    for h, hn in HEADS.items():
        has_snp = all(s in d.columns for s in M["leads"][h])
        parts = ["age"] + (["mri"] if has_mri else []) + (["q"] if has_q else []) + (["snp"] if has_snp else [])
        cn = next(k for k, v in M["combos"].items() if set(v) == set(parts)); used[h] = cn
        sn = np.stack([d[s].values for s in M["leads"][h]], axis=1) if has_snp else None
        dd = d if has_mri else d.assign(BV_icv=1.0, **{c: 1.0 for c in ref.vols + ms.CLIPPED})
        Xa, _ = blocks(ref, dd, sn, M["combos"][cn])
        s = M["models"][(h, cn)].predict_proba(Xa)[:, 1]
        if "calib" in M and (h, cn) in M["calib"]:   # 보정 확률: 표본 유병률 기준, --prevalence 가 있으면 그 유병률로 환산
            ca, cb, pi_s = M["calib"][(h, cn)]; lgt = ca * np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))) + cb
            pv = PREV_ARG.get(h)
            if pv: lgt = lgt + np.log(pv / (1 - pv)) - np.log(pi_s / (1 - pi_s))
            out[f"prob_{h}"] = (1 / (1 + np.exp(-lgt))).round(3)
        out[f"why_{h}(logit)"] = evidence(M["models"][(h, cn)], Xa, M["combos"][cn], len(M["Q"]), 0 if sn is None else sn.shape[1])
        if "thr_age" in M:   # 나이대별 기준(<50, 50-59, 60+): 나이에 따른 오탐·민감도 편향을 줄임
            thr = np.array(M["thr_age"][(h, cn)])[np.digitize(d.age.values, [50, 60])]; out[f"score_{h}"] = s.round(3); out[f"flag_{h}(spec90_by_age)"] = (s >= thr).astype(int)
        else: out[f"score_{h}"] = s.round(3); out[f"flag_{h}(spec90)"] = (s >= M["thr"][(h, cn)]).astype(int)
    out.to_csv(a.output, index_label="id"); print("사용된 모달리티 조합:", used); print("저장:", a.output)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    for k in ["mri", "scan", "kept", "geno", "tr", "te", "out", "model", "input", "output", "prevalence"]: ap.add_argument("--" + k)
    a = ap.parse_args()
    PREV_ARG = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.prevalence.split(",")} if a.prevalence else {}   # 예: any=0.28,htn=0.2
    {"fit": cmd_fit, "predict": cmd_predict}[a.cmd](a)
