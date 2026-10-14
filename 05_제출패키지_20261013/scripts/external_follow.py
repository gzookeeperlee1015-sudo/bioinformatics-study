"""FOLLOW 1·5차 외부 평가(SNP 제외 모델). 설문·체격 변수를 BASE 이름으로 매핑한다.
SNP 모델은 FOLLOW 와 유전체 연결 규칙이 없어 평가할 수 없다.
주의: 2~5차의 질환·가족력 응답은 '직전 이후 신규'이므로 이전 차수와 누적(OR)해서 쓴다."""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
import mri_screen as ms, multimodal as mm
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
for k in ["mri", "model", "out"]: ap.add_argument("--" + k)
a = ap.parse_args()
M = joblib.load(a.model); ref = M["ref"]; D = ms.load_all(a.mri); C = ms.CLIN
fl = {k: pd.read_csv(C + f"FOLLOW_{k:02d}_DATA.txt").set_index("T00_ID") for k in range(1, 6)}

def cum(col, k, val=2):
    out = None
    for j in range(1, k + 1):
        c = fl[j]; v = (c[f"T{j:02d}_{col}"] == val)
        out = v if out is None else (out.reindex(out.index.union(v.index)).fillna(False) | v.reindex(out.index.union(v.index)).fillna(False))
    return out

def prep(k):
    d = D[f"FOLLOW{k}"].copy(); p = f"T{k:02d}_"
    d["htn"] = cum("HTN", k).reindex(d.index).fillna(False).astype(int); d["dm"] = cum("DM", k).reindex(d.index).fillna(False).astype(int)
    d["lip"] = cum("LIP", k).reindex(d.index).fillna(False).astype(int)
    d["T_DRINK"], d["T_SMOKE"], d["T_EXER"], d["T_PSM"] = d[p + "DRINK"], d[p + "SMOKE"], d[p + "EXER"], d[p + "PSM"]
    for bn, fn in [("T_FMFHT1", "FMFHT"), ("T_FMFHT2", "FMMHT"), ("T_FMFDM1", "FMFDM"), ("T_FMFDM2", "FMMDM")]:
        d[bn] = np.where(cum(fn, k).reindex(d.index).fillna(False), 2, 1)       # 누적: 한 번이라도 있으면 2
    d["T_BMI"] = ms._clean(d[p + "WEIGHT"]) / (ms._clean(d[p + "HEIGHT"]) / 100.0) ** 2
    d["T_WAIST"] = d[p + "WAIST"]
    return d

COMBOS = ["나이·성별", "+MRI", "+설문·체격", "+MRI+설문·체격"]
HEADS = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
rows = []
for k in (1, 5):
    d = prep(k)
    for h, hn in HEADS.items():
        posm = (d.lab == "disease") if h == "any" else (d[h] == 1)
        dd = d[(d.lab == "normal") | posm]; y = posm.loc[dd.index].astype(int).values
        for cn in COMBOS:
            parts = M["combos"][cn]; Xa, _ = mm.blocks(ref, dd, None, parts)
            s = M["models"][(h, cn)].predict_proba(Xa)[:, 1]
            bs = [roc_auc_score(y[i], s[i]) for i in (np.random.RandomState(r).choice(len(y), len(y)) for r in range(300))]
            thr = M["thr"][(h, cn)]; flag = s >= thr
            rows.append(dict(wave=f"FOLLOW{k}", target=hn, model=cn, n=len(y), pos=int(y.sum()), auc=roc_auc_score(y, s), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             auc_age=ms.strat_auc(s, y, dd.age.values), sens=float(flag[y == 1].mean()), spec=float((~flag)[y == 0].mean())))
R = pd.DataFrame(rows); R.to_csv(a.out + "/external_follow_nosnp.csv", index=False); pd.set_option("display.width", 220)
for w, g in R.groupby("wave"):
    print("\n", w); print(g.pivot(index="model", columns="target", values="auc").round(3).loc[COMBOS, ["전체 질환", "고혈압", "당뇨", "고지혈"]].to_string())
print("\n95% 신뢰구간 (전체 질환, +MRI+설문·체격):")
print(R[(R.model == "+MRI+설문·체격")][["wave", "target", "n", "pos", "auc", "lo", "hi", "auc_age", "sens", "spec"]].round(3).to_string(index=False))
