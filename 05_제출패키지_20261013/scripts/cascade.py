"""순차적 모달리티 획득(cascade): 설문·체격 -> (애매하면) +MRI -> (애매하면) +SNP.
애매함의 기준은 보정 확률이 [lo, hi] 사이일 때. 검사를 얼마나 줄이고 성능이 얼마나 유지되는지 평가한다.
평가: 학습에 쓰지 않은 30% 검증 개체. 각 단계 확률은 5겹 교차 보정(Platt)한 값."""
import argparse, glob, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
import mri_screen as ms, multimodal as mm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
for k in ["mri", "kept", "geno", "te", "model", "out"]: ap.add_argument("--" + k)
a = ap.parse_args(); M = joblib.load(a.model); ref = M["ref"]; D = ms.load_all(a.mri); b = D["BASE"]
K = np.load(a.kept, allow_pickle=True); idx = {n: i for i, n in enumerate(K["names"])}; X = K["X"]
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
pos = {n: i for i, n in enumerate(fam)}; lab = b.reindex(pd.Index(fam)); te = np.load(a.te)

def platt_cv(s, y):
    lg = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))).reshape(-1, 1); out = np.zeros(len(y))
    for tr, tt in StratifiedKFold(5, shuffle=True, random_state=0).split(lg, y): out[tt] = LogisticRegression(C=1e6).fit(lg[tr], y[tr]).predict_proba(lg[tt])[:, 1]
    return out

def bacc(pred, y): return 0.5 * (pred[y == 1].mean() + (1 - pred[y == 0].mean()))
rows = []
for h, hn in [("any", "전체 질환"), ("htn", "고혈압"), ("dm", "당뇨"), ("lip", "고지혈")]:
    posf = (lambda d: d.lab == "disease") if h == "any" else (lambda d: d[h] == 1)
    ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf(ted)).values]; y = posf(ted).astype(int).values
    sn = np.stack([X[idx[s]][[pos[n] for n in ted.index]] for s in M["leads"][h]], axis=1)
    P = {}
    for st, cn in [(1, "+설문·체격"), (2, "+MRI+설문·체격"), (3, "전부")]:
        Xa, _ = mm.blocks(ref, ted, sn, M["combos"][cn]); P[st] = platt_cv(M["models"][(h, cn)].predict_proba(Xa)[:, 1], y)
    cut = y.mean()                                   # 최종 판정 기준: 표본 유병률
    full = bacc((P[3] >= cut).astype(float), y); s1 = bacc((P[1] >= cut).astype(float), y); s2 = bacc((P[2] >= cut).astype(float), y)
    rows.append(dict(target=hn, policy="1단계만(설문·체격)", mri=0.0, snp=0.0, bacc=s1)); rows.append(dict(target=hn, policy="2단계만(+MRI, SNP 없이)", mri=1.0, snp=0.0, bacc=s2))
    rows.append(dict(target=hn, policy="전부 사용(전원 MRI·SNP)", mri=1.0, snp=1.0, bacc=full))
    for lo_off, hi_off in [(.10, .15), (.15, .25), (.20, .35), (.25, .45)]:
        lo, hi = max(cut - lo_off, .02), min(cut + hi_off, .98)
        go2 = (P[1] >= lo) & (P[1] < hi); go3 = go2 & (P[2] >= lo) & (P[2] < hi)
        final = np.where(go3, P[3], np.where(go2, P[2], P[1])); rows.append(dict(target=hn, policy=f"순차 [{lo:.2f}, {hi:.2f}]", mri=float(go2.mean()), snp=float(go3.mean()), bacc=bacc((final >= cut).astype(float), y)))
R = pd.DataFrame(rows); R.to_csv(a.out + "/cascade.csv", index=False); pd.set_option("display.width", 200)
for t, g in R.groupby("target", sort=False): print("\n", t); print(g.assign(MRI촬영=lambda d: (d.mri * 100).round(0).astype(int).astype(str) + "%", SNP검사=lambda d: (d.snp * 100).round(0).astype(int).astype(str) + "%", 균형정확도=lambda d: d.bacc.round(3))[["policy", "MRI촬영", "SNP검사", "균형정확도"]].to_string(index=False))
