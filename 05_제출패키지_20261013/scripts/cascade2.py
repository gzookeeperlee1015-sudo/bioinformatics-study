"""질환별 순차 선별: 모달리티 순서와 구성을 질환마다 다르게 하고, 애매함 구간을 유병률에 비례하게 정한다.
정책 후보(질환별로 모두 평가): A 설문→MRI→SNP, B 설문→SNP→MRI, C 설문→SNP(MRI 생략), D 설문→MRI(SNP 생략)."""
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
    for key, cn in [("q", "+설문·체격"), ("qm", "+MRI+설문·체격"), ("qs", "+설문·체격+SNP"), ("all", "전부")]:
        Xa, _ = mm.blocks(ref, ted, sn, M["combos"][cn]); P[key] = platt_cv(M["models"][(h, cn)].predict_proba(Xa)[:, 1], y)
    cut = y.mean(); full = bacc((P["all"] >= cut).astype(float), y)
    pol = {"A 설문→MRI→SNP": ["q", "qm", "all"], "B 설문→SNP→MRI": ["q", "qs", "all"], "C 설문→SNP(MRI 생략)": ["q", "qs"], "D 설문→MRI(SNP 생략)": ["q", "qm"]}
    for name, stages in pol.items():
        for alo, bhi in [(.3, 2.0), (.5, 1.5), (.5, 2.5), (.7, 3.0)]:
            lo, hi = cut * alo, min(cut * bhi, .98); undecided = np.ones(len(y), bool); final = P[stages[0]].copy(); used = {"m": 0.0, "s": 0.0}
            amb = (P[stages[0]] >= lo) & (P[stages[0]] < hi)
            for st in stages[1:]:
                used["m"] += float(amb.mean()) if st in ("qm",) else 0.0; used["s"] += float(amb.mean()) if st in ("qs",) else 0.0
                final = np.where(amb, P[st], final)
                if st == "all":                      # 마지막 단계는 앞 단계에서 이미 가진 모달리티에 나머지를 추가
                    pass
                amb = amb & (P[st] >= lo) & (P[st] < hi)
            # 3단계 정책(A, B): 마지막 'all' 단계에서 나머지 모달리티를 추가 획득
            if len(stages) == 3:
                first_extra = "m" if stages[1] == "qm" else "s"; second_extra = "s" if first_extra == "m" else "m"
                used = {first_extra: float(((P[stages[0]] >= lo) & (P[stages[0]] < hi)).mean()), second_extra: float((((P[stages[0]] >= lo) & (P[stages[0]] < hi)) & (P[stages[1]] >= lo) & (P[stages[1]] < hi)).mean())}
            rows.append(dict(target=hn, policy=name, band=f"[{lo:.2f},{hi:.2f}]", mri=used["m"], snp=used["s"], bacc=bacc((final >= cut).astype(float), y), full=full, q_only=bacc((P["q"] >= cut).astype(float), y)))
R = pd.DataFrame(rows); R.to_csv(a.out + "/cascade2.csv", index=False)
pd.set_option("display.width", 220)
for t, g in R.groupby("target", sort=False):
    full, q0 = g.full.iloc[0], g.q_only.iloc[0]; print(f"\n[{t}] 전원 MRI·SNP 균형정확도 {full:.3f} | 설문·체격만 {q0:.3f}")
    gg = g[(g.bacc >= full - 0.012)].sort_values(["mri", "snp"]); gg = gg.assign(비용=lambda d: d.mri + d.snp)
    print(gg.sort_values("비용").head(5).assign(MRI=lambda d: (d.mri * 100).round().astype(int).astype(str) + "%", SNP=lambda d: (d.snp * 100).round().astype(int).astype(str) + "%", 균형정확도=lambda d: d.bacc.round(3))[["policy", "band", "MRI", "SNP", "균형정확도"]].to_string(index=False) if len(gg) else "  (전원 사용 대비 -0.012 이내 정책 없음)")
