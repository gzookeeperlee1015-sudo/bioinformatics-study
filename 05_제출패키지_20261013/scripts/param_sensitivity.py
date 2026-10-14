"""하이퍼파라미터 민감도: (a) 정상 참조에 쓰는 정상군 비율(40/60/80%), (b) 로지스틱 규제 강도(교차검증 선택 vs 고정 C).
입력은 나이·성별 + MRI 영역편차(z) + 설문·체격(SNP 제외, 빠른 비교). 5겹 교차검증 AUC. 매 폴드의 참조는 학습 개체 안에서만 만든다."""
import argparse, glob, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import mri_screen as ms, multimodal as mm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
for k in ["mri", "geno", "out"]: ap.add_argument("--" + k)
a = ap.parse_args(); b = ms.load_all(a.mri)["BASE"]
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
lab = b.reindex(pd.Index(fam)); HEADS = ["any", "htn", "dm", "lip"]; HN = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
P = {"any": (lab.lab == "disease").values, "htn": (lab.htn == 1).values, "dm": (lab.dm == 1).values, "lip": (lab.lip == 1).values}; ISN = (lab.lab == "normal").values
LAB = lab.lab.isin(["normal", "disease"]).values; ALL = np.where(LAB)[0]; y_any = P["any"][ALL].astype(int)
def split(trd, seed, frac):
    rng = np.random.RandomState(seed); nm = trd[trd.lab == "normal"]; idx = rng.permutation(len(nm)); k = int(frac * len(nm)); return nm.iloc[idx[:k]], nm.iloc[idx[k:]]
def clf(C):
    if C == "CV": return mm._lr()
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=C, max_iter=3000))
CFG = [("참조 정상군 40%", .4, "CV"), ("참조 정상군 60% (현재)", .6, "CV"), ("참조 정상군 80%", .8, "CV"), ("규제 C=0.001 (강함)", .6, .001), ("규제 C=0.01", .6, .01), ("규제 C=1 (약함)", .6, 1.0)]
sc = {(h, n): [] for h in HEADS for n, _, _ in CFG}
for f, (ta, tb) in enumerate(StratifiedKFold(5, shuffle=True, random_state=11).split(ALL, y_any)):
    tri, vai = ALL[ta], ALL[tb]; trd = lab.iloc[tri]
    for n, frac, C in CFG:
        rn, rest = split(trd, f, frac); ref = ms.Ref().fit(rn, f); cs = ms.colsets(ref.transform(rest.iloc[:5]))["age+z+clip"]
        for h in HEADS:
            pos_tr = trd[P[h][tri]]; ctr = pd.concat([rest, pos_tr]); yc = np.r_[np.zeros(len(rest)), np.ones(len(pos_tr))].astype(int)
            vm = np.array([i for i in vai if ISN[i] or P[h][i]]); dv = lab.iloc[vm]; yv = P[h][vm].astype(int)
            mk = lambda d, F: np.column_stack([F[cs].values] + [ms._clean(d[c]).values for c in mm.Q])
            sc[(h, n)].append(roc_auc_score(yv, clf(C).fit(mk(ctr, ref.transform(ctr)), yc).predict_proba(mk(dv, ref.transform(dv)))[:, 1]))
    print("폴드", f, "완료", flush=True)
R = pd.DataFrame([dict(질환=HN[h], 설정=n, auc=np.mean(sc[(h, n)]), sd=np.std(sc[(h, n)])) for h in HEADS for n, _, _ in CFG]); R.to_csv(a.out + "/param_sensitivity.csv", index=False)
pd.set_option("display.width", 200); print(R.pivot(index="설정", columns="질환", values="auc").round(3).loc[[n for n, _, _ in CFG], [HN[h] for h in HEADS]].to_string())
