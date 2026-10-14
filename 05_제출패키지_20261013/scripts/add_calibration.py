"""최종 모델에 확률 보정(Platt)을 붙인다. 5겹 교차검증 점수(cv_oof.npz)의 로짓과 실제 라벨로 조합·질환마다 로지스틱 보정을 학습하고
model 파일의 M["calib"][(질환, 조합)] = (기울기, 절편, 표본 유병률)로 저장한다. predict 는 prob_* 열(표본 유병률 기준 확률)을 낸다.
한계: 보정은 교차검증 모델의 점수에서 배웠고 최종 모델(100% 학습) 점수에 적용한다. 점수 분포가 약간 다를 수 있다."""
import argparse, glob, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
import mri_screen as ms
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
for k in ["mri", "geno", "oof", "model"]: ap.add_argument("--" + k)
a = ap.parse_args(); b = ms.load_all(a.mri)["BASE"]
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
lab = b.reindex(pd.Index(fam)); V = np.load(a.oof); REPS = sorted({int(k.split("|")[1]) for k in V.files if k.startswith("oof|")})
P = {"any": (lab.lab == "disease").values, "htn": (lab.htn == 1).values, "dm": (lab.dm == 1).values, "lip": (lab.lip == 1).values}; ISN = (lab.lab == "normal").values
M = joblib.load(a.model); calib = {}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
for (h, cn) in M["models"]:
    mk = ISN | P[h]; y = P[h][mk].astype(int); s = np.mean([V[f"oof|{r}|{h}|{cn}"] for r in REPS], axis=0)[mk]
    m = LogisticRegression(C=1e6).fit(lg(s).reshape(-1, 1), y); calib[(h, cn)] = (float(m.coef_[0, 0]), float(m.intercept_[0]), float(y.mean()))
M["calib"] = calib; joblib.dump(M, a.model); print("보정 저장:", len(calib), "개 (질환 x 조합)")
for h in ["any", "htn", "dm", "lip"]: print(h, "전부:", np.round(calib[(h, "전부")], 3))
