"""심실 상한(클리핑) 지시변수 의존성 점검. 합성자료는 심실 4개 변수가 고정 상한에 걸려 있고(②-1), 모델은 '상한에 걸렸는가' 지시변수(k_*)를 입력으로 쓴다.
이 지시변수를 빼면 5겹 AUC가 얼마나 떨어지는지 본다(MRI·설문·나이·성별 모델, SNP 제외). 매 폴드에서 정상 참조는 학습 개체 안에서만 만든다."""
import argparse, glob, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import mri_screen as ms, multimodal as mm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
for k in ["mri", "geno", "out"]: ap.add_argument("--" + k)
a = ap.parse_args(); b = ms.load_all(a.mri)["BASE"]
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
lab = b.reindex(pd.Index(fam)); HEADS = ["any", "htn", "dm", "lip"]; HN = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
P = {"any": (lab.lab == "disease").values, "htn": (lab.htn == 1).values, "dm": (lab.dm == 1).values, "lip": (lab.lip == 1).values}; ISN = (lab.lab == "normal").values
LAB = lab.lab.isin(["normal", "disease"]).values; ALL = np.where(LAB)[0]; y_any = P["any"][ALL].astype(int)
VAR = {"MRI 영역편차(z)만": ("z", False), "z + 심실 상한 지시변수": ("zk", False), "z만 + 설문·체격": ("z", True), "z + 지시변수 + 설문·체격": ("zk", True)}
sc = {(h, v): [] for h in HEADS for v in VAR}
for f, (ta, tb) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(ALL, y_any)):
    tri, vai = ALL[ta], ALL[tb]; trd = lab.iloc[tri]; rn, rest = ms.split_ref(trd, f); ref = ms.Ref().fit(rn, f)
    for h in HEADS:
        pos_tr = trd[P[h][tri]]; ctr = pd.concat([rest, pos_tr]); yc = np.r_[np.zeros(len(rest)), np.ones(len(pos_tr))].astype(int)
        vm = np.array([i for i in vai if ISN[i] or P[h][i]]); dv = lab.iloc[vm]; yv = P[h][vm].astype(int)
        Fc, Fv = ref.transform(ctr), ref.transform(dv); cs = ms.colsets(Fc); zc = [c for c in cs["age+z"][2:]]; kc = [c for c in cs["age+z+clip"][2:] if c not in zc]
        for v, (kind, useq) in VAR.items():
            cols = ["age", "sex"] + zc + (kc if kind == "zk" else [])
            Xc, Xv = Fc[cols].values, Fv[cols].values
            if useq: Xc = np.column_stack([Xc] + [ms._clean(ctr[c]).values for c in mm.Q]); Xv = np.column_stack([Xv] + [ms._clean(dv[c]).values for c in mm.Q])
            sc[(h, v)].append(roc_auc_score(yv, mm._lr().fit(Xc, yc).predict_proba(Xv)[:, 1]))
    print("폴드", f, "완료", flush=True)
rows = [dict(질환=HN[h], 입력=v, auc=np.mean(sc[(h, v)]), sd=np.std(sc[(h, v)])) for h in HEADS for v in VAR]
R = pd.DataFrame(rows); R.to_csv(a.out + "/clip_ablation.csv", index=False)
pd.set_option("display.width", 200); print(R.pivot(index="입력", columns="질환", values="auc").round(3).loc[list(VAR), [HN[h] for h in HEADS]].to_string())
