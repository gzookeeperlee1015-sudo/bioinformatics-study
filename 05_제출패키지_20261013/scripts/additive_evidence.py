"""가산 증거 모델: 진단이 같으면 모달리티가 서로 독립이라는 구조를 쓴 결합.
최종 로짓 = 나이·성별 로짓 + Σ_m [ (나이·성별+m 로짓) - (나이·성별 로짓) ]  (모달리티 m 의 증분 증거를 더함)
기존 방식(조합마다 따로 학습한 결합 로지스틱)과 같은 검증 30%에서 비교한다. 새로 학습하는 모델은 없다."""
import argparse, glob, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
import mri_screen as ms, multimodal as mm
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
for k in ["mri", "kept", "geno", "te", "model", "out"]: ap.add_argument("--" + k)
a = ap.parse_args(); M = joblib.load(a.model); ref = M["ref"]; D = ms.load_all(a.mri); b = D["BASE"]
K = np.load(a.kept, allow_pickle=True); idx = {n: i for i, n in enumerate(K["names"])}; X = K["X"]
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
pos = {n: i for i, n in enumerate(fam)}; lab = b.reindex(pd.Index(fam)); te = np.load(a.te)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
rows = []
for h, hn in [("any", "전체 질환"), ("htn", "고혈압"), ("dm", "당뇨"), ("lip", "고지혈")]:
    posf = (lambda d: d.lab == "disease") if h == "any" else (lambda d: d[h] == 1)
    ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf(ted)).values]; y = posf(ted).astype(int).values
    sn = np.stack([X[idx[s]][[pos[n] for n in ted.index]] for s in M["leads"][h]], axis=1)
    L = {}
    for cn in ["나이·성별", "+MRI", "+설문·체격", "+SNP", "+MRI+설문·체격", "+MRI+SNP", "+설문·체격+SNP", "전부"]:
        Xa, _ = mm.blocks(ref, ted, sn, M["combos"][cn]); L[cn] = logit(M["models"][(h, cn)].predict_proba(Xa)[:, 1])
    ev = {m: L["+" + m] - L["나이·성별"] for m in ["MRI", "설문·체격", "SNP"]}
    for cn, ms_ in [("+MRI+설문·체격", ["MRI", "설문·체격"]), ("+MRI+SNP", ["MRI", "SNP"]), ("+설문·체격+SNP", ["설문·체격", "SNP"]), ("전부", ["MRI", "설문·체격", "SNP"])]:
        add = L["나이·성별"] + sum(ev[m] for m in ms_)
        rows.append(dict(target=hn, 조합=cn, 결합로지스틱=roc_auc_score(y, L[cn]), 가산증거=roc_auc_score(y, add), 차이=roc_auc_score(y, add) - roc_auc_score(y, L[cn]),
                         증거간상관=float(np.mean([np.corrcoef(ev[m1], ev[m2])[0, 1] for i, m1 in enumerate(ms_) for m2 in ms_[i + 1:]]))))
    # 진단이 같을 때 증거 간 상관 (질환자 안, 정상 안)
    for g, mk in [("질환자 안", y == 1), ("정상 안", y == 0)]:
        c = {f"{m1}-{m2}": np.corrcoef(ev[m1][mk], ev[m2][mk])[0, 1] for m1, m2 in [("MRI", "SNP"), ("MRI", "설문·체격"), ("SNP", "설문·체격")]}
        print(f"[{hn}] 진단이 같을 때 증거 간 상관 ({g}):", {k: round(v, 3) for k, v in c.items()})
R = pd.DataFrame(rows); R.to_csv(a.out + "/additive_evidence.csv", index=False); pd.set_option("display.width", 200)
print("\n", R.round(4).to_string(index=False))
