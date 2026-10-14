"""나이대별 선별 기준 추가: 하나의 전체 기준은 고령 정상인을 과하게 선별하고 젊은 환자를 놓친다.
나이대(<50, 50-59, 60+)마다 정상군 점수의 90백분위를 기준으로 삼아 나이대별 특이도를 90%로 맞춘다.
정직한 효과 추정을 위해 검증 정상군을 반으로 나눠(한쪽에서 기준 결정, 다른 쪽에서 평가) 평균한 값도 함께 계산한다."""
import argparse, glob, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
import mri_screen as ms, multimodal as mm

ap = argparse.ArgumentParser()
for k in ["mri", "kept", "geno", "te", "model", "out"]: ap.add_argument("--" + k)
a = ap.parse_args(); M = joblib.load(a.model); ref = M["ref"]; D = ms.load_all(a.mri); b = D["BASE"]
K = np.load(a.kept, allow_pickle=True); idx = {n: i for i, n in enumerate(K["names"])}; X = K["X"]
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
pos = {n: i for i, n in enumerate(fam)}; lab = b.reindex(pd.Index(fam)); te = np.load(a.te)
HN = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
posf = {"any": lambda d: d.lab == "disease", "htn": lambda d: d.htn == 1, "dm": lambda d: d.dm == 1, "lip": lambda d: d.lip == 1}
BANDS = [("<50", 0, 50), ("50-59", 50, 60), ("60+", 60, 200)]
def band(age): return np.digitize(age, [50, 60])      # 0,1,2
thr_age = {}; rows = []; rng = np.random.RandomState(0)
for h in HN:
    ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf[h](ted)).values]; y = posf[h](ted).astype(int).values; ag = band(ted.age.values)
    for cn in M["combos"]:
        sn = np.stack([X[idx[s]][[pos[n] for n in ted.index]] for s in M["leads"][h]], axis=1)
        s = M["models"][(h, cn)].predict_proba(mm.blocks(ref, ted, sn, M["combos"][cn])[0])[:, 1]
        thr_age[(h, cn)] = [float(np.quantile(s[(y == 0) & (ag == k)], .90)) for k in range(3)]
        if cn != "전부": continue
        g_thr = M["thr"][(h, cn)]
        for k, (nm, lo, hi) in enumerate(BANDS):
            m0, m1 = (y == 0) & (ag == k), (y == 1) & (ag == k); cs, csp = [], []
            for _ in range(50):
                nid = np.where(m0)[0]; rng.shuffle(nid); A_, B_ = nid[:len(nid) // 2], nid[len(nid) // 2:]
                for f_, e_ in [(A_, B_), (B_, A_)]:
                    t = np.quantile(s[f_], .90); cs.append((s[m1] >= t).mean()); csp.append((s[e_] < t).mean())
            rows.append(dict(target=HN[h], age_band=nm, n_neg=int(m0.sum()), n_pos=int(m1.sum()), 전체기준_오탐=float((s[m0] >= g_thr).mean()), 전체기준_민감도=float((s[m1] >= g_thr).mean()),
                             나이대별기준_오탐=1 - float(np.mean(csp)), 나이대별기준_민감도=float(np.mean(cs))))
M["thr_age"] = thr_age; joblib.dump(M, a.model)
R = pd.DataFrame(rows); R.to_csv(a.out + "/age_threshold_effect.csv", index=False); pd.set_option("display.width", 200)
print(R.assign(**{c: (R[c] * 100).round(0).astype(int).astype(str) + "%" for c in ["전체기준_오탐", "전체기준_민감도", "나이대별기준_오탐", "나이대별기준_민감도"]}).to_string(index=False))
