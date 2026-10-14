"""성능 향상 실험. 방법 선택은 학습 70% 안의 5겹 교차검증으로만 하고, 검증 30%는 마지막에 한 번만 쓴다.
SNP 선택(전장 스캔)은 매 폴드의 학습 개체에서만 한다(후보 풀은 이전 스캔의 rs SNP 전부 + 약한 연관 배경 SNP, 검증 개체의 라벨은 쓰이지 않음)."""
import argparse, glob, os, sys, zipfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import mri_screen as ms, multimodal as mm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
for k in ["mri", "scan", "kept", "geno", "tr", "te", "out"]: ap.add_argument("--" + k)
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()
D = ms.load_all(a.mri); b = D["BASE"]
R = pd.read_csv(a.scan); K = np.load(a.kept, allow_pickle=True); names = list(K["names"]); X = K["X"]
meta = R.set_index("snp").loc[names]; chrv, posv = meta.chr.values, meta.pos.values
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]; N = len(fam)
lab = b.reindex(pd.Index(fam)); idpos = {n: i for i, n in enumerate(fam)}
tr_all, te_all = np.load(a.tr), np.load(a.te)
HEADS = ["any", "htn", "dm", "lip"]; NAMES = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
P = {"any": (lab.lab == "disease").values, "htn": (lab.htn == 1).values, "dm": (lab.dm == 1).values, "lip": (lab.lip == 1).values}
ISN = (lab.lab == "normal").values

def scan(mask, y, chunk=6000):
    out = np.empty(X.shape[0], np.float32); ym = y - y.mean(); sy = np.sqrt((ym ** 2).sum())
    for s in range(0, X.shape[0], chunk):
        Xm = X[s:s + chunk][:, mask]; Xc = Xm - Xm.mean(1, keepdims=True)
        out[s:s + chunk] = (Xc @ ym) / (np.sqrt((Xc ** 2).sum(1)) * sy + 1e-12)
    return out
def clump(r, K_, win=1_000_000):
    ch = []
    for i in np.argsort(-np.abs(r)):
        if len(ch) >= K_: break
        if all(not (chrv[i] == chrv[j] and abs(posv[i] - posv[j]) < win) for j in ch): ch.append(i)
    return ch
def scans(train_idx):
    out = {}
    for h in HEADS:
        trm = np.zeros(N, bool); trm[train_idx] = True; trm &= (ISN | P[h]); out[h] = scan(trm, P[h][trm].astype(float))
    return out
def snp_sets(rs, h):
    """변이 선택 방식 후보: 반환 {이름: SNP 인덱스 리스트}"""
    others = [t for t in ["htn", "dm", "lip"]]
    u15 = sorted(set(sum([clump(rs[t], 15) for t in others], []) + clump(rs[h], 15)))
    u30 = sorted(set(sum([clump(rs[t], 30) for t in others], []) + clump(rs[h], 30)))
    return {"K15(기존)": clump(rs[h], 15), "K30": clump(rs[h], 30), "K60": clump(rs[h], 60), "합집합15": u15, "합집합30": u30,
            "상위300(규제)": list(np.argsort(-np.abs(rs[h]))[:300])}

def fit_eval(ref, rest_pos, train_idx, val_idx, h, rs, models):
    pos_tr = np.array([i for i in train_idx if P[h][i]]); ctr_pos = np.concatenate([rest_pos, pos_tr]); yc = np.r_[np.zeros(len(rest_pos)), np.ones(len(pos_tr))].astype(int)
    val_pos = np.array([i for i in val_idx if (ISN[i] or P[h][i])]); yv = P[h][val_pos].astype(int)
    base_tr, _ = mm.blocks(ref, lab.iloc[ctr_pos], None, ("age", "mri", "q")); base_va, _ = mm.blocks(ref, lab.iloc[val_pos], None, ("age", "mri", "q"))
    res = {}
    for nm, ix in snp_sets(rs, h).items():
        s_tr, s_va = X[ix][:, ctr_pos].T, X[ix][:, val_pos].T
        for mn in models:
            if mn == "LR" or (nm in ("합집합15", "K15(기존)") and mn != "LR"):
                A, B = np.hstack([base_tr, s_tr]), np.hstack([base_va, s_va])
                if mn == "LR": s = mm._lr().fit(A, yc).predict_proba(B)[:, 1]
                elif mn == "GBM": s = ms._hgb().fit(np.nan_to_num(A, nan=-999), yc).predict_proba(np.nan_to_num(B, nan=-999))[:, 1]
                else:
                    s = (mm._lr().fit(A, yc).predict_proba(B)[:, 1] + ms._hgb().fit(np.nan_to_num(A, nan=-999), yc).predict_proba(np.nan_to_num(B, nan=-999))[:, 1]) / 2
                res[(nm, mn)] = roc_auc_score(yv, s)
    return res

if __name__ == "__main__":
    t0 = time.time(); models = ["LR", "GBM", "LR+GBM"]
    # 학습 70% 안에서 5겹 교차검증 (질환 라벨 기준 층화)
    lab_tr = tr_all[lab.lab.iloc[tr_all].isin(["normal", "disease"]).values]
    y_strat = (lab.lab.iloc[lab_tr] == "disease").astype(int).values
    rows = []
    for f, (a_, b_) in enumerate(StratifiedKFold(a.folds, shuffle=True, random_state=7).split(lab_tr, y_strat)):
        tri, vai = lab_tr[a_], lab_tr[b_]
        trd = lab.iloc[tri]; rn, rest = ms.split_ref(trd, f); ref = ms.Ref().fit(rn, f); rest_pos = np.array([idpos[n] for n in rest.index])
        rs = scans(tri)
        for h in HEADS:
            for (nm, mn), auc in fit_eval(ref, rest_pos, tri, vai, h, rs, models).items(): rows.append(dict(fold=f, target=NAMES[h], snp=nm, model=mn, auc=auc))
        print(f"fold {f} 완료 ({time.time() - t0:.0f}s)", flush=True)
    C = pd.DataFrame(rows); C.to_csv(a.out + "/tune_cv.csv", index=False)
    pd.set_option("display.width", 220)
    S = C.groupby(["target", "snp", "model"]).auc.mean().round(4).reset_index()
    for t, g in S.groupby("target", sort=False): print(f"\n[{t}] 교차검증 AUC (5겹 평균)"); print(g.sort_values("auc", ascending=False).head(8).to_string(index=False))
    print(f"\n기존 설정(K15 + LR): ", {t: float(S[(S.target == t) & (S.snp == 'K15(기존)') & (S.model == 'LR')].auc.iloc[0]) for t in NAMES.values()})
