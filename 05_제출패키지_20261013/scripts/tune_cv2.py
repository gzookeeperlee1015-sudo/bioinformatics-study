"""성능 향상 실험 2차(추가 특징). 1차(tune_cv.py)에서 정한 질환별 SNP 집합에 요약 특징·유전 점수·나이 상호작용을 더해 비교.
성능 향상 실험. 방법 선택은 학습 70% 안의 5겹 교차검증으로만 하고, 검증 30%는 마지막에 한 번만 쓴다.
SNP 선택(전장 스캔)은 매 폴드의 학습 개체에서만 한다(후보 풀은 이전 스캔의 rs SNP 전부 + 약한 연관 배경 SNP, 검증 개체의 라벨은 쓰이지 않음)."""
import argparse, glob, os, sys, zipfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import mri_screen as ms, multimodal as mm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
for k in ["mri", "scan", "kept", "geno", "tr", "te", "out"]: ap.add_argument("--" + k)
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--final", action="store_true")
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

BEST = {"any": "합집합15", "htn": "K15(기존)", "dm": "K15(기존)", "lip": "상위300(규제)"}
def extras(ref, d, snp_rows_dosage, w300, ix300, kinds):
    F = ref.transform(d); zc = F[[c for c in F.columns if c.startswith("z_")]].values; cols = []
    zm = zc.mean(1, keepdims=True); age = F["age"].values.reshape(-1, 1)
    if "mri" in kinds: cols += [zm, zc.min(1, keepdims=True), zc.max(1, keepdims=True), (np.abs(zc) > 2).sum(1, keepdims=True), np.sqrt((zc ** 2).sum(1, keepdims=True))]
    if "prs" in kinds: cols.append((X[ix300][:, snp_rows_dosage].T @ w300).reshape(-1, 1))
    if "age" in kinds: cols += [(age - 55) * zm, (age - 55) * (np.abs(zc) > 2).sum(1, keepdims=True)]
    return np.hstack(cols) if cols else np.zeros((len(d), 0))

def fit_eval(ref, rest_pos, train_idx, val_idx, h, rs, models):
    pos_tr = np.array([i for i in train_idx if P[h][i]]); ctr_pos = np.concatenate([rest_pos, pos_tr]); yc = np.r_[np.zeros(len(rest_pos)), np.ones(len(pos_tr))].astype(int)
    val_pos = np.array([i for i in val_idx if (ISN[i] or P[h][i])]); yv = P[h][val_pos].astype(int)
    base_tr, _ = mm.blocks(ref, lab.iloc[ctr_pos], None, ("age", "mri", "q")); base_va, _ = mm.blocks(ref, lab.iloc[val_pos], None, ("age", "mri", "q"))
    ix = snp_sets(rs, h)[BEST[h]]; ix300 = list(np.argsort(-np.abs(rs[h]))[:300]); w300 = rs[h][ix300].astype(np.float64)
    res = {}
    for kinds in [(), ("mri",), ("prs",), ("age",), ("mri", "prs"), ("mri", "prs", "age")]:
        A = np.hstack([base_tr, X[ix][:, ctr_pos].T, extras(ref, lab.iloc[ctr_pos], ctr_pos, w300, ix300, kinds)])
        B = np.hstack([base_va, X[ix][:, val_pos].T, extras(ref, lab.iloc[val_pos], val_pos, w300, ix300, kinds)])
        res[("+".join(kinds) or "없음", "LR")] = roc_auc_score(yv, mm._lr().fit(A, yc).predict_proba(B)[:, 1])
    return res


FINAL = {"any": ("합집합15", ()), "htn": ("K15(기존)", ("prs",)), "dm": ("K15(기존)", ("prs",)), "lip": ("상위300(규제)", ())}
def final_eval():
    """교차검증으로 정한 설정을 학습 70% 전체로 학습하고 검증 30%에서 한 번만 평가(기존 설정과 같은 표본에서 쌍체 비교)."""
    tri = tr_all[lab.lab.iloc[tr_all].isin(["normal", "disease"]).values]; trd = lab.iloc[tri]
    rn, rest = ms.split_ref(trd, 0); ref = ms.Ref().fit(rn, 0); rest_pos = np.array([idpos[n] for n in rest.index]); rs = scans(tri); rows = []
    for h in HEADS:
        snpname, kinds = FINAL[h]; pos_tr = np.array([i for i in tri if P[h][i]]); ctr_pos = np.concatenate([rest_pos, pos_tr]); yc = np.r_[np.zeros(len(rest_pos)), np.ones(len(pos_tr))].astype(int)
        te_pos = np.array([i for i in te_all if (ISN[i] or P[h][i])]); yt = P[h][te_pos].astype(int)
        base_tr, _ = mm.blocks(ref, lab.iloc[ctr_pos], None, ("age", "mri", "q")); base_te, _ = mm.blocks(ref, lab.iloc[te_pos], None, ("age", "mri", "q"))
        ix300 = list(np.argsort(-np.abs(rs[h]))[:300]); w300 = rs[h][ix300].astype(np.float64)
        def feats(ix, rows, base, kd): return np.hstack([base, X[ix][:, rows].T, extras(ref, lab.iloc[rows], rows, w300, ix300, kd)])
        ix_base = snp_sets(rs, h)["K15(기존)"]; ix_new = snp_sets(rs, h)[snpname]
        s0 = mm._lr().fit(feats(ix_base, ctr_pos, base_tr, ()), yc).predict_proba(feats(ix_base, te_pos, base_te, ()))[:, 1]
        s1 = mm._lr().fit(feats(ix_new, ctr_pos, base_tr, kinds), yc).predict_proba(feats(ix_new, te_pos, base_te, kinds))[:, 1]
        rng = np.random.RandomState(0); d = []; b0 = []; b1 = []
        for _ in range(500):
            i = rng.choice(len(yt), len(yt)); a0, a1 = roc_auc_score(yt[i], s0[i]), roc_auc_score(yt[i], s1[i]); d.append(a1 - a0); b0.append(a0); b1.append(a1)
        rows.append(dict(target=NAMES[h], 기존AUC=roc_auc_score(yt, s0), 개선AUC=roc_auc_score(yt, s1), 차이=roc_auc_score(yt, s1) - roc_auc_score(yt, s0), 차이_하한=np.percentile(d, 2.5), 차이_상한=np.percentile(d, 97.5),
                         개선_하한=np.percentile(b1, 2.5), 개선_상한=np.percentile(b1, 97.5), 설정=f"{snpname} + {'+'.join(kinds) or '추가특징 없음'}"))
    R_ = pd.DataFrame(rows); R_.to_csv(a.out + "/tuned_final_test.csv", index=False); pd.set_option("display.width", 240); print(R_.round(4).to_string(index=False))

if __name__ == "__main__":
    if a.final:
        final_eval(); sys.exit()
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
    C = pd.DataFrame(rows); C.to_csv(a.out + "/tune_cv2.csv", index=False)
    pd.set_option("display.width", 220)
    S = C.groupby(["target", "snp", "model"]).auc.mean().round(4).reset_index()
    for t, g in S.groupby("target", sort=False): print(f"\n[{t}] 교차검증 AUC (5겹 평균)"); print(g.sort_values("auc", ascending=False).head(8).to_string(index=False))
