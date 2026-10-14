"""모델 자체의 편향·오류 점검: 임계값 누수, 스펙트럼 편향, 하위집단 편향, 중복 개체, 정답을 담은 입력."""
import argparse, glob, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
from scipy.spatial import cKDTree
import mri_screen as ms, multimodal as mm
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
for k in ["mri", "kept", "geno", "tr", "te", "model", "out"]: ap.add_argument("--" + k)
a = ap.parse_args(); M = joblib.load(a.model); ref = M["ref"]; D = ms.load_all(a.mri); b = D["BASE"]
K = np.load(a.kept, allow_pickle=True); idx = {n: i for i, n in enumerate(K["names"])}; X = K["X"]
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
pos = {n: i for i, n in enumerate(fam)}; lab = b.reindex(pd.Index(fam)); tr, te = np.load(a.tr), np.load(a.te)
HN = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
posf = {"any": lambda d: d.lab == "disease", "htn": lambda d: d.htn == 1, "dm": lambda d: d.dm == 1, "lip": lambda d: d.lip == 1}
def sn(d, h): return np.stack([X[idx[s]][[pos[n] for n in d.index]] for s in M["leads"][h]], axis=1)
def sc(d, h, cn="전부"): return M["models"][(h, cn)].predict_proba(mm.blocks(ref, d, sn(d, h), M["combos"][cn])[0])[:, 1]
amb = lab[lab.lab == "ambiguous"]

print("=== 1. 스펙트럼 편향: 진단 전 의심군(모호군)을 음성에 포함하면 성능이 얼마나 떨어지나 ===")
print(f"(모호군 {len(amb)}명은 학습·검증에 쓰이지 않음. 정상군과 모집단 비율(정상:모호 = 3.0:1)을 맞추려고 가중치 적용)")
for h in HN:
    ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf[h](ted)).values]; y = posf[h](ted).astype(int).values
    for cn in ["나이·성별", "전부"]:
        s_te, s_am = sc(ted, h, cn), sc(amb, h, cn)
        n_norm = int((y == 0).sum()); w_amb = (n_norm * len(amb) / (b.lab == "normal").sum()) / len(amb)   # 모집단 비율에 맞춘 가중치
        yy = np.r_[y, np.zeros(len(amb))]; ss = np.r_[s_te, s_am]; ww = np.r_[np.ones(len(y)), np.full(len(amb), w_amb)]
        print(f"[{HN[h]} | {cn}] 보고한 AUC(정상만 음성) {roc_auc_score(y, s_te):.3f} -> 모호군 포함 {roc_auc_score(yy, ss, sample_weight=ww):.3f} | 평균 점수 정상 {s_te[y == 0].mean():.3f}, 모호군 {s_am.mean():.3f}, 질환 {s_te[y == 1].mean():.3f}")

print("\n=== 2. 임계값 누수: 선별 기준(특이도 90%)을 검증 정상군에서 정함 -> 반으로 나눠 보정한 값과 비교 ===")
rng = np.random.RandomState(0)
for h in HN:
    ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf[h](ted)).values]; y = posf[h](ted).astype(int).values; s = sc(ted, h)
    thr = M["thr"][(h, "전부")]; rep_s, rep_sp = (s[y == 1] >= thr).mean(), (s[y == 0] < thr).mean(); cs, csp = [], []
    for _ in range(50):
        nid = np.where(y == 0)[0]; rng.shuffle(nid); A_, B_ = nid[:len(nid) // 2], nid[len(nid) // 2:]
        for fit_, ev_ in [(A_, B_), (B_, A_)]:
            t = np.quantile(s[fit_], .90); idxp = np.where(y == 1)[0]; cs.append((s[idxp] >= t).mean()); csp.append((s[ev_] < t).mean())
    print(f"[{HN[h]}] 보고(같은 표본에서 정함): 민감도 {rep_s:.3f}, 특이도 {rep_sp:.3f} | 반으로 나눠 정함: 민감도 {np.mean(cs):.3f}, 특이도 {np.mean(csp):.3f} (특이도 목표 0.900)")

print("\n=== 3. 하위집단 편향: 같은 기준에서 오탐률(정상을 선별)과 민감도가 집단마다 다른가 ===")
for h in HN:
    ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf[h](ted)).values]; y = posf[h](ted).astype(int).values; s = sc(ted, h); fl = s >= M["thr"][(h, "전부")]
    out = []
    for nm, mk in [("나이<50", ted.age.values < 50), ("50-59", (ted.age.values >= 50) & (ted.age.values < 60)), ("60+", ted.age.values >= 60), ("남", ted.sex.values == 1), ("여", ted.sex.values == 2)]:
        out.append(f"{nm}: 오탐 {fl[mk & (y == 0)].mean():.0%}/민감도 {fl[mk & (y == 1)].mean():.0%}")
    print(f"[{HN[h]}] " + " | ".join(out))

print("\n=== 4. 중복 개체(누수 가능성): MRI 벡터의 최근접 거리 ===")
vols = ref.vols; mu = lab[vols].mean().values; sd_ = lab[vols].std().values
Z = lambda d: (d[vols].values - mu) / sd_
trd, ted_all = lab.iloc[tr], lab.iloc[te]; tree = cKDTree(Z(trd))
d_te = tree.query(Z(ted_all))[0]; print(f"검증 개체 -> 학습 개체 최근접 거리: 최솟값 {d_te.min():.3f}, 5백분위 {np.percentile(d_te, 5):.3f}, 중앙값 {np.median(d_te):.3f}; 거리<0.01 인 개체 {int((d_te < .01).sum())}명")
d_in = cKDTree(Z(trd)).query(Z(trd), k=2)[0][:, 1]; print(f"학습 내부 최근접 거리: 최솟값 {d_in.min():.3f}, 중앙값 {np.median(d_in):.3f}; 거리<0.01 {int((d_in < .01).sum())}쌍")
f1 = D["FOLLOW1"]; d_f = cKDTree(Z(lab)).query(Z(f1))[0]; print(f"FOLLOW 1차 -> BASE 최근접 거리: 최솟값 {d_f.min():.3f}, 중앙값 {np.median(d_f):.3f}; 거리<0.01 {int((d_f < .01).sum())}명")

print("\n=== 5. 정답을 담은 입력이 있나: 입력별 단변량 AUC 최댓값 (검증) ===")
for h in HN:
    ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf[h](ted)).values]; y = posf[h](ted).astype(int).values
    Xm, F = mm.blocks(ref, ted, sn(ted, h), M["combos"]["전부"]); Xm = np.nan_to_num(Xm, nan=np.nanmedian(Xm)); u = [max(roc_auc_score(y, Xm[:, j]), 1 - roc_auc_score(y, Xm[:, j])) for j in range(Xm.shape[1]) if Xm[:, j].std() > 0]
    print(f"[{HN[h]}] 입력 {Xm.shape[1]}개 중 단변량 AUC 최댓값 {max(u):.3f} (0.9 이상이면 정답 누출 의심)")
