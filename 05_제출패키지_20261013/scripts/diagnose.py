"""성능이 덜 나오는 원인 진단: 과적합/정보 부족, 하위집단, 놓친 환자의 특성, 동반 질환."""
import argparse, glob, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
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
def snp_of(d, h): return np.stack([X[idx[s]][[pos[n] for n in d.index]] for s in M["leads"][h]], axis=1)
def mat(d, h, parts=("age", "mri", "q", "snp")): return mm.blocks(ref, d, snp_of(d, h), parts)[0]

trd = lab.iloc[tr]; rn, rest = ms.split_ref(trd, 0)
print("=== D1. 과적합 vs 정보 부족: 학습(표본 내) AUC 와 검증 AUC, 학습 곡선 ===")
rng = np.random.RandomState(0)
for h in ["any", "htn", "dm", "lip"]:
    ctr = pd.concat([rest, trd[posf[h](trd)]]); yc = ctr.index.isin(trd[posf[h](trd)].index).astype(int)
    ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf[h](ted)).values]; yt = posf[h](ted).astype(int).values
    A_, T_ = mat(ctr, h), mat(ted, h); m = M["models"][(h, "전부")]
    a_tr = roc_auc_score(yc, m.predict_proba(A_)[:, 1]); a_te = roc_auc_score(yt, m.predict_proba(T_)[:, 1])
    lc = []
    for fr in (.1, .25, .5, 1.0):
        ii = rng.choice(len(yc), int(fr * len(yc)), replace=False); ii = ii if len(set(yc[ii])) == 2 else np.arange(len(yc))
        lc.append(f"{int(fr * 100)}%={roc_auc_score(yt, mm._lr().fit(A_[ii], yc[ii]).predict_proba(T_)[:, 1]):.3f}")
    print(f"[{HN[h]}] 학습 AUC {a_tr:.3f} / 검증 AUC {a_te:.3f} (차이 {a_tr - a_te:+.3f}) | 학습 곡선(학습 자료 비율): " + ", ".join(lc))

print("\n=== D2. 하위집단별 AUC (검증, 전부 사용) ===")
for h in ["any", "htn", "dm", "lip"]:
    ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf[h](ted)).values]; yt = posf[h](ted).astype(int).values
    s = M["models"][(h, "전부")].predict_proba(mat(ted, h))[:, 1]; out = []
    for nm, mk in [("나이<50", ted.age.values < 50), ("50-59", (ted.age.values >= 50) & (ted.age.values < 60)), ("60+", ted.age.values >= 60), ("남", ted.sex.values == 1), ("여", ted.sex.values == 2)]:
        out.append(f"{nm} {roc_auc_score(yt[mk], s[mk]):.3f}(n={int(mk.sum())})" if len(set(yt[mk])) == 2 else f"{nm} n/a")
    print(f"[{HN[h]}] " + " | ".join(out))

print("\n=== D3. 놓친 환자(특이도 90% 기준 음성)는 어떤 사람인가 ===")
for h, agc, mc in [("htn", "T_HTNAG", lambda d: (ms._clean(d.T_SBP) >= 140) | (ms._clean(d.T_DBP) >= 90)), ("dm", "T_DMAG", lambda d: (ms._clean(d.T_GLU0) >= 126) | (ms._clean(d.T_HBA1C) >= 6.5)), ("lip", "T_LIPAG", lambda d: ms._clean(d.T_TCHL) >= 240)]:
    ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf[h](ted)).values]; yt = posf[h](ted).astype(int).values
    s = M["models"][(h, "전부")].predict_proba(mat(ted, h))[:, 1]; flag = s >= M["thr"][(h, "전부")]; dis = ted[yt == 1]; fl = flag[yt == 1]
    dur = dis.age - ms._clean(dis[agc]); meas = mc(dis).values
    print(f"[{HN[h]}] 환자 {len(dis)}명 중 놓침 {int((~fl).sum())}명 | 투병 기간 중앙값 포착 {np.nanmedian(dur[fl]):.1f}년 vs 놓침 {np.nanmedian(dur[~fl]):.1f}년 | 측정값이 비정상인 비율 포착 {np.nanmean(meas[fl]):.0%} vs 놓침 {np.nanmean(meas[~fl]):.0%} | 평균 나이 {dis.age[fl].mean():.1f} vs {dis.age[~fl].mean():.1f}")
print("\n=== D4. 투병 기간과 신호 크기 / 동반 질환 ===")
dis = lab[(lab.lab == "disease")]
Fz = ref.transform(lab.assign(BV_icv=lab.BV_icv)); atr = -Fz[[c for c in Fz.columns if c.startswith("z_")]].mean(axis=1)
for h, agc in [("htn", "T_HTNAG"), ("dm", "T_DMAG"), ("lip", "T_LIPAG")]:
    d_ = lab[lab[h] == 1]; dur = d_.age - ms._clean(d_[agc]); ok = dur.notna() & (dur >= 0)
    print(f"[{HN[h]}] 투병 기간과 MRI 위축 점수 Spearman {pd.Series(atr[d_.index][ok]).corr(dur[ok], method='spearman'):+.3f} (n={int(ok.sum())}); 투병 기간 중앙값 {dur[ok].median():.1f}년")
nd = (lab.htn + lab.dm + lab.lip)
print("동반 질환 수 분포(전체 BASE):", nd.value_counts().sort_index().to_dict())
print("질환 라벨 간 상관:", lab[["htn", "dm", "lip"]].corr().round(3).values[np.triu_indices(3, 1)], "(htn-dm, htn-lip, dm-lip)")
ted = lab.iloc[te]; ted = ted[ted.lab.isin(["normal", "disease"])]; s = M["models"][("any", "전부")].predict_proba(mat(ted, "any"))[:, 1]
ndt = (ted.htn + ted.dm + ted.lip).values
print("전체 질환 점수 중앙값 (질환 수별):", {int(k): round(float(np.median(s[ndt == k])), 3) for k in sorted(set(ndt))})
