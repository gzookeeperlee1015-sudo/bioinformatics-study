"""5겹 교차검증 점수(cv_oof.npz)로 보정·유용성, SNP 효과 축소, 순차 선별, 나이 편향, 스펙트럼 편향을 다시 계산한다.
모든 환자가 한 번씩 평가에 쓰인 점수만 사용하므로 70/30 분할의 한 검증 표본에 의존하지 않는다."""
import argparse, glob, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import mri_screen as ms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
for k in ["mri", "geno", "oof", "out"]: ap.add_argument("--" + k)
a = ap.parse_args(); D = ms.load_all(a.mri); b = D["BASE"]
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
lab = b.reindex(pd.Index(fam)); V = np.load(a.oof); REPS = sorted({int(k.split("|")[1]) for k in V.files if k.startswith("oof|")})
HEADS = ["any", "htn", "dm", "lip"]; HN = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
P = {"any": (lab.lab == "disease").values, "htn": (lab.htn == 1).values, "dm": (lab.dm == 1).values, "lip": (lab.lip == 1).values}; ISN = (lab.lab == "normal").values
PREV = {"any": float((b.lab == "disease").mean()), "htn": float(b.htn.mean()), "dm": float(b.dm.mean()), "lip": float(b.lip.mean())}
def oof(h, cn): return np.mean([V[f"oof|{r}|{h}|{cn}"] for r in REPS], axis=0)       # 반복 평균 점수
def members(h): return (ISN | P[h])
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def platt(s, y):
    lg = logit(s).reshape(-1, 1); out = np.zeros(len(y))
    for tr, tt in StratifiedKFold(5, shuffle=True, random_state=0).split(lg, y): out[tt] = LogisticRegression(C=1e6).fit(lg[tr], y[tr]).predict_proba(lg[tt])[:, 1]
    return out
def ece(p, y, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9; e = 0.0
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e
def bacc(pred, y): return 0.5 * (pred[y == 1].mean() + (1 - pred[y == 0].mean()))
pd.set_option("display.width", 220)

# ---------- 1. 보정·유용성 ----------
rows = []
for h in HEADS:
    mk = members(h); y = P[h][mk].astype(int)
    for cn in ["나이·성별", "+MRI+설문·체격", "전부"]:
        s = oof(h, cn)[mk]; cal = platt(s, y); thr = np.quantile(s[y == 0], .90); fl = s >= thr
        sens, spec = fl[y == 1].mean(), (~fl)[y == 0].mean(); p = PREV[h]
        top = np.argsort(-s)[:int(np.ceil(.2 * len(s)))]; cap = y[top].sum() / y.sum()
        rows.append(dict(target=HN[h], model=cn, auc=roc_auc_score(y, s), ece_raw=ece(s, y), ece_cal=ece(cal, y), brier_cal=brier_score_loss(y, cal), sens_spec90=sens,
                         ppv_pop=sens * p / (sens * p + (1 - spec) * (1 - p)), npv_pop=spec * (1 - p) / (spec * (1 - p) + (1 - sens) * p), capture_top20=cap, lift_top20=cap / .2))
U = pd.DataFrame(rows); U.to_csv(a.out + "/cv_utility_table.csv", index=False)
print("=== 1. 보정·유용성 (5겹 점수) ==="); print(U[U.model == "전부"].round(3).to_string(index=False))

# ---------- 2. SNP 효과 축소 ----------
FS = [1, .75, .5, .35, .25, .15, .1, .05, 0]; rows = []
for h in HEADS:
    mk = members(h); y = P[h][mk].astype(int)
    for cn in ("+SNP", "+MRI+SNP", "전부"):
        for f in FS:
            lg = np.mean([V[f"shr|{r}|{h}|{cn}|{f}"] for r in REPS], axis=0)[mk]; rows.append(dict(target=HN[h], model=cn, scale=f, auc=roc_auc_score(y, lg)))
S = pd.DataFrame(rows); S.to_csv(a.out + "/cv_snp_shrink.csv", index=False)
print("\n=== 2. SNP 효과를 f배로 줄였을 때 AUC (전부) ==="); print(S[S.model == "전부"].pivot(index="scale", columns="target", values="auc").round(3).loc[FS, [HN[h] for h in HEADS]].to_string())

# ---------- 3. 순차 선별 (질환별 정책) ----------
rows = []
for h in HEADS:
    mk = members(h); y = P[h][mk].astype(int); Pc = {k: platt(oof(h, cn)[mk], y) for k, cn in [("q", "+설문·체격"), ("qm", "+MRI+설문·체격"), ("qs", "+설문·체격+SNP"), ("all", "전부")]}
    cut = y.mean(); full = bacc((Pc["all"] >= cut).astype(float), y); q0 = bacc((Pc["q"] >= cut).astype(float), y)
    pol = {"설문→MRI→SNP": ["q", "qm", "all"], "설문→SNP→MRI": ["q", "qs", "all"], "설문→SNP(MRI 생략)": ["q", "qs"], "설문→MRI(SNP 생략)": ["q", "qm"]}
    for name, st in pol.items():
        for alo, bhi in [(.3, 2.0), (.5, 1.5), (.5, 2.5), (.7, 3.0), (.2, 1.5), (.4, 3.5)]:
            lo, hi = cut * alo, min(cut * bhi, .98); a1 = (Pc[st[0]] >= lo) & (Pc[st[0]] < hi); final = Pc[st[0]].copy(); a2 = np.zeros(len(y), bool)
            final = np.where(a1, Pc[st[1]], final); a2 = a1 & (Pc[st[1]] >= lo) & (Pc[st[1]] < hi)
            if len(st) == 3: final = np.where(a2, Pc[st[2]], final)
            if len(st) == 3: m_, s_ = (a1.mean(), a2.mean()) if st[1] == "qm" else (a2.mean(), a1.mean())
            else: m_, s_ = (a1.mean(), 0.0) if st[1] == "qm" else (0.0, a1.mean())
            rows.append(dict(target=HN[h], policy=name, band=f"[{lo:.2f},{hi:.2f}]", mri=float(m_), snp=float(s_), bacc=bacc((final >= cut).astype(float), y), full=full, q_only=q0))
Cc = pd.DataFrame(rows); Cc.to_csv(a.out + "/cv_cascade.csv", index=False)
print("\n=== 3. 순차 선별: 전원 사용 대비 균형 정확도 손실 허용치별 최저 비용 정책 ===")
for t, g in Cc.groupby("target", sort=False):
    full = g.full.iloc[0]; print(f"[{t}] 전원 사용 {full:.3f} | 설문·체격만 {g.q_only.iloc[0]:.3f}")
    for tol in (0.01, 0.02, 0.03):
        gg = g[g.bacc >= full - tol].assign(c=lambda d: d.mri + d.snp).sort_values("c")
        if len(gg): r = gg.iloc[0]; print(f"   허용 손실 {tol}: {r.policy} {r.band} → MRI {r.mri:.0%}, SNP {r.snp:.0%}, 균형정확도 {r.bacc:.3f}")
        else: print(f"   허용 손실 {tol}: 해당 정책 없음")

# ---------- 4. 나이 편향 ----------
rows = []; rng = np.random.RandomState(0)
for h in HEADS:
    mk = members(h); y = P[h][mk].astype(int); s = oof(h, "전부")[mk]; ag = np.digitize(lab.age.values[mk], [50, 60]); gthr = np.quantile(s[y == 0], .90)
    for k, nm in enumerate(["<50", "50-59", "60+"]):
        m0, m1 = (y == 0) & (ag == k), (y == 1) & (ag == k); cs, csp = [], []
        for _ in range(50):
            nid = np.where(m0)[0]; rng.shuffle(nid); A_, B_ = nid[:len(nid) // 2], nid[len(nid) // 2:]
            for f_, e_ in [(A_, B_), (B_, A_)]:
                t = np.quantile(s[f_], .90); cs.append((s[m1] >= t).mean()); csp.append((s[e_] < t).mean())
        rows.append(dict(target=HN[h], age_band=nm, n_neg=int(m0.sum()), n_pos=int(m1.sum()), 전체기준_오탐=float((s[m0] >= gthr).mean()), 전체기준_민감도=float((s[m1] >= gthr).mean()), 나이대별_오탐=1 - float(np.mean(csp)), 나이대별_민감도=float(np.mean(cs))))
A = pd.DataFrame(rows); A.to_csv(a.out + "/cv_age_threshold.csv", index=False)
print("\n=== 4. 나이대별 기준 효과 (5겹 점수) ==="); print(A.assign(**{c: (A[c] * 100).round(0).astype(int).astype(str) + "%" for c in ["전체기준_오탐", "전체기준_민감도", "나이대별_오탐", "나이대별_민감도"]}).to_string(index=False))

# ---------- 5. 스펙트럼 편향 ----------
AMB = V["amb_idx"]; rows = []
for h in HEADS:
    mk = members(h); y = P[h][mk].astype(int); n_norm = int((ISN).sum())
    for cn in ["나이·성별", "전부"]:
        s = oof(h, cn)[mk]; sa = V[f"amb|{h}|{cn}"]; w_amb = (int((y == 0).sum()) * len(AMB) / n_norm) / len(AMB)
        yy = np.r_[y, np.zeros(len(AMB))]; ss = np.r_[s, sa]; ww = np.r_[np.ones(len(y)), np.full(len(AMB), w_amb)]
        rows.append(dict(target=HN[h], model=cn, auc_정상만음성=roc_auc_score(y, s), auc_모호군포함=roc_auc_score(yy, ss, sample_weight=ww)))
Sp = pd.DataFrame(rows); Sp.to_csv(a.out + "/cv_spectrum.csv", index=False); print("\n=== 5. 스펙트럼 편향 (5겹 점수) ==="); print(Sp.round(3).to_string(index=False))

# ---------- 6. 그림 (5겹 점수 기준) ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
os.makedirs(a.out + "/figures", exist_ok=True)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for k, h in enumerate(HEADS):
    t = HN[h]; g = S[(S.target == t) & (S.model == "전부")].sort_values("scale")
    ax[k].plot(g.scale, g.auc, "o-", color="#265974", label="전부(MRI+설문·체격+SNP)"); ax[k].axhline(float(g[g.scale == 0].auc.iloc[0]), color="#999", ls="--", label="SNP 없이")
    ax[k].axvspan(.08, .2, color="#e7b04a", alpha=.35, label="실제 GWAS 효과 수준(추정)"); ax[k].set_title(t); ax[k].set_xlabel("SNP 효과 배율 (1 = 합성자료 그대로)"); ax[k].set_ylim(.7, .95)
ax[0].set_ylabel("5겹 교차검증 AUC"); ax[0].legend(fontsize=7, loc="lower right"); plt.suptitle("SNP 효과가 실제 수준으로 작아지면 성능 이득은 얼마나 남는가 (5겹 교차검증)"); plt.tight_layout(); plt.savefig(a.out + "/figures/snp_shrink.png", dpi=150, bbox_inches="tight"); plt.close()
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for k, h in enumerate(HEADS):
    mk = members(h); y = P[h][mk].astype(int)
    for cn, col in [("나이·성별", "#999"), ("+MRI+설문·체격", "#4182a6"), ("전부", "#265974")]:
        cal = platt(oof(h, cn)[mk], y); q = np.quantile(cal, np.linspace(0, 1, 9)); q[-1] += 1e-9; xs, ys = [], []
        for lo, hi in zip(q[:-1], q[1:]):
            m = (cal >= lo) & (cal < hi); xs.append(cal[m].mean()); ys.append(y[m].mean())
        ax[k].plot(xs, ys, "o-", color=col, label=cn)
    ax[k].plot([0, 1], [0, 1], "k:", lw=1); ax[k].set_title(HN[h]); ax[k].set_xlabel("예측 확률(보정 후)"); ax[k].set_ylabel("실제 비율")
ax[0].legend(fontsize=7); plt.suptitle("보정곡선 (5겹 교차검증 점수, 교차 보정, 표본 내 유병률 기준)"); plt.tight_layout(); plt.savefig(a.out + "/figures/calibration.png", dpi=150, bbox_inches="tight"); plt.close()
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6)); pts = np.linspace(.05, .7, 27)
nb = lambda p, y, ts: [((p >= t) & (y == 1)).sum() / len(y) - ((p >= t) & (y == 0)).sum() / len(y) * t / (1 - t) for t in ts]
for k, h in enumerate(HEADS):
    mk = members(h); y = P[h][mk].astype(int); prev = y.mean()
    ax[k].plot(pts, [prev - (1 - prev) * t / (1 - t) for t in pts], color="#bbb", label="전원 검사"); ax[k].axhline(0, color="k", lw=.8, label="아무도 안 함")
    for cn, col in [("나이·성별", "#999"), ("+MRI+설문·체격", "#4182a6"), ("전부", "#265974")]: ax[k].plot(pts, nb(platt(oof(h, cn)[mk], y), y, pts), color=col, label=cn)
    ax[k].set_title(HN[h]); ax[k].set_xlabel("임계 확률"); ax[k].set_ylim(-.05, max(.4, prev + .05))
ax[0].set_ylabel("순이득"); ax[0].legend(fontsize=7); plt.suptitle("의사결정곡선 (5겹 교차검증 점수)"); plt.tight_layout(); plt.savefig(a.out + "/figures/decision_curve.png", dpi=150, bbox_inches="tight"); plt.close(); print("그림 저장 완료")
