"""임상적 유용성 분석: 보정(calibration), PPV/NPV, 상위 20% 포착률, 의사결정곡선, SNP 효과 축소 민감도.
multimodal.py 로 학습·저장한 모델과 같은 30% 검증 자료를 사용한다(학습에 쓰지 않은 개체)."""
import argparse, glob, json, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
import mri_screen as ms, multimodal as mm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
for k in ["mri", "kept", "geno", "te", "model", "out"]: ap.add_argument("--" + k)
a = ap.parse_args()
M = joblib.load(a.model); ref = M["ref"]
D = ms.load_all(a.mri); b = D["BASE"]
K = np.load(a.kept, allow_pickle=True); idx = {n: i for i, n in enumerate(K["names"])}; X = K["X"]
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
pos = {n: i for i, n in enumerate(fam)}; lab = b.reindex(pd.Index(fam)); te = np.load(a.te)
PREV = {h: float(v) for h, v in {"any": ((b.lab == "disease").mean()), "htn": b.htn.mean(), "dm": b.dm.mean(), "lip": b.lip.mean()}.items()}

def test_set(h):
    posf = (lambda d: d.lab == "disease") if h == "any" else (lambda d: d[h] == 1)
    ted = lab.iloc[te]; ted = ted[((ted.lab == "normal") | posf(ted)).values]
    sn = np.stack([X[idx[s]][[pos[n] for n in ted.index]] for s in M["leads"][h]], axis=1)
    return ted, posf(ted).astype(int).values, sn

def scores(h, cn, ted, sn, snp_scale=None):
    parts = M["combos"][cn]; Xa, _ = mm.blocks(ref, ted, sn, parts); m = M["models"][(h, cn)]
    if snp_scale is None: return m.predict_proba(Xa)[:, 1], None
    imp, sc, lr = m.steps[0][1], m.steps[1][1], m.steps[2][1]
    Xs = sc.transform(imp.transform(Xa)); contrib = Xs * lr.coef_[0]
    n_orig = Xa.shape[1]; ns = sn.shape[1]; contrib[:, n_orig - ns:n_orig] *= snp_scale
    logit = contrib.sum(1) + lr.intercept_[0]; return 1 / (1 + np.exp(-logit)), logit

def platt_cv(s, y, seed=0):
    lg = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))).reshape(-1, 1); out = np.zeros(len(y))
    for tr, tt in StratifiedKFold(5, shuffle=True, random_state=seed).split(lg, y):
        out[tt] = LogisticRegression(C=1e6).fit(lg[tr], y[tr]).predict_proba(lg[tt])[:, 1]
    return out

def ece(p, y, bins=10):
    e = 0.0; q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e

def net_benefit(p, y, pts):
    n = len(y); return [((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t) for t in pts]

rows, curves, dca, shrink = [], {}, {}, []
HEADS = ["any", "htn", "dm", "lip"]; NAMES = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
FS = [1, .75, .5, .35, .25, .15, .1, .05, 0]
for h in HEADS:
    ted, y, sn = test_set(h)
    for cn in M["combos"]:
        s, _ = scores(h, cn, ted, sn)
        cal = platt_cv(s, y); thr = M["thr"][(h, cn)]; flag = s >= thr
        sens = float(flag[y == 1].mean()); spec = float((~flag)[y == 0].mean()); p = PREV[h]
        ppv_pop = sens * p / (sens * p + (1 - spec) * (1 - p)); npv_pop = spec * (1 - p) / (spec * (1 - p) + (1 - sens) * p)
        top = np.argsort(-s)[:int(np.ceil(.2 * len(s)))]; capture = float(y[top].sum() / y.sum())
        rows.append(dict(target=NAMES[h], model=cn, n=len(y), pos=int(y.sum()), auc=roc_auc_score(y, s), brier_raw=brier_score_loss(y, s), brier_cal=brier_score_loss(y, cal),
                         ece_raw=ece(s, y), ece_cal=ece(cal, y), sens_spec90=sens, spec=spec, ppv_sample=float(y[flag].mean()) if flag.any() else np.nan,
                         ppv_pop=ppv_pop, npv_pop=npv_pop, capture_top20=capture, lift_top20=capture / .2, pop_prev=p))
        if cn in ("나이·성별", "+설문·체격", "+MRI+설문·체격", "전부"): curves[(h, cn)] = (cal, y)
    if "전부" in M["combos"]:
        for cn in ("+SNP", "+MRI+SNP", "전부"):
            base_s, _ = scores(h, cn, ted, sn)
            for f in FS:
                _, lg = scores(h, cn, ted, sn, snp_scale=f); shrink.append(dict(target=NAMES[h], model=cn, scale=f, auc=roc_auc_score(y, lg)))
R = pd.DataFrame(rows); R.to_csv(a.out + "/utility_table.csv", index=False)
S = pd.DataFrame(shrink); S.to_csv(a.out + "/snp_shrink.csv", index=False)
pd.set_option("display.width", 250)
print(R[R.model.isin(["나이·성별", "+MRI+설문·체격", "전부"])][["target", "model", "auc", "brier_raw", "brier_cal", "ece_raw", "ece_cal", "sens_spec90", "ppv_sample", "ppv_pop", "npv_pop", "capture_top20", "lift_top20"]].round(3).to_string(index=False))
print("\nSNP 효과를 f배로 줄였을 때 테스트 AUC")
print(S[S.model == "전부"].pivot(index="scale", columns="target", values="auc").round(3).loc[FS].to_string())
print("나이·성별 기준:", {NAMES[h]: round(float(R[(R.target == NAMES[h]) & (R.model == '나이·성별')].auc.iloc[0]), 3) for h in HEADS}, "| SNP 없이 MRI+설문·체격:", {NAMES[h]: round(float(R[(R.target == NAMES[h]) & (R.model == '+MRI+설문·체격')].auc.iloc[0]), 3) for h in HEADS})

# ---------- 그림 ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
os.makedirs(a.out + "/figures", exist_ok=True)
# (1) SNP 축소 민감도
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
for k, h in enumerate(HEADS):
    t = NAMES[h]; g = S[(S.target == t) & (S.model == "전부")].sort_values("scale")
    ax[k].plot(g.scale, g.auc, "o-", color="#265974", label="전부(MRI+설문·체격+SNP)")
    ax[k].axhline(float(R[(R.target == t) & (R.model == "+MRI+설문·체격")].auc.iloc[0]), color="#999", ls="--", label="SNP 없이")
    ax[k].axvspan(.08, .2, color="#e7b04a", alpha=.35, label="실제 GWAS 효과 수준(추정)")
    ax[k].set_title(t); ax[k].set_xlabel("SNP 효과 배율 (1 = 합성자료 그대로)"); ax[k].set_ylim(0.7, 0.95)
ax[0].set_ylabel("테스트 AUC"); ax[0].legend(fontsize=7, loc="lower right"); plt.suptitle("SNP 효과가 실제 수준으로 작아지면 성능 이득은 얼마나 남는가"); plt.tight_layout(); plt.savefig(a.out + "/figures/snp_shrink.png", dpi=150, bbox_inches="tight"); plt.close()
# (2) 보정곡선
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for k, h in enumerate(HEADS):
    for cn, col in [("나이·성별", "#999"), ("+MRI+설문·체격", "#4182a6"), ("전부", "#265974")]:
        cal, y = curves[(h, cn)]; q = np.quantile(cal, np.linspace(0, 1, 9)); q[-1] += 1e-9; xs, ys = [], []
        for lo, hi in zip(q[:-1], q[1:]):
            m = (cal >= lo) & (cal < hi); xs.append(cal[m].mean()); ys.append(y[m].mean())
        ax[k].plot(xs, ys, "o-", color=col, label=cn)
    ax[k].plot([0, 1], [0, 1], "k:", lw=1); ax[k].set_title(NAMES[h]); ax[k].set_xlabel("예측 확률(보정 후)"); ax[k].set_ylabel("실제 비율")
ax[0].legend(fontsize=7); plt.suptitle("보정곡선 (검증 30%, 교차 보정, 표본 내 유병률 기준)"); plt.tight_layout(); plt.savefig(a.out + "/figures/calibration.png", dpi=150, bbox_inches="tight"); plt.close()
# (3) 의사결정곡선
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6)); pts = np.linspace(.05, .7, 27)
for k, h in enumerate(HEADS):
    _, y = curves[(h, "전부")]; prev = y.mean()
    ax[k].plot(pts, [prev - (1 - prev) * t / (1 - t) for t in pts], color="#bbb", label="전원 검사")
    ax[k].axhline(0, color="k", lw=.8, label="아무도 안 함")
    for cn, col in [("나이·성별", "#999"), ("+MRI+설문·체격", "#4182a6"), ("전부", "#265974")]:
        cal, yy = curves[(h, cn)]; ax[k].plot(pts, net_benefit(cal, yy, pts), color=col, label=cn)
    ax[k].set_title(NAMES[h]); ax[k].set_xlabel("임계 확률"); ax[k].set_ylim(-0.05, max(.4, prev + .05))
ax[0].set_ylabel("순이득"); ax[0].legend(fontsize=7); plt.suptitle("의사결정곡선 (검증 30%)"); plt.tight_layout(); plt.savefig(a.out + "/figures/decision_curve.png", dpi=150, bbox_inches="tight"); plt.close()
print("그림 저장:", os.listdir(a.out + "/figures"))
