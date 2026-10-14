"""DLBS(실제 자료)에서 대회 자료의 임상 신호 방향 재현 확인.
대회(합성) 결과: 고혈압·당뇨 자기보고가 있으면 초기 해마가 낮고(-0.3~-0.4mL), 이후 위축은 오히려 적었다(-0.26~-0.35mL).
여기서는 측정 혈압을 노출로 쓴다(자기보고 고혈압이 DLBS 파생자료에 없음).
"""
import numpy as np, pandas as pd, sys
R = "C:/junny/bioinformaticsstudy/bioinfo대회/03_DLBS_원자료/확장제공본/"
mri = pd.ExcelFile(R + "derivatives/brainsummary/Template_Structural_MRI.xlsx")
ph = pd.ExcelFile(R + "derivatives/surveys/Template8_Physical_Health.xlsx")
p = pd.read_csv(R + "participants.tsv", sep="\t")

def num(s): return pd.to_numeric(s, errors="coerce")
def wave(w):
    sv = mri.parse(f"SubcorticalVolume-W{w}"); gv = mri.parse(f"GlobalVariables-W{w}")
    h = (num(sv.LhHippocampusVol) + num(sv.RhHippocampusVol)) / 1000.0
    out = pd.DataFrame({"S": sv["S#"], f"h{w}": h})
    out = out.merge(pd.DataFrame({"S": gv["S#"], f"icv{w}": num(gv.TotalIntracranialVol) / 1000.0}), on="S")
    return out
W = wave(1).merge(wave(2), on="S", how="left").merge(wave(3), on="S", how="left")
b = ph.parse("Wave1")
sys_c = [c for c in b.columns if c.startswith("BPDay") and "Sys" in c]; dia_c = [c for c in b.columns if c.startswith("BPDay") and "Dia" in c]
bp = pd.DataFrame({"S": b["S#"], "sbp": b[sys_c].apply(num).mean(axis=1), "dbp": b[dia_c].apply(num).mean(axis=1),
                   "nbp": b[sys_c].apply(num).notna().sum(axis=1)})
W = W.merge(bp, on="S", how="left")
p["S"] = p.participant_id.str.replace("sub-", "", regex=False)
p["S"] = pd.to_numeric(p["S"], errors="coerce")
W = W.merge(p[["S", "AgeMRI_W1", "Sex", "MRIW1toW2", "MRIW1toW3", "BMI_W1"]], on="S", how="left")
print("병합 후 n =", len(W), "| 나이/성별 결측:", int(W.AgeMRI_W1.isna().sum()), int(W.Sex.isna().sum()), "| BP 유효:", int((W.nbp >= 2).sum()))
W["sex"] = W.Sex.map({"F": 1, "M": 0, "f": 1, "m": 0}).fillna(W.Sex.map({1: 0, 2: 1}) if W.Sex.dtype != object else np.nan)
print("Sex 값:", W.Sex.value_counts().to_dict())
W["htn"] = ((W.sbp >= 140) | (W.dbp >= 90)).astype(float).where(W.nbp >= 2)
W["sbp10"] = W.sbp / 10.0

def ols(y, X, names):
    X1 = np.c_[np.ones(len(X)), np.asarray(X, float)]; y = np.asarray(y, float)
    b_ = np.linalg.lstsq(X1, y, rcond=None)[0]; e = y - X1 @ b_
    Xi = np.linalg.inv(X1.T @ X1); meat = X1.T @ (X1 * (e ** 2)[:, None]) * len(y) / (len(y) - X1.shape[1])
    se = np.sqrt(np.diag(Xi @ meat @ Xi)); return pd.DataFrame({"coef": b_[1:], "se": se[1:]}, index=names), e

def run(k, expo, amin=40, amax=200, label=""):
    hk, yrs = f"h{k}", f"MRIW1to{'W'+str(k)}"
    d = W.dropna(subset=["h1", hk, "icv1", "AgeMRI_W1", "sex", expo, yrs]).copy()
    d = d[(d.AgeMRI_W1 >= amin) & (d.AgeMRI_W1 < amax)]
    d["a"] = d.AgeMRI_W1 - 55; d["a2"] = d.a ** 2
    cov = ["a", "a2", "sex", "icv1", yrs]
    names = [expo] + cov
    if len(d) < 30 or d[expo].nunique() < 2: print(f"{label} n={len(d)} 부족"); return
    n1 = int((d[expo] == 1).sum()) if expo == "htn" else None
    off, _ = ols(d.h1, d[[expo, "a", "a2", "sex", "icv1"]], [expo, "a", "a2", "sex", "icv1"])
    raw, _ = ols(d.h1 - d[hk], d[names], names)                     # 클수록 위축
    anc, e = ols(d[hk], d[names + ["h1"]], names + ["h1"])         # h5 에 대한 계수 -> 부호 반전 = 위축
    gm = d.groupby(expo).h1.transform("mean") if expo == "htn" else d.h1.mean()
    h1c = gm + 0.9 * (d.h1 - gm)
    cal, _ = ols(d[hk], d[names].assign(h1c=h1c), names + ["h1c"])
    sd_r = e.std()
    f = lambda t, s=1: f"{s*t.loc[expo,'coef']:+.3f}±{1.96*t.loc[expo,'se']:.3f}"
    print(f"{label:<34s} n={len(d):<4d}{'(고혈압 '+str(n1)+')' if n1 is not None else '':<11s} 기저해마 {f(off)} | 단순위축 {f(raw)} | h1보정 {f(anc,-1)} | λ=.9 {f(cal,-1)} | h1보정/SD {-anc.loc[expo,'coef']/sd_r:+.2f}")

print("\n(mL, ±95%CI; 위축 쪽 양수 = 더 많이 줄어듦. 대회 합성자료: 고혈압 기저 -0.29~-0.41, 단순위축 -0.35, h1보정 -0.26)")
for k, nm in [(3, "1→3차(약9년, 대회와 가장 비슷)"), (2, "1→2차(약4년)")]:
    print(f"\n== {nm} ==")
    run(k, "htn", 40, 200, "고혈압범위(≥140/90), 나이≥40")
    run(k, "htn", 40, 75, "고혈압범위, 40≤나이<75")
    run(k, "sbp10", 40, 200, "수축기혈압(10mmHg당), 나이≥40")
    run(k, "sbp10", 40, 75, "수축기혈압(10mmHg당), 40≤나이<75")
