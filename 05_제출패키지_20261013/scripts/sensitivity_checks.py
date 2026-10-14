"""질문 대비 민감도 점검 3가지.
(1) 설문·체격 변수별 기여: 변수 하나를 뺐을 때와 하나만 썼을 때의 5겹 AUC 변화 (나이·성별 포함 모델, 질환마다).
(2) 최종 모델의 표준화 계수 상위 항목(MRI 영역, SNP, 설문): 연관의 방향과 크기이며 인과가 아니다.
(3) 선별 기준 민감도: 특이도 80~99%에서의 민감도·양성예측도·권고 인원, 나이대 경계를 바꿨을 때의 효과(5겹 교차검증 점수).
한계: (1)은 설문·체격 모델(MRI·SNP 제외)에서 계산했다. (3)의 기준은 같은 정상군 점수에서 정해 약간 낙관적이다."""
import argparse, glob, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
import mri_screen as ms, multimodal as mm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
for k in ["mri", "geno", "oof", "model", "out"]: ap.add_argument("--" + k)
a = ap.parse_args(); b = ms.load_all(a.mri)["BASE"]
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
lab = b.reindex(pd.Index(fam)); V = np.load(a.oof); REPS = sorted({int(k.split("|")[1]) for k in V.files if k.startswith("oof|")})
HEADS = ["any", "htn", "dm", "lip"]; HN = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
P = {"any": (lab.lab == "disease").values, "htn": (lab.htn == 1).values, "dm": (lab.dm == 1).values, "lip": (lab.lip == 1).values}; ISN = (lab.lab == "normal").values
PREV = {"any": float((b.lab == "disease").mean()), "htn": float(b.htn.mean()), "dm": float(b.dm.mean()), "lip": float(b.lip.mean())}
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
QN = {"T_DRINK": "음주", "T_SMOKE": "흡연", "T_EXER": "운동", "T_FMFHT1": "고혈압 가족력(부)", "T_FMFHT2": "고혈압 가족력(모)", "T_FMFDM1": "당뇨 가족력(부)", "T_FMFDM2": "당뇨 가족력(모)",
      "T_PSM": "심리 스트레스(PSM)", "T_BMI": "BMI", "T_WAIST": "허리둘레"}

# ---------- (1) 설문·체격 변수별 기여 ----------
def cv_auc(X, y, reps=2):
    out = []
    for r in range(reps):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=300 + r).split(X, y): s[te] = mm._lr().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out.append(roc_auc_score(y, s))
    return float(np.mean(out))
rows = []
for h in HEADS:
    mk = ISN | P[h]; d = lab[mk]; y = P[h][mk].astype(int); base = d[["age", "sex"]].values.astype(float)
    Qm = np.column_stack([ms._clean(d[c]).values for c in mm.Q]); full = cv_auc(np.hstack([base, Qm]), y); a0 = cv_auc(base, y)
    row = dict(질환=HN[h], 나이성별만=a0, 설문전체=full)
    for j, c in enumerate(mm.Q):
        row[f"뺐을때_{QN[c]}"] = cv_auc(np.hstack([base, np.delete(Qm, j, axis=1)]), y) - full
        row[f"하나만_{QN[c]}"] = cv_auc(np.hstack([base, Qm[:, [j]]]), y) - a0
    grp = {"체격(BMI·허리)": ["T_BMI", "T_WAIST"], "가족력 전체": ["T_FMFHT1", "T_FMFHT2", "T_FMFDM1", "T_FMFDM2"], "생활습관(음주·흡연·운동·PSM)": ["T_DRINK", "T_SMOKE", "T_EXER", "T_PSM"]}
    for g, cs in grp.items(): row[f"뺐을때_{g}"] = cv_auc(np.hstack([base, np.delete(Qm, [mm.Q.index(c) for c in cs], axis=1)]), y) - full
    rows.append(row); print(h, "완료", flush=True)
Q1 = pd.DataFrame(rows); Q1.to_csv(a.out + "/sens_questionnaire_vars.csv", index=False)
print("\n=== (1) 설문·체격 변수를 하나 뺐을 때 AUC 변화 (음수 = 그 변수가 기여) ==="); print(Q1[["질환", "나이성별만", "설문전체"] + [c for c in Q1.columns if c.startswith("뺐을때_")]].round(4).T.to_string(header=False))
print("\n=== 하나만 더했을 때 나이·성별 대비 AUC 이득 ==="); print(Q1[["질환"] + [c for c in Q1.columns if c.startswith("하나만_")]].round(4).T.to_string(header=False))

# ---------- (2) 최종 모델 표준화 계수 ----------
M = joblib.load(a.model); ref = M["ref"]; F = ref.transform(lab.iloc[:10]); mc = ms.colsets(F)["age+z+clip"][2:]
rows = []
for h in HEADS:
    m = M["models"][(h, "전부")]; coef = m.steps[2][1].coef_[0]; n0 = 2 + len(mc) + len(mm.Q) + len(M["leads"][h])
    names = ["나이", "성별"] + [("MRI " + c) for c in mc] + [("설문 " + QN[c]) for c in mm.Q] + [("SNP " + s) for s in M["leads"][h]]
    assert len(names) == n0, (len(names), n0, coef.shape)
    for nm, c in zip(names, coef[:n0]): rows.append(dict(질환=HN[h], 변수=nm, 표준화계수=float(c)))
C = pd.DataFrame(rows); C.to_csv(a.out + "/sens_final_coefficients.csv", index=False)
print("\n=== (2) 최종 모델 표준화 계수 상위 8 (양수 = 질환 쪽, 설명용이며 인과 아님) ===")
for h in HEADS: g = C[C.질환 == HN[h]].assign(abs_=lambda d: d.표준화계수.abs()).sort_values("abs_", ascending=False).head(8); print(HN[h], ":", "; ".join(f"{r.변수}({r.표준화계수:+.2f})" for r in g.itertuples()))

# ---------- (3) 선별 기준 민감도 ----------
oof = lambda h, cn="전부": np.mean([V[f"oof|{r}|{h}|{cn}"] for r in REPS], axis=0)
rows = []
for h in HEADS:
    mk = ISN | P[h]; y = P[h][mk].astype(int); s = oof(h)[mk]; age = lab.age.values[mk]; p = PREV[h]
    for spec in (.80, .85, .90, .95, .99):
        bd = np.digitize(age, [50, 60]); thr = np.array([np.quantile(s[(y == 0) & (bd == k)], spec) for k in range(3)]); fl = s >= thr[bd]
        sens, fpr = fl[y == 1].mean(), fl[y == 0].mean(); ref_n = 100 * (p * sens + (1 - p) * fpr)
        rows.append(dict(질환=HN[h], 특이도=spec, 민감도=sens, 양성예측도=sens * p / (sens * p + fpr * (1 - p)), 권고_100명당=ref_n, 환자_100명당_발견=100 * p * sens, 놓친_100명당=100 * p * (1 - sens)))
T = pd.DataFrame(rows); T.to_csv(a.out + "/sens_threshold.csv", index=False)
print("\n=== (3) 특이도를 바꿨을 때 (나이대별 기준, 100명당) ==="); print(T.round(3).to_string(index=False))
rows = []
for h in HEADS:
    mk = ISN | P[h]; y = P[h][mk].astype(int); s = oof(h)[mk]; age = lab.age.values[mk]
    for nm, cuts in {"기준 하나(나이 무시)": [], "50·60 (현재)": [50, 60], "45·55·65": [45, 55, 65], "40·50·60·70": [40, 50, 60, 70]}.items():
        bd = np.digitize(age, cuts) if cuts else np.zeros(len(age), int); K = bd.max() + 1
        thr = np.array([np.quantile(s[(y == 0) & (bd == k)] if ((y == 0) & (bd == k)).sum() >= 30 else s[y == 0], .90) for k in range(K)]); fl = s >= thr[bd]   # 정상 30명 미만인 구간은 전체 정상군 기준
        fp_by = [fl[(y == 0) & (np.digitize(age, [50, 60]) == k)].mean() for k in range(3)]        # 보고 기준 나이대로 비교
        rows.append(dict(질환=HN[h], 나이대_나누기=nm, 민감도=fl[y == 1].mean(), 오탐_50미만=fp_by[0], 오탐_50대=fp_by[1], 오탐_60이상=fp_by[2]))
A = pd.DataFrame(rows); A.to_csv(a.out + "/sens_age_bands.csv", index=False)
print("\n=== 나이대를 어떻게 나누느냐에 따른 효과 (정상인 오탐률은 50미만/50대/60이상으로 비교) ==="); print(A.round(3).to_string(index=False))

# ---------- (4) 모달리티 추가 효과의 짝지은 부트스트랩 (같은 사람의 점수 차이) ----------
rng = np.random.RandomState(1); rows = []
for h in HEADS:
    mk = ISN | P[h]; y = P[h][mk].astype(int); full = oof(h, "전부")[mk]
    for nm, cn in {"MRI·설문·SNP 모두 vs 나이·성별만": "나이·성별", "SNP 추가 효과 (vs +MRI+설문·체격)": "+MRI+설문·체격", "MRI 추가 효과 (vs +설문·체격+SNP)": "+설문·체격+SNP", "설문·체격 추가 효과 (vs +MRI+SNP)": "+MRI+SNP"}.items():
        o = oof(h, cn)[mk]; d = [roc_auc_score(y[i], full[i]) - roc_auc_score(y[i], o[i]) for i in (rng.choice(len(y), len(y)) for _ in range(300))]
        rows.append(dict(질환=HN[h], 비교=nm, 차이=roc_auc_score(y, full) - roc_auc_score(y, o), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
D = pd.DataFrame(rows); D.to_csv(a.out + "/sens_paired_gain.csv", index=False)
print("\n=== (4) '전부' 모델의 AUC 이득 (같은 사람 짝지은 부트스트랩 95% 구간) ==="); print(D.round(4).to_string(index=False))
