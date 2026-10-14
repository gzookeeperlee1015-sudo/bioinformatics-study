"""현장 적용 가능성 근거: 100명당 시나리오, 유병률 민감도, 성별·나이대 하위군 성능, 현장 흐름도.
5겹 교차검증 점수(cv_oof.npz)만 사용한다. 나이대별 선별 기준은 정상군 점수의 90백분위(모델 배포 기준과 같다).
한계: 음성은 '정상'만이므로 다른 질환자는 음성 집단에 없다. 유병률 환산은 BASE 유병률 또는 아래 배율을 가정한 것이다."""
import argparse, glob, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import mri_screen as ms
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
for k in ["mri", "geno", "oof", "out"]: ap.add_argument("--" + k)
a = ap.parse_args(); b = ms.load_all(a.mri)["BASE"]
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]
lab = b.reindex(pd.Index(fam)); V = np.load(a.oof); REPS = sorted({int(k.split("|")[1]) for k in V.files if k.startswith("oof|")})
HEADS = ["any", "htn", "dm", "lip"]; HN = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
P = {"any": (lab.lab == "disease").values, "htn": (lab.htn == 1).values, "dm": (lab.dm == 1).values, "lip": (lab.lip == 1).values}; ISN = (lab.lab == "normal").values
PREV = {"any": float((b.lab == "disease").mean()), "htn": float(b.htn.mean()), "dm": float(b.dm.mean()), "lip": float(b.lip.mean())}
oof = lambda h, cn="전부": np.mean([V[f"oof|{r}|{h}|{cn}"] for r in REPS], axis=0)
band = lambda age: np.digitize(age, [50, 60])
pd.set_option("display.width", 220)

# ---------- 1. 100명당 시나리오 + 유병률 민감도 ----------
rows, prow = [], []
for h in HEADS:
    mk = ISN | P[h]; y = P[h][mk].astype(int); s = oof(h)[mk]; bd = band(lab.age.values[mk])
    thr = [np.quantile(s[(y == 0) & (bd == k)], .90) for k in range(3)]; fl = s >= np.array(thr)[bd]
    sens = fl[y == 1].mean(); fpr = fl[y == 0].mean(); p = PREV[h]
    tp, fn, fp = 100 * p * sens, 100 * p * (1 - sens), 100 * (1 - p) * fpr; ref = tp + fp
    rows.append(dict(질환=HN[h], 유병률=p, 환자수_100명당=100 * p, 정밀검사_권고=ref, 권고_중_실제환자=tp, 권고_중_정상=fp, 놓친_환자=fn, 양성예측도=tp / ref,
                     권고1명당_환자발견=tp / ref, 환자1명_찾는데_권고수=ref / tp, 민감도=sens, 오탐률=fpr))
    for mult in (0.5, 1, 2):
        pp = min(p * mult, .95); prow.append(dict(질환=HN[h], 유병률_배율=mult, 유병률=pp, 양성예측도=sens * pp / (sens * pp + fpr * (1 - pp)), 음성예측도=(1 - fpr) * (1 - pp) / ((1 - fpr) * (1 - pp) + (1 - sens) * pp)))
R = pd.DataFrame(rows); R.to_csv(a.out + "/cv_deploy_per100.csv", index=False); Pv = pd.DataFrame(prow); Pv.to_csv(a.out + "/cv_deploy_prevalence.csv", index=False)
print("=== 1. MRI·설문·유전 정보가 모두 있는 100명당 (나이대별 특이도 90% 기준) ==="); print(R.round(3).to_string(index=False))
print("\n=== 유병률이 절반/그대로/두 배일 때 양성·음성 예측도 ==="); print(Pv.round(3).to_string(index=False))

# ---------- 2. 하위군 성능 (성별 x 나이대) ----------
rng = np.random.RandomState(0); rows = []
SEX = lab.sex.values
def sub(h, m, name, grp):
    mk = (ISN | P[h]) & m; y = P[h][mk].astype(int); s = oof(h)[mk]
    if y.sum() < 30 or (1 - y).sum() < 30: return
    bs = [roc_auc_score(y[i], s[i]) for i in (rng.choice(len(y), len(y)) for _ in range(200)) if 0 < y[i].sum() < len(i)]
    rows.append(dict(질환=HN[h], 구분=grp, 하위군=name, n=len(y), 환자=int(y.sum()), auc=roc_auc_score(y, s), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
AGE = lab.age.values
for h in HEADS:
    sub(h, np.ones(len(lab), bool), "전체", "전체")
    for sx, nm in ((1, "남성"), (2, "여성")): sub(h, SEX == sx, nm, "성별")
    for k, nm in enumerate(["50세 미만", "50대", "60세 이상"]): sub(h, band(AGE) == k, nm, "나이대")
H = pd.DataFrame(rows); H.to_csv(a.out + "/cv_subgroup_auc.csv", index=False)
print("\n=== 2. 하위군 AUC (5겹 점수, 95% 신뢰구간) ==="); print(H.round(3).to_string(index=False))
print("\n성별 코딩 값:", sorted(pd.Series(SEX).dropna().unique()))

# ---------- 3. 현장 흐름도 ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
os.makedirs(a.out + "/figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(13, 5.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")
def box(x, y, w, h, t, fc, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="#265974", lw=1.2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None, linespacing=1.45)
arrow = lambda x1, y1, x2, y2, t="": (ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#265974", lw=1.5)), t and ax.text((x1 + x2) / 2, (y1 + y2) / 2 + .13, t, ha="center", fontsize=8, color="#555"))
box(.2, 3.6, 2.5, 1.2, "대상자\n(MRI를 이미 촬영했거나\n설문·체격만 있음)", "#eef3f7", bold=True)
box(3.4, 3.6, 2.7, 1.2, "1단계 입력 확인\n있는 정보로 모델 조합 자동 선택\n(나이·성별 / 설문·체격 / MRI / SNP)", "#f3eee2")
box(6.9, 3.6, 2.7, 1.2, "점수 산출\n질환별 점수 + 개인별 근거\n(MRI·설문·SNP 기여)", "#f3eee2")
box(10.2, 3.6, 2.6, 1.2, "나이대별 기준 비교\n(50 미만 / 50대 / 60+)\n정상인 90% 통과 기준", "#f3eee2")
arrow(2.7, 4.2, 3.4, 4.2); arrow(6.1, 4.2, 6.9, 4.2); arrow(9.6, 4.2, 10.2, 4.2)
box(8.2, 1.7, 2.3, 1.1, "선별 표시 1\n해당 질환 정밀검사 권고\n(혈압·혈당·지질검사)", "#d9ecd9", bold=True)
box(10.9, 1.7, 1.9, 1.1, "선별 표시 0\n정기 검진 유지", "#eeeeee")
arrow(11.1, 3.6, 9.4, 2.8); arrow(11.5, 3.6, 11.85, 2.8)
box(.2, 1.55, 7.6, 1.4, "비용을 고려한 검사 순서 (질환별)\n고혈압·전체: 설문·체격 → MRI → SNP\n당뇨: 설문·체격 → SNP → MRI      고지혈: 설문·체격 → SNP (MRI 생략)\n애매한 점수대만 다음 검사로 진행. 성능 손실 0.01 이내에서 MRI 29~100%, SNP 24~66% 절감", "#fff7e0", fs=9)
arrow(6.9, 3.6, 5.5, 2.95)
ax.text(.2, .95, "※ 이 도구는 혈압·혈당·지질 검사를 대체하지 않는다. 이미 확보된 정보로 정밀검사를 권할 대상을 1차로 고르는 보조 도구이다.", fontsize=9.5, color="#265974", fontweight="bold")
ax.text(.2, .45, "※ 합성자료로 개발·검증했으며, 실제 코호트 적용 전에는 사전 고정된 기준으로 외부 검증이 필요하다(보고서 ②-13).", fontsize=9, color="#a33")
plt.title("현장 적용 흐름: 입력 → 점수·근거 → 나이대별 기준 → 정밀검사 권고", fontsize=12, pad=8)
plt.savefig(a.out + "/figures/deployment_flow.png", dpi=150, bbox_inches="tight"); plt.close(); print("\n흐름도 저장")
