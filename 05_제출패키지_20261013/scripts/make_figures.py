"""보고서용 그림: (1) 모달리티별 기여, (2) 유전→진단→MRI 매개 구조도."""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
P = sys.argv[1]; os.makedirs(P + "/results/figures", exist_ok=True)
E = pd.read_csv(P + "/model/cv_full_results.csv")
order = ["나이·성별", "+설문·체격", "+MRI", "+SNP", "+MRI+설문·체격", "+MRI+SNP", "전부"]
cols = ["#bfc8cf", "#9ec5d8", "#4182a6", "#e7b04a", "#2f6f93", "#c98a2b", "#1d3f55"]
targets = ["전체 질환", "고혈압", "당뇨", "고지혈"]
fig, ax = plt.subplots(figsize=(11, 4.2)); w = .11
for i, (m, c) in enumerate(zip(order, cols)):
    v = [float(E[(E.target == t) & (E.model == m)].auc.iloc[0]) for t in targets]
    lo = [float(E[(E.target == t) & (E.model == m)].lo.iloc[0]) for t in targets]; hi = [float(E[(E.target == t) & (E.model == m)].hi.iloc[0]) for t in targets]
    x = np.arange(len(targets)) + (i - 3) * w
    ax.bar(x, v, w, color=c, label=m, yerr=[np.array(v) - lo, np.array(hi) - v], error_kw=dict(lw=.8, capsize=1.5))
ax.set_xticks(np.arange(len(targets))); ax.set_xticklabels(targets); ax.set_ylim(.6, 1.0); ax.set_ylabel("테스트 AUC (95% 신뢰구간)")
ax.set_title("질환별로 기여하는 모달리티가 다르다 (검증 30%, 학습에 쓰지 않은 개체)"); ax.legend(ncol=4, fontsize=8, loc="upper left"); plt.tight_layout()
plt.savefig(P + "/results/figures/modality_contribution.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(10, 3.8)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
def box(x, y, t, c):
    ax.add_patch(FancyBboxPatch((x, y), 2.4, 1.1, boxstyle="round,pad=0.05", fc=c, ec="#233541")); ax.text(x + 1.2, y + .55, t, ha="center", va="center", fontsize=12)
box(.3, 1.4, "유전 위험 점수\n(SNP)", "#f3dfb3"); box(3.8, 1.4, "진단 여부\n(고혈압·당뇨·고지혈)", "#cfe3ee"); box(7.3, 1.4, "MRI 위축\n(영역 편차)", "#d9e6dd")
ax.annotate("", xy=(3.75, 1.95), xytext=(2.75, 1.95), arrowprops=dict(arrowstyle="->", lw=2)); ax.annotate("", xy=(7.25, 1.95), xytext=(6.25, 1.95), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(8.5, 1.35), xytext=(1.5, 1.35), arrowprops=dict(arrowstyle="->", lw=1.5, ls="--", color="#b44", connectionstyle="arc3,rad=0.28"))
ax.text(5.0, .15, "직접 경로: 진단을 보정하면 사라짐 (고혈압 +0.103 → +0.010)", ha="center", color="#b44", fontsize=10.5)
ax.text(3.25, 2.75, "SNP 점수는 진단과 강하게 연관\n(오즈비 대립유전자당 1.55~1.98)", ha="center", fontsize=9.5); ax.text(6.75, 2.75, "MRI 위축은 진단과 연관\n(진단군-정상군 d=0.46)", ha="center", fontsize=9.5)
ax.set_title("유전 → 진단 → 뇌 변화: 두 모달리티는 진단이 같으면 서로 무관하므로 정보가 독립적으로 더해진다", fontsize=11)
plt.tight_layout(); plt.savefig(P + "/results/figures/mediation_structure.png", dpi=150); plt.close(); print("ok")
