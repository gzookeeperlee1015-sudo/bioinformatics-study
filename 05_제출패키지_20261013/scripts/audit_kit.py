"""합성 코호트 충실도 감사 키트 (한 명령으로 점검표 생성).

  python audit_kit.py --mri <BrainVol csv 폴더> --clin <임상 txt 폴더> --dlbs <DLBS 확장제공본 폴더> \
                      --adni <ADNIMERGE.csv> --out <출력 폴더> [--geno <유전체 zip>]

점검 항목: 임상 라벨 코딩(누적/신규), 상한 클리핑, 횡단 효과(나이·ICV)의 실제 자료 대조, 종단 위축 속도의 나이 의존성,
영역 간 상관 구조, MRI(·SNP) 신호가 진단 라벨에 묶이는지, 추적 탈락의 선택성, 고령 표본 수, (선택) SNP 품질.
판정 기준은 코드 안의 규칙이며 임의 조정 없이 같은 규칙을 어떤 합성 코호트에도 적용한다."""
import argparse, glob, json, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
for k in ["mri", "clin", "dlbs", "adni", "out", "geno"]: ap.add_argument("--" + k)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rows = []
def add(item, synth, real, verdict, note=""): rows.append(dict(항목=item, 합성자료=synth, 실제자료=real, 판정=verdict, 비고=note))

def num(s): return pd.to_numeric(s, errors="coerce")
def ols(y, X):
    X1 = np.c_[np.ones(len(X)), np.asarray(X, float)]; return np.linalg.lstsq(X1, np.asarray(y, float), rcond=None)[0][1:]

# ---------------- 자료 적재 ----------------
mri_files = {os.path.basename(f).replace("BrainVol_", "").replace(".csv", ""): f for f in glob.glob(a.mri + "/BrainVol_*.csv")}
bm = pd.read_csv(mri_files["BASE"]).set_index("T_ID")
cl = {k: pd.read_csv(a.clin + f"/FOLLOW_{k:02d}_DATA.txt").set_index("T00_ID") for k in range(1, 6)}
bc = pd.concat([pd.read_csv(a.clin + f"/BASE_DATA{i}.txt").set_index("T_ID") for i in (1, 2, 3)], axis=1); bc = bc.loc[:, ~bc.columns.duplicated()]
B = bc.join(bm, how="inner")
vols = [c for c in bm.columns if c.startswith("BV_")]
H = "BV_subctx_hippocampus"
DL = a.dlbs + "/derivatives/"
dl_mri = pd.ExcelFile(DL + "brainsummary/Template_Structural_MRI.xlsx"); dl_p = pd.read_csv(a.dlbs + "/participants.tsv", sep="\t")
dl_p["S"] = pd.to_numeric(dl_p.participant_id.str.replace("sub-", ""))

# 1) 임상 라벨 코딩
prev = {d: [float((num(cl[k][f"T{k:02d}_{d}"]) == 2).mean()) for k in range(1, 6)] for d in ["HTN", "DM", "LIP"]}
drop = prev["HTN"][1] / prev["HTN"][0]
add("임상 질환 응답의 의미", "고혈압 응답률 1~5차: " + " → ".join(f"{v:.0%}" for v in prev["HTN"]), "누적 유병률은 시간이 갈수록 증가",
    "주의" if drop < .5 else "양호", "2차 이후 응답이 신규 진단이면 이전 차수와 누적해서 써야 함" if drop < .5 else "")

# 2) 상한 클리핑
clip = {}
for nm, d in [("BASE", bm)] + [(k, pd.read_csv(f).set_index(pd.read_csv(f).columns[0])) for k, f in mri_files.items() if k != "BASE"]:
    for c in vols:
        if c in d.columns:
            fr = float((d[c] == d[c].max()).mean())
            if fr > .05: clip.setdefault(c, {})[nm] = round(fr, 3)
if clip:
    ex = max(clip.items(), key=lambda kv: max(kv[1].values())); worst = max(ex[1].values())
    add("체적 변수의 상한 절단", f"{len(clip)}개 변수가 최댓값에 5% 초과 몰림 (예: {ex[0]} " + ", ".join(f"{k} {v:.0%}" for k, v in ex[1].items()) + ")", "해당 없음(연속 분포)", "결함", "변수 목록: " + ", ".join(clip))
else: add("체적 변수의 상한 절단", "없음", "-", "양호")

# 3) 횡단 효과: 나이, ICV (해마)
age = B.T_AGE.values; sx = B.T_SEX.values; icv = B.BV_icv.values
c = ols(B[H].values, np.c_[age - 55, sx, icv]); s_age, s_icv = c[0] * 10, c[2] * 100
s_dl = dl_mri.parse("SubcorticalVolume-W1"); g_dl = dl_mri.parse("GlobalVariables-W1")
D1 = pd.DataFrame({"S": s_dl["S#"], "h": (num(s_dl.LhHippocampusVol) + num(s_dl.RhHippocampusVol)) / 1000}).merge(pd.DataFrame({"S": g_dl["S#"], "icv": num(g_dl.TotalIntracranialVol) / 1000}), on="S").merge(dl_p[["S", "AgeMRI_W1", "Sex"]], on="S")
D1["sex"] = D1.Sex.map({"m": 1, "f": 2, "M": 1, "F": 2}); D1 = D1[D1.AgeMRI_W1 >= 40].dropna()
c2 = ols(D1.h.values, np.c_[D1.AgeMRI_W1 - 55, D1.sex, D1.icv]); r_age, r_icv = c2[0] * 10, c2[2] * 100
add("횡단 해마의 나이 효과 (mL/10년)", f"{s_age:+.3f} (n={len(B)})", f"{r_age:+.3f} (DLBS n={len(D1)})", "양호" if abs(s_age - r_age) / abs(r_age) < .25 else "주의")
add("횡단 해마의 ICV 효과 (mL/100mL)", f"{s_icv:+.3f}", f"{r_icv:+.3f}", "양호" if abs(s_icv - r_icv) / abs(r_icv) < .25 else "주의")

# 4) 종단 위축 속도의 나이 의존성
bands = [(40, 50), (50, 60), (60, 70), (70, 90)]
def band_rates(age_v, pct):
    return [float(np.nanmedian(pct[(age_v >= lo) & (age_v < hi)])) if ((age_v >= lo) & (age_v < hi)).sum() >= 10 else np.nan for lo, hi in bands]
fk = sorted(k for k in mri_files if k.startswith("FOLLOW"))
f_first, f_last = pd.read_csv(mri_files[fk[0]]).set_index("T00_ID"), pd.read_csv(mri_files[fk[-1]]).set_index("T00_ID")
kf, kl = int(fk[0][-2:]), int(fk[-1][-2:])
ids = f_first.index.intersection(f_last.index); yrs = (num(cl[kl][f"T{kl:02d}_AGE"]) - num(cl[kf][f"T{kf:02d}_AGE"])).loc[ids]
syn = band_rates(num(cl[kf][f"T{kf:02d}_AGE"]).loc[ids].values, ((f_last.loc[ids, H] - f_first.loc[ids, H]) / f_first.loc[ids, H] * 100 / yrs).values)
def dl_rates(w):
    def hp(x): s_ = dl_mri.parse(f"SubcorticalVolume-W{x}"); return pd.DataFrame({"S": s_["S#"], f"h{x}": num(s_.LhHippocampusVol) + num(s_.RhHippocampusVol)})
    W = hp(1).merge(hp(w), on="S", how="left").merge(dl_p[["S", "AgeMRI_W1", f"MRIW1toW{w}"]], on="S").dropna(); W = W[W.AgeMRI_W1 >= 40]
    return band_rates(W.AgeMRI_W1.values, ((W[f"h{w}"] - W.h1) / W.h1 * 100 / W[f"MRIW1toW{w}"]).values)
real = dl_rates(3)
adni = pd.read_csv(a.adni, low_memory=False); adni = adni[adni.Hippocampus.notna() & (adni.DX_bl == "CN")].copy(); adni["date"] = pd.to_datetime(adni.EXAMDATE)
ag = adni.sort_values(["RID", "date"]).groupby("RID"); fa, la = ag.first(), ag.last(); yy = (la.date - fa.date).dt.days / 365.25; ok = yy >= 1.5
adni_r = band_rates(fa.AGE[ok].values, ((la.Hippocampus - fa.Hippocampus) / fa.Hippocampus * 100 / yy)[ok].values)
fmt = lambda v: " / ".join("n/a" if np.isnan(x) else f"{x:+.2f}" for x in v)
def trend(v): v = [x for x in v if not np.isnan(x)]; return v[-1] - v[0] if len(v) >= 2 else np.nan
t_s, t_r = trend(syn), trend(real)
add("종단 해마 위축률의 나이 의존성 (%/년, 40·50·60·70대)", fmt(syn), f"DLBS {fmt(real)}; ADNI CN(60·70대) {fmt(adni_r[2:])}",
    "결함" if (t_s * t_r < 0) else "양호", "실제는 나이가 들수록 위축이 가속(음수가 커짐)" if t_r < 0 else "")

# 5) 영역 간 상관 구조 (공통 피질하 영역)
pairs = {"hippocampus": ("BV_subctx_hippocampus", ["Hippocampus"]), "amygdala": ("BV_subctx_amygdala", ["Amygdala"]), "thalamus": ("BV_subctx_thalamus", ["ThalamusProper"]),
         "caudate": ("BV_subctx_caudate", ["Caudate"]), "putamen": ("BV_subctx_putamen", ["Putamen"]), "pallidum": ("BV_subctx_pallidum", ["Pallidum"]), "accumbens": ("BV_subctx_accumbens", ["Accumbens"])}
def resid_corr(Y, age_v, sex_v, icv_v):
    X1 = np.c_[np.ones(len(Y)), age_v - 55, (age_v - 55) ** 2, sex_v, icv_v]; R = Y - X1 @ np.linalg.lstsq(X1, Y, rcond=None)[0]; return np.corrcoef(R.T)
sY = np.column_stack([B[v[0]].values for v in pairs.values()])
dY = pd.DataFrame({"S": s_dl["S#"]});
for k, v in pairs.items(): dY[k] = sum(num(s_dl["Lh" + n + "Vol"]) + num(s_dl["Rh" + n + "Vol"]) for n in v[1]) / 1000
dY = dY.merge(D1[["S", "AgeMRI_W1", "sex", "icv"]], on="S").dropna()
Cs = resid_corr(sY, age, sx, icv); Cd = resid_corr(dY[list(pairs)].values, dY.AgeMRI_W1.values, dY.sex.values, dY.icv.values)
iu = np.triu_indices(len(pairs), 1); rho = spearmanr(Cs[iu], Cd[iu])[0]
add("피질하 영역 간 상관 구조 (나이·성별·ICV 보정 후 7개 영역 21쌍)", f"평균 |상관| {np.abs(Cs[iu]).mean():.2f}", f"평균 |상관| {np.abs(Cd[iu]).mean():.2f}; 두 구조의 순위상관 {rho:+.2f}", "양호" if rho > .6 else "주의",
    "공통 영역만 비교(분할 방식이 달라 절대값 비교는 제한적)")

# 6) 진단 라벨 결합 vs 측정 수치
import mri_screen as ms
Dd = ms.load_all(a.mri); bs = Dd["BASE"]; nm = bs[bs.lab == "normal"]; dis = bs[bs.lab == "disease"]
ref = ms.Ref().fit(nm.sample(frac=.5, random_state=0), 0); rest = nm.drop(nm.sample(frac=.5, random_state=0).index)
def atr(d): F = ref.transform(d); return -F[[c for c in F.columns if c.startswith("z_")]].mean(axis=1)
sbp_n = ms._clean(rest.T_SBP); okn = sbp_n.notna(); rho_n = spearmanr(atr(rest)[okn], sbp_n[okn])[0]
dn, dd_ = atr(rest), atr(dis); d_eff = (dd_.mean() - dn.mean()) / np.sqrt((dd_.var() + dn.var()) / 2)
add("MRI 위축 점수가 반영하는 것", f"진단군-정상군 차이 d={d_eff:.2f}; 정상군 내 수축기혈압과 상관 ρ={rho_n:+.2f}", "측정 혈압과 해마 위축이 연관", "결함" if (d_eff > .3 and abs(rho_n) < .1) else "양호", "진단 여부에는 반응하나 측정 수치에는 반응하지 않음")

# 7) 추적 탈락의 선택성
st = {}; ids1 = cl[1].index; idsL = cl[kl].index
flag1 = ((num(cl[1]["T01_HTN"]) == 2) | (num(cl[1]["T01_DM"]) == 2) | (num(cl[1]["T01_LIP"]) == 2))
stay_d = float(ids1[flag1.values].isin(idsL).mean()); stay_n = float(ids1[~flag1.values].isin(idsL).mean())
add(f"추적 탈락의 선택성 (1차→{kl}차 잔류율)", f"질환 응답군 {stay_d:.0%}, 비응답군 {stay_n:.0%}", "-", "주의" if abs(stay_n - stay_d) > .1 else "양호", "질환군이 더 많이 탈락하면 종단 결과가 편향될 수 있음")

# 8) 고령 표본
a1 = num(cl[kf][f"T{kf:02d}_AGE"]); aL = num(cl[kl][f"T{kl:02d}_AGE"])
add("70세 이상 표본", f"BASE {float((age >= 70).mean()):.1%}({int((age >= 70).sum())}명); 종단 시작 시점 {int((a1 >= 70).sum())}명/{len(a1)}명, 마지막 추적 시점 {int((aL >= 70).sum())}명/{len(aL)}명", f"DLBS 1차 MRI {int((dl_p.AgeMRI_W1 >= 70).sum())}명/{len(dl_p)}명", "주의" if (age >= 70).mean() < .05 else "양호", "나이대별 종단 위축률은 시작 시점 나이 기준이라 70대 구간이 비어 있음")

# 9) (선택) SNP 품질
if a.geno:
    z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr22.fam").decode().splitlines()]; N = len(fam)
    bim = pd.read_csv(__import__("io").BytesIO(z.read("synthetic_kchip_chr22.bim")), sep="\t", header=None, names=["chr", "snp", "cm", "pos", "a1", "a2"])
    raw = np.frombuffer(z.read("synthetic_kchip_chr22.bed"), dtype=np.uint8)[3:].reshape(len(bim), (N + 3) // 4)
    bits = np.stack([(raw >> (2 * k)) & 3 for k in range(4)], axis=2).reshape(len(bim), -1)[:, :N]
    n0, n1, n2 = (bits == 3).sum(1), (bits == 2).sum(1), (bits == 0).sum(1); nn = n0 + n1 + n2
    p = (2 * n2 + n1) / (2 * nn); exp0, exp1, exp2 = nn * (1 - p) ** 2, 2 * nn * p * (1 - p), nn * p ** 2
    chi = np.where(nn > 0, (n0 - exp0) ** 2 / np.maximum(exp0, 1e-9) + (n1 - exp1) ** 2 / np.maximum(exp1, 1e-9) + (n2 - exp2) ** 2 / np.maximum(exp2, 1e-9), 0)
    hwe_fail = float((chi > 23.9).mean()); maf = np.minimum(p, 1 - p)
    nrs = ntot = 0
    for ch in range(1, 23):
        nm_ = pd.read_csv(__import__("io").BytesIO(z.read(f"synthetic_kchip_chr{ch}.bim")), sep="	", header=None, usecols=[1])[1]; ntot += len(nm_); nrs += int((~nm_.str.startswith("bg_")).sum())
    rs = nrs / ntot
    add("SNP 품질 (chr22 표본)", f"HWE 위반(p<1e-6 근사) {hwe_fail:.2%}; MAF 중앙값 {np.median(maf):.2f}; rs 번호 SNP {nrs}개/{ntot:,}개({rs:.2%}, 전체 염색체)", "대부분 HWE 유지, 위반 1% 미만", "양호" if hwe_fail < .01 else "주의",
        "rs 번호가 붙은 소수 SNP에 질환 연관이 집중되는지는 별도 스캔 필요")

T = pd.DataFrame(rows); T.to_csv(a.out + "/audit_table.csv", index=False, encoding="utf-8-sig")
icon = {"양호": "양호", "주의": "주의", "결함": "**결함**"}
md = ["# 합성 코호트 충실도 점검표", "", f"입력: {a.mri}", "", "| 항목 | 합성자료 | 실제 자료 | 판정 |", "|---|---|---|---|"]
for r in rows: md.append(f"| {r['항목']} | {r['합성자료']} | {r['실제자료']} | {icon[r['판정']]}{' — ' + r['비고'] if r['비고'] else ''} |")
n_f = sum(r["판정"] == "결함" for r in rows); n_w = sum(r["판정"] == "주의" for r in rows)
md += ["", f"요약: 점검 {len(rows)}개 중 결함 {n_f}, 주의 {n_w}, 양호 {len(rows) - n_f - n_w}."]
open(a.out + "/충실도_점검표.md", "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))
