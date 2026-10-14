"""BASE 라벨 개체 전체(8,223명)에 대한 5겹 교차검증 + 100% 재학습 최종 모델.
매 폴드에서 정상 참조 구성, SNP 선택(전장 스캔), 점수 학습을 모두 폴드의 학습 개체에서만 한다.
교차검증 점수(OOF)로 평가하고, 나이대별 선별 기준도 OOF 점수에서 정해 학습과 분리한다.
최종 모델은 전체 라벨 개체로 다시 학습한다(multimodal.py predict 와 호환).
한계: 같은 BASE 자료를 나눈 내부 검증이다(외부 검증 아님). 후보 SNP 풀은 이전 스캔의 rs SNP 전부 + 약한 연관 배경 SNP."""
import argparse, glob, os, sys, zipfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
import mri_screen as ms, multimodal as mm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
for k in ["mri", "scan", "kept", "geno", "out"]: ap.add_argument("--" + k)
ap.add_argument("--reps", type=int, default=2)
a = ap.parse_args()
D = ms.load_all(a.mri); b = D["BASE"]
R = pd.read_csv(a.scan); K = np.load(a.kept, allow_pickle=True); names = list(K["names"]); X = K["X"]; idx = {n: i for i, n in enumerate(names)}
meta = R.set_index("snp").loc[names]; chrv, posv = meta.chr.values, meta.pos.values
z = zipfile.ZipFile(glob.glob(a.geno)[0]); fam = [l.split()[0] for l in z.read("synthetic_kchip_chr1.fam").decode().splitlines()]; N = len(fam)
lab = b.reindex(pd.Index(fam)); idpos = {n: i for i, n in enumerate(fam)}
HEADS = ["any", "htn", "dm", "lip"]; HN = {"any": "전체 질환", "htn": "고혈압", "dm": "당뇨", "lip": "고지혈"}
P = {"any": (lab.lab == "disease").values, "htn": (lab.htn == 1).values, "dm": (lab.dm == 1).values, "lip": (lab.lip == 1).values}
ISN = (lab.lab == "normal").values; LAB = lab.lab.isin(["normal", "disease"]).values; ALL = np.where(LAB)[0]

def scan(mask, y, chunk=6000):
    out = np.empty(X.shape[0], np.float32); ym = y - y.mean(); sy = np.sqrt((ym ** 2).sum())
    for s in range(0, X.shape[0], chunk):
        Xm = X[s:s + chunk][:, mask]; Xc = Xm - Xm.mean(1, keepdims=True); out[s:s + chunk] = (Xc @ ym) / (np.sqrt((Xc ** 2).sum(1)) * sy + 1e-12)
    return out
def clump(r, K_=15, win=1_000_000):
    ch = []
    for i in np.argsort(-np.abs(r)):
        if len(ch) >= K_: break
        if all(not (chrv[i] == chrv[j] and abs(posv[i] - posv[j]) < win) for j in ch): ch.append(i)
    return ch

def train_models(train_idx, seed, combos):
    """학습 개체에서 참조 구성 + SNP 선택 + 점수 학습. 반환: ref, leads(인덱스), models"""
    trd = lab.iloc[train_idx]; rn, rest = ms.split_ref(trd, seed); ref = ms.Ref().fit(rn, seed); rest_pos = np.array([idpos[n] for n in rest.index])
    leads, models = {}, {}
    for h in HEADS:
        trm = np.zeros(N, bool); trm[train_idx] = True; trm &= (ISN | P[h]); leads[h] = clump(scan(trm, P[h][trm].astype(float)))
        pos_tr = np.array([i for i in train_idx if P[h][i]]); ctr = np.concatenate([rest_pos, pos_tr]); yc = np.r_[np.zeros(len(rest_pos)), np.ones(len(pos_tr))].astype(int)
        snp = X[leads[h]][:, ctr].T
        for cn in combos: models[(h, cn)] = mm._lr().fit(mm.blocks(ref, lab.iloc[ctr], snp, mm.COMBOS[cn])[0], yc)
    return ref, leads, models

def predict(ref, leads, models, rows, h, cn):
    return models[(h, cn)].predict_proba(mm.blocks(ref, lab.iloc[rows], X[leads[h]][:, rows].T, mm.COMBOS[cn])[0])[:, 1]

if __name__ == "__main__":
    t0 = time.time(); combos = list(mm.COMBOS); y_any = P["any"][ALL].astype(int)
    oof = {r: {(h, cn): np.full(N, np.nan) for h in HEADS for cn in combos} for r in range(a.reps)}
    FS = [1, .75, .5, .35, .25, .15, .1, .05, 0]; SHR = ("+SNP", "+MRI+SNP", "전부")
    shr = {(r, h, cn, f): np.full(N, np.nan) for r in range(a.reps) for h in HEADS for cn in SHR for f in FS}
    AMB = np.where((lab.lab == "ambiguous").values)[0]; amb = {(h, cn): np.zeros(len(AMB)) for h in HEADS for cn in ("나이·성별", "전부")}; amb_n = 0
    for r in range(a.reps):
        for f, (ta, tb) in enumerate(StratifiedKFold(5, shuffle=True, random_state=100 + r).split(ALL, y_any)):
            tri, vai = ALL[ta], ALL[tb]; ref, leads, models = train_models(tri, 10 * r + f, combos)
            for h in HEADS:
                vp = np.array([i for i in vai if (ISN[i] or P[h][i])])
                for cn in combos: oof[r][(h, cn)][vp] = predict(ref, leads, models, vp, h, cn)
                for cn in SHR:   # SNP 효과를 f배로 줄인 로짓(진단 민감도 분석)
                    Xa = mm.blocks(ref, lab.iloc[vp], X[leads[h]][:, vp].T, mm.COMBOS[cn])[0]; m_ = models[(h, cn)]
                    Xs = m_.steps[1][1].transform(m_.steps[0][1].transform(Xa)); contrib = Xs * m_.steps[2][1].coef_[0]; ns = len(leads[h]); n0 = Xa.shape[1]
                    for f_ in FS:
                        c2 = contrib.copy(); c2[:, n0 - ns:n0] *= f_; shr[(r, h, cn, f_)][vp] = c2.sum(1) + m_.steps[2][1].intercept_[0]
                for cn in ("나이·성별", "전부"): amb[(h, cn)] += predict(ref, leads, models, AMB, h, cn)
            amb_n += 1
            print(f"반복 {r} 폴드 {f} 완료 ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for h in HEADS:
        mk = ISN | P[h]; y = P[h][mk].astype(int); age = lab.age.values[mk]
        for cn in combos:
            aucs, auc_ages = [], []
            for r in range(a.reps):
                s = oof[r][(h, cn)][mk]; aucs.append(roc_auc_score(y, s)); auc_ages.append(ms.strat_auc(s, y, age))
            sm = np.mean([oof[r][(h, cn)][mk] for r in range(a.reps)], axis=0); rng = np.random.RandomState(0); bs = [roc_auc_score(y[i], sm[i]) for i in (rng.choice(len(y), len(y)) for _ in range(300))]
            rows.append(dict(target=HN[h], model=cn, n=len(y), pos=int(y.sum()), auc=np.mean(aucs), auc_sd_reps=np.std(aucs), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), auc_age=np.mean(auc_ages)))
    C = pd.DataFrame(rows); C.to_csv(a.out + "/cv_full_results.csv", index=False)
    sv = {f"oof|{r}|{h}|{cn}": oof[r][(h, cn)] for r in range(a.reps) for h in HEADS for cn in combos}
    sv.update({f"shr|{r}|{h}|{cn}|{f}": v for (r, h, cn, f), v in shr.items()}); sv.update({f"amb|{h}|{cn}": v / amb_n for (h, cn), v in amb.items()}); sv["amb_idx"] = AMB
    np.savez_compressed(a.out + "/cv_oof.npz", **sv)
    # OOF 점수에서 선별 기준(정상군 90백분위, 나이대별) 결정
    thr, thr_age = {}, {}
    for h in HEADS:
        for cn in combos:
            nm_ = ISN; sc = np.concatenate([oof[r][(h, cn)][nm_] for r in range(a.reps)]); ag = np.concatenate([lab.age.values[nm_]] * a.reps)
            thr[(h, cn)] = float(np.quantile(sc, .90)); bd = np.digitize(ag, [50, 60]); thr_age[(h, cn)] = [float(np.quantile(sc[bd == k], .90)) for k in range(3)]
    # 최종 모델: 전체 라벨 개체로 재학습
    ref, leads, models = train_models(ALL, 999, combos)
    M = dict(ref=ref, models=models, leads={h: [names[i] for i in leads[h]] for h in HEADS}, thr=thr, thr_age=thr_age, combos=mm.COMBOS, Q=mm.Q)
    joblib.dump(M, a.out + "/multimodal_full.joblib")
    pd.set_option("display.width", 220); print(C.pivot(index="model", columns="target", values="auc").round(3).loc[combos, [HN[h] for h in HEADS]].to_string())
    print(f"\n총 {time.time() - t0:.0f}s | 최종 모델 저장")
