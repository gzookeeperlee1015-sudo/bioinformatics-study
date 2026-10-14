"""매개 검정: 유전 위험 점수 -> MRI 위축이 '진단 여부'를 거쳐서만 나타나는가.
verify.py 의 자료 준비부를 재사용한다(학습 70%로 SNP 선택, 나머지 30%에서 계수 추정)."""
src = open('verify.py', encoding='utf-8').read().split("print('\\n=== A.")[0]
exec(src)
rng = np.random.RandomState(0); ids = np.where(lab_ok)[0]; rng.shuffle(ids); tr = np.sort(ids[:int(.7 * len(ids))])
te = np.ones(N, bool); te[tr] = False; te &= lab_ok
trd = lab.iloc[tr]; rn, rest = ms.split_ref(trd, 0); ref = ms.Ref().fit(rn, 0)
F = ref.transform(lab); zc = F[[c for c in F.columns if c.startswith('z_')]]; atr = -zc.mean(axis=1).values
age = lab.age.values; sex = lab.sex.values

def fit(yv, cols, mask):
    A = np.c_[np.ones(mask.sum()), *[c[mask] for c in cols]]; b_ = np.linalg.lstsq(A, yv[mask], rcond=None)[0]; e = yv[mask] - A @ b_
    XtXi = np.linalg.inv(A.T @ A); se = np.sqrt(np.diag(XtXi @ (A.T @ (A * (e ** 2)[:, None])) @ XtXi)); return b_, se

for trait, name in [('htn', '고혈압'), ('dm', '당뇨'), ('lip', '고지혈')]:
    pos = (lab[trait] == 1).values; mem = isn | pos
    trm = np.zeros(N, bool); trm[tr] = True; trm &= mem
    r = scan(trm, pos[trm].astype(float)); L = leads_from(r)
    m = lrm().fit(np.c_[X[L][:, trm].T], pos[trm].astype(int)); sc = m.decision_function(np.c_[X[L].T])
    sel = te & mem
    s = (sc - sc[sel].mean()) / sc[sel].std(); a = (atr - atr[sel].mean()) / atr[sel].std(); y = pos.astype(float)
    b1, se1 = fit(a, [s, age, sex], sel)            # 라벨 보정 전
    b2, se2 = fit(a, [s, y, age, sex], sel)         # 라벨(진단 여부) 보정 후
    out = []
    for g, mk in [('질환자 안', pos & sel), ('정상 안', isn & sel)]:
        bb, se = fit(a, [s, age, sex], mk); out.append(f'{g} {bb[1]:+.3f}±{1.96 * se[1]:.3f}')
    print(f'[{name}] SNP점수→MRI위축(표준화 계수): 진단 보정 전 {b1[1]:+.3f}±{1.96 * se1[1]:.3f} | 진단 보정 후 {b2[1]:+.3f}±{1.96 * se2[1]:.3f} (진단 계수 {b2[2]:+.3f}) | ' + ' | '.join(out))
