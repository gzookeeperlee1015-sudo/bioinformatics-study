# Bioinformatics Study

생물정보학 학습 과정에서 작성한 코드와 Jupyter Notebook을 기록하는 저장소입니다.

## Repository structure

- `notebooks/`: 단계별 실습 Notebook
- `results/`: 로컬 분석 결과(Git 추적 제외)
- `logs/`: 프로그램 실행 로그(Git 추적 제외)

## Data policy

FASTQ, BAM, VCF, reference genome 및 환자·샘플 수준 데이터는 이 저장소에 포함하지 않습니다. 실습 코드는 서버의 허가된 공용 데이터를 읽되, 데이터 자체는 GitHub에 업로드하지 않습니다.

## Learning log

### 01. FASTQ basics

- paired-end FASTQ의 R1/R2 경로 확인
- FASTQ 한 read의 4줄 구조 확인
- 염기서열 길이와 품질 문자열 길이 비교

Notebook: `notebooks/01_fastq_basic.ipynb`

## FastQC summary

- R1/R2: 1,449,575 reads each
- Read length: 76 bp
- Per-base quality: PASS
- Adapter content: PASS
- R1 GC/content bias: WARN
- R2 overall QC: PASS

---

## 해마 감소 예측 모델 (BAG 기준범위)

### BAG 기준범위 내 미래 해마 감소 예측

2026-10-06 정리 완료. [현재 모델 시작 화면](현재모델_보기.html)에서 설명·결과·사용법을 볼 수 있습니다.

| 폴더 | 내용 |
|---|---|
| `01_대회원자료` | 대회에서 제공한 전체 임상·MRI·유전체·코드북·공식문서 |
| `02_ADNI_원자료` | ADNIMERGE·TADPOLE 전체 원본 사본, QC·가공 입력표·자료정의 |
| `03_DLBS_원자료` | 기존/확장 제공본, 원자료 출처·메타정보·코드북 |
| `04_brain_imbalance` | 현재 해마 예측·BAG 선별 모델, 오류 보정 실험, 필수 의존 파일 |
| `00_정리기록` | 삭제 목록, 데이터 이관 경로·해시, 독립 검토와 실행 기록 |
| `.venv` | 모델을 실행하는 Python 환경 |

현재 주모델: `04_brain_imbalance/brain_blindspot_20261006`.
후속 오류 보정은 `brain_blindspot_error_upgrade_20261006`에 보존했습니다. 안정적인 추가 검출은 확인되지 않았습니다.
models/results/src의 부모 파일은 현재 모델의 실행·검증 의존성 때문에 유지했습니다.

첫 학습외 평가: BAG 기준범위575명 / 실제 큰 감소117명 / 선별118명 / 정탐61명.
이 범위와 큰 감소는 상대적인 연구 기준입니다. 실제자료 외부검증은 미완료입니다.

프로젝트 루트 PowerShell에서 예제 추론:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 .\04_brain_imbalance\brain_blindspot_20261006\blindspot.py predict --input .\04_brain_imbalance\brain_blindspot_20261006\models\example_input.csv --output .\04_brain_imbalance\brain_blindspot_20261006\models\my_predictions.csv
```

출력 파일은 새 이름을 사용합니다. 입력열은 models/input_contract.json에서 확인합니다.
정리 후 기존·후속 모델의 verify와 새 프로세스 예측, 각각2094개 OOF 재생을 통과했습니다.

기존 ADNI Downloads 원본은 그대로 뒀습니다. 프로젝트의02폴더에는 byte/hash가 같은 사본이 있습니다.
DLBS 코드북과 원자료는03폴더로 옮겼습니다. 예전 문서에 기록된 경로는00_정리기록/migrations.json에서 새 위치를 확인합니다.
취득목록은00_정리기록/원자료취득목록.csv입니다.

이전 미팅자료·노트북·모델 실험·SCODA·옛 발표PDF 등3808파일(총1,961,875,998bytes)을 삭제했습니다.
불필요한 PDF6개를 삭제하고 공식가이드·설명회·DLBS코드북3개를 유지했습니다.
대회 제공 파일·데이터 사본·모델의 보존해시는 독립 검토에서 확인했습니다.

---

#### GitHub 저장소 구성 안내

이 저장소에는 모델·코드·결과만 있습니다. 용량과 데이터 이용 조건 때문에 `01_대회원자료`, `02_ADNI_원자료`, `03_DLBS_원자료`, `00_정리기록`, `.venv`는 제외했습니다.
재현하려면 서버에 같은 이름의 원자료 폴더를 프로젝트 루트에 두고 `pip install -r requirements.txt`로 환경을 만듭니다.
`05_제출패키지_20261013/scripts` 일부 스크립트는 `C:/junny/bioinformaticsstudy/bioinfo대회/` 절대경로를 직접 씁니다. 리눅스 서버에서는 해당 경로를 서버 경로로 바꿔야 합니다.
