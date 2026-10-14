MRI 단독 대사질환 선별 모델 (2026-10-13)
목적: 임상 검사값 없이 뇌 MRI 체적표만으로 고혈압·당뇨·고지혈 보유 가능성이 높은 사람을 선별하고, 정상 참조 대비 영역 편차를 보여준다.

구성: 정상군 참조(뇌연령 3모델 앙상블+나이편향 보정, 영역별 정상 기댓값) -> 영역 편차 z + 상한 클리핑 변수 지시자 -> 규제 로지스틱 선별 점수.
참조를 만든 사람과 선별 점수를 학습한 사람은 분리했다(참조 정상군 60% / 점수 학습 정상군 40% + 질환군).
정상/질환 정의는 이 연구에서 정한 규칙이다(mri_screen.py 상단). 2~5차 질환 표시는 신규 진단이라 이전 차수와 누적했다.

실행(프로젝트 .venv 는 원본 Python 삭제로 동작하지 않아 별도 환경 사용: numpy, pandas, scikit-learn 1.9.1, scipy, joblib):
  python mri_screen.py cv      --mri <BrainVol csv 폴더>            # BASE 개발 교차검증
  python mri_screen.py fit     --mri <BrainVol csv 폴더> --out models  # 학습 + FOLLOW 1·5차 동결 평가
  python mri_screen.py predict --model models/model.joblib --input models/example_input.csv --output my_out.csv

출력 열: bag_years, bag_percentile_in_age, top3_deviant_regions(z), score_any_disease/htn/dm/lip(순위용, 확률 아님), screen_flag(spec90).
입력 형식: models/input_contract.json. 예제: models/example_input.csv, example_output.csv.
