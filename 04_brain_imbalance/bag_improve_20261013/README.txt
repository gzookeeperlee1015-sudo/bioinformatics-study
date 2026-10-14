BAG 개선 비교 실험 (2026-10-13)
실행: <python> -X utf8 bag_improve.py <작업폴더(안에 mri/ 압축해제본)> <출력폴더>
 - mri/ 에는 뇌영상체적_합성데이터.zip 을 풀어 둔다. 프로젝트 .venv 는 원본 Python 이 삭제돼 실행 불가라
   별도 venv(numpy, pandas, scikit-learn 1.9.1, scipy)를 만들어 실행했다.
 - 정상/질환 규칙은 이 실험에서 임의로 정한 것이다(팀원의 1393명 기준 아님). 자기보고 HTN/DM/LIP 없음 + SBP<140, DBP<90,
   공복혈당<126, HbA1c<6.5, 총콜레스테롤<240 이면 정상. 자기보고 있으면 질환. 나머지는 모호군으로 제외.
 - FOLLOW 2~5차 질환 표시는 '신규 진단'이라 이전 차수와 누적했다.
 - 결과: bag_compare_summary.csv(5반복 평균/표준편차), bag_compare_raw.csv, bag_incident.json
