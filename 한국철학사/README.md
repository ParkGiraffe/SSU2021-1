# SSU_2021-1_KoreanPhilosophyEL_final
한국철학사 EngagedLearning 기말프로젝트 



1. 크롤링 수집기간

2021.6.14 13:59 - 2021.6.16.19:59 | 30분 단위로 수집.


2. 수집한 데이터 파일

￼<img width="917" alt="Pasted Graphic 2" src="https://user-images.githubusercontent.com/63358871/122339566-39599a80-cf7c-11eb-984a-33d9d989e8f4.png">





# 데이터 시각화 : 진영 간의 기사 노출 빈도수 차이

3. 데이터 중에서 언론사와 등장 빈도수를 추출하여 세로운 데이터셋 제작


￼![알 수 없음](https://user-images.githubusercontent.com/63358871/122339643-4b3b3d80-cf7c-11eb-9a29-262e0a6fad30.png)



4. 워드클라우드 제작


￼![알 수 없음](https://user-images.githubusercontent.com/63358871/122339629-46768980-cf7c-11eb-894d-1b092ac2939f.png)



5. 보수와 중도와 진보 나누기.


￼![알 수 없음](https://user-images.githubusercontent.com/63358871/122339665-53937880-cf7c-11eb-889e-fbcb1e2cf827.png)



구분 기준 : 4년 간 각종 언론사를 들여다 본 개인적 소견 + MBC ‘스트레이트’에서 지정한 보수, 진보, 중도 언론 구분 + 각종 연구자료 참고

진보와 중도 : ‘경향신문', '한겨레', '프레시안', '오마이뉴스', '뉴스1', '머니투데이', '이데일리', '미디어오늘', 'YTN',  'MBC', '뉴시스', '서울신문', '지디넷코리아', '내일신문'
나머지는 보수



6. 트리맵 이란?

어떠한 값으 크기에 따라 커다란 사각형으로 표시됨.
각 그룹간의 크기, 비율을 한 눈에 확인할 수 있어 유용한 데이터 시각화 툴 중 하나로 뽑힘.

위의 사진은 미국 오바마 정부의 2016년도 예산을 시각적으로 분류한 것이다.

7. 제작한 트리맵

￼![알 수 없음](https://user-images.githubusercontent.com/63358871/122339682-59895980-cf7c-11eb-9cb8-0b4681348531.png)





# 머신러닝 인공지능 : 기사 제목을 통해 언론사의 정치 진영 예측

1. 다시 수집 데이터 불러오기


2. 기사제목과 언론사가 포함된 데이터셋으로 정제


￼<img width="510" alt="428 rows x 2 columns" src="https://user-images.githubusercontent.com/63358871/122339770-6f971a00-cf7c-11eb-8171-4da6c89de26e.png">



3. 한국어를 자동으로 형태소 단위로 분리해주는 오픈소스 활용


￼<img width="987" alt="Pasted Graphic 7" src="https://user-images.githubusercontent.com/63358871/122339786-745bce00-cf7c-11eb-8f2a-f593c416ce9e.png">



4. 수집된 기사 제목에 등장한 모든 단어들에 고유한 벡터값을 부여


￼<img width="186" alt="0 0 011" src="https://user-images.githubusercontent.com/63358871/122339807-79b91880-cf7c-11eb-9689-3166de5602ca.png">


총 972개의 단어가 등장함. (중복된 단어는 한 개로 통일) 
이 모든 단어에 벡터값을 넣을 수 있는 자리를 만듦.
여기서 행은 기사의 제목이고, 열은 단어이다. 
만약 1행에서 1번 단어가 포함되면, 첫번째 칸에 어떠한 값이 들어갈 것이다.

(예시)


￼<img width="650" alt="Pasted Graphic 11" src="https://user-images.githubusercontent.com/63358871/122339840-82a9ea00-cf7c-11eb-95ba-ae884a2f5159.png">

해설 : 첫번째 기사제목에서 n번째 단어가 적혀있다. > 0.3580333의 벡터값 부여.

5. 만약 언론사의 진영이 진보나 중도이면 1, 보수이면 0을 부여


6. 학습데이터와 테스트데이터를 나눔


￼<img width="780" alt="Pasted Graphic 13" src="https://user-images.githubusercontent.com/63358871/122339882-8f2e4280-cf7c-11eb-8e0e-be28e6218f09.png">


7. 학습데이터와 테스트데이터란?

￼<img width="582" alt="Pasted Graphic 15" src="https://user-images.githubusercontent.com/63358871/122339874-8b9abb80-cf7c-11eb-8068-b8d574546791.png">

학습데이터 : 문제와 정답을 동시에 줌.
인공지능은 학습데이터를 공부해서 정답을 맞출 수 있는 그래프를 그림.
테스트데이터 : 테스트데이터의 문제를 그래프에 집어넣고, 그래프를 이용해서 추려낸 답이 실제 테스트데이터 안에 적힌 정답과 일치한 지 확인.

정확도가 높을수록 좋은 인공지능이다. (정확도 = 정답을 맞춘 데이터 수 / 총 테스트 데이터 수)

8. 학습 및 테스트 결과

￼<img width="149" alt="accuracy 0 78" src="https://user-images.githubusercontent.com/63358871/122339900-935a6000-cf7c-11eb-85c0-d2c179d43210.png">


Accuracy (정확도) : 전체 분류 중에 얼마나 제대로 분류했어?
Precision (정밀도) : 모델이 True로 분류한 것 중에, 실제로 True인건 얼마야?
Recall (재현율) : 실제로 True인 것 중에, 모델이 True로 분류한 건 얼마야?
F1-score : 재현율과 정밀도의 조화평균은?   

=> 높을 수록 좋다.
=> 12시간정도 크롤링한 데이터로 시험을 해봤더니 정확도가 0.1이 나왔다. 즉, 데이터 수가 많을 수록 학습량이 늘어나고 정확도가 높아진다.


9. 학습 모델의 오차행렬


￼![Confusion Matrix](https://user-images.githubusercontent.com/63358871/122339912-96ede700-cf7c-11eb-82f4-c63b7e61a151.png)


그래프 해설 : 
0은 보수, 1은 중도와 진보

0으로 예측했을 때 실제로 0인 경우가 52.
1으로  예측했을 때 실제로 1인 경우가 49.

진한 색이 많으면 정답률(정확도)이 높다.
연한 색이 많으면 오답률이 높다.

10. 각 단어의 정치 진영 분포 정도

￼![알 수 없음](https://user-images.githubusercontent.com/63358871/122339923-9a816e00-cf7c-11eb-94ab-e060bc382c23.png)

972개의 단어중, 위를 향하면 진보 및 중도, 아래를 향하면 보수 키워드이다.
막대가 없는 단어는 진영에 상관없이 쓰이는 키워드.



11. 각 진영의 키워드로 워드 클라우드 제작
중도 및 진보진영


￼![알 수 없음](https://user-images.githubusercontent.com/63358871/122339964-a79e5d00-cf7c-11eb-9b0e-72959a8140cb.png)

보수진영


￼![알 수 없음](https://user-images.githubusercontent.com/63358871/122339977-ab31e400-cf7c-11eb-80de-20d98311e973.png)


12. 각 진영의 키워드로 트리맵 제작
중도 및 진보진영


￼![알 수 없음](https://user-images.githubusercontent.com/63358871/122339984-ae2cd480-cf7c-11eb-9e87-4977f5ef7f73.png)

보수진영


￼![알 수 없음](https://user-images.githubusercontent.com/63358871/122339994-b1c05b80-cf7c-11eb-8e84-1dbaa74ce3f8.png)


13. 연구 의의
- 데이터 시각화를 통해, 직관적으로 정보를 제공하는 경험
- 데이터와 머신러닝을 이용하여, 어떠한 입력값을 넣었을 때 그것이 어떤 분류에 속하는 지 예측하는 인공지능을 제작하는 경험
-  부적절한 알고리즘으로 이루어진 프로그램을 저격하는 프로그래밍을 통해, IT윤리의식을 높이는 데에 기여

14. 연구 한계
- 진보, 중도, 보수를 나누는 기준이 애매모호하다.
- 더 오랫동안 크롤링을 하지 못하였다.
- 협업을 하지 못함.

15. 향후 연구 과제

- 딥러닝 기술을 활용하여, 위와 같은 예측프로그램을 제작해본다.
- 딥러닝 기술을 활용하여, 유사한 키워드를 많이 쓰는 언론사끼리 분류를 하는 인공지능을 개발한다.
- 이렇게 하면, 진보와 보수 진영을 나누는 기준의 정확성이 매우 향상될 것으로 기대가 된다.
