# LSTM을 이용한 기사 성향 분석 인공지능 만들기

## 데이터 전처리
1. 라이브러리 불러오기

<img width="565" alt="import pandas as pd" src="https://user-images.githubusercontent.com/63358871/123212891-5522ea80-d500-11eb-9fc8-b1fd0fe42bbb.png">


2. 데이터 불러오기

<img width="997" alt="Pasted Graphic 38" src="https://user-images.githubusercontent.com/63358871/123212910-59e79e80-d500-11eb-8348-75cb2ebc3b70.png">


3. 학습 데이터와 테스트 데이터 생성

<img width="560" alt="def train_val_split(df)" src="https://user-images.githubusercontent.com/63358871/123212920-5ce28f00-d500-11eb-9a4a-6014b86855d5.png">


4. 기사 진영에 따라 0 또는 1 부여

<img width="1023" alt="Pasted Graphic 40" src="https://user-images.githubusercontent.com/63358871/123212939-64a23380-d500-11eb-835c-50378dbc515a.png">


0과 1의 계산 범위를 맞추기 위해서
학습 데이터와 테스트 데이터에서 진보 또는 중도 진영일 경우 1, 보수진영일 경우 0으로 ‘press’의 값들을 변경한다.


5. 0과 1의 비율


<img width="983" alt="Pasted Graphic 41" src="https://user-images.githubusercontent.com/63358871/123212997-71bf2280-d500-11eb-9948-f933adf1f916.png">


![알 수 없음](https://user-images.githubusercontent.com/63358871/123213009-74217c80-d500-11eb-9ca9-0deb13d1ddff.png)

<img width="998" alt="Pasted Graphic 42" src="https://user-images.githubusercontent.com/63358871/123213021-771c6d00-d500-11eb-9e48-ae66e8dfa33a.png">

![알 수 없음](https://user-images.githubusercontent.com/63358871/123213024-797ec700-d500-11eb-92a7-d82874437213.png)



6. 정규표현식 함수

<img width="690" alt="Pasted Graphic 43" src="https://user-images.githubusercontent.com/63358871/123213034-7c79b780-d500-11eb-8c5d-157dab185fe1.png">


기사제목에서 불필요한 텍스트를 제거하기 위한 정규표현식 함수를 만든다.

7. 정규표현식 적용 & 명사단위로 나누기

<img width="675" alt="Pasted Graphic 44" src="https://user-images.githubusercontent.com/63358871/123213068-869bb600-d500-11eb-9238-85e3bb2d9ba5.png">


전에 만든 정규표현식으로 불필요한 텍스트를 걸러내고, Konlpy의 Okt 함수를 이용해서 명사 형태소 단위로 문장을 나눈 뒤 저장한다.

<img width="1014" alt="Pasted Graphic 45" src="https://user-images.githubusercontent.com/63358871/123213055-84395c00-d500-11eb-939a-c06a11823c8e.png">


8. 단어에 번호 부여

<img width="1010" alt="Pasted Graphic 47" src="https://user-images.githubusercontent.com/63358871/123213086-8b606a00-d500-11eb-9020-6621a6f65f04.png">


tokenizer을 이용하면 명사형태소 단위로 나눈 문자열에 고유한 번호를 부여할 수 있다.

<img width="947" alt="Pasted Graphic 48" src="https://user-images.githubusercontent.com/63358871/123213103-91564b00-d500-11eb-82c9-7705e5cfcbb8.png">
<img width="1022" alt="Pasted Graphic 49" src="https://user-images.githubusercontent.com/63358871/123213109-93b8a500-d500-11eb-80b7-930fc614e694.png">



이걸 이용해서 ‘X_train’과 ‘X_test’의 문자열을 단어의 번호로 이루어진 리스트로 변환한다.

9. ‘y_train’과 ‘y_test’에는 정답이 적힌 ‘press’의 값들을 넣는다.

<img width="619" alt="y train" src="https://user-images.githubusercontent.com/63358871/123213119-974c2c00-d500-11eb-909f-821ad5dab72e.png">


10. 패딩

<img width="942" alt="Pasted Graphic 52" src="https://user-images.githubusercontent.com/63358871/123213126-99ae8600-d500-11eb-963e-ed0b006d2e70.png">
<img width="540" alt="167, 168 ), dtype=int32)" src="https://user-images.githubusercontent.com/63358871/123213131-9ca97680-d500-11eb-9fd8-9d6ac0303f74.png">


자연어처리를 위해 LSTM모델을 사용할 예정인데, 이때 조건으로 input이 일정한 크기를 가져야 한다. 하지만 기사 제목의 명사의 개수는 제각각이기에 통일시키는 작업이 필요하다. 따라서 padding을 이용해 지정값에 초과하는 길이는 잘라내고, 부족한 부분은 0으로 메꾸는 작업을 한다.

## 딥러닝 모델
1. 라이브러리 불러오기

<img width="605" alt="tensorflow" src="https://user-images.githubusercontent.com/63358871/123213246-bc409f00-d500-11eb-8dee-3134dfcaa535.png">


2. 모델 구성하기

<img width="963" alt="Pasted Graphic 56" src="https://user-images.githubusercontent.com/63358871/123213256-bf3b8f80-d500-11eb-9d65-a8eb52a7aae3.png">


Embedding을 통해 input 값을 지정하고, LSTM레이어와 Dense(Fully-Connected)레이어를 생성합니다. 이때 노드의 개수와 레이어의 개수, 활성함수의 종료에 따라
총 9가지의 모델을 비교한다.
그리고 가장 좋은 성능을 가진 모델을 채택한다.

<img width="899" alt="Pasted Graphic 59" src="https://user-images.githubusercontent.com/63358871/123213264-c2cf1680-d500-11eb-8589-e4b10876f3fa.png">
<img width="857" alt="Pasted Graphic 58" src="https://user-images.githubusercontent.com/63358871/123213271-c5317080-d500-11eb-9d59-8c3fb4091cb8.png">



optimizer는 adam, 손실함수는 binary_crossentropy를 사용한다.
epoch는 각각의 모델마다 과잉적합이 일어나지 않을 정도로 조정하였고, batch_size는 10으로 정하였다.
validation_split을 통해 학습데이터 중 20%를 검증데이터로 활용하여 정확도를 측정한다.
keras의 callbakcs와 ModelCheckpoint를 이용해서, 검증데이터의 정확도가 높아질 경우에만 모델을 저장하는 방식으로 설정하였다.

모델학습 출력 예시 :

<img width="1100" alt="Pasted Graphic 62" src="https://user-images.githubusercontent.com/63358871/123213282-c8c4f780-d500-11eb-976b-f3b46a18d62a.png">



3. 모델들
1번 모델 : LSTM(32) - Dense(16, sigmoid) - Dense(1, sigmoid)
<img width="963" alt="Pasted Graphic 60" src="https://user-images.githubusercontent.com/63358871/123213304-cf536f00-d500-11eb-926b-2a8889a3985b.png">
<img width="729" alt="Pasted Graphic 63" src="https://user-images.githubusercontent.com/63358871/123213311-d24e5f80-d500-11eb-96fc-fb49330af6f4.png">

테스트 데이터로 모델 평가 결과 58퍼센트의 정확도를 보였다.




2번 모델 : LSTM(32) - Dense(1, sigmoid)

<img width="914" alt="Pasted Graphic 64" src="https://user-images.githubusercontent.com/63358871/123213323-d5e1e680-d500-11eb-9939-ec7d5d117efc.png">
<img width="727" alt="Pasted Graphic 66" src="https://user-images.githubusercontent.com/63358871/123213328-d8dcd700-d500-11eb-90bd-ff0ce7a29d48.png">



테스트 데이터로 모델 평가 결과 56퍼센트의 정확도를 보였다.

3번 모델 : LSTM(32) - Dense(1, relu)

<img width="922" alt="Pasted Graphic 67" src="https://user-images.githubusercontent.com/63358871/123213402-edb96a80-d500-11eb-9170-be59bcd26507.png">
<img width="728" alt="(1 3635327816009521, 0 3529411852359772" src="https://user-images.githubusercontent.com/63358871/123213411-ef832e00-d500-11eb-8fc0-dabebb5c4d68.png">


활성함수를 sigmoid에서 relu로 변경해보았다.
테스트 데이터로 모델 평가 결과 35퍼센트의 정확도를 보였다.

4번 모델 : LSTM(64) - Dense(16, sigmoid) - Dense(8, sigmoid) - Dense(1, sigmoid)

<img width="932" alt="Pasted Graphic 69" src="https://user-images.githubusercontent.com/63358871/123213419-f316b500-d500-11eb-944d-96ced06dfba2.png">
<img width="731" alt="0 6941 - accuracy 0 6000" src="https://user-images.githubusercontent.com/63358871/123213427-f5790f00-d500-11eb-95aa-612d828d569b.png">


LSTM레이어의 노드를 64개로 증가시키고, sigmoid 레이어를 활용한 은닉층 2개를 생성했다.
테스트 데이터로 모델 평가 결과 60퍼센트의 정확도를 보였다.

5번 모델 : LSTM(64) - Dense(32, relu) - Dense(16, relu) - Dense(8, relu) - Dense(1, relu)

<img width="916" alt="Pasted Graphic 71" src="https://user-images.githubusercontent.com/63358871/123213444-fa3dc300-d500-11eb-802f-62cfcdfba342.png">
<img width="721" alt="Pasted Graphic 72" src="https://user-images.githubusercontent.com/63358871/123213458-fca01d00-d500-11eb-9cc9-ced9d328f678.png">


LSTM레이어의 노드를 64개로 증가시키고, relu 레이어를 활용한 은닉층 3개를 생성했다.
테스트 데이터로 모델 평가 결과 41퍼센트의 정확도를 보였다.

6번 모델 : LSTM(64) - Dense(16, sigmoid) - Dense(8, relu) - Dense(1, relu)

<img width="917" alt="Pasted Graphic 73" src="https://user-images.githubusercontent.com/63358871/123213471-00cc3a80-d501-11eb-89ba-a92c46467b5e.png">
<img width="719" alt="Pasted Graphic 74" src="https://user-images.githubusercontent.com/63358871/123213480-03c72b00-d501-11eb-8948-046710fa125d.png">


sigmoid 레이어와 relu 레이어를 각각 은닉층으로 1개 씩 생성했다.
출력층은 relu 함수로 지정하였다.
테스트 데이터로 모델 평가 결과 38퍼센트의 정확도를 보였다.

7번 모델 : LSTM(64) - Dense(16, sigmoid) - Dense(8, relu) - Dense(1, sigmoid)

<img width="923" alt="Pasted Graphic 75" src="https://user-images.githubusercontent.com/63358871/123213498-088bdf00-d501-11eb-816c-f04b5ed73b91.png">
<img width="720" alt="Pasted Graphic 76" src="https://user-images.githubusercontent.com/63358871/123213515-0b86cf80-d501-11eb-8bab-35d39835e24d.png">


sigmoid 레이어와 relu 레이어를 각각 은닉층으로 1개 씩 생성했다.
출력층은 sigmoid 함수로 지정하였다.
테스트 데이터로 모델 평가 결과 58퍼센트의 정확도를 보였다.

8번 모델 : LSTM(32) - Dense(16, sigmoid) - Dense(8, relu) - Dense(1, relu)

<img width="895" alt="Pasted Graphic 79" src="https://user-images.githubusercontent.com/63358871/123213523-0e81c000-d501-11eb-9adb-5ebdc3e9733a.png">
<img width="720" alt="Pasted Graphic 80" src="https://user-images.githubusercontent.com/63358871/123213534-104b8380-d501-11eb-81a6-03b039d67fe6.png">


6번모델에서 LSTM 노드를 32개로 줄였다.
테스트 데이터로 모델 평가 결과 58퍼센트의 정확도를 보였다.

7번모델과 정확도는 같지만, 손실률은 적어, 더 좋은 모델이라고 평가할 수 있다.

4. 모델평가 결과


<img width="731" alt="0 6941 - accuracy 0 6000" src="https://user-images.githubusercontent.com/63358871/123213548-15103780-d501-11eb-9ba0-7d6ccd65865c.png">


 LSTM(64) - Dense(16, sigmoid) - Dense(8, sigmoid) - Dense(1, sigmoid)로 구성된 4번 모델이 60%의 정확도를 지닌 가장 성능이 좋은 모델이다.

5. 모델 사용하기

<img width="616" alt="Pasted Graphic 81" src="https://user-images.githubusercontent.com/63358871/123213594-21949000-d501-11eb-8955-d794dc7ea84e.png">


제작한 모델을 통해, 실제 기사 제목을 입력하면 어떤 성향의 뉴스인지를 예측하는 함수를 만든다.

<img width="549" alt="Pasted Graphic 82" src="https://user-images.githubusercontent.com/63358871/123213603-248f8080-d501-11eb-9bc7-9f62fca175b5.png">


출력 결과이다.



——

처음으로 딥러닝 인공지능 프로그램을 구성해봤습니다. 아마 데이터 수집이 더 오랫동안 이루어진다면, 보다 높은 정확도를 가진 학습 인공지능 모델을 만들어낼 수 있을 것으로 기대가 됩니다.
