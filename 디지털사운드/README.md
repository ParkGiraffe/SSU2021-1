
##디지털사운드 ANN과 CNN을 활용하여 음성 분류 인공지능 제작

## ANN (Artificial Neural Network)
1. 라이브러리 불러오기

<img width="307" alt="import librosa" src="https://user-images.githubusercontent.com/63358871/123209980-594d0900-d4fc-11eb-9381-59db138d9463.png">


librosa : 오디오 분석 라이브러리.
numpy : ndarray를 제공해주는 라이브러리.
pandas : 판다스 dataframe을 제공해주는 라이브러리.
matplotlib : 주어진 값에 맞추어서 그래프를 그려주는 라이브러리.

입니다.

2. 오디오 파일의 분류 데이터를 불러옵니다

<img width="659" alt="Pasted Graphic 2" src="https://user-images.githubusercontent.com/63358871/123210050-6cf86f80-d4fc-11eb-9e98-5f6f2cc19ed4.png">

<img width="201" alt="drilling" src="https://user-images.githubusercontent.com/63358871/123210066-71248d00-d4fc-11eb-9da1-56fd9cddf5e1.png">


여기에서 train.csv에는 총 5435개의 행이 있을 확인할 수 있습니다.

3. 동일한 class끼리 묶어서 고유한 숫자를 부여합니다.

<img width="529" alt="Pasted Graphic 4" src="https://user-images.githubusercontent.com/63358871/123210092-7aadf500-d4fc-11eb-8db1-dfa90dc51cda.png">

<img width="345" alt="Class numeric class" src="https://user-images.githubusercontent.com/63358871/123210100-7da8e580-d4fc-11eb-9961-4683ba5ed7d2.png">


‘cat.codes’함수를 넣어서 글자를 그대로 복사하는 것이 아니라 고유한 숫자를 부여하도록 합니다. 

4. train dataset과 test dataset을 분류

<img width="595" alt="Pasted Graphic 6" src="https://user-images.githubusercontent.com/63358871/123210114-813c6c80-d4fc-11eb-9240-a1a47182ab29.png">


총 5435개의 데이터에서 8:2의 비율로 학습데이터와 테스트데이터를 나눕니다.
(사실 이전에 ‘test.csv’파일을 ‘test_df’로 불러왔으나, 이 파일에는 정답이 따로 적혀있지 않아서 정확도를 측정하기에는 무리가 있다.) 





5. torch 라이브러리 불러옵니다.

<img width="329" alt="import torch" src="https://user-images.githubusercontent.com/63358871/123210285-c06abd80-d4fc-11eb-8700-0768fd7d3292.png">


PyTorch는 nn(neural network)을 생성하고 학습시키는 것을 도와주는 각종 모듈과 클래스를 제공해주는 라이브러리이다.
라이브러리 명은 ‘torch’이다.

6. 오디오 데이터 불러오기.

<img width="774" alt="Pasted Graphic 10" src="https://user-images.githubusercontent.com/63358871/123210298-c496db00-d4fc-11eb-843a-4afb4aca497e.png">


이전에 불러온 데이터는 n번째 오디오 파일이 어떤 class인지를 구별해주는 목록이 적힌 csv 파일이었다.
이번에는 진짜 음원이 담긴 ‘wav’파일을 불러오는 것이다.

만약 음성 길이가 4초 미만이라면 ‘reflect’를 통해서 복제를 한 후, 파이썬 리스트 슬라이싱을 통해 4초로 잘라낸다.



4sr인 이유

sr = 샘플링 레이트
1초 동안에 취한 표본 수 (단위 : Hz)를 의미합니다.
샘플링 레이트가 높을 수록 소리의 해상도, 즉 음질이 좋습니다.
44100Hz는 1초에 44,100 샘플을 측정한다는 것입니다.
(44.1kHz는 CD음질과 같다.)

<img width="333" alt="8 samples" src="https://user-images.githubusercontent.com/63358871/123210314-c82a6200-d4fc-11eb-9a83-37f9be61e54b.png">


따라서 n*sr은 n초의 음원을 의미한다.


오디오 데이터를 ‘np.stack’으로 다시 처리하는 이유

<img width="617" alt="audio data" src="https://user-images.githubusercontent.com/63358871/123210329-cc567f80-d4fc-11eb-85d1-031d0951f108.png">
<img width="634" alt="(-1 1386772e-01, -1 5588881e-01, -1 3303000e-01," src="https://user-images.githubusercontent.com/63358871/123210333-ceb8d980-d4fc-11eb-9f8f-e57f34836761.png">


그냥 오디오 데이터를 바로 불러오면 array가 여러 개로 나누어지지만, stack을 통해 하나씩 쌓아서 하나의 array로 만들어야 ‘torch’ 자료 유형으로 바뀐다.

7. 불러온 음성데이터와 label을 tensor 형식으로 바꾸기 

<img width="407" alt="Pasted Graphic 15" src="https://user-images.githubusercontent.com/63358871/123210356-d6787e00-d4fc-11eb-87bd-03aaa6c30c99.png">
<img width="592" alt="train data" src="https://user-images.githubusercontent.com/63358871/123210372-daa49b80-d4fc-11eb-9f33-7978d52aa21a.png">


numpy array에서 tensor로 바꾸어야 torch를 적용할 수 있다.


8. train_data와 train_label을 하나의 TensorDataset으로 묶고, DataLoader에 올리기

<img width="691" alt="Pasted Graphic" src="https://user-images.githubusercontent.com/63358871/123210381-ded0b900-d4fc-11eb-898c-d7ea99e54390.png">


이때 batch_size를 32로 정하여서, 하나의 loader에 총 32개의 TensorDataset이 들어가게 한다.


9. 모델클래스 정의

<img width="306" alt="NUM FEATURES" src="https://user-images.githubusercontent.com/63358871/123210389-e2644000-d4fc-11eb-9d65-8067965aa8d2.png">
<img width="721" alt="Pasted Graphic 1" src="https://user-images.githubusercontent.com/63358871/123210392-e42e0380-d4fc-11eb-8da7-57fa56c81ad8.png">


wav라는 1d input값을 연산할 때 사용하는 FC(Fully-Connected)레이어를 생성한다.
음원의 길이는 4초로 지정했고 sr은 44100이므로, 한 음원이 들어간다면 176400의 샘플이 들어오기에 입력값을 176400으로 정하였다.
그리고 class의 종류는 총 10가지로, 위의 10가지 중 하나의 정답을 맞추면 되는 것이기에 최종 결과값은 10개로 지정하였다.
(input과 output의 크기를 잘 맞춰야 한다.)

2개의 히든레이어를 가지고 있고, 활성함수(Activation Function)는 Relu를 사용했다.

10. 인스턴스와 optimizer 생성

<img width="472" alt="Pasted Graphic 3" src="https://user-images.githubusercontent.com/63358871/123210396-e85a2100-d4fc-11eb-9dd3-6c30ea2becb7.png">


net이라는 ‘DNN’클래스가 담긴 인스턴스를 생성하고, Learning rate가 0.001인 optimzer(역전파)를 생선한다.
loss_fn은 딥러닝 예측 모델링에 있어 실제값과 예측값의 차이를 계산하기 위해서 cross-entropy 손실함수를 적용하고 값을 저장한다.

11. 알고리즘 학습

<img width="804" alt="Pasted Graphic 4" src="https://user-images.githubusercontent.com/63358871/123210407-ec863e80-d4fc-11eb-8946-fa0b9e0f4d2a.png">


이제 loader를 불러와서 net인스턴스에 넣어서 학습을 한다.
이때 epoch는 모든 테스트데이터를 학습하는 횟수를 의미하고, 5 epoch를 돌리면 총 5번 동안 약 4300개의 테스트 데이터를 학습했다는 것을 의미한다.
밑에 출력 결과에서 ‘136/136’이라고 나오는데 이는 batchsize 를 정해놓고 loader에 train dataset을 넣었기 때문이다.

(10 epoch로 하고 싶었으나, 하드웨어 가속기를 변경하여도 colab에서 자꾸 오류가 발생하여 5로 설정하였다.) 

<img width="225" alt="loss 2 2507" src="https://user-images.githubusercontent.com/63358871/123210423-f14af280-d4fc-11eb-810b-b352b638e8a6.png">


모델 학습 중간 과정으로 backward(역전파)과정을 거치면서 손실률을 계산한다. 
epoch가 반복될 수록 학습모델이 정교해지고 있음을 위의 결과값을 통해서 알 수 있다.

12. Valid Dataset 불러오기

<img width="513" alt="val_labels append(df(nuneric_class idx )" src="https://user-images.githubusercontent.com/63358871/123210455-fc9e1e00-d4fc-11eb-9240-7b2d101d6957.png">
<img width="535" alt="val_loader data utils  DataLoader (val data, batch size = 32, shuffle - True)" src="https://user-images.githubusercontent.com/63358871/123210458-ff007800-d4fc-11eb-99c1-8f63ee36e3a2.png">


유효성 검사를 위해 train_loader을 만든 방식 그대로 val_loader를 만든다.

13. 평가를 해주는 함수 만들기

<img width="398" alt="correct = 0" src="https://user-images.githubusercontent.com/63358871/123210484-045dc280-d4fd-11eb-9239-1eec11ecc2cb.png">


val_loader를 통해 모델을 평가하는 함수를 제작한다.
이때는 모델이 이미 생성된 상태이므로, 경사하강법/역전파 방식을 사용하지 않는다.

14. 모델 평가 결과

<img width="465" alt="Validation accuracy 0 1886" src="https://user-images.githubusercontent.com/63358871/123210495-07f14980-d4fd-11eb-9a6d-cff3f64c339f.png">


테스트를 해보니 정확도가 18.86%밖에 되지 않음을 확인할 수 있다.



## CNN (Convolutional Neural Network)

CNN은 FC와 다르게 2D 이미지 처리가 가능한 레이어를 생성한다.

1. Tensor Dataset 구축

<img width="450" alt="train_labels = torch  from_numpy(train_labels) long()" src="https://user-images.githubusercontent.com/63358871/123210598-2fe0ad00-d4fd-11eb-97c7-f169b28ff799.png">


ANN방식의 7번까지는 비슷한 방식으로 train_data와 val_data 두 개의 TensorDataset을 제작한다.

차이점 : spectrogram에서의 y축은 나이키스트 이론에 따라 원본 샘플링 레이트의 1/2이다.

<img width="498" alt="Pasted Graphic 13" src="https://user-images.githubusercontent.com/63358871/123210602-32db9d80-d4fd-11eb-8b78-2b5d03760052.png">
<img width="748" alt="contentdriveNyDriveColab" src="https://user-images.githubusercontent.com/63358871/123210610-34a56100-d4fd-11eb-9280-ef3e4d8cb466.png">


따라서 CNN 모델에 적용될 음성파일의 sr은 22050이다.

<img width="238" alt="( {22050}, {22050})" src="https://user-images.githubusercontent.com/63358871/123210617-3707bb00-d4fd-11eb-8508-117dcc8ebf53.png">


2. 스펙트로그램 생성 함수와 표준화&정규화 함수

<img width="725" alt="Pasted Graphic 11" src="https://user-images.githubusercontent.com/63358871/123210625-3a02ab80-d4fd-11eb-9280-038748358733.png">

<img width="427" alt="from sklearn preprocessing import MinMaxScaler" src="https://user-images.githubusercontent.com/63358871/123210639-3e2ec900-d4fd-11eb-872e-2004d66a4b85.png">


스팩트로그램을 생성하는 함수, 표준화(평균이 0이고, 표준편차가 1)와 정규화(갑의 범위가 0~1)를 하는 함수를 만든다.

3. Train_DataLoader 구축

<img width="1074" alt="Pasted Graphic 16" src="https://user-images.githubusercontent.com/63358871/123210645-41c25000-d4fd-11eb-86be-4664c7d40efd.png">
<img width="812" alt="Pasted Graphic 20" src="https://user-images.githubusercontent.com/63358871/123210651-4424aa00-d4fd-11eb-98e4-99d5a927efa2.png">
<img width="850" alt="Pasted Graphic 19" src="https://user-images.githubusercontent.com/63358871/123210656-46870400-d4fd-11eb-8252-f5d91f2f0720.png">


ANN에서는 음성파일과 라벨을 바로 loader로 사용했지만, CNN에서는 스펙트로그램 이미지를 loader로 사용한다.
스펙트로그램을 생성하고, 표준화와 정규화를 거치게 한 다음 Tensor로 저장한다.
그리고 255(이미지 값)을 곱한 후 TensorDataset으로 모은다.
만들어진 TensorDataset을 모델에 집적 입력가능한 형태인 loader로 변환한다.

배치 사이즈(BATCH_SIZE)는 32로 설정했다.

<img width="926" alt="Pasted Graphic 18" src="https://user-images.githubusercontent.com/63358871/123210667-4ab32180-d4fd-11eb-9a49-78f644bb89a9.png">



loader에 저장된 feature는 이러한 형식으로 저장되어 있다.

4. Valid_DataLoader 구축

<img width="843" alt="Pasted Graphic 21" src="https://user-images.githubusercontent.com/63358871/123210678-4edf3f00-d4fd-11eb-8816-61668f15d069.png">


Train_DataLoader를 만드는 과정과 동일하게 Valid_DataLoader를 제작한다.


5. CNN 모델 구축

<img width="1045" alt="Pasted Graphic 23" src="https://user-images.githubusercontent.com/63358871/123210689-51da2f80-d4fd-11eb-9277-5f6fa85f68db.png">


3개의 레이어를 생성한다.

Conv2d와 MaxPool2d:

<img width="466" alt="convolution + pooling layers" src="https://user-images.githubusercontent.com/63358871/123210698-569ee380-d4fd-11eb-97fd-98dcbb960376.png">


conv2d에서 이미지를 입력 받은 다음 filter를 이용해서 이미지 feature을 계산한다
계산결과를 activation function인 relu를 이용해서 정리해준다.
그리고 maxpool2d를 이용해서 위의 계산 결과를 pooling한다.
이때 Max Pooling은 정해진 filter크기 안에서 가장 큰 값만 뽑아내는 것으로, filter마다 특징을 찾아내는 데에 용이하다는 장점이 있다.

<img width="777" alt="Pasted Graphic 24" src="https://user-images.githubusercontent.com/63358871/123210705-5a326a80-d4fd-11eb-8ca2-fec3e1a535b4.png">



<img width="326" alt="Pasted Graphic 30" src="https://user-images.githubusercontent.com/63358871/123210712-5c94c480-d4fd-11eb-939d-833c3fd6908a.png">



convolution과 pooling작업을 거치고 나온 이미지를 flatten한 다음에 1d 처리에 유용한 FC(fully-connected)작업을 시행한다.


<img width="344" alt="for s in size" src="https://user-images.githubusercontent.com/63358871/123210716-5f8fb500-d4fd-11eb-8bd2-678cb6128a04.png">



flatten화 시키는 함수이다.

<img width="543" alt="self layer2(x)" src="https://user-images.githubusercontent.com/63358871/123210724-61f20f00-d4fd-11eb-936e-2fd1d11f3f5f.png">




모델에 지정된 layer 연산을 실행하는 함수를 제작한다. 
이때 평탄화 이후 FC 레이어에 연산된 값이 활성함수인 relu를 거치게 한다.

6. 클래스 불러오기

<img width="540" alt="LEARNING_RATE)" src="https://user-images.githubusercontent.com/63358871/123210743-69b1b380-d4fd-11eb-8209-04a0c0fcfd0f.png">


클래스를 받아올 model이라는 인스턴스를 형성하고, 역전파 optimizer를 생성한다.



7. 모델 학습

<img width="991" alt="correct - (predicted" src="https://user-images.githubusercontent.com/63358871/123210755-6c140d80-d4fd-11eb-950d-a584c34b5462.png">
<img width="295" alt="loss 1 1755" src="https://user-images.githubusercontent.com/63358871/123210767-7209ee80-d4fd-11eb-87b4-68a4643b076f.png">


학습을 거듭할수록 정확도가 높아지는 것을 확인할 수 있다.


8. 평가

<img width="522" alt="num_test_batches = len(test_loader)" src="https://user-images.githubusercontent.com/63358871/123210777-759d7580-d4fd-11eb-9bfd-063a9dc8bb47.png">


전에 만들어 놓은 val_loader를 이용해서 학습모델의 정확도를 평가하기 위한 함수를 만든다.
(이미 완성된 모델이므로 역전파 과정을 제외힌다.)

9. 모델 평가 결과

<img width="649" alt="Validation accuracy 0 9121" src="https://user-images.githubusercontent.com/63358871/123210786-7930fc80-d4fd-11eb-9a7e-d9cc436345bb.png">


스펙트로그램을 활용한 CNN 모델의 경우, Wav파일을 이용한 ANN모델보다 훨씬 더 높은 정확도를 나타냄을 알 수 있다.
