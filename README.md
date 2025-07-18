# 가스계량기 객체 탐지 및 숫자 인식 자동화 시스템
<br>

## 💬 개요
YOLOv5 기반 객체 탐지 모델과 신경망 기반 숫자 인식 모델을 활용하여 가정용 가스계량기의 수치를 자동으로 인식하는 시스템
<br><br>

## 📌 목표
- 수기 검침 업무를 자동화 하기 위한 시스템 개발
- 가스계량기 영역 자동 탐지 모델 구현 (YOLOv5)
- 가스계량기 숫자 인식 모델 구현 (MLP, CNN)
- MNIST 손글씨 데이터 학습 성능과 Custom 데이터 학습 성능 비교
<br><br>

## 🙋🏻‍♀️ 수행 역할
- 전체 파이프라인 설계 및 단독 구현 (데이터 수집 -> 전처리 -> 모델 학습 -> 성능 평가)
- YOLOv5 기반 객체 탐지 모델 학습 및 최적화
- 숫자 인식 모델(MLP, CNN) 구현 및 성능 평가
<br><br>

## 🗂️ 데이터
- **구성** : 가스계량기 이미지 약 400장 (직접 촬영)
- **전처리** : OpenCV를 활용한 숫자 이미지 생성 (0~9 클래스당 75장 -> 총 750장) <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 객체 탐지를 위한 바운딩 박스 추출
- 수집된 전체 데이터셋은 비공개지만, 예시 이미지 일부를 포함하였습니다.
<br><br>

## 🔍 모델 구조
- **주요 기술** : Python, OpenCV, PyTorch, YOLOv5, TensorFlow/Keras
- **YOLOv5 (객체 탐지)** :
  - 사전 학습된 'yolov5s' 모델 사용
  - *Input* : 416 *416 (가스계량기 이미지)
  - *Epochs* : 50
  - *Batchsize* : 16
  - 'train.txt', 'val.txt', 'data.yaml' 커스터마이징 후 학습 진행
  - [YOLOv5 커스텀 학습 튜토리얼 영상 참고](https://youtu.be/T0DO1C8uYP8?si=dSr4nJK_Cg9B-Bf9) <br><br>
- **MLP (숫자 인식)**:
  - <img width="600" height="300" alt="Image" src="https://github.com/user-attachments/assets/fd8b7913-c11c-45da-8e18-11b2a063cf1e"/>
  - *Input* : 28 *28 (숫자 이미지) 
  - *Hidden* : 1024 units, 'tanh' activation
  - *Output* : 10 classes(0~9), 'tanh' activation
  - *Loss Function* : Mean Squared Error
  - *Optimizer* : Adam (learning_rate=0.001)
  - *Epochs* : 30
  - *Batchsize* : 128
  <br>
- **CNN (숫자 인식)**:
   <img width="1280" height="207" alt="Image" src="https://github.com/user-attachments/assets/160f65cb-af25-44d1-98a4-838a7264e1a0" />
   - *Input* : 28 *28 *1 (숫자 이미지)
   - *Output* : 10 classes, Softmax
   - *Loss Function* : Categorical Crossentropy
   - *Optimizer* : Adam (learning_rate=0.001)
   - *Epochs* : 30
   - *Batchsize* : 128
  <br><br>

## 📊 주요 결과
- 객체 탐지 결과 (**YOLOv5**)
  |-|Precision|Recall|mAP50|mAP50-90|
  |---|---|---|---|---|
  |YOLOv5|0.98|1.0|0.995|0.754|
    - 가스계량기 탐지 이미지 예시
  <img width="800" height="400" alt="Image" src="https://github.com/user-attachments/assets/98c6cb40-317e-4aab-8721-fe2bc0a66dba" />

- 숫자 인식 결과 (숫자 인식에 실패한 데이터/총 데이터)
  |-|MLP|CNN|
  |---|---|---|
  |MNIST|646 / 750|494 / 750|
  |Custom|52 / 750|7 / 750|
    - 숫자 인식 실패한 이미지 예시 (**MLP**)
      <img width="1500" height="500" alt="Image" src="https://github.com/user-attachments/assets/ff2d0ab9-5f98-4342-b4ea-651899068fa7" />
    - 숫자 인식 실패한 이미지 예시 (**CNN**)
      <img width="1370" height="217" alt="Image" src="https://github.com/user-attachments/assets/a87e68a0-ccf4-43d6-8e48-46b9483fbf66" />
- 실시간 계량기 수치 확인 시스템, IoT 기반 자동 검침 시스템(가스/전기/수도) 등으로 확장 가능성을 제시함.
<br><br>

## 🔁 회고
YOLO 학습 시 바운딩 박스 설정이 객체 탐지 성능에 미치는 영향력이 크다는 것을 직접 체감하였으며, 결과 후처리 로직과 웹/앱 연동 기술을 사용하여 시스템을 통합하면 실제 사용이 가능할 것이라고 판단함.
