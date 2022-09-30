# 0. intro
 - PDF를 먼저 보시는걸 권장드립니다.
 - 대회 : https://dacon.io/competitions/official/235927/overview/description
## 대회결과
 - 압도적인 차이로 Private Score 1위
 - AWARD 발표(2022.09.27) 전날(2022.09.26)에 Data Leakage로 실격 통보를 받음. **심사측 내부에서도 의견이 갈려** 결정이 늦었다고 함.
 - 심사측의 주장 : 쥬혁이 팀은, test의 이동평균, 이동 중앙값을 사용했습니다. 이는 test의 **통계량**을 사용되는 것이기에 Data Leakage에 해당합니다. 또한 test의 입력을 분할하여 사용하는 것은 test dataset의 **사이즈**를 가늠할 수 있어야 하므로 test 정보가 누설된 경우입니다. 결과적으로 test 데이터셋을 **random shuffle**했을 때  동일한 성능이 나올 수 없습니다.
 - 쥬혁이팀 반론 
 1. 분할 크기인 30은, 최소 샘플크기가 30이상이면 모집단의 분포모양에 관계없이 근사적으로 정규분포를 이룬다는 **중심극한 정리**를 활용한 것. 만약 통계정보를 활용했다면 최소수치인 30을 적용하지 않았을것임. 실제로도 분할크기로 A/B테스트 안해봄. train_Data_set을 통해 **정규분포를 이루며 잘 관리되고있는 공정**이라는 insight와, **대량생산되는 자율주행용 안테나**라는 insight에 의해 중심극한 정리를 적용한 것 뿐이다.
 2. test 데이터셋을 random shuffle했을 때  동일한 성능이 나와야한다면, **시계열접근법(칼만필터 또한) 자체가, 과거 test데이터(통계량)를 사용하기 때문에 Data Leakage**이다. 코드검정(2022.09.04~2022.09.15)을 통해 **의견이 갈릴 필요도 없이, 당연히** Data Leakage check에서 실격됐어야함. 즉, 해당 기준은 **원래(코드검정) 적용되지 않았던 기준**임. **타 팀의 발표에 영향**받은게 아닌가 의심되는데, **불공정**하다고 생각함.
 3. **실제공정에 적용시, 전혀 제한없는 접근법임**.

# 1. 파일 설명

### 1-1. [Code] Feature_engineering.ipynb : 피쳐엔지니어링 코드

- 가설을 바탕으로 feature engineering된 csv 파일이 data폴더에 생성

### 1-2. [Code] Modeling.py : 학습 및 예측 코드

- 실행시 FOLD1 폴더가 생성, 하위에 Y_01~Y_14 모델 가중치 저장(약 3시간 소요)

### 1-3. [Code] Modeling_load.py : 미리 학습된 모델로드, 예측코드

- 가중치 로드 및 csv 파일 생성(약 10분 소요)

### 1-4. [Code] Modeling_load_2.py : 미리 학습된 모델로드, 예측코드

- [Code] Modeling_load.py 에서 에러가 발생했을때, [Code] Modeling_load.py 을 대체하는 예비용 파일. 가중치 로드 및 csv 파일 생성(약 20~30분 소요)


# 2. 진행방법

### 2-1 학습/예측 진행방법

- [Code] Feature_engineering.ipynb 실행하여 데이터 생성
- [Code] modeling.py 실행

### 2-2 사전학습 가중치/예측 진행방법

- 학습한 모델의 폴더의 경로(위치)는 [Code] Modeling_load.py, [Code] Modeling_load_2.py 스크립트들의 경로(위치)와 같아야 함.
- [Code] modeling_load.py 실행

# 3. 기타

### 3-1 **학습환경 권장사항**

- NVIDIA Geforce RTX 3070 이상
- 더 낮은 GPU로도 학습은 가능하나 학습중간 끊어지는 경우가 있음

### 3-2 학습이 끊어졌을시

- case1) [Code] Modeling.py로 학습하다 중간(예로들면 Y_06)에 메모리 문제로 학습이 중단되고 run이 끊겼다
    
    - FOLD1폴더의 Y_06Models-predict폴더를 삭제하고, 70라인의 주석을 참고하여 range(6, 15)로 변경하여 이어서 학습완료. 그러나 sample_submission을 다시 read 하므로, 생성되는 csv는 Y_01~Y_05 prediction이 누락되어 있음 
    
    - [Code] Modeling_load.py 14라인의 weights_path를 weights_path='./FOLD1/'으로 경로만 바꿔주고 실행하면 해당 모델로 Y_01 ~ Y_14 모두를 예측한 csv 생성가능.  
    OR [Code] Modeling_load_2.py 의 23라인 weights_path = './weights/ 를  weights_path='./FOLD1/'으로 경로만 바꿔주고 실행하면 해당 모델로 Y_01 ~ Y_14 모두를 예측한 csv 생성가능.
    

- case2) [Code] Modeling.py로 학습중인데 error가 발생하여 run중이지만 학습이 안되어(시간이 오래걸리고 cpu, gpu가 사용안되고 있음) 시간만 계속 흐르는 경우
→ run을 멈추고, case1 진행

### 3-3 [Code] Modeling_load.py 에서 ERROR가 발생했을시

- 4. [Code] Modeling_load_2.py 를 실행한다.

### 3-3 Requirements

```python
conda create -n LB1 python=3.9.0
conda activate LB1
conda install ipykernel


pip3 install -U pip
pip3 install -U setuptools wheel
pip3 install filterpy
pip3 install jupyter notebook


pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchtext==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install autogluon==0.5.3b20220811
```

### 3-4 Test list

- NVIDIA GeForce RTX 3070 - LB : 1.9090032119 (1등)
- NVIDIA GeForce GTX 1060 - LB : 1.9109867458 (1등 유지)
- NVIDIA GeForce RTX 3090 - LB : 1.9118771404 (1등 유지)
- **학습시 seed는 유지되나 하드웨어별로 결과가 바뀔 수 있음**
