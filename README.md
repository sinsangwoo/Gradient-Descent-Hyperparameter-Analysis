# Gradient-Descent-Hyperparameter-Analysis

## 경사하강법(Gradient Descent) 하이퍼파라미터가 수렴 속도에 미치는 영향 분석(고등학교 3학년 인공지능 기초 교과목 심화탐구 프로젝트)

---

### 💡 프로젝트 개요

이 프로젝트는 머신러닝 모델 학습의 핵심인 **경사하강법(Gradient Descent)**의 수렴 속도에 영향을 미치는 주요 하이퍼파라미터(Hyperparameter)들을 탐구하고 시각적으로 분석하기 위해 개발되었습니다. 특히, **학습률(Learning Rate), 배치 크기(Batch Size), 그리고 옵티마이저(Optimizer)의 종류**가 모델의 손실(Loss) 감소와 학습 과정에 미치는 영향을 직접 실험하고 그 결과를 그래프로 시각화합니다.

이 프로젝트는 AI 및 컴퓨터 과학 분야 진학을 희망하는 학생으로서, 머신러닝의 기초 개념과 모델 최적화 원리를 깊이 이해하기 위한 학습 목표로 수행되었습니다.

---

### ✨ 주요 특징 및 학습 목표

* **경사하강법 원리 이해:** 가장 기본적인 최적화 알고리즘인 경사하강법의 동작 원리를 코드를 통해 직접 확인합니다.
* **하이퍼파라미터 영향 분석:**
    * **학습률(Learning Rate):** 너무 높거나 낮을 때 학습에 어떤 영향을 미치는지 실험합니다.
    * **배치 크기(Batch Size):** 확률적 경사하강법(SGD), 미니 배치 경사하강법(Mini-Batch GD), 배치 경사하강법(Batch GD)의 특성을 비교합니다.
    * **옵티마이저(Optimizer):** SGD와 Adam 옵티마이저의 수렴 성능 차이를 비교합니다.
* **TensorFlow/Keras 활용:** 실제 딥러닝 프레임워크인 TensorFlow와 Keras를 사용하여 간단한 선형 회귀 모델을 구축하고 학습하는 방법을 익힙니다.
* **데이터 생성 및 전처리:** Scikit-learn을 사용하여 가상 선형 회귀 데이터셋을 생성하고, StandardScaler를 이용한 데이터 표준화 과정을 경험합니다.
* **결과 시각화:** Matplotlib을 활용하여 에포크(Epoch)에 따른 손실(Loss) 변화를 그래프로 시각화하여, 하이퍼파라미터의 영향을 직관적으로 이해합니다.
* **코드 재현성 확보:** `numpy`와 `tensorflow`의 난수 시드를 고정하여 실험 결과의 재현성을 보장합니다.

---

### 🛠️ 기술 스택

* **Python**
* **TensorFlow / Keras** (딥러닝 모델 구축 및 학습)
* **NumPy** (수치 계산)
* **Scikit-learn** (데이터셋 생성 및 전처리)
* **Matplotlib** (결과 시각화)

---

### 🚀 프로젝트 실행 방법

1.  **레포지토리 클론:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/Gradient-Descent-Hyperparameter-Analysis.git](https://github.com/YourGitHubUsername/Gradient-Descent-Hyperparameter-Analysis.git)
    cd Gradient-Descent-Hyperparameter-Analysis
    ```
2.  **가상 환경 설정 (권장):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **필요한 라이브러리 설치:**
    ```bash
    pip install tensorflow numpy scikit-learn matplotlib
    ```
4.  **프로젝트 실행:**
    ```bash
    python main.py # 또는 파이썬 파일명
    ```
    *스크립트 실행 시 학습률, 배치 크기, 옵티마이저별 손실 변화 그래프가 자동으로 생성됩니다.*



---

### 📝 학습 및 개선점

이 프로젝트를 통해 경사하강법 기반 최적화 과정에서 하이퍼파라미터 튜닝의 중요성을 깊이 이해하게 되었습니다. 특히, 각 하이퍼파라미터가 모델의 수렴 속도와 최종 성능에 미치는 영향을 시각적으로 확인하며 직관적인 이해를 높일 수 있었습니다.

향후에는 이 프로젝트를 통해 얻은 지식을 바탕으로 더 복잡한 딥러닝 모델(예: CNN, RNN)에 다양한 옵티마이저와 학습률 스케줄링 기법을 적용해보고, 과적합(Overfitting) 방지 기법(예: 정규화, 드롭아웃)에 대해서도 탐구할 계획입니다. 최종적으로는 주식 시장 분석 AI 서비스 개발이라는 장기 목표를 위해 꾸준히 학습하고 발전해나갈 것입니다.
