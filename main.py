# -*- coding: utf-8 -*-
"""
주제: 경사하강법의 수렴 속도에 영향을 미치는 하이퍼파라미터 분석
- 보고서 3항 '탐구 절차'에 따른 프로그래밍 실험 구현
- 작성자: AI 프로그래밍 전문가 (최종 오류 수정 및 권장사항 반영 버전)
"""

# [수정 1] TensorFlow의 정보성 로그 메시지를 숨기기 위한 설정
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import platform

# ==============================================================================
# 0. 실험 환경 설정 및 재현성 확보
# ==============================================================================
np.random.seed(42)
tf.random.set_seed(42)

# [수정 2] 그래프의 한글 폰트 깨짐 방지 설정
try:
    if platform.system() == "Windows":
        font_name = "Malgun Gothic"
    elif platform.system() == "Darwin":  # macOS
        font_name = "AppleGothic"
    else:  # Linux
        font_name = "NanumGothic"
    plt.rc("font", family=font_name)
    plt.rc("axes", unicode_minus=False)
    print(f"성공: 한글 폰트 '{font_name}'로 설정되었습니다.")
except Exception:
    print("경고: 한글 폰트 설정에 실패했습니다. 그래프의 한글이 깨질 수 있습니다.")

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BATCH_SIZE = 32
DEFAULT_OPTIMIZER = "sgd"
NUM_EPOCHS = 100


# ==============================================================================
# 3.1 데이터셋 및 모델 구조 정의
# ==============================================================================
def generate_dataset(n_samples=200, n_features=1):
    """Scikit-learn을 사용하여 선형 회귀용 데이터셋을 생성합니다."""
    print(f"\n[데이터 생성] 샘플 수: {n_samples}, 특성 수: {n_features}")
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=20, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = y.reshape(-1, 1)
    return X_scaled, y_scaled


# [수정 3] Keras 권장사항을 반영하여 모델 구조 정의 함수 개선
def build_linear_regression_model(n_features=1):
    """
    단층 선형 회귀 모델을 생성합니다. (Keras 권장 방식 적용)
    - Input Layer를 명시적으로 추가하여 'UserWarning'을 해결합니다.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(n_features,)),  # 입력 형태를 명시
            tf.keras.layers.Dense(units=1),
        ]
    )
    return model


# ==============================================================================
# 3.2 실험 설계 및 실행
# ==============================================================================
# [최종 수정] KeyError를 근본적으로 해결한 범용 실험 함수
def run_and_plot_experiment(title, X, y, configs, varying_param_key, display_name, y_limit=None):
    """
    주어진 설정에 따라 실험을 실행하고 결과를 시각화하는 범용 함수
    - varying_param_key: 실제 config 딕셔너리에 사용된 키 (e.g., 'learning_rate')
    - display_name: 그래프 범례에 표시될 이름 (e.g., 'η')
    """
    plt.figure(figsize=(12, 7))
    plt.title(title, fontsize=16)

    print(f"\n--- 실험 시작: {title} ---")

    for config in configs:
        # 매 실험마다 새로운 모델 생성 및 컴파일
        model = build_linear_regression_model(n_features=X.shape[1])

        if config["optimizer_name"] == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=config["learning_rate"])

        model.compile(optimizer=optimizer, loss="mean_squared_error")

        # 정확한 키를 사용하여 값 추출 및 레이블 생성
        value_to_display = config[varying_param_key]
        label = f"{display_name} = {value_to_display}"
        print(f"실행 조건: {label}")

        start_time = time.time()
        history = model.fit(X, y, epochs=NUM_EPOCHS, batch_size=config["batch_size"], verbose=0)
        end_time = time.time()

        final_loss = history.history["loss"][-1]
        print(f"  -> 최종 Loss: {final_loss:.4f}, 학습 시간: {end_time - start_time:.2f}초")

        plt.plot(history.history["loss"], label=label, lw=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Mean Squared Error)", fontsize=12)
    plt.legend()
    plt.grid(True)

    if y_limit:
        plt.ylim(0, y_limit)
    else:
        plt.ylim(bottom=0)

    plt.show()


# --- 각 실험 정의 ---
def experiment_1_learning_rate(X, y):
    """학습률(Learning Rate)에 따른 수렴 속도 비교"""
    configs = []
    for lr in [0.001, 0.01, 0.1, 1.0]:
        configs.append(
            {
                "learning_rate": lr,  # 실제 키
                "batch_size": DEFAULT_BATCH_SIZE,
                "optimizer_name": DEFAULT_OPTIMIZER,
            }
        )
    run_and_plot_experiment(
        title="학습률(Learning Rate)에 따른 Loss 변화",
        X=X,
        y=y,
        configs=configs,
        varying_param_key="learning_rate",  # config 딕셔너리의 실제 키
        display_name="η",  # 그래프에 표시될 이름
        y_limit=5000,
    )


def experiment_2_batch_size(X, y):
    """배치 크기(Batch Size)에 따른 수렴 속도 비교"""
    configs = []
    for bs in [1, 32, len(X)]:
        configs.append(
            {
                "learning_rate": DEFAULT_LEARNING_RATE,
                "batch_size": bs,  # 실제 키
                "optimizer_name": DEFAULT_OPTIMIZER,
            }
        )
    run_and_plot_experiment(
        title="배치 크기(Batch Size)에 따른 Loss 변화",
        X=X,
        y=y,
        configs=configs,
        varying_param_key="batch_size",  # config 딕셔너리의 실제 키
        display_name="배치 크기",  # 그래프에 표시될 이름
    )


def experiment_3_optimizer(X, y):
    """옵티마이저(Optimizer) 종류에 따른 수렴 속도 비교"""
    configs = []
    for opt_name in ["sgd", "adam"]:
        configs.append(
            {
                "learning_rate": DEFAULT_LEARNING_RATE,
                "batch_size": DEFAULT_BATCH_SIZE,
                "optimizer_name": opt_name,  # 실제 키
            }
        )
    run_and_plot_experiment(
        title="옵티마이저(Optimizer)에 따른 Loss 변화",
        X=X,
        y=y,
        configs=configs,
        varying_param_key="optimizer_name",  # config 딕셔너리의 실제 키
        display_name="옵티마이저",  # 그래프에 표시될 이름
    )


# ==============================================================================
# 메인 실행 블록
# ==============================================================================
if __name__ == "__main__":
    X_train, y_train = generate_dataset()

    experiment_1_learning_rate(X_train, y_train)
    experiment_2_batch_size(X_train, y_train)
    experiment_3_optimizer(X_train, y_train)

    print("\n--- 모든 실험이 성공적으로 완료되었습니다. ---")
