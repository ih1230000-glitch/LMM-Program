import io
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="AI 데이터 분석 & 챗봇 통합 플랫폼",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# 우측 하단 플로팅 말풍선(Popover) 스타일 CSS 적용
# ---------------------------------------------------------
st.markdown("""
<style>
div[data-testid="stPopover"] {
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 9999;
}
div[data-testid="stPopover"] > button {
    border-radius: 50px !important;
    padding: 14px 22px !important;
    background: linear-gradient(135deg, #2563EB, #4F46E5) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stPopover"] > button:hover {
    transform: translateY(-3px) scale(1.03) !important;
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.55) !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 선형·비선형 회귀 & 🧠 인공신경망(ANN) 통합 분석 플랫폼")
st.markdown("""
하나의 데이터셋으로 **① 경사하강법 기반 선형·비선형 회귀 분석**과 **② 다층 인공신경망(ANN) 예측**을 수행하고, 우측 하단의 **💬 AI 챗봇 말풍선**을 눌러 분석 결과와 알고리즘 원리를 실시간으로 질문할 수 있습니다.
""")

# ---------------------------------------------------------
# 1. 공통 데이터 업로드 및 기본 샘플 데이터 로드
# ---------------------------------------------------------
st.sidebar.header("🛠️ 1. 데이터 파일 관리")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드 (미업로드 시 샘플 자동 적용)", type=["csv"])

if uploaded_file is None:
    st.sidebar.info("💡 현재 **기본 비선형 샘플 데이터**가 적용되어 있습니다. 본인 CSV 파일이 있다면 위에서 업로드하세요.")
    sample_data = """X,Y
1.0,2.1
2.0,3.8
3.0,7.9
4.0,15.2
5.0,24.8
6.0,36.1
7.0,49.5
8.0,64.2
9.0,81.0
10.0,100.5
11.0,121.2
12.0,143.8
13.0,168.1
14.0,196.0
15.0,225.2
16.0,256.4
17.0,289.1
18.0,324.5
19.0,361.0
20.0,400.2"""
    df = pd.read_csv(io.StringIO(sample_data))
    data_source_name = "기본 샘플 데이터 (Quadratic Dataset)"
else:
    df = pd.read_csv(uploaded_file)
    data_source_name = f"사용자 업로드 파일 ({uploaded_file.name})"

with st.expander(f"📋 현재 분석 중인 데이터셋 미리보기 [{data_source_name}]", expanded=True):
    c_info1, c_info2 = st.columns([3, 1])
    with c_info1:
        st.dataframe(df.head(), use_container_width=True)
    with c_info2:
        st.metric("전체 데이터 행 수", f"{df.shape[0]}개")
        st.metric("전체 변수(열) 수", f"{df.shape[1]}개")

columns = df.columns.tolist()

# ---------------------------------------------------------
# 2. 기능별 탭 분리 (회귀 분석 vs 인공신경망)
# ---------------------------------------------------------
tab_reg, tab_ann = st.tabs(["📊 1. 선형 & 비선형 회귀 분석", "🧠 2. 인공신경망(ANN) 데이터 예측"])

# =========================================================
# TAB 1: 선형 & 비선형 회귀 분석
# =========================================================
with tab_reg:
    st.subheader("📊 경사하강법 기반 선형 & 2차 비선형 회귀 시뮬레이터")
    st.caption("독립 변수(X)와 종속 변수(Y)를 선택하고 평균제곱오차(MSE)를 최소화하는 최적의 직선 또는 곡선 수식을 도출합니다.")

    reg_ctrl1, reg_ctrl2, reg_ctrl3, reg_ctrl4 = st.columns([1, 1, 1.5, 1.5])
    with reg_ctrl1:
        reg_x_col = st.selectbox("X축 독립 변수", columns, index=0, key="reg_x")
    with reg_ctrl2:
        reg_y_col = st.selectbox("Y축 종속 변수", columns, index=len(columns) - 1 if len(columns) > 1 else 0, key="reg_y")
    with reg_ctrl3:
        reg_model_type = st.radio(
            "회귀 모델 선택",
            ["1차 선형 회귀 (y = ax + b)", "2차 비선형 회귀 (y = w2·x² + w1·x + b)"],
            key="reg_model"
        )
    with reg_ctrl4:
        reg_lr = st.slider("학습률 (Learning Rate)", 0.001, 0.1, 0.02, step=0.005, format="%.3f", key="reg_lr")
        reg_epochs = st.slider("학습 횟수 (Epochs)", 100, 10000, 3000, step=100, key="reg_epochs")

    reg_df = df[[reg_x_col, reg_y_col]].copy()
    reg_df[reg_x_col] = pd.to_numeric(reg_df[reg_x_col], errors="coerce")
    reg_df[reg_y_col] = pd.to_numeric(reg_df[reg_y_col], errors="coerce")
    reg_df = reg_df.dropna()

    if len(reg_df) >= 2:
        X_orig = reg_df[reg_x_col].values
        y_orig = reg_df[reg_y_col].values
        N = len(X_orig)

        x_mean, x_std = X_orig.mean(), X_orig.std()
        if x_std == 0:
            x_std = 1.0
        X_norm = (X_orig - x_mean) / x_std

        reg_loss_history = []

        if reg_model_type.startswith("1차"):
            w_s, b_s = 0.0, 0.0
            for _ in range(reg_epochs):
                y_pred = w_s * X_norm + b_s
                error = y_orig - y_pred
                reg_loss_history.append(np.mean(error ** 2))
                dw = (-2 / N) * np.sum(X_norm * error)
                db = (-2 / N) * np.sum(error)
                w_s -= reg_lr * dw
                b_s -= reg_lr * db

            w_final = w_s / x_std
            b_final = b_s - w_final * x_mean
            y_hat = w_final * X_orig + b_final

            x_dense = np.linspace(X_orig.min(), X_orig.max(), 200)
            y_dense = w_final * x_dense + b_final
            formula_str = f"y = {w_final:.4f}x + {b_final:.4f}"
        else:
            w2_s, w1_s, b_s = 0.0, 0.0, 0.0
            for _ in range(reg_epochs):
                y_pred = w2_s * (X_norm ** 2) + w1_s * X_norm + b_s
                error = y_orig - y_pred
                reg_loss_history.append(np.mean(error ** 2))
                dw2 = (-2 / N) * np.sum((X_norm ** 2) * error)
                dw1 = (-2 / N) * np.sum(X_norm * error)
                db = (-2 / N) * np.sum(error)
                w2_s -= reg_lr * dw2
                w1_s -= reg_lr * dw1
                b_s -= reg_lr * db

            w2_final = w2_s / (x_std ** 2)
            w1_final = (w1_s / x_std) - (2 * x_mean * w2_final)
            b_final = b_s - (w1_s * x_mean / x_std) + (w2_final * (x_mean ** 2))
            y_hat = w2_final * (X_orig ** 2) + w1_final * X_orig + b_final

            x_dense = np.linspace(X_orig.min(), X_orig.max(), 200)
            y_dense = w2_final * (x_dense ** 2) + w1_final * x_dense + b_final
            formula_str = f"y = {w2_final:.4f}x² + {w1_final:.4f}x + {b_final:.4f}"

        reg_mse = np.mean((y_orig - y_hat) ** 2)
        ss_tot_r = np.sum((y_orig - y_orig.mean()) ** 2)
        ss_res_r = np.sum((y_orig - y_hat) ** 2)
        reg_r2 = 1 - (ss_res_r / ss_tot_r) if ss_tot_r != 0 else 0.0

        st.session_state["last_reg_summary"] = (
            f"모델: {reg_model_type}, 변수: {reg_x_col}->{reg_y_col}, "
            f"도출 수식: {formula_str}, MSE: {reg_mse:.4f}, R2: {reg_r2:.4f}"
        )

        r_col1, r_col2 = st.columns([2, 1])
        with r_col1:
            fig_r, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(12, 4.5))
            ax_r1.scatter(X_orig, y_orig, color="crimson", s=50, label="Actual Data", zorder=5)
            ax_r1.plot(x_dense, y_dense, color="royalblue", linewidth=2.5, label="Regression Curve")
            ax_r1.set_title("Data Distribution & Regression Fit")
            ax_r1.set_xlabel(reg_x_col)
            ax_r1.set_ylabel(reg_y_col)
            ax_r1.legend()
            ax_r1.grid(True, linestyle="--", alpha=0.5)

            ax_r2.plot(reg_loss_history, color="darkorange", linewidth=1.8)
            ax_r2.set_title("Gradient Descent MSE Loss")
            ax_r2.set_xlabel("Epochs")
            ax_r2.set_ylabel("MSE Loss")
            ax_r2.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig_r)

        with r_col2:
            st.markdown("#### 📌 회귀 학습 결과 요약")
            st.success(f"**도출된 최종 수식:**\n\n${formula_str}$")
            st.metric("평균제곱오차 (MSE)", f"{reg_mse:.4f}")
            st.metric("결정 계수 ($R^2$ Score)", f"{reg_r2:.4f}")
            st.metric("분석 데이터 수", f"{N}개")
    else:
        st.warning("⚠️ 선택한 컬럼에 유효한 숫자 데이터가 2개 이상 필요합니다.")


# =========================================================
# TAB 2: 인공신경망(ANN) 데이터 예측
# =========================================================
with tab_ann:
    st.subheader("🧠 다층 인공신경망(ANN) 기반 데이터 예측 및 시뮬레이터")
    st.caption("입력층-은닉층-출력층 구조와 비선형 활성화 함수(Sigmoid/ReLU/Tanh), 역전파 알고리즘으로 복잡한 데이터 패턴을 학습합니다.")

    ann_ctrl1, ann_ctrl2, ann_ctrl3 = st.columns([1.5, 1.5, 1.5])
    with ann_ctrl1:
        ann_x_cols = st.multiselect(
            "X축 입력 변수 (다중 선택 가능)",
            columns,
            default=[columns[0]] if len(columns) > 0 else [],
            key="ann_x"
        )
        ann_y_col = st.selectbox(
            "Y축 출력 변수 (Target)",
            columns,
            index=len(columns) - 1 if len(columns) > 1 else 0,
            key="ann_y"
        )
    with ann_ctrl2:
        activation_name = st.selectbox(
            "활성화 함수 (Activation Function)",
            ["Sigmoid", "ReLU", "Tanh"],
            key="ann_act"
        )
        hidden_units = st.slider("은닉층 뉴런 수 (Hidden Neurons)", 1, 32, 8, step=1, key="ann_hidden")
    with ann_ctrl3:
        test_size_ratio = st.slider("테스트 데이터 분할 비율 (Test Split)", 0.1, 0.4, 0.2, step=0.05, key="ann_split")
        ann_lr = st.number_input("학습률 (Learning Rate)", value=0.01, format="%.4f", key="ann_lr")
        ann_epochs = st.slider("학습 반복 횟수 (Epochs)", 100, 10000, 2000, step=100, key="ann_epochs")

    if len(ann_x_cols) > 0 and ann_y_col:
        if ann_y_col in ann_x_cols:
            st.warning("⚠️ 입력 변수(X)와 출력 변수(Y)가 겹치지 않도록 다르게 선택해 주세요.")
        else:
            temp_df = df[ann_x_cols + [ann_y_col]].copy()
            for col in ann_x_cols + [ann_y_col]:
                temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")
            temp_df = temp_df.dropna()

            if len(temp_df) > 5:
                X_raw = temp_df[ann_x_cols].values
                y_raw = temp_df[ann_y_col].values.reshape(-1, 1)

                np.random.seed(42)
                indices = np.random.permutation(len(X_raw))
                test_cnt = max(1, int(len(X_raw) * test_size_ratio))
                test_idx, train_idx = indices[:test_cnt], indices[test_cnt:]

                X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
                y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]

                X_mean, X_std = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
                X_std[X_std == 0] = 1.0
                X_train = (X_train_raw - X_mean) / X_std
                X_test = (X_test_raw - X_mean) / X_std

                y_mean, y_std = y_train_raw.mean(), y_train_raw.std()
                if y_std == 0:
                    y_std = 1.0
                y_train = (y_train_raw - y_mean) / y_std
                y_test = (y_test_raw - y_mean) / y_std

                N_train, D = X_train.shape
                H = hidden_units

                np.random.seed(42)
                W1 = np.random.randn(D, H) * np.sqrt(2.0 / D)
                b1 = np.zeros((1, H))
                W2 = np.random.randn(H, 1) * np.sqrt(2.0 / H)
                b2 = np.zeros((1, 1))

                def activate(z, name):
                    if name == "Sigmoid":
                        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                    elif name == "ReLU":
                        return np.maximum(0, z)
                    elif name == "Tanh":
                        return np.tanh(z)

                def activate_derivative(a, name):
                    if name == "Sigmoid":
                        return a * (1 - a)
                    elif name == "ReLU":
                        return (a > 0).astype(float)
                    elif name == "Tanh":
                        return 1 - a ** 2

                train_loss_history = []
                test_loss_history = []

                for _ in range(ann_epochs):
                    Z1 = np.dot(X_train, W1) + b1
                    A1 = activate(Z1, activation_name)
                    Y_pred_train = np.dot(A1, W2) + b2

                    train_error = Y_pred_train - y_train
                    train_loss_history.append(np.mean(train_error ** 2))

                    Z1_test = np.dot(X_test, W1) + b1
                    A1_test = activate(Z1_test, activation_name)
                    Y_pred_test = np.dot(A1_test, W2) + b2
                    test_loss_history.append(np.mean((Y_pred_test - y_test) ** 2))

                    dY = 2 * train_error / N_train
                    dW2 = np.dot(A1.T, dY)
                    db2 = np.sum(dY, axis=0, keepdims=True)

                    dA1 = np.dot(dY, W2.T)
                    dZ1 = dA1 * activate_derivative(A1, activation_name)
                    dW1 = np.dot(X_train.T, dZ1)
                    db1 = np.sum(dZ1, axis=0, keepdims=True)

                    W1 -= ann_lr * dW1
                    b1 -= ann_lr * db1
                    W2 -= ann_lr * dW2
                    b2 -= ann_lr * db2

                Z1_final = np.dot(X_test, W1) + b1
                A1_final = activate(Z1_final, activation_name)
                y_pred_test_orig = (np.dot(A1_final, W2) + b2) * y_std + y_mean

                mse = np.mean((y_test_raw - y_pred_test_orig) ** 2)
                mae = np.mean(np.abs(y_test_raw - y_pred_test_orig))
                ss_tot = np.sum((y_test_raw - y_test_raw.mean()) ** 2)
                ss_res = np.sum((y_test_raw - y_pred_test_orig) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

                st.session_state["last_ann_summary"] = (
                    f"구조: 입력 {D}개 -> 은닉 {H}개({activation_name}) -> 출력 1개, "
                    f"테스트 MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}"
                )

                a_col1, a_col2 = st.columns([2, 1])
                with a_col1:
                    fig_a, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
                    if D == 1:
                        x_grid = np.linspace(X_raw[:, 0].min(), X_raw[:, 0].max(), 200).reshape(-1, 1)
                        x_grid_scaled = (x_grid - X_mean) / X_std
                        y_grid_pred = (np.dot(activate(np.dot(x_grid_scaled, W1) + b1, activation_name), W2) + b2) * y_std + y_mean

                        ax1.scatter(X_train_raw[:, 0], y_train_raw, color="dodgerblue", alpha=0.6, label="Train Data", s=30)
                        ax1.scatter(X_test_raw[:, 0], y_test_raw, color="crimson", alpha=0.9, label="Test Data", s=45, zorder=5)
                        ax1.plot(x_grid[:, 0], y_grid_pred[:, 0], color="rebeccapurple", linewidth=2.5, label="ANN Prediction", zorder=6)
                        ax1.set_xlabel(ann_x_cols[0])
                        ax1.set_ylabel(ann_y_col)
                    else:
                        ax1.scatter(y_test_raw, y_pred_test_orig, color="teal", alpha=0.8, s=45)
                        ax1.plot([y_test_raw.min(), y_test_raw.max()], [y_test_raw.min(), y_test_raw.max()], "r--", label="Ideal Fit")
                        ax1.set_xlabel("Actual Value")
                        ax1.set_ylabel("Predicted Value")

                    ax1.set_title("Actual vs ANN Prediction")
                    ax1.legend()
                    ax1.grid(True, linestyle="--", alpha=0.5)

                    ax2.plot(train_loss_history, color="royalblue", label="Train Loss", linewidth=1.5)
                    ax2.plot(test_loss_history, color="darkorange", label="Test Loss", linewidth=1.5, linestyle="--")
                    ax2.set_title("Train & Test MSE Loss Curve")
                    ax2.set_xlabel("Epochs")
                    ax2.set_ylabel("MSE Loss")
                    ax2.legend()
                    ax2.grid(True, linestyle="--", alpha=0.5)
                    st.pyplot(fig_a)

                with a_col2:
                    st.markdown("#### 📌 신경망 성능 평가 및 요약")
                    st.success(f"**네트워크 구조:**\n- 입력층: {D}개 변수\n- 은닉층: {H}개 뉴런 ({activation_name})\n- 출력층: 1개")
                    st.metric("테스트 MSE (평균제곱오차)", f"{mse:.4f}")
                    st.metric("테스트 MAE (평균절대오차)", f"{mae:.4f}")
                    st.metric("결정 계수 ($R^2$ Score)", f"{r2:.4f}")

                st.markdown("---")
                st.markdown("#### 🔮 실시간 단건 예측 시뮬레이터")
                st.write("입력 변수($X$) 값을 직접 조절하여 학습된 인공신경망의 예측 결과($Y$)를 즉시 확인해 보세요.")

                user_inputs = []
                pred_cols = st.columns(min(len(ann_x_cols), 4))
                for i, col_name in enumerate(ann_x_cols):
                    col_min = float(temp_df[col_name].min())
                    col_max = float(temp_df[col_name].max())
                    col_mean = float(temp_df[col_name].mean())
                    margin = (col_max - col_min) * 0.2 if col_max != col_min else 10.0
                    with pred_cols[i % 4]:
                        val = st.number_input(
                            f"{col_name} 입력",
                            value=col_mean,
                            min_value=col_min - margin,
                            max_value=col_max + margin,
                            key=f"sim_{col_name}"
                        )
                        user_inputs.append(val)

                if st.button("🚀 인공신경망 예측 수행하기", use_container_width=True, key="ann_predict_btn"):
                    single_x_raw = np.array(user_inputs).reshape(1, -1)
                    single_x_scaled = (single_x_raw - X_mean) / X_std
                    s_a1 = activate(np.dot(single_x_scaled, W1) + b1, activation_name)
                    single_y_pred = (np.dot(s_a1, W2) + b2) * y_std + y_mean
                    st.info(f"✨ **인공신경망 예측 결과 ({ann_y_col}):** **{single_y_pred[0, 0]:.4f}**")
            else:
                st.warning("⚠️ 유효한 숫자 데이터가 6개 이상 필요합니다.")
    else:
        st.warning("⚠️ X축 입력 변수와 Y축 출력 변수를 각각 선택해 주세요.")


# =========================================================
# 3. 우측 하단 플로팅 말풍선 AI 챗봇 (OpenRouter)
# =========================================================
def get_api_key():
    try:
        return (
            st.secrets.get("API_KEY", "")
            or st.secrets.get("OPENROUTER_API_KEY", "")
            or st.secrets.get("DEEPSEEK_API_KEY", "")
        )
    except Exception:
        return ""

@st.cache_data(ttl=600, show_spinner=False)
def fetch_active_free_models():
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            models = json.loads(res.read().decode("utf-8")).get("data", [])

        free_models = []
        for m in models:
            mid = m.get("id", "")
            pricing = m.get("pricing", {})
            is_free = mid.endswith(":free") or (
                str(pricing.get("prompt", "1")) == "0"
                and str(pricing.get("completion", "1")) == "0"
            )
            if "deepseek" in mid.lower() and is_free and mid != "openrouter/free":
                free_models.append(mid)

        return ["openrouter/free"] + free_models
    except Exception:
        return ["openrouter/free"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "안녕하세요! **OpenRouter AI 분석 도우미**입니다. 😊\n\n현재 화면에서 실행한 **회귀 분석 수식**이나 **인공신경망(ANN) 예측 결과**에 대해 궁금한 점을 물어보시면 바로 해석해 드릴게요!"
        }
    ]

default_key = get_api_key()
available_models = fetch_active_free_models()

with st.popover("💬 AI 도우미"):
    st.markdown("### 🤖 OpenRouter AI 분석 도우미")
    st.caption("이전 대화 맥락과 현재 화면의 회귀·신경망 분석 결과를 기억하여 답변합니다.")

    with st.expander("⚙️ 모델 및 API 설정", expanded=False):
        api_key = st.text_input(
            "OpenRouter API Key",
            value=default_key,
            type="password",
            key="bubble_api_key",
            help="Streamlit Secrets에 등록했다면 자동으로 입력됩니다."
        )
        model_name = st.selectbox(
            "모델 선택 (기본값: openrouter/free)",
            available_models,
            index=0,
            key="bubble_model"
        )
        if st.button("🗑️ 대화 기억 지우기", use_container_width=True, key="bubble_clear"):
            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "content": "대화 기억을 초기화했습니다. 새로운 질문을 입력해 주세요!"
                }
            ]

    # 대화 내용이 표시될 스크롤 영역
    chat_container = st.container(height=320)

    # 말풍선 내부 질문 입력 폼
    with st.form(key="bubble_chat_form", clear_on_submit=True):
        f_col1, f_col2 = st.columns([4, 1])
        with f_col1:
            user_prompt = st.text_input(
                "질문 입력",
                placeholder="예: 방금 나온 회귀 수식과 R2 점수 해석해줘",
                label_visibility="collapsed"
            )
        with f_col2:
            send_clicked = st.form_submit_button("전송", use_container_width=True)

    if send_clicked and user_prompt.strip():
        if not api_key:
            st.error("⚠️ Streamlit Secrets 또는 [모델 및 API 설정]에 API Key를 입력해 주세요.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_prompt.strip()})

            reg_ctx = st.session_state.get("last_reg_summary", "아직 실행된 회귀 분석 결과 없음")
            ann_ctx = st.session_state.get("last_ann_summary", "아직 실행된 인공신경망 결과 없음")

            system_instruction = (
                "너는 데이터 처리 개론 웹 플랫폼에 내장된 유능한 AI 데이터 분석 어시스턴트야. "
                "사용자의 이전 대화 맥락과 앞서 내린 조건을 정확히 기억하고 일관되게 답변해야 해. "
                f"또한 현재 사용자가 웹 화면에서 실행 중인 분석 결과는 다음과 같으니, 관련 질문을 받으면 이 수치를 근거로 친절히 해석해 줘.\n"
                f"- [현재 회귀 분석 결과]: {reg_ctx}\n"
                f"- [현재 인공신경망(ANN) 결과]: {ann_ctx}"
            )

            api_messages = [{"role": "system", "content": system_instruction}] + st.session_state.chat_history

            try:
                client = OpenAI(
                    api_key=api_key.strip(),
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://streamlit.app",
                        "X-Title": "AI Data Platform Bubble Chatbot"
                    }
                )
                with st.spinner("답변을 생성 중입니다..."):
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=api_messages
                    )
                    reply = response.choices[0].message.content
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state.chat_history.pop()
                st.error(f"오류가 발생했습니다: {e}")

    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
