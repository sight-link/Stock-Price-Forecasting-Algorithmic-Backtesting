import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from src.config import SVR_C, SVR_GAMMA, SVR_EPS, SVR_KERNEL, LSTM_UNITS, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH

def train_logistic_regression(X_train, X_test, y_train_class, actual_today):
    print("[4/6] 正在训练分类模型: 逻辑回归 (Logistic Regression)...")
    clf_model = LogisticRegression(max_iter=1000)
    clf_model.fit(X_train, y_train_class)
    clf_pred_class = clf.predict(X_test)
    lr_pred_price = np.where(clf_pred_class == 1, actual_today * 1.006, actual_today * 0.998).reshape(-1, 1)
    return lr_pred_price

def train_svr(X_train, X_test, y_train_price, scaler_y):
    print("[5/6] 正在训练回归模型: 支持向量回归 (SVR)...")
    svr_model = SVR(kernel=SVR_KERNEL, C=SVR_C, gamma=SVR_GAMMA, epsilon=SVR_EPS)
    svr_model.fit(X_train, y_train_price)
    svr_pred_scaled = svr_model.predict(X_test)
    svr_pred = scaler_y.inverse_transform(svr_pred_scaled.reshape(-1, 1))
    return svr_pred

def train_lstm(X_train, X_test, y_train_price, scaler_y):
    print("[6/6] 正在构建并训练时序模型: LSTM 网络...")
    has_lstm = False
    lstm_pred = None
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_train.shape[1]))

        lstm_model = Sequential([
            Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
            LSTM(LSTM_UNITS, activation="relu", return_sequences=False),
            Dropout(LSTM_DROPOUT),
            Dense(1)
        ])
        lstm_model.compile(optimizer="adam", loss="mse")
        lstm_model.fit(X_train_lstm, y_train_price, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=0)
        lstm_pred_scaled = lstm_model.predict(X_test_lstm, verbose=0)
        lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)
        has_lstm = True
    except ImportError:
        print("TensorFlow 未检测到，跳过 LSTM 评估")
    return lstm_pred, has_lstm
