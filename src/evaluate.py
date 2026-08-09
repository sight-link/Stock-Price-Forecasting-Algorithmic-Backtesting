import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.config import FIG_WIDTH, FIG_HEIGHT

def calculate_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} -> RMSE: ${rmse:.4f} USD | R²: {r2:.4f}")
    return rmse, r2

def plot_price_comparison(test_dates, y_test_true, lr_pred, svr_pred, lstm_pred, lstm_available):
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.plot(test_dates, y_test_true, label="真实市场收盘价", color="black", linewidth=1.5)
    plt.plot(test_dates, lr_pred, label="逻辑回归预测价 (分类模型)", color="green", linestyle="-.", alpha=0.7)
    plt.plot(test_dates, svr_pred, label="SVR 预测价 (回归模型)", color="crimson", linestyle="--", linewidth=1.2)
    if lstm_available:
        plt.plot(test_dates, lstm_pred, label="LSTM 预测价 (时序模型)", color="royalblue", linestyle=":")
    plt.title("标普500指数ETF (SPY) 跨领域多模型预测对比", fontsize=14, fontweight="bold")
    plt.xlabel("交易日期")
    plt.ylabel("价格 (USD)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="upper left")
    plt.gcf().autofmt_xdate()
    plt.show()
