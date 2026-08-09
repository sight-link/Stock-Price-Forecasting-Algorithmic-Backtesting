import pandas as pd
import numpy as np
from src.config import RETURN_THRESHOLD, ANNUAL_RF, TRADING_DAYS_PER_YEAR

def manual_backtest_calc(y_test_true, svr_pred, test_dates):
    print("\n正在启动模拟交易回测引擎，计算金融实战指标...")
    df_backtest = pd.DataFrame({
        "Actual_Next": y_test_true.ravel(),
        "Pred_Next": svr_pred.ravel()
    }, index=test_dates)
    df_backtest["Actual_Today"] = df_backtest["Actual_Next"].shift(1)
    df_backtest["Actual_Today"] = df_backtest.bfill()
    df_backtest["Pred_Return"] = (df_backtest["Pred_Next"] - df_backtest["Actual_Today"]) / df_backtest["Actual_Today"]
    df_backtest["Signal"] = np.where(df_backtest["Pred_Return"] > RETURN_THRESHOLD, 1, 0)
    df_backtest["Market_Return"] = (df_backtest["Actual_Next"] - df_backtest["Actual_Today"]) / df_backtest["Actual_Today"]
    df_backtest["Strategy_Return"] = df_backtest["Signal"].shift(1) * df_backtest["Market_Return"]
    df_backtest["Strategy_Return"].fillna(0, inplace=True)
    df_backtest["Cum_Strategy"] = (1 + df_backtest["Strategy_Return"]).cumprod() - 1

    total_return = df_backtest["Cum_Strategy"].iloc[-1]
    daily_rf = ANNUAL_RF / TRADING_DAYS_PER_YEAR
    excess_returns = df_backtest["Strategy_Return"] - daily_rf
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if excess_returns.std() != 0 else 0.0

    cum_returns_plus_one = df_backtest["Cum_Strategy"] + 1
    running_max = cum_returns_plus_one.cummax()
    max_drawdown = ((cum_returns_plus_one - running_max) / running_max).min()
    return df_backtest, total_return, sharpe_ratio, max_drawdown

# Optional Backtrader module
try:
    import backtrader as bt
    class SvrSignalStrategy(bt.Strategy):
        def __init__(self):
            self.signal = self.datas[0].signal
        def next(self):
            if self.signal[0] == 1 and not self.position:
                self.buy(size=self.broker.get_cash() / self.data.close[0])
            elif self.signal[0] == 0 and self.position:
                self.close()

    class PandasDataFeed(bt.feeds.PandasData):
        lines = ("signal",)
        params = (("signal", -1),)

    def run_backtrader_simulation(df_backtest, initial_cap=100000.0):
        df_bt_input = pd.DataFrame({
            "open": df_backtest["Actual_Today"],
            "high": df_backtest["Actual_Next"],
            "low": df_backtest["Actual_Today"],
            "close": df_backtest["Actual_Next"],
            "volume": 0.0,
            "signal": df_backtest["Signal"]
        }, index=df_backtest.index)
        cerebro = bt.Cerebro()
        data_feed = PandasDataFeed(dataname=df_bt_input)
        cerebro.adddata(data_feed)
        cerebro.addstrategy(SvrSignalStrategy)
        cerebro.broker.setcash(initial_cap)
        print(f'注入 Backtrader 引擎成功！初始资产净值: {cerebro.broker.getvalue():.2f}')
        cerebro.run()
        final_val = cerebro.broker.getvalue()
        print(f'Backtrader 回测结束！期末资产净值: {final_val:.2f}')
        return final_val
except ImportError:
    def run_backtrader_simulation(*args):
        print("Backtrader 未安装，跳过事件驱动回测")
        return None
