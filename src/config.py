# Global fixed hyperparameters & time range
START_DATE = "2016-01-01"
END_DATE = "2026-01-01"
TRAIN_SPLIT_RATIO = 0.85
RETURN_THRESHOLD = 0.005  # 0.5% trading signal threshold
ANNUAL_RF = 0.02
TRADING_DAYS_PER_YEAR = 252

# Moving average window
MA10_WINDOW = 10
MA50_WINDOW = 50

# SVR hyperparams
SVR_KERNEL = "rbf"
SVR_C = 10.0
SVR_GAMMA = 0.1
SVR_EPS = 0.01

# LSTM config
LSTM_UNITS = 50
LSTM_DROPOUT = 0.1
LSTM_EPOCHS = 10
LSTM_BATCH = 32

# Plot size
FIG_WIDTH = 14
FIG_HEIGHT = 7
