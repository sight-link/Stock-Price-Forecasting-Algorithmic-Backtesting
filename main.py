import matplotlib.pyplot as plt
from src.config import RETURN_THRESHOLD, ANNUAL_RF, TRADING_DAYS_PER_YEAR
from src.data_loader import full_data_pipeline
from src.model_train import train_logistic_regression, train_svr, train_lstm
from src.evaluate import calculate_metrics, plot_price_com
