# src/pipeline.py
import pandas as pd
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from window_ops.rolling import rolling_mean
from mlforecast.target_transforms import Differences
from src.config import FREQ, HORIZON, RANDOM_SEED

class RocketLaunchPipeline:
    def __init__(self):
        # Инициализируем ML-модели и Feature Engineering
        self.ml_fcst = MLForecast(
            models={
                'LightGBM': LGBMRegressor(random_state=RANDOM_SEED, verbose=-1)
                # Сюда можно добавить CatBoost и другие
            },
            freq=FREQ,
            target_transforms=[Differences([1])],
            lags=[1, 2, 3, 12],
            lag_transforms={1: [(rolling_mean, 6), (rolling_mean, 12)]},
            date_features=['month']
        )
        # Здесь также можно добавить инициализацию NeuralForecast

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обучает пайплайн и возвращает прогноз"""
        print("[Pipeline]: Разделение данных...")
        train = df.iloc[:-HORIZON].copy()
        test = df.iloc[-HORIZON:].copy()

        print("[Pipeline]: Обучение моделей...")
        self.ml_fcst.fit(train)
        
        print("[Pipeline]: Генерация прогноза...")
        forecast = self.ml_fcst.predict(h=HORIZON)
        
        # Склейка с фактом для удобства
        result = test[['ds', 'y']].merge(forecast, on='ds', how='left')
        return result