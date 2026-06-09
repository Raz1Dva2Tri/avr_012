import pandas as pd
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from window_ops.rolling import rolling_mean
from mlforecast.target_transforms import Differences

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from src.config import FREQ, HORIZON, RANDOM_SEED

class RocketLaunchPipeline:
    def __init__(self):
        """Инициализация пайплайна с двумя типами моделей."""
        
        # 1. Настройка MLForecast (на базе LightGBM)
        self.ml_fcst = MLForecast(
            models={
                'LightGBM': LGBMRegressor(random_state=RANDOM_SEED, verbose=-1)
            },
            freq=FREQ,
            target_transforms=[Differences([1])], # Удаление тренда
            lags=[1, 2, 3, 12],
            lag_transforms={1: [(rolling_mean, 6), (rolling_mean, 12)]},
            date_features=['month']
        )
        
        # 2. Настройка NeuralForecast (на базе NHITS)
        # input_size устанавливаем равным горизонту или больше (например, 2*HORIZON)
        self.nhits = NeuralForecast(
            models=[NHITS(input_size=HORIZON, h=HORIZON, max_steps=100)],
            freq=FREQ
        )

    def fit_predict_lightgbm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обучение и прогноз с помощью градиентного бустинга."""
        print("[Pipeline]: Обучение LightGBM...")
        self.ml_fcst.fit(df)
        
        print("[Pipeline]: Генерация прогноза LightGBM...")
        forecast = self.ml_fcst.predict(h=HORIZON)
        return forecast

    def fit_predict_nhits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обучение и прогноз с помощью нейросети NHITS."""
        print("[Pipeline]: Обучение NHITS...")
        # Нейросети в NeuralForecast ожидают формат ['unique_id', 'ds', 'y']
        self.nhits.fit(df)
        
        print("[Pipeline]: Генерация прогноза NHITS...")
        forecast = self.nhits.predict()
        return forecast

    def run_all(self, df: pd.DataFrame):
        """Универсальный метод для получения прогнозов от обеих моделей."""
        lgbm_res = self.fit_predict_lightgbm(df)
        nhits_res = self.fit_predict_nhits(df)
        
        # Объединяем результаты для сравнения
        return lgbm_res.merge(nhits_res, on=['unique_id', 'ds'], how='outer')

#