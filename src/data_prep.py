# src/data_prep.py
import pandas as pd
from src.config import RAW_DATE_COL, DATE_COL, TARGET_COL, ID_COL




def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    print(f"[*] Чтение сырых данных из {filepath}...")
    # Загружаем файл
    df = pd.read_csv(filepath)
    
    # 1. Извлекаем дату и приводим к формату datetime
    df[DATE_COL] = pd.to_datetime(df[RAW_DATE_COL], utc=True, errors='coerce')
    df = df.dropna(subset=[DATE_COL])
    df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)
    
    print("[*] Агрегация данных: подсчет количества запусков по месяцам...")
    # 2. Группируем по фактическим датам запусков
    df_monthly = df.groupby(df[DATE_COL].dt.to_period('M')).size().reset_index(name=TARGET_COL)
    df_monthly[DATE_COL] = df_monthly[DATE_COL].dt.to_timestamp()
    
    print("[*] Заполнение пропущенных месяцев нулями...")
    # 3. Делаем дату индексом, чтобы применить resample
    df_monthly = df_monthly.set_index(DATE_COL)
    
    # Ресемплинг ('MS') находит все дыры в датах от самой первой до самой последней 
    # и заполняет их нулями (0 запусков в пустые месяцы)
    df_monthly = df_monthly.resample('MS').asfreq().fillna(0)
    
    # Возвращаем дату обратно в колонку
    df_monthly = df_monthly.reset_index()
    
    # 4. Добавляем обязательный уникальный идентификатор ряда
    df_monthly[ID_COL] = 'Rocket_Launch'
    
    # Сортируем от старых к новым
    df_monthly = df_monthly.sort_values(DATE_COL).reset_index(drop=True)
    
    print(f"✅ Данные успешно восстановлены без пропусков!")
    print(f"[*] Период: с {df_monthly[DATE_COL].min().strftime('%Y-%m')} по {df_monthly[DATE_COL].max().strftime('%Y-%m')}")
    print(f"[*] Всего месяцев для обучения (включая пустые): {len(df_monthly)}")
    
    return df_monthly