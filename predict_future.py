# predict_future.py
import os
import pandas as pd
from src.config import HORIZON
from src.data_prep import load_and_prepare_data
from src.pipeline import RocketLaunchPipeline

def run_menu_forecast():
    print("="*60)
    print("🚀 МОДУЛЬ ПРОГНОЗИРОВАНИЯ КОСМИЧЕСКИХ ЗАПУСКОВ В БУДУЩЕЕ")
    print("="*60)
    
    # 1. Загрузка данных для определения последней точки
    data_path = os.path.join("data", "space_corrected.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "dataset.csv")
        
    df = load_and_prepare_data(data_path)
    last_date = df['ds'].max()
    
    print(f"\n[*] Последняя историческая дата в датасете: {last_date.strftime('%Y-%m-%d')}")
    
    # 2. Интерактивное меню выбора года
    print("\nВыбери целевой год для прогноза:")
    print("1. Сделать стандартный прогноз (на 24 месяца вперед)")
    print("2. Сделать прогноз до конца 2024 года")
    print("3. Ввести свой год")
    
    choice = input("Введи номер варианта (1/2/3): ").strip()
    
    if choice == '2':
        target_year = 2024
    elif choice == '3':
        target_year = int(input("Введи интересующий год (например, 2025): ").strip())
    else:
        target_year = None
        current_horizon = HORIZON

    # Расчет горизонта, если выбран конкретный год
    if target_year:
        target_date = pd.Timestamp(f"{target_year}-12-01")
        if target_date <= last_date:
            print("❌ Ошибка: Выбранный год уже есть в исторической базе данных!")
            return
        
        # Считаем разницу в месяцах между последней точкой и концом целевого года
        current_horizon = ((target_date.year - last_date.year) * 12) + (target_date.month - last_date.month)
    
    print(f"\n[~] Расчитанный горизонт прогнозирования: {current_horizon} мес.")
    
    # 3. Инициализация и запуск пайплайна
    pipeline = RocketLaunchPipeline()
    
    print("\n[Pipeline]: Обучение моделей на ВСЕХ исторических данных...")
    pipeline.ml_fcst.fit(df)
    
    print(f"\n[Pipeline]: Генерация прогноза вперед на {current_horizon} месяцев...")
    future_forecast = pipeline.ml_fcst.predict(h=current_horizon)
    
    # 4. Фильтруем результаты, если заказывали конкретный 2024 год
    if target_year:
        # Оставляем в финальном выводе только строчки за выбранный год
        final_forecast = future_forecast[future_forecast['ds'].dt.year == target_year].copy()
        output_filename = f"forecast_{target_year}.csv"
    else:
        final_forecast = future_forecast
        output_filename = "future_space_forecast.csv"
        
    # 5. Сохранение результатов
    output_path = os.path.join("data", output_filename)
    final_forecast.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print(f"✅ Успешно! Прогноз сохранен в: {output_path}")
    print("="*60)
    
    print(f"\nФрагмент прогноза на {target_year if target_year else 'запрошенный период'}:")
    print(final_forecast.to_string(index=False))

if __name__ == "__main__":
    run_menu_forecast()