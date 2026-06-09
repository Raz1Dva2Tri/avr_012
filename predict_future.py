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
    first_date = df['ds'].min()
    historical_years = sorted(df['ds'].dt.year.unique())
    
    print(f"\n[*] Период исторических данных: {first_date.strftime('%Y-%m-%d')} - {last_date.strftime('%Y-%m-%d')}")
    print(f"[*] Доступные годы в истории: {historical_years}")
    
    # 2. Интерактивное меню выбора года
    print("\nВыбери целевой год для прогноза:")
    print("1. Сделать стандартный прогноз (на 24 месяца вперед)")
    print("2. Сделать прогноз до конца 2024 года")
    print("3. Ввести свой год")
    
    choice = input("Введи номер варианта (1/2/3): ").strip()
    
    target_year = None
    current_horizon = HORIZON
    
    if choice == '2':
        target_year = 2024
    elif choice == '3':
        target_year = int(input("Введи интересующий год (например, 2025): ").strip())
    else:
        # Вариант 1: стандартный прогноз
        target_year = None
    
    # Расчет горизонта, если выбран конкретный год
    if target_year:
        target_date = pd.Timestamp(f"{target_year}-12-01")
        
        # Считаем разницу в месяцах между последней точкой и концом целевого года
        if target_date > last_date:
            current_horizon = ((target_date.year - last_date.year) * 12) + (target_date.month - last_date.month)
            print(f"\n[~] Целевой год {target_year} в БУДУЩЕМ относительно исторических данных")
        else:
            print(f"\n[~] Целевой год {target_year} уже присутствует в исторических данных")
            current_horizon = HORIZON
    
    print(f"\n[~] Рассчитанный горизонт прогнозирования для LightGBM: {current_horizon} мес.")
    
    # 3. Инициализация и запуск пайплайна
    pipeline = RocketLaunchPipeline()
    
    print("\n[Pipeline]: Обучение моделей на ВСЕХ исторических данных...")
    
    # Обучаем модели
    pipeline.ml_fcst.fit(df)
    pipeline.nhits.fit(df)
    
    print(f"\n[Pipeline]: Генерация прогноза...")
    
    # Получаем прогноз от LightGBM на нужный горизонт
    lgbm_full_forecast = pipeline.ml_fcst.predict(h=current_horizon)
    
    # Получаем прогноз от NHITS (всегда на HORIZON шагов вперед)
    nhits_full_forecast = pipeline.nhits.predict(h=current_horizon)
    
    # 4. Фильтруем результаты в зависимости от выбранного года
    if target_year:
        # LightGBM - фильтруем прогноз по году
        lgbm_filtered = lgbm_full_forecast[lgbm_full_forecast['ds'].dt.year == target_year].copy()
        
        # NHITS - ПРЯМАЯ ПРОВЕРКА: есть ли год в исторических данных?
        if target_year in historical_years:
            # Год есть в истории - берем фактические данные из df
            print(f"\n📂 NHITS: Беру ФАКТИЧЕСКИЕ данные за {target_year} год из history")
            nhits_filtered = df[df['ds'].dt.year == target_year][['unique_id', 'ds', 'y']].copy()
            nhits_filtered = nhits_filtered.rename(columns={'y': 'NHITS'})
        else:
            # Года нет в истории - берем прогноз NHITS
            print(f"\n🔮 NHITS: Беру ПРОГНОЗ на {target_year} год")
            nhits_filtered = nhits_full_forecast[nhits_full_forecast['ds'].dt.year == target_year].copy()
        
        # Объединяем прогнозы для сравнения
        combined_filtered = lgbm_filtered.merge(
            nhits_filtered[['unique_id', 'ds', 'NHITS']], 
            on=['unique_id', 'ds'], 
            how='outer'
        )
        
        # Формируем имена файлов с подписью года
        output_filename_lgbm = f"forecast_lightgbm_{target_year}.csv"
        output_filename_nhits = f"forecast_nhits_{target_year}.csv"
        output_filename_combined = f"forecast_combined_{target_year}.csv"
        
        print(f"\n🎯 Прогноз сформирован для {target_year} года")
        print(f"   LightGBM: {'прогноз' if not lgbm_filtered.empty else 'нет данных'}")
        print(f"   NHITS: {'исторические данные' if target_year in historical_years else 'прогноз' if not nhits_filtered.empty else 'нет данных'}")
        
    else:
        # Стандартный прогноз без фильтрации по году
        lgbm_filtered = lgbm_full_forecast
        nhits_filtered = nhits_full_forecast
        
        # Объединяем для сравнения
        combined_filtered = lgbm_filtered.merge(
            nhits_filtered[['unique_id', 'ds', 'NHITS']], 
            on=['unique_id', 'ds'], 
            how='outer'
        )
        
        # Формируем имена файлов для стандартного прогноза
        output_filename_lgbm = "forecast_lightgbm_future.csv"
        output_filename_nhits = "forecast_nhits_future.csv"
        output_filename_combined = "forecast_combined_future.csv"
        
        print(f"\n📈 Сформирован стандартный прогноз на {current_horizon} месяцев")
    
    # 5. Сохранение результатов
    output_dir = os.path.join("data", "forecasts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем прогноз LightGBM
    lgbm_path = os.path.join(output_dir, output_filename_lgbm)
    lgbm_filtered.to_csv(lgbm_path, index=False)
    print(f"\n✅ Прогноз LightGBM сохранен в: {lgbm_path}")
    print(f"   Содержит {len(lgbm_filtered)} записей")
    
    # Сохраняем прогноз NHITS
    nhits_path = os.path.join(output_dir, output_filename_nhits)
    if not nhits_filtered.empty:
        nhits_filtered.to_csv(nhits_path, index=False)
        print(f"✅ Прогноз NHITS сохранен в: {nhits_path}")
        print(f"   Содержит {len(nhits_filtered)} записей")
    else:
        print(f"⚠️ NHITS: Нет данных для сохранения (файл не создан)")
    
    # Сохраняем объединенный прогноз
    combined_path = os.path.join(output_dir, output_filename_combined)
    combined_filtered.to_csv(combined_path, index=False)
    print(f"✅ Объединенный прогноз сохранен в: {combined_path}")
    print(f"   Содержит {len(combined_filtered)} записей")
    
    # 6. Вывод результатов
    print("\n" + "="*60)
    if target_year:
        print(f"📊 РЕЗУЛЬТАТЫ ДЛЯ {target_year} ГОДА:")
    else:
        print(f"📊 ПРОГНОЗ НА {current_horizon} МЕСЯЦЕВ ВПЕРЕД:")
    print("="*60)
    
    if not lgbm_filtered.empty:
        print("\n🔵 LightGBM прогноз:")
        # Показываем первые 12 записей или все, если их меньше
        display_rows = min(12, len(lgbm_filtered))
        print(lgbm_filtered[['ds', 'LightGBM']].head(display_rows).to_string(index=False))
        
        print(f"\n📈 Среднее значение LightGBM: {lgbm_filtered['LightGBM'].mean():.4f}")
        print(f"📈 Медианное значение LightGBM: {lgbm_filtered['LightGBM'].median():.4f}")
        print(f"📈 Стандартное отклонение LightGBM: {lgbm_filtered['LightGBM'].std():.4f}")
    else:
        print("\n⚠️ Нет данных для прогноза LightGBM")
    
    if not nhits_filtered.empty:
        print("\n🟢 NHITS данные:")
        display_rows = min(12, len(nhits_filtered))
        print(nhits_filtered[['ds', 'NHITS']].head(display_rows).to_string(index=False))
        
        print(f"\n📈 Среднее значение NHITS: {nhits_filtered['NHITS'].mean():.4f}")
        print(f"📈 Медианное значение NHITS: {nhits_filtered['NHITS'].median():.4f}")
        print(f"📈 Стандартное отклонение NHITS: {nhits_filtered['NHITS'].std():.4f}")
        
        # Если это исторические данные, покажем это
        if target_year and target_year in historical_years:
            print(f"\n💡 Примечание: Это ФАКТИЧЕСКИЕ исторические данные за {target_year} год")
    else:
        print("\n⚠️ Нет данных для NHITS")
    
    # 7. Пояснительный блок
    if target_year:
        print("\n" + "="*60)
        print("📌 ПОЯСНЕНИЕ ПО ДАННЫМ:")
        print("="*60)
        if target_year in historical_years:
            print(f"• {target_year} год присутствует в исторических данных")
            print("• LightGBM показывает ПРОГНОЗ (может отличаться от факта)")
            print("• NHITS показывает ФАКТИЧЕСКИЕ данные из истории")
            print("• Для прогноза на будущий год выберите год > максимального в истории")
        else:
            print(f"• {target_year} год отсутствует в исторических данных")
            print("• Обе модели показывают ПРОГНОЗ")
    
    print("\n" + "="*60)
    print("✨ ПРОГНОЗИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО ✨")
    print("="*60)

if __name__ == "__main__":
    run_menu_forecast()