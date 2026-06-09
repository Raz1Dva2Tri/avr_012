# main.py
import os
from src.data_prep import load_and_prepare_data
from src.pipeline import RocketLaunchPipeline

def main():
    # 1. Пути к файлам
    input_file = 'data/space_corrected.csv'
    output_file = 'data/space_results.csv'
    
    # Проверка наличия файла
    if not os.path.exists(input_file):
        print(f"❌ Ошибка: Положите ваш файл с данными по пути {input_file}")
        return

    # 2. Запуск процесса
    df = load_and_prepare_data(input_file)
    
    pipeline = RocketLaunchPipeline()
    results_df = pipeline.fit_predict(df)
    
    # 3. Сохранение результата
    results_df.to_csv(output_file, index=False)
    print(f"✅ Готово! Прогноз успешно сохранен в {output_file}")
    
    # Вывод первых строк для проверки
    print("\nФрагмент прогноза:")
    print(results_df.head())

if __name__ == "__main__":
    main()