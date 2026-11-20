# Этап 4. Валидация, модель и эксперименты — Student Performance

Предсказание индекса успеваемости студента (`Performance Index`) классическими алгоритмами машинного обучения.
Работа — 4-й этап проекта (после аудита и обработки данных, EDA и подготовки признаков).

## Задача
- **Тип:** регрессия (целевая переменная — индекс успеваемости 10–100).
- **Метрики:** RMSE (главная, в баллах индекса), MAE, R². Обоснование — в разделе 0 ноутбука.
- **Данные:** `data/Student_Performance_cleaned.csv`, 9 873 строки, 5 признаков (`Hours Studied`, `Previous Scores`, `Extracurricular Activities`, `Sleep Hours`, `Sample Question Papers Practiced`), пропусков и полных дубликатов нет.

## Схема эксперимента
| Часть | Доля | Назначение |
|---|---|---|
| fit | 60% | обучение при подборе гиперпараметров |
| val | 20% | валидация (пункты 3–4) |
| test | 20% | **одна** оценка лучшей модели (пункт 8) |

`SEED = 42` фиксируется для `random`, `numpy`, разбиений, `KFold` и всех моделей со случайностью. Кросс-валидация: `KFold(5, shuffle=True, random_state=42)` на train-части (fit + val).

Baseline: среднее, медиана, линейная регрессия только по `Previous Scores`.
Модели (по 6–8 значений гиперпараметра): **Ridge** (`alpha`), **KNN** (`n_neighbors`), **Random Forest** (`max_depth`), **Gradient Boosting** (`n_estimators`) — всего 26 конфигураций.

## Главные результаты
| Модель | CV RMSE (среднее ± std) | Тест RMSE | Тест MAE | Тест R² |
|---|---|---|---|---|
| Baseline: среднее | 19.19 ± 0.10 | 19.30 | 16.25 | −0.001 |
| Baseline: LinReg по Previous Scores | 7.73 ± 0.04 | 7.82 | 6.71 | 0.836 |
| **Ridge (alpha=0.01) — лучшая** | **2.04 ± 0.03** | **2.08** | **1.65** | **0.988** |

- Лучшая по кросс-валидации — линейная модель (Ridge); бустинг (CV RMSE 2.11), случайный лес (2.28) и KNN (2.98) её не обгоняют.
- Остатки лучшей модели нормальны, центрированы и не имеют структуры: оставшаяся ошибка ≈ 2 балла — шум данных, а не нехватка сложности модели.
- Полные таблицы, графики и выводы — в ноутбуке и папке `results/`.

## Структура репозитория
```
data/                 исходный подготовленный датасет
notebooks/            04_validation_models_experiments.ipynb — главный отчёт (код, вывод, графики, выводы)
src/stage4_experiments.py   тот же код в виде скрипта (формат «percent»; единый источник для ноутбука)
src/build_notebook.py       пересборка и выполнение ноутбука без Jupyter
results/              summary_table.csv/.md, validation_results.csv, cv_results.csv, test_results.csv
results/figures/      графики: метрики vs гиперпараметры (val и CV), остатки, важность признаков
```

## Как воспроизвести
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/04_validation_models_experiments.ipynb   # Run All
# либо без Jupyter:
python src/stage4_experiments.py          # печатает результаты и пересоздаёт results/
python src/build_notebook.py              # пересобирает ноутбук с выводами
```
Результаты детерминированы (`SEED = 42`). При другой версии scikit-learn последние знаки метрик могут слегка отличаться.
