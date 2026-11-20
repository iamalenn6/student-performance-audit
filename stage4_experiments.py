# %% [markdown]
# # Этап 4. Валидация, модель и эксперименты
# **Датасет:** `Student_Performance_cleaned.csv` (успеваемость студентов, после аудита, очистки и подготовки признаков на этапах 1–3).
#
# **Что делаем в этом ноутбуке (по пунктам задания):**
# 0. Определяем тип задачи и метрики, фиксируем `seed`, делим данные на train / validation / test.
# 1. Baseline (наивные модели для сравнения).
# 2. Обучаем 4 семейства алгоритмов, для каждого перебираем 6–8 значений ключевого гиперпараметра.
# 3. Оцениваем на валидационной части обучающих данных (**тест не трогаем**).
# 4. Графики «метрика – гиперпараметр» + выводы.
# 5–6. Повторяем всё с 5-кратной кросс-валидацией: среднее и стандартное отклонение, графики.
# 7. Итоговая таблица.
# 8. Лучшая модель: обучение на всех train-данных, **единственная** оценка на тесте.
# 9. Анализ ошибок (график остатков).

# %%
# ---------- 0. Импорты и воспроизводимость ----------
import os, random, warnings
from pathlib import Path

SEED = 42                                    # единый seed для всего эксперимента
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

import numpy as np
np.random.seed(SEED)
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

try:
    display
except NameError:                            # запуск как обычного .py-скрипта
    def display(x):
        print(x.to_string() if hasattr(x, "to_string") else x)

ROOT = Path.cwd()
if not (ROOT / "data").exists() and (ROOT.parent / "data").exists():
    ROOT = ROOT.parent
FIG, RES = ROOT / "results" / "figures", ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
print(f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}; SEED = {SEED}")

# %%
# ---------- Загрузка данных ----------
df = pd.read_csv(ROOT / "data" / "Student_Performance_cleaned.csv")
TARGET = "Performance Index"
print("Размер:", df.shape, "| пропусков:", int(df.isna().sum().sum()), "| полных дубликатов:", int(df.duplicated().sum()))
display(df.describe().T.round(2))
X, y = df.drop(columns=TARGET), df[TARGET]

# %% [markdown]
# ## 0. Тип задачи и метрики
#
# **Тип задачи — регрессия.** Целевая переменная `Performance Index` — числовой индекс успеваемости от 10 до 100
# (91 различное значение, упорядоченная количественная шкала, расстояние между значениями имеет смысл).
# Превращать её в классы значило бы терять информацию, поэтому предсказываем число.
#
# **Метрики** (обоснование; сверьте формулировки с вашим паспортом датасета и при необходимости добавьте ссылки на него):
#
# | Метрика | Зачем | Как читать |
# |---|---|---|
# | **RMSE** (главная) | В тех же единицах, что и целевая переменная (баллы индекса). Квадрат штрафует крупные промахи сильнее мелких: сильно ошибиться в оценке студента хуже, чем несколько раз ошибиться чуть-чуть. | Типичная величина ошибки в баллах. Меньше – лучше. |
# | **MAE** | Средний модуль ошибки. Прозрачна для интерпретации и меньше зависит от редких выбросов. Если MAE ≈ RMSE, то грубых промахов почти нет. | «В среднем ошибаемся на N баллов». |
# | **R²** | Доля дисперсии цели, объяснённая моделью. Безразмерна, а значит удобна для сравнения с baseline: модель «всегда предсказывать среднее» даёт R² ≈ 0. | 0 – не лучше среднего, 1 – идеал. |
#
# Не используем accuracy/F1 (это метрики классификации) и MAPE (не нужна: цель не бывает около нуля, а относительная ошибка здесь не важна).
# **Модели сравниваем по RMSE**, MAE и R² – для контроля и интерпретации.

# %%
# ---------- Разбиение данных ----------
# 80% – train_full (на нём живут и валидация, и кросс-валидация), 20% – test (хранится «в сейфе» до пункта 8).
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
# Из train_full выделяем валидацию (25% от него = 20% от всех данных).
X_fit, X_val, y_fit, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=SEED)

split = pd.DataFrame({"строк": [len(X_fit), len(X_val), len(X_test)],
                      "доля": [len(X_fit) / len(X), len(X_val) / len(X), len(X_test) / len(X)],
                      "среднее цели": [y_fit.mean(), y_val.mean(), y_test.mean()],
                      "std цели": [y_fit.std(), y_val.std(), y_test.std()]},
                     index=["fit (обучение)", "val (валидация)", "test (тест, не трогаем)"]).round(3)
display(split)

# %% [markdown]
# **Схема данных.** Данные делятся один раз с фиксированным `random_state=42`:
#
# ```
# все данные ─┬─ train_full (80%) ─┬─ fit (60%)   – обучение при подборе гиперпараметров (пункты 2–4)
#             │                    └─ val (20%)   – валидация (пункты 3–4)
#             │                    (вся train_full → 5-кратная кросс-валидация, пункты 5–6, и итоговое обучение, пункт 8)
#             └─ test (20%)        – одна оценка лучшей модели в пункте 8
# ```
# Среднее и разброс цели в трёх частях почти совпадают, значит разбиение репрезентативно.

# %%
# ---------- Вспомогательные функции ----------
def metrics(y_true, y_pred):
    return {"rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)}

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}

def cv_scores(model, Xa, ya):
    """5-кратная кросс-валидация: среднее и std метрик на валидационных фолдах (+ среднее на обучающих)."""
    r = cross_validate(model, Xa, ya, cv=CV, scoring=SCORING, return_train_score=True, n_jobs=1)
    out = {}
    for m in ("rmse", "mae", "r2"):
        sign = -1 if m != "r2" else 1
        out[f"cv_{m}_mean"] = sign * r[f"test_{m}"].mean()
        out[f"cv_{m}_std"] = r[f"test_{m}"].std()
        out[f"cv_train_{m}_mean"] = sign * r[f"train_{m}"].mean()
    return out

# %% [markdown]
# ## 1. Baseline
# Baseline нужен, чтобы понимать, **что значит «хороший» результат**. Берём три ориентира разной силы:
#
# 1. `Mean` – всегда предсказывать среднее обучающей выборки (**основной baseline**, нижняя граница; R² ≈ 0).
# 2. `Median` – то же с медианой.
# 3. `LinReg по Previous Scores` – «простое правило»: смотрим только на предыдущие баллы. Это уже содержательный ориентир: любая серьёзная модель обязана его обыграть.
#
# Все baseline обучаются на тех же `fit`-данных и проверяются на `val`.

# %%
baselines = {
    "Baseline: Mean": DummyRegressor(strategy="mean"),
    "Baseline: Median": DummyRegressor(strategy="median"),
    "Baseline: LinReg по Previous Scores": Pipeline([
        ("sel", ColumnTransformer([("p", "passthrough", ["Previous Scores"])])),
        ("lr", LinearRegression())]),
}
rows = []
for name, mdl in baselines.items():
    mdl.fit(X_fit, y_fit)
    rows.append({"Модель": name, **{f"val_{k}": v for k, v in metrics(y_val, mdl.predict(X_val)).items()},
                 **cv_scores(mdl, X_train_full, y_train_full)})
baseline_df = pd.DataFrame(rows)
display(baseline_df[["Модель", "val_rmse", "val_mae", "val_r2"]].round(3))

# %% [markdown]
# ## 2. Модели и сетки гиперпараметров
# Четыре семейства алгоритмов, принципиально по-разному устроенных:
#
# | Модель | Идея | Варьируемый гиперпараметр | Что он регулирует |
# |---|---|---|---|
# | **Ridge** (линейная с L2-штрафом) | Взвешенная сумма признаков | `alpha` | Сила штрафа за большие коэффициенты |
# | **KNN** | Усреднение целей ближайших похожих студентов | `n_neighbors` | Сколько соседей усредняем (гладкость) |
# | **Random Forest** | Усреднение множества решающих деревьев | `max_depth` | Глубина дерева (сложность) |
# | **Gradient Boosting** | Деревья, последовательно исправляющие ошибки друг друга | `n_estimators` | Число шагов бустинга |
#
# Для Ridge и KNN признаки стандартизуются внутри `Pipeline` (`StandardScaler`), иначе «баллы» и «часы сна» будут влиять на расстояния и штраф несравнимо.
# Масштабирование внутри `Pipeline` ещё и защищает от **утечки данных**: на каждой валидации/фолде оно обучается только на обучающей части.
# У всех моделей со случайностью зафиксирован `random_state=42`.

# %%
GRID = {
    "Ridge": dict(param="alpha", values=[0.01, 1, 10, 100, 1000, 10000], log=True,
                  build=lambda v: Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=v, random_state=SEED))]),
                  fixed="StandardScaler"),
    "KNN": dict(param="n_neighbors", values=[1, 3, 5, 10, 20, 50, 100, 200], log=True,
                build=lambda v: Pipeline([("scaler", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=v))]),
                fixed="weights=uniform, StandardScaler"),
    "Random Forest": dict(param="max_depth", values=[2, 4, 6, 8, 12, None], log=False,
                          build=lambda v: RandomForestRegressor(n_estimators=100, max_depth=v, random_state=SEED, n_jobs=-1),
                          fixed="n_estimators=100"),
    "Gradient Boosting": dict(param="n_estimators", values=[10, 25, 50, 100, 200, 400], log=True,
                              build=lambda v: GradientBoostingRegressor(n_estimators=v, learning_rate=0.1, max_depth=3, random_state=SEED),
                              fixed="learning_rate=0.1, max_depth=3"),
}
n_cfg = sum(len(g["values"]) for g in GRID.values())
print(f"Моделей: {len(GRID)}, всего конфигураций: {n_cfg}")
for k, g in GRID.items():
    print(f"  {k:18s} {g['param']:13s} = {g['values']}   (фиксировано: {g['fixed']})")

# %% [markdown]
# ## 3. Оценка на валидационной части (без кросс-валидации)
# Для каждой конфигурации: обучаем на `fit`, считаем метрики на `fit` (train) и на `val`.
# Сравнение train и val показывает **недо- и переобучение**: если на train ошибка сильно меньше, чем на val, модель запоминает данные.

# %%
val_rows = []
for mname, g in GRID.items():
    for v in g["values"]:
        mdl = g["build"](v).fit(X_fit, y_fit)
        tr, va = metrics(y_fit, mdl.predict(X_fit)), metrics(y_val, mdl.predict(X_val))
        val_rows.append({"model": mname, "param": g["param"], "value": str(v),
                         **{f"train_{k}": x for k, x in tr.items()}, **{f"val_{k}": x for k, x in va.items()}})
val_df = pd.DataFrame(val_rows)
for mname in GRID:
    print(f"\n{mname} (по {GRID[mname]['param']})")
    display(val_df[val_df.model == mname][["value", "train_rmse", "val_rmse", "val_mae", "val_r2"]].round(4).reset_index(drop=True))

# %% [markdown]
# ## 4. Графики: метрики в зависимости от гиперпараметров (валидация)
# Сплошная линия – валидация, пунктир – обучающая часть. Три строки: RMSE, MAE, R².

# %%
LABEL = {"rmse": "RMSE (баллы)", "mae": "MAE (баллы)", "r2": "R²"}

def xpos(g):
    vals = g["values"]
    if any(v is None for v in vals):
        return list(range(len(vals))), [str(v) for v in vals]
    return vals, None

def plot_sweeps(table, kind, fname, title):
    """kind = 'val' (одно разбиение) или 'cv' (кросс-валидация, с полосой ±std)."""
    fig, axes = plt.subplots(3, 4, figsize=(17, 9.5))
    for j, (mname, g) in enumerate(GRID.items()):
        sub = table[table.model == mname].reset_index(drop=True)
        x, ticks = xpos(g)
        for i, m in enumerate(("rmse", "mae", "r2")):
            ax = axes[i, j]
            if kind == "val":
                ax.plot(x, sub[f"val_{m}"], "o-", color="C0", label="validation")
                ax.plot(x, sub[f"train_{m}"], "s--", color="C1", alpha=.8, label="train")
            else:
                mu, sd = sub[f"cv_{m}_mean"].to_numpy(), sub[f"cv_{m}_std"].to_numpy()
                ax.plot(x, mu, "o-", color="C0", label="CV: среднее по 5 фолдам")
                ax.fill_between(x, mu - sd, mu + sd, color="C0", alpha=.25, label="±1 std")
                ax.plot(x, sub[f"cv_train_{m}_mean"], "s--", color="C1", alpha=.8, label="train (среднее)")
            if g["log"]: ax.set_xscale("log")
            if ticks: ax.set_xticks(x); ax.set_xticklabels(ticks)
            if i == 0: ax.set_title(f"{mname}\n(по {g['param']})", fontsize=10)
            if i == 2: ax.set_xlabel(g["param"])
            if j == 0: ax.set_ylabel(LABEL[m])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130, bbox_inches="tight")
    plt.show()

plot_sweeps(val_df, "val", "01_validation_metrics_vs_hyperparams.png",
            "Валидация: качество в зависимости от гиперпараметра (сплошная – val, пунктир – train)")

# %% [markdown]
# ### Выводы по пункту 4 (валидация)
# **Ориентиры.** Baseline «всегда среднее» даёт RMSE ≈ 19.1 и R² ≈ 0: среднее ничего не знает о студенте. Уже один признак `Previous Scores` снижает RMSE до 7.7 (R² 0.84). Всё, что хуже 7.7, смысла не имеет.
#
# **Ridge (`alpha`).** При `alpha` от 0.01 до 100 кривая плоская (RMSE ≈ 2.02–2.03), train и val совпадают. Штраф начинает вредить при `alpha ≥ 1000`: RMSE 3.4, а при 10 000 – 12.1 (R² падает с 0.99 до 0.60). Это **недообучение**: слишком сильный штраф прижимает коэффициенты к нулю, модель перестаёт пользоваться признаками, и ошибка велика и на train, и на val. Вывод: этим данным регуляризация не нужна, у линейной модели с 5 признаками и ~6 тыс. строк нет проблемы переобучения.
#
# **KNN (`n_neighbors`).** U-образная кривая – классический компромисс смещения и разброса. При k=1 train RMSE 0.60, а val 4.23: **переобучение** (модель повторяет единственного соседа вместе с его шумом). При k ≥ 50 train и val сходятся, но обе растут: **недообучение** (усредняются слишком разные студенты). Оптимум k ≈ 10–20 (val RMSE 2.94–2.96).
#
# **Random Forest (`max_depth`).** При глубине 2 RMSE 8.5 (недообучение: 4 листа не описывают плавную зависимость). Быстрое улучшение до глубины 8 (RMSE 2.26). Дальше train продолжает падать (1.98 → 1.13 → 0.94), а val слегка растёт (2.26 → 2.32 → 2.35): начинается переобучение, но мягкое, так как усреднение 100 деревьев его сглаживает.
#
# **Gradient Boosting (`n_estimators`).** 10 деревьев – недообучение (RMSE 7.9), быстрый спад до 50–100 деревьев, затем плато (100: 2.101; 200: 2.100; 400: 2.112). Больше 200 шагов ничего не даёт; train и val почти совпадают, то есть при `learning_rate=0.1`, `max_depth=3` переобучения практически нет.
#
# **Общее.** Три принципиально разные модели независимо упираются в потолок ≈ 2.0–2.1 RMSE, причём лучшая из деревьев (бустинг, 2.10) не обгоняет простую линейную (2.02). Это намёк на то, что остаточная ошибка – шум, а не нехватка сложности модели.

# %% [markdown]
# ## 5–6. Повторяем эксперимент с 5-кратной кросс-валидацией
# **Зачем.** Одно разбиение на `val` даёт одну оценку, и она зависит от того, какие именно 20% строк попали в валидацию.
# `KFold(n_splits=5, shuffle=True, random_state=42)` делит `train_full` на 5 частей; каждая часть по очереди становится валидацией,
# остальные четыре – обучением. Получаем 5 оценок: **среднее** показывает качество, **std** – насколько оно устойчиво.
# Тестовая выборка по-прежнему не используется.

# %%
cv_rows = []
for mname, g in GRID.items():
    for v in g["values"]:
        cv_rows.append({"model": mname, "param": g["param"], "value": str(v),
                        **cv_scores(g["build"](v), X_train_full, y_train_full)})
cv_df = pd.DataFrame(cv_rows)
for mname in GRID:
    print(f"\n{mname} (по {GRID[mname]['param']})")
    display(cv_df[cv_df.model == mname][["value", "cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_mae_std", "cv_r2_mean", "cv_r2_std"]]
            .round(4).reset_index(drop=True))

# %%
plot_sweeps(cv_df, "cv", "02_cv_metrics_vs_hyperparams.png",
            "Кросс-валидация (5 фолдов): среднее ± std в зависимости от гиперпараметра")

# %% [markdown]
# ### Выводы по пунктам 5–6 (кросс-валидация)
# **Те же оптимумы, что и на валидации:** KNN – k ≈ 20, Random Forest – глубина 8, Gradient Boosting – 100–200 деревьев, Ridge – `alpha ≤ 10`. Значит, выводы не артефакт одного разбиения.
#
# **Устойчивость.** Std между фолдами для хороших конфигураций 0.025–0.06 RMSE (1–2% от значения), полоса ±std на графиках узкая. Больше всего «шумят» слабые модели (Random Forest с глубиной 2: 0.115). Метрики на val и среднее по CV отличаются у хороших моделей не более чем на ~0.05 – валидационное разбиение было репрезентативным.
#
# **Зачем нужна кросс-валидация – наглядный пример.** На одном val-разбиении лучшей оказалась Ridge с `alpha=10` (RMSE 2.0178), а по CV значения `alpha` = 0.01, 1 и 10 неразличимы (2.0375, 2.0375, 2.0377: разница 0.0002 при стандартной ошибке 0.013). Различия в третьем знаке на одном разбиении – случайность, и только CV позволяет это увидеть.
#
# **Ранжирование семейств по CV RMSE:** Ridge 2.04 < Gradient Boosting 2.11 < Random Forest 2.28 < KNN 2.98 (лучшие конфигурации). Разрыв Ridge–бустинг (0.07) больше стандартного отклонения (≈ 0.03), то есть устойчив, но практически мал: 0.07 балла из 100.

# %% [markdown]
# ## 7. Итоговая таблица
# Объединяем: метрики на одном валидационном разбиении и среднее ± std по кросс-валидации. Baseline включён для сравнения.

# %%
def hp_string(mname, v):
    g = GRID[mname]
    return f"{g['param']}={v}; {g['fixed']}"

summary = []
for _, r in baseline_df.iterrows():
    summary.append({"Модель": r["Модель"], "Гиперпараметры": "—", "val RMSE": r.val_rmse, "val MAE": r.val_mae, "val R²": r.val_r2,
                    "CV RMSE": f"{r.cv_rmse_mean:.3f} ± {r.cv_rmse_std:.3f}", "CV MAE": f"{r.cv_mae_mean:.3f} ± {r.cv_mae_std:.3f}",
                    "CV R²": f"{r.cv_r2_mean:.4f} ± {r.cv_r2_std:.4f}"})
for (_, a), (_, b) in zip(val_df.iterrows(), cv_df.iterrows()):
    summary.append({"Модель": a.model, "Гиперпараметры": hp_string(a.model, a.value),
                    "val RMSE": a.val_rmse, "val MAE": a.val_mae, "val R²": a.val_r2,
                    "CV RMSE": f"{b.cv_rmse_mean:.3f} ± {b.cv_rmse_std:.3f}", "CV MAE": f"{b.cv_mae_mean:.3f} ± {b.cv_mae_std:.3f}",
                    "CV R²": f"{b.cv_r2_mean:.4f} ± {b.cv_r2_std:.4f}"})
summary_df = pd.DataFrame(summary)
summary_df[["val RMSE", "val MAE"]] = summary_df[["val RMSE", "val MAE"]].round(3)
summary_df["val R²"] = summary_df["val R²"].round(4)

def to_markdown(t):
    def fmt(col, v):
        return (f"{v:.4f}" if "R²" in col else f"{v:.3f}") if isinstance(v, float) else str(v)
    head = "| " + " | ".join(t.columns) + " |\n|" + "|".join("---" for _ in t.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(fmt(c, v) for c, v in zip(t.columns, row)) + " |" for row in t.itertuples(index=False))

summary_df.to_csv(RES / "summary_table.csv", index=False, encoding="utf-8-sig")
(RES / "summary_table.md").write_text(to_markdown(summary_df), encoding="utf-8")
val_df.to_csv(RES / "validation_results.csv", index=False)
cv_df.to_csv(RES / "cv_results.csv", index=False)
with pd.option_context("display.max_colwidth", 60, "display.width", 250):
    display(summary_df)

# %% [markdown]
# ## 8. Выбор лучшей модели и оценка на тесте
# **Критерий выбора:** минимальный средний RMSE по кросс-валидации. Кросс-валидация надёжнее одного валидационного разбиения: она усредняет 5 оценок и показывает разброс.
# Если несколько конфигураций различаются меньше, чем на величину шума (стандартная ошибка = std / √5), то они **статистически неразличимы**; это нужно честно отметить в выводах – выбор между ними на качество не влияет.

# %%
cv_sorted = cv_df.sort_values("cv_rmse_mean").reset_index(drop=True)
cv_sorted["SE (std/√5)"] = cv_sorted["cv_rmse_std"] / np.sqrt(5)
best = cv_sorted.iloc[0]
cv_sorted["в пределах 1 SE от лучшей"] = cv_sorted["cv_rmse_mean"] <= best["cv_rmse_mean"] + best["SE (std/√5)"]
display(cv_sorted[["model", "param", "value", "cv_rmse_mean", "cv_rmse_std", "SE (std/√5)", "cv_r2_mean", "в пределах 1 SE от лучшей"]].head(8).round(4))
print("\nКонфигурации, неотличимые от лучшей в пределах 1 SE:")
print(cv_sorted[cv_sorted["в пределах 1 SE от лучшей"]][["model", "param", "value"]].to_string(index=False))

# %%
BEST_MODEL = best["model"]
BEST_VALUE = next(v for v in GRID[BEST_MODEL]["values"] if str(v) == best["value"])   # возвращаем исходный тип (int / float / None)
print(f"ЛУЧШАЯ МОДЕЛЬ: {BEST_MODEL}, {GRID[BEST_MODEL]['param']} = {BEST_VALUE}")

final = GRID[BEST_MODEL]["build"](BEST_VALUE).fit(X_train_full, y_train_full)     # обучение на ВСЕХ обучающих данных
pred_test = final.predict(X_test)                                                 # единственное обращение к тесту
test_m = metrics(y_test, pred_test)

# для честного сравнения baseline тоже переобучаем на train_full
test_rows = [{"Модель": f"{BEST_MODEL} ({GRID[BEST_MODEL]['param']}={BEST_VALUE})", **{k.upper(): v for k, v in test_m.items()}}]
for name, mdl in baselines.items():
    mdl.fit(X_train_full, y_train_full)
    test_rows.append({"Модель": name, **{k.upper(): v for k, v in metrics(y_test, mdl.predict(X_test)).items()}})
test_df = pd.DataFrame(test_rows)
display(test_df.round(4))
test_df.to_csv(RES / "test_results.csv", index=False)

# доверительный интервал RMSE на тесте (bootstrap по остаткам)
rng = np.random.default_rng(SEED)
res_test = (y_test - pred_test).to_numpy()
boot = [np.sqrt(np.mean(rng.choice(res_test, len(res_test)) ** 2)) for _ in range(2000)]
print(f"\nRMSE на тесте = {test_m['rmse']:.3f}, 95% bootstrap-интервал [{np.percentile(boot, 2.5):.3f}; {np.percentile(boot, 97.5):.3f}]")
print(f"RMSE по кросс-валидации для этой модели = {best['cv_rmse_mean']:.3f} ± {best['cv_rmse_std']:.3f}")

# %% [markdown]
# ## 9. Анализ ошибок лучшей модели (график остатков)
# Остаток = фактическое – предсказанное. У хорошей регрессии остатки: центрированы около нуля, не зависят от предсказания и признаков (нет структуры), приблизительно нормальны, без тяжёлых «хвостов».

# %%
resid = y_test.to_numpy() - pred_test
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
ax[0, 0].scatter(pred_test, y_test, s=8, alpha=.4)
lim = [min(pred_test.min(), y_test.min()) - 2, max(pred_test.max(), y_test.max()) + 2]
ax[0, 0].plot(lim, lim, "r--", lw=1); ax[0, 0].set(xlabel="Предсказание", ylabel="Факт", title="Факт vs предсказание")
ax[0, 1].scatter(pred_test, resid, s=8, alpha=.4); ax[0, 1].axhline(0, color="r", lw=1)
ax[0, 1].set(xlabel="Предсказание", ylabel="Остаток", title="Остатки vs предсказание")
ax[0, 2].hist(resid, bins=40, density=True, alpha=.7)
xs = np.linspace(resid.min(), resid.max(), 200); ax[0, 2].plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()), "r-", label="нормальное")
ax[0, 2].set(xlabel="Остаток", title="Распределение остатков"); ax[0, 2].legend()
stats.probplot(resid, dist="norm", plot=ax[1, 0]); ax[1, 0].set_title("Q-Q график остатков")
ax[1, 1].scatter(X_test["Previous Scores"], resid, s=8, alpha=.4); ax[1, 1].axhline(0, color="r", lw=1)
ax[1, 1].set(xlabel="Previous Scores", ylabel="Остаток", title="Остатки vs Previous Scores")
hours = sorted(X_test["Hours Studied"].unique())
groups = [resid[X_test["Hours Studied"].to_numpy() == h] for h in hours]
ax[1, 2].boxplot(groups, tick_labels=hours); ax[1, 2].axhline(0, color="r", lw=1)
ax[1, 2].set(xlabel="Hours Studied", ylabel="Остаток", title="Остатки по часам занятий")
fig.suptitle(f"Анализ ошибок: {BEST_MODEL} ({GRID[BEST_MODEL]['param']}={BEST_VALUE}), тестовая выборка", fontsize=13)
fig.tight_layout(); fig.savefig(FIG / "03_residuals_best_model.png", dpi=130, bbox_inches="tight"); plt.show()

# %%
summary_err = pd.Series({
    "Средний остаток (смещение)": resid.mean(),
    "Std остатков": resid.std(),
    "Доля |ошибка| ≤ 2 баллов": np.mean(np.abs(resid) <= 2),
    "Доля |ошибка| ≤ 5 баллов": np.mean(np.abs(resid) <= 5),
    "Максимальная |ошибка|": np.abs(resid).max(),
    "Асимметрия остатков": stats.skew(resid),
    "Эксцесс остатков": stats.kurtosis(resid),
    "Предсказаний > 100": int((pred_test > 100).sum()),
    "Корреляция остатков и предсказаний": np.corrcoef(resid, pred_test)[0, 1],
}).round(4)
display(summary_err.to_frame("значение"))
print("Округление цели до целого само даёт шум со std ≈ 0.289 (1/√12); остатки не могут быть меньше этого порога.")

worst = X_test.assign(Факт=y_test, Предсказание=pred_test.round(1), Остаток=resid.round(1))
worst = worst.reindex(worst["Остаток"].abs().sort_values(ascending=False).index).head(8)
display(worst)

# %%
# Что выучила модель: перестановочная важность признаков + (для линейной модели) коэффициенты в исходных единицах
pi = permutation_importance(final, X_test, y_test, scoring="neg_root_mean_squared_error", n_repeats=20, random_state=SEED)
imp = pd.DataFrame({"признак": X.columns, "рост RMSE при перемешивании": pi.importances_mean, "std": pi.importances_std}).sort_values("рост RMSE при перемешивании", ascending=False)
display(imp.round(3).reset_index(drop=True))
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(imp["признак"][::-1], imp["рост RMSE при перемешивании"][::-1], xerr=imp["std"][::-1])
ax.set(xlabel="Рост RMSE при перемешивании признака", title="Важность признаков (тест)")
fig.tight_layout(); fig.savefig(FIG / "04_permutation_importance.png", dpi=130, bbox_inches="tight"); plt.show()
if BEST_MODEL == "Ridge":
    sc, rg = final.named_steps["scaler"], final.named_steps["m"]
    coef = pd.DataFrame({"признак": X.columns, "коэффициент (баллов индекса на 1 ед. признака)": rg.coef_ / sc.scale_})
    print(f"Свободный член: {rg.intercept_ - (rg.coef_ / sc.scale_ * sc.mean_).sum():.3f}")
    display(coef.round(3))

# %% [markdown]
# ## Итоговые выводы
# 1. **Задача и метрики.** Регрессия по `Performance Index`; главная метрика RMSE (баллы индекса), дополнительно MAE и R².
# 2. **Лучшая модель – Ridge** (`alpha = 0.01`; значения 0.01–10 равнозначны, то есть по сути обычная линейная регрессия на стандартизованных признаках). CV RMSE 2.038 ± 0.030. **На тесте:** RMSE 2.075 (95% bootstrap-интервал 2.01–2.14), MAE 1.65, R² 0.988. Тестовая оценка лежит в пределах разброса CV, значит, выбор по кросс-валидации не привёл к «подгонке под валидацию».
# 3. **Сравнение с baseline на тесте.** Среднее: RMSE 19.3 → модель лучше ≈ в 9 раз. Линейная регрессия по одному `Previous Scores`: RMSE 7.8 → модель лучше ≈ в 3.8 раза, значит, остальные признаки вносят реальный вклад.
# 4. **Анализ ошибок.** Смещение 0.06 балла, std остатков 2.07; 67% предсказаний ошибаются не более чем на 2 балла, 98% – не более чем на 5, максимальная ошибка 7.6. Остатки практически нормальны (асимметрия 0.006, эксцесс 0.06), не зависят от предсказания (корреляция 0.004), от `Previous Scores` и `Hours Studied`. Систематических ошибок и предсказаний выше 100 нет.
# 5. **Почему простая линейная модель победила.** Зависимость цели от признаков по сути линейна плюс случайный шум со std ≈ 2. Деревья и KNN аппроксимируют плоскость ступеньками и усреднением соседей и платят за это дополнительной ошибкой; нелинейности, которую они могли бы уловить, в данных нет. Остаточная ошибка ≈ 2 балла – неустранимый шум (в данных есть 665 строк с совпадающими признаками, но разной целью). Дальнейшее усложнение модели на этих признаках не поможет; помочь могли бы только новые признаки или данные.
# 6. **Что выучила модель.** Почти всё определяют `Previous Scores` и `Hours Studied` (рост RMSE при перемешивании: 23.1 и 8.6 балла); `Sleep Hours` (0.31), `Sample Question Papers Practiced` (0.15) и `Extracurricular Activities` (0.06) влияют слабо, но не нулево. Коэффициенты: +2.85 балла за каждый час занятий, +1.02 за балл предыдущих результатов, +0.57 за внеучебную активность, +0.47 за час сна, +0.19 за решённый пробник. Это ассоциации, а не причинно-следственные связи.
# 7. **Ограничения.** (а) Данные выглядят синтетическими (почти идеальная линейность, нормальный шум, целочисленные признаки), поэтому на реальных студентах R² ≈ 0.99 ожидать нельзя. (б) Гиперпараметры подбирались по одному за раз, остальные фиксировались (`learning_rate`, `max_depth`, `n_estimators` леса); многомерный поиск (GridSearchCV) возможен, но при таком потолке шума выигрыша не даст. (в) Метрики стоит сверить с паспортом датасета и при необходимости дополнить обоснование ссылками на него.
