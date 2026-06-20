## Что такое FreqAI и зачем он нужен

Обычная стратегия Freqtrade использует **детерминированные индикаторы**: RSI, EMA, MACD — формулы, которые всегда дают одинаковый результат на одних и тех же данных. FreqAI добавляет **обученные ML-модели**: они строят признаки из исторических данных, обучаются на них и возвращают предсказание — например, насколько изменится цена через 24 свечи.

Результат предсказания становится обычной колонкой в DataFrame, и стратегия работает с ней так же, как с любым другим индикатором.

---

## 1. Связь FreqAI и стратегии

Архитектурно FreqAI **встраивается внутрь** `populate_indicators()` — он не заменяет стратегию, а расширяет её:

```python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe = self.freqai.start(dataframe, metadata, self)
    return dataframe
```

`self.freqai.start()` выполняет полный цикл:

1. Вызывает методы `feature_engineering_*()` и собирает матрицу признаков
2. Обучает модель (если нет актуального кэша)
3. Делает предсказание на текущем DataFrame
4. Возвращает результаты как колонки с префиксами `&-` и служебными полями

После этого `populate_entry_trend()` читает предсказания:

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (dataframe["do_predict"] == 1)       # предсказание признано надёжным
        & (dataframe["&-s_close"] > 0.01),   # цена вырастет больше чем на 1%
        "enter_long",
    ] = 1
    return dataframe
```

### Специальные колонки FreqAI

| Колонка          | Направление       | Смысл                                                                     |
| ---------------- | ----------------- | ------------------------------------------------------------------------- |
| `%-имя`          | strategy → FreqAI | Входной признак (feature) для модели                                      |
| `&-имя`          | FreqAI → strategy | Предсказание модели (target)                                              |
| `do_predict`     | FreqAI → strategy | `1` — предсказание в норме; `0` — данные аномальны, не торговать          |
| `&-s_close_mean` | FreqAI → strategy | Среднее предсказания за тренировочный период                              |
| `&-s_close_std`  | FreqAI → strategy | Стандартное отклонение предсказания                                       |

Префикс `%` помечает признаки, `&` — целевые переменные. FreqAI распознаёт их именно по этим префиксам.

---

## 2. Методы feature engineering

Вместо монолитного `populate_indicators()` в FreqAI-стратегии пишутся специализированные методы. FreqAI вызывает их автоматически в нужном порядке.

### `feature_engineering_expand_all` — признаки с переменным периодом

Вызывается **несколько раз** — по одному разу для каждого значения из `indicator_periods_candles`. Одна строка кода автоматически превращается в N признаков:

```python
def feature_engineering_expand_all(
    self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
) -> DataFrame:
    dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
    dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
    dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
    dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
    return dataframe
```

При `«indicator_periods_candles»: [14, 21, 50]` FreqAI автоматически создаст колонки `%-rsi-14`, `%-rsi-21`, `%-rsi-50` и так далее для каждого индикатора.

Затем эти признаки ещё раз размножаются по:
- `include_timeframes` — те же признаки на 5m, 15m, 1h
- `include_shifted_candles` — лаговые копии (признак смещённый на 1, 2 свечи назад)
- `include_corr_pairlist` — те же признаки для коррелированных пар (BTC, ETH)

Итог: один индикатор → десятки колонок-признаков.

### `feature_engineering_expand_basic` — признаки без периода

Расширяются по `include_timeframes`, `include_shifted_candles`, `include_corr_pairlist`, но **не** по `indicator_periods_candles`. Подходит для признаков, у которых нет смысла в переменном периоде:

```python
def feature_engineering_expand_basic(
    self, dataframe: DataFrame, metadata: dict, **kwargs
) -> DataFrame:
    dataframe["%-pct-change"] = dataframe["close"].pct_change()
    dataframe["%-raw-volume"] = dataframe["volume"]
    dataframe["%-raw-price"] = dataframe["close"]

    # Соотношение объёма к скользящей средней объёма
    dataframe["%-volume-mean-ratio"] = (
        dataframe["volume"] / dataframe["volume"].rolling(20).mean()
    )
    return dataframe
```

### `feature_engineering_standard` — уникальные признаки

Вызывается **один раз**, уже после всех расширений. Для признаков, которые не нужно или нельзя дублировать:

```python
def feature_engineering_standard(
    self, dataframe: DataFrame, metadata: dict, **kwargs
) -> DataFrame:
    # Временны́е признаки — нет смысла множить по таймфреймам
    dataframe["%-day-of-week"] = dataframe["date"].dt.dayofweek
    dataframe["%-hour-of-day"] = dataframe["date"].dt.hour

    # Глобальный контекст рынка (если пара — не BTC)
    if metadata["pair"] != "BTC/USDT":
        btc_df = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
        if len(btc_df) > 0:
            dataframe["%-btc-close"] = btc_df["close"].iloc[-1]

    return dataframe
```

### `set_freqai_targets` — что предсказывать (обязательный)

Определяет целевые переменные (`&`-колонки) — то, что модель будет предсказывать:

```python
def set_freqai_targets(
    self, dataframe: DataFrame, metadata: dict, **kwargs
) -> DataFrame:
    # Регрессия: предсказать среднюю относительную цену через N свечей
    label_period = self.freqai_info["feature_parameters"]["label_period_candles"]
    dataframe["&-s_close"] = (
        dataframe["close"]
        .shift(-label_period)
        .rolling(label_period)
        .mean()
        / dataframe["close"]
        - 1
    )
    return dataframe
```

`shift(-N)` берёт данные из будущего — это допустимо **только при обучении** на исторических данных. В live-торговле последние N строк автоматически исключаются FreqAI.

**Вариант с классификацией** — если нужно предсказать направление, а не величину:

```python
def set_freqai_targets(
    self, dataframe: DataFrame, metadata: dict, **kwargs
) -> DataFrame:
    # Классификация: "up" / "down" через 10 свечей
    dataframe["&-direction"] = np.where(
        dataframe["close"].shift(-10) > dataframe["close"],
        "up",
        "down",
    )
    return dataframe
```

**Мульти-таргет** — модель предсказывает несколько целей одновременно:

```python
def set_freqai_targets(
    self, dataframe: DataFrame, metadata: dict, **kwargs
) -> DataFrame:
    label_period = self.freqai_info["feature_parameters"]["label_period_candles"]

    # Будущая доходность
    dataframe["&-s_close"] = (
        dataframe["close"].shift(-label_period).rolling(label_period).mean()
        / dataframe["close"]
        - 1
    )
    # Будущая волатильность (стандартное отклонение)
    dataframe["&-s_volatility"] = (
        dataframe["close"]
        .shift(-label_period)
        .rolling(label_period)
        .std()
        / dataframe["close"]
    )
    return dataframe
```

Для мульти-таргета используйте `LightGBMRegressorMultiTarget` или `XGBoostRegressorMultiTarget`.

---

## 3. Конфигурация FreqAI в config.json

```json
"freqai": {
    "enabled": true,
    "purge_old_models": 2,
    "train_period_days": 30,
    "backtest_period_days": 7,
    "identifier": "my_model_v1",
    "live_retrain_hours": 0,
    "expiration_hours": 0,

    "feature_parameters": {
        "include_timeframes": ["5m", "15m", "1h"],
        "include_corr_pairlist": ["BTC/USDT", "ETH/USDT"],
        "label_period_candles": 24,
        "include_shifted_candles": 2,
        "indicator_periods_candles": [10, 20],
        "DI_threshold": 0.9,
        "weight_factor": 0.9
    },

    "data_split_parameters": {
        "test_size": 0.25,
        "shuffle": false
    },

    "model_training_parameters": {
        "n_estimators": 200,
        "learning_rate": 0.05
    }
}
```

### Ключевые параметры

| Параметр                               | Описание                                                                                             |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `purge_old_models`                     | Хранить только N последних моделей на диске                                                          |
| `train_period_days`                    | Сколько дней истории использовать для обучения                                                       |
| `backtest_period_days`                 | В бэктесте — шаг переобучения (каждые N дней модель обучается заново)                                |
| `identifier`                           | Уникальное имя модели для кэширования. При смене — модель обучается с нуля                           |
| `live_retrain_hours`                   | Принудительное переобучение каждые N часов в live; `0` — по факту устаревания                        |
| `expiration_hours`                     | Модель считается устаревшей через N часов; `0` — не устаревает принудительно                         |
| `include_timeframes`                   | Доп. таймфреймы для признаков (требуют скачанных данных)                                             |
| `include_corr_pairlist`                | Пары, признаки которых добавляются как фичи (корреляционный контекст рынка)                          |
| `label_period_candles`                 | Горизонт предсказания: за сколько свечей вперёд считается таргет                                     |
| `include_shifted_candles`              | Сколько лаговых копий признаков добавить (0 = не добавлять)                                          |
| `indicator_periods_candles`            | Периоды для `feature_engineering_expand_all`                                                         |
| `DI_threshold`                         | Порог Dissimilarity Index: если DI > порога → `do_predict = 0`                                       |
| `weight_factor`                        | Экспоненциальное взвешивание свежих данных: 1.0 = равный вес, 0.9 = свежие данные важнее             |
| `data_split_parameters.shuffle: false` | **Обязательно `false` для временных рядов** — иначе данные из будущего попадут в обучение            |

---

## 4. Как происходит обучение и валидация

### Walk-forward training

FreqAI использует **скользящее окно обучения** — ключевой механизм против look-ahead bias:

```
──────────────────────────────────────── время ───────────────────────────────────────→

[  train: 30 дней  ][  predict: 7 дней  ]
                [  train: 30 дней  ][  predict: 7 дней  ]
                                [  train: 30 дней  ][  predict: 7 дней  ]
```

Модель никогда не «видит» данные из периода, на котором делает предсказание. В live-торговле переобучение запускается автоматически при устаревании модели.

### Train / test split

Внутри каждого тренировочного окна `DataKitchen` делает split:

```
[──────────── 30 дней обучения ────────────]
[──────── 75% train ────────][── 25% test ──]
```

На тестовой части FreqAI считает метрики: RMSE, accuracy, feature importances. `shuffle: false` обязателен — без него данные из будущего попадут в тренировочную выборку.

### DI (Dissimilarity Index) и `do_predict`

FreqAI вычисляет, насколько текущие рыночные данные похожи на обучающую выборку. Если данные слишком «чужие» — рынок изменился, модель работает вне зоны обученности — `do_predict` устанавливается в `0`.

Стратегия **обязана** проверять это:

```python
dataframe.loc[
    (dataframe["do_predict"] == 1)       # данные похожи на обучающую выборку
    & (dataframe["&-s_close"] > 0.01),
    "enter_long",
] = 1
```

Значение порога `DI_threshold` подбирается экспериментально: слишком низкий — бот не торгует вовсе, слишком высокий — торгует в непредсказуемых условиях.

---

## 5. Полный пример стратегии с FreqAI

```python
import numpy as np
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class FreqAIExampleStrategy(IStrategy):
    """
    Пример FreqAI-стратегии с LightGBMRegressor.
    Предсказывает среднюю доходность через label_period_candles свечей.
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = True
    stoploss = -0.05
    minimal_roi = {"0": 0.1}
    startup_candle_count = 40
    process_only_new_candles = True

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw-volume"] = dataframe["volume"]
        dataframe["%-volume-mean-ratio"] = (
            dataframe["volume"] / dataframe["volume"].rolling(20).mean()
        )
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-day-of-week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour-of-day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-label_period)
            .rolling(label_period)
            .mean()
            / dataframe["close"]
            - 1
        )
        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe.loc[
            (dataframe["do_predict"] == 1) & (dataframe["&-s_close"] > 0.01),
            "enter_long",
        ] = 1
        dataframe.loc[
            (dataframe["do_predict"] == 1) & (dataframe["&-s_close"] < -0.01),
            "enter_short",
        ] = 1
        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe.loc[
            (dataframe["do_predict"] == 1) & (dataframe["&-s_close"] < 0),
            "exit_long",
        ] = 1
        dataframe.loc[
            (dataframe["do_predict"] == 1) & (dataframe["&-s_close"] > 0),
            "exit_short",
        ] = 1
        return dataframe
```

---

## 6. Пример с классификацией

Для классификационных задач (предсказать класс, а не число) используйте `LightGBMClassifier` и стройте таргет как строковые метки:

```python
def set_freqai_targets(
    self, dataframe: DataFrame, metadata: dict, **kwargs
) -> DataFrame:
    # "up" — цена вырастет за 10 свечей, "down" — упадёт
    dataframe["&-direction"] = np.where(
        dataframe["close"].shift(-10) > dataframe["close"] * 1.005,
        "up",
        "down",
    )
    return dataframe
```

В `populate_entry_trend` чтение результата:

```python
def populate_entry_trend(
    self, dataframe: DataFrame, metadata: dict
) -> DataFrame:
    dataframe.loc[
        (dataframe["do_predict"] == 1)
        & (dataframe["&-direction"] == "up"),
        "enter_long",
    ] = 1
    return dataframe
```

При классификации FreqAI также добавляет колонки с вероятностями классов: `&-direction_up_probability`, `&-direction_down_probability`. Их можно использовать для фильтрации сигналов с низкой уверенностью модели:

```python
dataframe.loc[
    (dataframe["do_predict"] == 1)
    & (dataframe["&-direction"] == "up")
    & (dataframe["&-direction_up_probability"] > 0.65),  # уверенность > 65%
    "enter_long",
] = 1
```

---

## 7. Пример с кастомным stoploss на основе предсказания

FreqAI-предсказание можно использовать и в `custom_stoploss()` — например, ужесточать стоп, когда модель теряет уверенность:

```python
def custom_stoploss(
    self,
    pair: str,
    trade,
    current_time,
    current_rate: float,
    current_profit: float,
    **kwargs,
) -> float:
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    if dataframe.empty:
        return self.stoploss

    last = dataframe.iloc[-1]

    # Если модель неуверена — ужесточить стоп до -1%
    if last["do_predict"] == 0:
        return -0.01

    # Если предсказание разворачивается против позиции — выйти раньше
    if last["&-s_close"] < -0.005:
        return -0.02

    return self.stoploss
```

---

## 8. Доступные модели

| Модель                            | Тип задачи               | Библиотека      |
| --------------------------------- | ------------------------ | --------------- |
| `LightGBMRegressor`               | Регрессия                | LightGBM        |
| `LightGBMClassifier`              | Классификация            | LightGBM        |
| `LightGBMRegressorMultiTarget`    | Мульти-регрессия         | LightGBM        |
| `LightGBMClassifierMultiTarget`   | Мульти-классификация     | LightGBM        |
| `XGBoostRegressor`                | Регрессия                | XGBoost         |
| `XGBoostClassifier`               | Классификация            | XGBoost         |
| `XGBoostRFRegressor`              | Регрессия (Random Forest)| XGBoost         |
| `SKLearnRandomForestClassifier`   | Классификация            | scikit-learn    |
| `PyTorchMLPRegressor`             | Регрессия (нейросеть)    | PyTorch         |
| `PyTorchMLPClassifier`            | Классификация (нейросеть)| PyTorch         |
| `PyTorchTransformerRegressor`     | Регрессия (Transformer)  | PyTorch         |
| `ReinforcementLearner`            | RL (агент)               | Stable-Baselines3|

**Выбор модели** определяется флагом `--freqaimodel` при запуске:

```bash
freqtrade trade \
    --strategy MyFreqAIStrategy \
    --freqaimodel LightGBMRegressor \
    --config user_data/config.json
```

---

## 9. Запуск

**Бэктест:**

```bash
freqtrade backtesting \
    --strategy FreqAIExampleStrategy \
    --freqaimodel LightGBMRegressor \
    --config user_data/config.json \
    --timerange 20230101-20240101
```

**Dry run / Live:**

```bash
freqtrade trade \
    --strategy FreqAIExampleStrategy \
    --freqaimodel LightGBMRegressor \
    --config user_data/config.json
```

**Через Docker:**

```bash
docker compose run --rm freqtrade backtesting \
    --strategy FreqAIExampleStrategy \
    --freqaimodel LightGBMRegressor \
    --config /freqtrade/user_data/config.json \
    --timerange 20230101-20240101
```

Обученные модели сохраняются в `user_data/models/<identifier>/`. При повторном запуске FreqAI проверяет кэш — если модель не устарела, переобучение не происходит.

---

## 10. Pipeline одним взглядом

```
populate_indicators()
    → freqai.start(dataframe, metadata, strategy)
        │
        ├─ feature_engineering_expand_all()
        │       × indicator_periods_candles [14, 21, 50]
        │       × include_timeframes [5m, 15m, 1h]
        │       × include_shifted_candles [1, 2]
        │       × include_corr_pairlist [BTC/USDT, ETH/USDT]
        │       → %-rsi-14, %-rsi-21, %-ema-14-15m, %-rsi-14-shift-1, ...
        │
        ├─ feature_engineering_expand_basic()
        │       × include_timeframes
        │       × include_shifted_candles
        │       × include_corr_pairlist
        │       → %-pct-change, %-raw-volume, ...
        │
        ├─ feature_engineering_standard()
        │       × 1 раз
        │       → %-day-of-week, %-hour-of-day, ...
        │
        ├─ set_freqai_targets()
        │       → &-s_close (y для обучения)
        │
        ├─ DataKitchen.normalize_data()
        │       → StandardScaler на X и y
        │
        ├─ DataKitchen.train_test_split()
        │       → 75% train / 25% test, shuffle=False
        │
        ├─ IFreqaiModel.fit(X_train, y_train)
        │       → обученная модель (LightGBM / XGBoost / PyTorch / ...)
        │
        ├─ IFreqaiModel.predict(X_live)
        │       → &-s_close (предсказание на текущих данных)
        │
        ├─ DataKitchen.denormalize()
        │       → обратное масштабирование предсказания
        │
        └─ DI-score → do_predict (0 или 1)
        
    → DataFrame + [&-s_close, do_predict, &-s_close_mean, &-s_close_std]

populate_entry_trend()
    → do_predict == 1 AND &-s_close > 0.01 → enter_long = 1
    → do_predict == 1 AND &-s_close < -0.01 → enter_short = 1
```

---

## 11. Ограничения и подводные камни

| Проблема                              | Описание и решение                                                                                                      |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Переобучение (overfitting)**        | Модель идеально описывает прошлое, но не обобщается. Контролируйте через test-метрики и out-of-sample бэктест           |
| **Слишком мало данных**               | `train_period_days` слишком мал для сложной модели — увеличьте или уменьшите число признаков                           |
| **Нарушение shuffle**                 | `shuffle: true` при обучении на временных рядах — данные из будущего утекают в тренировку, результаты нереалистичны     |
| **Игнорирование `do_predict`**        | Торговля без проверки `do_predict == 1` — модель работает на данных вне зоны обученности                                |
| **`shift(-N)` в live**                | `set_freqai_targets` с `shift(-N)` работает только при обучении. FreqAI автоматически исключает последние N строк — не переопределяйте это поведение вручную |
| **Высокая нагрузка при обучении**     | Обучение запускается в основном процессе. При большом числе признаков и длинном `train_period_days` — занимает минуты    |
| **Кэш и `identifier`**                | При изменении архитектуры признаков обязательно меняйте `identifier` — иначе FreqAI загрузит устаревшую модель из кэша  |
