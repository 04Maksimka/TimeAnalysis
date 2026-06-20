# Архитектурный анализ Freqtrade

> Анализ папки `freqtrade/` репозитория [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade), ветка `main`, commit `9acdcac`.

---

## Обзор

Freqtrade — Python-бот алгоритмической торговли с поддержкой spot/futures-бирж через CCXT. Архитектура проекта не декларирована явно как слоистая, но при изучении кода уверенно выделяются **четыре логических уровня**:

| Уровень                 | Freqtrade-аналог                        | Основные пакеты                                                              |
| ----------------------- | --------------------------------------- | ---------------------------------------------------------------------------- |
| **Фундамент** (Core)    | Типы, перечисления, исключения          | `enums/`, `ft_types/`, `exceptions.py`, `constants.py`                       |
| **Ядро** (Domain)       | Стратегия + торговая бизнес-логика      | `strategy/`, `persistence/`, `wallets.py`, `plugins/`, `leverage/`, `freqai/`|
| **Оркестраторы** (Application) | Главный бот, бэктест, гиперопт, воркер | `freqtradebot.py`, `worker.py`, `optimize/`, `commands/`, `resolvers/`  |
| **Инфраструктура**      | Биржи, уведомления, API, хранение       | `exchange/`, `rpc/`, `data/`, `loggers/`, `configuration/`                   |

---

## 1. Core — Фундамент

Нижний уровень. Содержит **контракты** — типы, перечисления, исключения и константы. Не содержит бизнес-логики. Все остальные уровни зависят от него, сам он ни от кого не зависит.

**Состав:**

- **`enums/`** — перечисления ключевых концептов: `RunMode`, `TradingMode`, `MarginMode`, `SignalType`, `ExitType`, `CandleType`, `RPCMessageType`, `State`
- **`ft_types/`** — type aliases (`TypedDict` для сигналов DataFrame, результатов бэктеста и т. д.)
- **`exceptions.py`** — иерархия исключений: `FreqtradeException`, `OperationalException`, `TemporaryError`, `InvalidOrderException` и другие
- **`constants.py`** — глобальные константы (`DEFAULT_DB_PROD_URL`, `DEFAULT_EXCHANGE`, поддерживаемые таймфреймы)
- **`misc.py`** — общие утилиты (JSON-хелперы, шифрование, `safe_value_fallback`)

---

## 2. Domain — Ядро бизнес-логики

Центральный слой, который **владеет понятиями** алготрейдинга: стратегия, сделка, портфель, риски, плечо, ML-модель. Не знает о конкретных биржах или каналах уведомлений.

### 2.1 `strategy/` — Стратегии

- **`interface.py`** (`IStrategy`, ~81 KB) — главный абстрактный класс. Определяет контракт: обязательные `populate_indicators()`, `populate_entry_trend()`, `populate_exit_trend()`, и опциональные callbacks (`custom_stoploss`, `custom_exit`, `confirm_trade_entry`, `adjust_trade_position` и другие)
- **`hyper.py`** — `HyperStrategyMixin`: хранение и загрузка Hyperopt-параметров
- **`parameters.py`** — `IntParameter`, `DecimalParameter`, `BooleanParameter`, `CategoricalParameter`
- **`informative_decorator.py`** — декоратор `@informative()` для автоматического merge информационных таймфреймов
- **`strategy_helper.py`** — утилиты: `merge_informative_pair()`, `stoploss_from_open()`, `stoploss_from_absolute()`

### 2.2 `persistence/` — Доменные модели

- **`trade_model.py`** (`Trade`, `Order`, ~85 KB) — центральная доменная модель. `Trade` — агрегат-корень с полной информацией о сделке и расчётами прибыли
- **`pairlock.py`** — `PairLock`: блокировки торговых пар
- **`key_value_store.py`** — KV-хранилище для стратегий (persist custom данные)
- **`custom_data.py`** — хранение произвольных данных трейда

### 2.3 `plugins/` — Расширяемые модули

- **`PairListManager`** + `pairlist/` — pipeline из 19 фильтров (Chain of Responsibility): `VolumePairList`, `AgeFilter`, `PriceFilter`, `SpreadFilter` и другие
- **`ProtectionManager`** + `protections/` — 4 механизма защиты: `StoplossGuard`, `MaxDrawdown`, `LowProfitPairs`, `CooldownPeriod`

### 2.4 `freqai/` — ML-подсистема

- **`freqai_interface.py`** (`IFreqaiModel`) — абстрактный интерфейс ML-модели
- **`data_kitchen.py`** — feature engineering, нормализация, train/test split
- **`data_drawer.py`** — кэширование обученных моделей и предсказаний
- **`prediction_models/`** — LightGBM, XGBoost, CatBoost, PyTorch, ReinforcementLearner

---

## 3. Application — Оркестраторы

Координирует взаимодействие между доменным слоем и инфраструктурой. Не содержит бизнес-логики сделок, но знает, в каком порядке вызывать компоненты.

### 3.1 `freqtradebot.py` — Главный оркестратор

`FreqtradeBot` (~112 KB) — центральный класс live/dry режима. Реализует основной торговый цикл: обновление pairlist → получение OHLCV → вызов `populate_*()` → обработка выходов → обработка входов → position adjust.

### 3.2 `worker.py` — Event loop

`Worker` — throttled loop с обработкой исключений, graceful shutdown и reload-config.

### 3.3 `optimize/` — Режимы оптимизации

- **`backtesting.py`** (~79 KB) — загружает исторические данные, прогоняет стратегию, симулирует ордера
- **`hyperopt/`** — Bayesian search по пространству параметров через Optuna, параллельный запуск через joblib
- **`hyperopt_loss/`** — сменные функции потерь: `SharpeHyperOptLoss`, `CalmarHyperOptLoss`, `SortinoHyperOptLoss` и другие

### 3.4 `resolvers/` — Dynamic loading

`StrategyResolver`, `ExchangeResolver`, `FreqaiModelResolver` — загрузка пользовательских классов по имени через `importlib`.

### 3.5 `commands/` и `configuration/`

CLI-команды через argparse и загрузка/валидация конфигурационных файлов по JSON Schema.

---

## 4. Infrastructure — Внешний мир

Изолированный слой адаптеров к внешним системам. Не содержит бизнес-логики.

### 4.1 `exchange/` — Биржевые адаптеры

`Exchange` (~172 KB) — обёртка над CCXT с retry-логикой (`@retrier`), rate limiting и нормализацией данных. Около 20 конкретных адаптеров: Binance, Bybit, Kraken, OKX, Gate, Hyperliquid и другие.

### 4.2 `rpc/` — Уведомления и управление

- **`rpc.py`** (`RPC`) — базовые команды: status, profit, forceexit, reload_config
- **`rpc_manager.py`** (`RPCManager`) — шина событий, рассылает `send_msg()` всем обработчикам
- **`telegram.py`** — Telegram-бот (~93 KB) с командами и уведомлениями
- **`api_server/`** — FastAPI REST + WebSocket для FreqUI
- **`webhook.py`**, **`discord.py`** — push-уведомления
- **`external_message_consumer.py`** — Producer/Consumer режим

### 4.3 `data/` — Работа с данными

- **`dataprovider.py`** (`DataProvider`) — единая точка доступа к свечам для стратегий
- **`history/`** — загрузка и сохранение исторических OHLCV
- **`datahandlers/`** — `JsonDataHandler`, `HDF5DataHandler`, `ParquetDataHandler`