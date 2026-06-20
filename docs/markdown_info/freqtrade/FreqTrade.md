Проект [Freqtrade](https://www.freqtrade.io/en/stable/) · [GitHub](https://github.com/freqtrade/freqtrade)
# Проект

**Freqtrade** — Python-фреймворк для алгоритмической торговли криптовалютами. Поддерживает spot- и futures-биржи через CCXT, управляется через Telegram и WebUI. Пользователь пишет **стратегию** (Python-класс), а фреймворк берёт на себя всё остальное: подключение к биржам, управление ордерами, оценку риска, бэктестинг, оптимизацию гиперпараметров и уведомления.

Архитектуру проекта удобно разложить на четыре логических уровня:

| **Уровень**                    | Freqtrade-аналог                        | Основные пакеты                                                                      |
| ------------------------------ | --------------------------------------- | ------------------------------------------------------------------------------------ |
| **Фундамент** (Core)           | Типы, перечисления, исключения          | `enums/`, `ft_types/`, `exceptions.py`, `constants.py`                               |
| **Ядро** (Domain)              | Стратегия + торговая бизнес-логика      | `strategy/`, `persistence/`, `wallets.py`, `plugins/`, `leverage/`, `freqai/`        |
| **Оркестраторы** (Application) | Главный бот, бэктест, гиперопт, воркер | `freqtradebot.py`, `worker.py`, `optimize/`, `commands/`, `resolvers/`               |
| **Инфраструктура**             | Биржи, уведомления, API, хранение       | `exchange/`, `rpc/`, `data/`, `loggers/`, `configuration/`, `persistence/models.py`  |

Пояснение к структуре:

```
L1 Core             — что существует (типы, контракты без логики)
  enums, ft_types, exceptions, constants

L2 Domain           — как работает бизнес (логика, правила)
  IStrategy         — Template Method с реализацией ROI/stoploss/trailing
  IFreqaiModel      — абстрактный ML-пайплайн с DataKitchen/DataDrawer
  Trade/Order       — агрегаты с бизнес-правилами
  Wallets           — расчёт доступного капитала
  PairListManager   — фильтрация пар
  ProtectionManager — защитные блокировки

L3 Application      — кто координирует (оркестраторы)
L4 Infrastructure   — с чем общается (I/O)
```

# Полная карта пакетов

```
freqtrade/
├── enums/              L1 · Core
├── ft_types/           L1 · Core
├── exceptions.py       L1 · Core
├── constants.py        L1 · Core
├── misc.py             L1 · Core
├── util/               L1 · Core
├── mixins/             L1 · Core (LoggingMixin)
│
├── strategy/           L2 · Domain (IStrategy)
├── persistence/        L2 · Domain (Trade, Order, PairLock)
├── wallets.py          L2 · Domain
├── plugins/
│   ├── pairlist/       L2 · Domain (19 фильтров)
│   └── protections/    L2 · Domain (4 защиты)
├── leverage/           L2 · Domain
├── freqai/             L2 · Domain (ML)
│
├── freqtradebot.py     L3 · Application (главный оркестратор)
├── worker.py           L3 · Application (process-loop)
├── optimize/           L3 · Application (Backtesting, Hyperopt)
├── resolvers/          L3 · Application (6 фабрик)
├── commands/           L3 · Application (Click CLI)
├── configuration/      L3 · Application
├── main.py             L3 · Application (точка входа)
│
├── exchange/           L4 · Infrastructure (ccxt + 20 адаптеров)
├── rpc/                L4 · Infrastructure (Telegram, FastAPI...)
├── data/               L4 · Infrastructure (DataProvider, history, metrics)
└── loggers/            L4 · Infrastructure
```

# Терминология

## Архитектура

| Термин                        | Суть                                                                                                                                                              |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Domain Layer**              | Уровень бизнес-логики — правила торговли, не зависящие от конкретной биржи или хранилища                                                                          |
| **Application Layer**         | Уровень оркестрации — координирует вызовы Domain и Infrastructure, сам бизнес-логики не содержит                                                                  |
| **Infrastructure Layer**      | Всё, что связано с I/O: сеть, БД, файлы, уведомления                                                                                                             |
| **Core / Foundation Layer**   | Самый нижний уровень: типы, контракты, исключения — без внешних зависимостей                                                                                      |
| **Look-ahead bias**           | Ошибка бэктеста: стратегия случайно «видит» данные из будущего (например, индикатор считается по ещё не закрытым свечам). Делает результаты нереалистично хорошими |
| **Process-loop / Event loop** | Бесконечный цикл: на каждой итерации выполняется набор действий, затем пауза (throttle)                                                                          |
| **Throttle**                  | Ограничитель частоты: пауза между итерациями цикла, чтобы не перегружать биржу запросами                                                                          |

## Технологии

| Термин / Аббревиатура             | Расшифровка / Суть                                                                                                                |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **ccxt**                          | CryptoCurrency eXchange Trading — Python/JS-библиотека с единым API для 100+ криптобирж                                          |
| **SQLAlchemy**                    | Python ORM (Object-Relational Mapping) — маппинг Python-классов на таблицы БД                                                    |
| **SQLite / PostgreSQL**           | Реляционные СУБД. SQLite — файловая (по умолчанию), PostgreSQL — серверная (для продакшна)                                        |
| **FastAPI**                       | Современный Python-фреймворк для REST API на основе Pydantic и ASGI                                                              |
| **Uvicorn**                       | ASGI-сервер для запуска FastAPI                                                                                                   |
| **Pydantic**                      | Библиотека валидации данных через аннотации типов. В Freqtrade — схемы запросов/ответов API                                       |
| **Apache Feather / Parquet**      | Колоночные бинарные форматы хранения DataFrame. Feather — быстрое чтение/запись, Parquet — компактнее, подходит для больших данных |
| **WebSocket**                     | Протокол двустороннего соединения поверх HTTP. Используется для real-time котировок и архитектуры Producer/Consumer                |
| **Stable-Baselines3**             | Python-библиотека алгоритмов Reinforcement Learning (PPO, A2C, DQN и др.)                                                        |
| **LightGBM / XGBoost / CatBoost** | Библиотеки градиентного бустинга для задач регрессии и классификации                                                              |
| **PyTorch**                       | Фреймворк глубокого обучения (нейронные сети)                                                                                    |

## Торговые термины

| Термин / Аббревиатура | Суть                                                                                                                                    |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **OHLCV**             | Open, High, Low, Close, Volume — стандартный формат свечи: цена открытия, максимум, минимум, закрытие, объём                            |
| **Timeframe**         | Период одной свечи: 1m, 5m, 15m, 1h, 4h, 1d                                                                                           |
| **Stoploss**          | Автоматическое закрытие позиции при достижении заданного убытка (например, −5%)                                                         |
| **Trailing stop**     | Скользящий стоп-лосс: стоп следует за ценой вверх, фиксируя прибыль, но срабатывает при откате                                         |
| **ROI**               | Return On Investment — в Freqtrade: таблица минимальной прибыли по времени удержания (например: через 5 мин — +1%, через 1 ч — +0.5%)  |
| **Dry run**           | «Сухой прогон» — режим с реальными котировками, но без реальных ордеров (виртуальные деньги)                                            |
| **Live**              | Реальная торговля с реальными деньгами                                                                                                 |
| **Backtesting**       | Тестирование стратегии на исторических данных — оценка без реального риска                                                              |
| **Stake amount**      | Размер ставки — сумма на одну сделку                                                                                                   |
| **Stake currency**    | Базовая валюта торговли (обычно USDT, BTC)                                                                                             |
| **Open trade**        | Открытая позиция — сделка, которая ещё не закрыта                                                                                      |
| **Long / Short**      | Long — ставка на рост (купить дешевле → продать дороже). Short — ставка на падение (продать дороже → откупить дешевле)                  |
| **Spot**              | Спотовая торговля — покупка реального актива без плеча                                                                                  |
| **Futures / Margin**  | Торговля фьючерсами или с маржой — с кредитным плечом                                                                                  |
| **Leverage**          | Кредитное плечо — множитель позиции. Плечо 10× превращает 100 USDT в позицию на 1000 USDT                                              |
| **Liquidation**       | Принудительное закрытие позиции биржей при исчерпании маржи                                                                             |
| **Funding rate**      | Ставка финансирования — периодические платежи между long и short на фьючерсах                                                          |
| **Whitelist**         | Список разрешённых торговых пар (например, BTC/USDT, ETH/USDT)                                                                         |
| **Blacklist**         | Список запрещённых пар                                                                                                                 |
| **Pairlist**          | Текущий список пар для анализа и торговли, формируется PairListManager                                                                 |
| **Ticker**            | Текущая котировка пары: last price, bid, ask, volume за 24 ч                                                                           |
| **Orderbook**         | Стакан заявок — список активных bid- и ask-ордеров на бирже                                                                            |
| **Fee / Commission**  | Комиссия биржи за исполнение ордера (обычно 0.1%)                                                                                      |
| **P&L**               | Profit & Loss — прибыль и убытки                                                                                                       |
| **Drawdown**          | Просадка — максимальное падение капитала от пика до минимума                                                                           |
| **Sharpe ratio**      | Отношение доходности к волатильности: чем выше — тем лучше соотношение доходность/риск                                                 |
| **Sortino ratio**     | Аналог Sharpe, но учитывает только нисходящую волатильность                                                                            |
| **Calmar ratio**      | Отношение годовой доходности к максимальной просадке                                                                                   |
| **CAGR**              | Compound Annual Growth Rate — среднегодовой темп роста капитала                                                                        |
| **Win rate**          | Процент прибыльных сделок                                                                                                              |
| **Profit factor**     | Отношение суммарной прибыли к суммарному убытку                                                                                        |
| **Expectancy**        | Математическое ожидание прибыли с одной сделки                                                                                         |

## Специфика Freqtrade

| Термин                             | Суть                                                                                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **IStrategy**                      | Абстрактный базовый класс стратегии. Префикс `I` — от «Interface»                                                                                           |
| **INTERFACE_VERSION**              | Версия контракта между фреймворком и стратегией. Текущая — `3`. Определяет, какие методы и имена колонок ожидаются                                            |
| **populate_indicators**            | Метод стратегии: добавляет технические индикаторы в DataFrame (RSI, MACD, BB и т. д.)                                                                        |
| **populate_entry_trend**           | Метод стратегии: проставляет сигналы входа (`enter_long = 1`)                                                                                                |
| **populate_exit_trend**            | Метод стратегии: проставляет сигналы выхода (`exit_long = 1`)                                                                                                |
| **startup_candle_count**           | Количество свечей для «прогрева» индикаторов. Эти свечи не участвуют в торговле и бэктесте                                                                   |
| **LocalTrade**                     | In-memory аналог `Trade` без SQLAlchemy — используется только в бэктесте                                                                                     |
| **DataKitchen**                    | Компонент FreqAI: препроцессинг, feature engineering, нормализация, train/test split                                                                         |
| **DataDrawer**                     | Компонент FreqAI: сериализация и загрузка обученных моделей и предсказаний                                                                                   |
| **FreqAI**                         | Встроенная ML-подсистема для обучения моделей на торговых данных                                                                                              |
| **RPCManager**                     | Remote Procedure Call Manager — менеджер уведомлений и команд. RPC здесь в широком смысле: любая внешняя коммуникация (Telegram, REST, Webhook)               |
| **Producer/Consumer**              | Архитектура из двух ботов: Producer генерирует сигналы и отправляет по WebSocket, Consumer получает их и торгует                                              |
| **force_entry / force_exit**       | Принудительный ручной вход/выход через API или Telegram, в обход сигналов стратегии                                                                           |
| **@retrier**                       | Декоратор в `exchange/common.py` — автоматически повторяет вызов биржи при временных сетевых ошибках                                                         |
| **Hyperopt**                       | Оптимизация гиперпараметров стратегии через автоматический перебор (байесовская оптимизация)                                                                  |
| **HyperOptable parameters**       | Параметры стратегии, объявленные через `IntParameter`, `DecimalParameter` и т. д. — заменяют устаревшие `buy_params`/`sell_params`                            |

# Установка и настройка

## Полное руководство по настройке Freqtrade

Всё, что делает пользователь, укладывается в **три шага**:

1. Установка
2. Конфигурация (`config.json`)
3. Написание стратегии

---

## 1. Установка

Три способа, всё через CLI:

**Docker (рекомендуется для продакшна):**

```bash
mkdir freqtrade && cd freqtrade
curl https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml -o docker-compose.yml
docker compose run --rm freqtrade create-userdir --userdir user_data
docker compose run --rm freqtrade new-config --config user_data/config.json
docker compose up -d
```

**Скрипт (Linux/macOS):**

```bash
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
./setup.sh -i          # создаёт Python venv и ставит зависимости
source .venv/bin/activate
```

**pip (внутри venv):**

```bash
pip install freqtrade
```

После установки нужно создать структуру `user_data/`:

```bash
freqtrade create-userdir --userdir user_data
```

Результат:

```
user_data/
├── strategies/          # .py файлы стратегий
├── data/                # исторические OHLCV-данные
├── logs/                # логи
├── backtest_results/
├── hyperopts/           # кастомные loss-функции
└── freqaimodels/        # кастомные FreqAI-модели
```

---

## 2. `config.json`

Генерация интерактивным мастером:

```bash
freqtrade new-config --config user_data/config.json
```

### 2.1 Ключевые параметры

| Параметр           | Тип             | По умолчанию | Описание                                                        |
| ------------------ | --------------- | ------------ | --------------------------------------------------------------- |
| `dry_run`          | Boolean         | `true`       | `true` — симуляция, `false` — реальная торговля                 |
| `dry_run_wallet`   | Float           | 1000         | Стартовый виртуальный баланс                                    |
| `max_open_trades`  | Int / -1        | —            | Максимум открытых позиций; `-1` — без ограничений               |
| `stake_amount`     | Float / `"unlimited"` | —      | Сумма на сделку; `"unlimited"` — делить баланс поровну          |
| `stake_currency`   | String          | —            | Валюта торговли (`USDT`, `BTC`, …)                              |
| `minimal_roi`      | Dict            | —            | Порог прибыли для автовыхода (Strategy Override)                |
| `stoploss`         | Float           | —            | Стоп-лосс как доля: `-0.05` = −5% (Strategy Override)          |
| `exchange.key` / `secret` | String   | `""`         | API-ключи биржи; **не нужны для dry run и backtesting**         |

### 2.2 Управление рисками

```json
{
    "stoploss": -0.10,
    "trailing_stop": false,
    "trailing_stop_positive": 0.005,
    "trailing_stop_positive_offset": 0.01,
    "minimal_roi": {
        "40": 0.0,
        "30": 0.01,
        "20": 0.02,
        "0": 0.04
    }
}
```

- **`stoploss`** — задаётся в стратегии или конфиге (конфиг переопределяет стратегию)
- **`minimal_roi`** — таблица «время удержания → минимальная прибыль». Ключ — минуты с момента входа, значение — порог прибыли. Если прибыль достигает порога — бот выходит
- **`trailing_stop_positive_offset`** — trailing начинает работать только после достижения этой прибыли

### 2.3 Ордеры и цены

```json
{
    "order_types": {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": false
    },
    "entry_pricing": {
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1
    },
    "unfilledtimeout": {
        "entry": 10,
        "exit": 10,
        "unit": "minutes"
    }
}
```

> **`stoploss_on_exchange`** — если `true`, стоп-лосс размещается непосредственно на бирже как отдельный ордер. Это защищает от ситуации, когда бот упал или потерял связь, но не все биржи поддерживают это одинаково хорошо.

### 2.4 Pairlist — список торговых пар

Самый гибкий раздел конфига. Фильтры выполняются **последовательно**, каждый сужает список — паттерн Chain of Responsibility:

```json
{
    "pairlists": [
        {"method": "VolumePairList", "number_assets": 20, "sort_key": "quoteVolume"},
        {"method": "AgeFilter", "min_days_listed": 10},
        {"method": "PrecisionFilter"},
        {"method": "PriceFilter", "low_price_ratio": 0.01},
        {"method": "SpreadFilter", "max_spread_ratio": 0.005},
        {"method": "RangeStabilityFilter", "lookback_days": 10, "min_rate_of_change": 0.01}
    ]
}
```

Основные методы:

| Метод                   | Функция                                              |
| ----------------------- | ---------------------------------------------------- |
| `StaticPairList`        | Фиксированный список из `pair_whitelist`             |
| `VolumePairList`        | Топ-N по объёму торгов                               |
| `AgeFilter`             | Исключает монеты моложе N дней                       |
| `PrecisionFilter`       | Исключает пары с неудобным шагом цены                |
| `PriceFilter`           | Фильтр по минимальной/максимальной цене              |
| `SpreadFilter`          | Исключает пары с широким bid/ask-спредом             |
| `RangeStabilityFilter`  | Исключает «замёрзшие» пары без движения              |
| `VolatilityFilter`      | Фильтр по волатильности                             |
| `DelistFilter`          | Исключает токены, запланированные к делистингу        |
| `FullTradesFilter`      | Исключает пары, если `max_open_trades` уже заполнен  |

### 2.5 Telegram

```json
{
    "telegram": {
        "enabled": true,
        "token": "ваш_bot_token",
        "chat_id": "ваш_chat_id",
        "notification_settings": {
            "entry": "on",
            "exit": {"roi": "off", "stop_loss": "on"},
            "status": "on"
        }
    }
}
```

Как получить токен: `@BotFather` в Telegram → `/newbot`.  
Chat ID: отправить любое сообщение `@userinfobot`.

### 2.6 API Server и FreqUI

```json
{
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "jwt_secret_key": "случайная_строка_32+_символов",
        "username": "freqtrader",
        "password": "StrongPassword123!",
        "ws_token": "secret_ws_token",
        "CORS_origins": ["http://localhost:3000"],
        "enable_openapi": false
    }
}
```

Установка веб-интерфейса:

```bash
freqtrade install-ui
```

### 2.7 Торговый режим (spot / futures)

```json
{
    "trading_mode": "futures",
    "margin_mode": "isolated"
}
```

Допустимые значения: `trading_mode` — `spot`, `futures`, `margin`; `margin_mode` — `isolated`, `cross`.

### 2.8 FreqAI (ML)

```json
{
    "freqai": {
        "enabled": true,
        "purge_old_models": 2,
        "train_period_days": 30,
        "backtest_period_days": 7,
        "identifier": "unique_id",
        "feature_parameters": {
            "include_timeframes": ["5m", "15m", "1h"],
            "include_corr_pairlist": ["BTC/USDT", "ETH/USDT"],
            "label_period_candles": 24,
            "include_shifted_candles": 2,
            "DI_threshold": 0.9
        },
        "data_split_parameters": {"test_size": 0.33},
        "model_training_parameters": {"n_estimators": 200}
    }
}
```

---

## 3. Написание стратегии

Создание шаблона:

```bash
freqtrade new-strategy --strategy MyStrategy
# → user_data/strategies/MyStrategy.py
```

Минимальная структура класса (паттерн Template Method):

```python
from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta


class MyStrategy(IStrategy):
    # Версия контракта с фреймворком.
    # v3 использует enter_long/exit_long вместо старых buy/sell.
    INTERFACE_VERSION = 3

    # ── Обязательные атрибуты ──────────────────────────
    timeframe = "5m"
    stoploss = -0.10
    minimal_roi = {"60": 0.01, "30": 0.02, "0": 0.04}
    startup_candle_count = 200

    # ── Опциональные ───────────────────────────────────
    trailing_stop = False
    can_short = False  # True для шорта на фьючерсах

    # Hyperopt-параметры
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True)

    # ── 3 обязательных метода ──────────────────────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Добавляет технические индикаторы в DataFrame."""
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Проставляет сигналы входа."""
        dataframe.loc[
            (dataframe["rsi"] < self.buy_rsi.value) & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Проставляет сигналы выхода."""
        dataframe.loc[
            (dataframe["rsi"] > 70) & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
```

### Дополнительные callback-методы стратегии

| Метод                      | Когда вызывается                                          |
| -------------------------- | --------------------------------------------------------- |
| `custom_stoploss()`        | Каждый тик — возвращает динамический стоп-лосс            |
| `custom_exit()`            | Каждый тик — возвращает строку-причину кастомного выхода  |
| `custom_entry_price()`     | При входе — переопределяет цену ордера                    |
| `custom_exit_price()`      | При выходе — переопределяет цену ордера                   |
| `custom_stake_amount()`    | Рассчитывает размер ставки для каждой сделки              |
| `confirm_trade_entry()`    | Финальное подтверждение входа (можно отменить)            |
| `confirm_trade_exit()`     | Финальное подтверждение выхода                            |
| `adjust_trade_position()`  | DCA — докупка или сокращение позиции                      |
| `leverage()`               | Задаёт плечо для фьючерсов                               |
| `informative_pairs()`      | Объявляет дополнительные пары/таймфреймы для индикаторов  |
| `bot_start()`              | Вызывается один раз при запуске бота                      |

---

## 4. Рабочий процесс

**Шаг 1 — Скачать данные:**

```bash
freqtrade download-data --exchange binance --pairs BTC/USDT ETH/USDT \
    --timeframes 5m 1h --days 365
```

**Шаг 2 — Бэктест:**

```bash
freqtrade backtesting --strategy MyStrategy --config user_data/config.json \
    --timerange 20240101-20241231
```

**Шаг 3 — Оптимизация гиперпараметров:**

```bash
freqtrade hyperopt --strategy MyStrategy --hyperopt-loss SharpeHyperOptLoss \
    --spaces buy sell roi stoploss --epochs 500
```

**Шаг 4 — Dry run (реальные котировки, виртуальные деньги):**

```bash
freqtrade trade --config user_data/config.json --strategy MyStrategy --dry-run
```

**Шаг 5 — Live trading:**

```bash
# В config.json: "dry_run": false, прописаны API-ключи биржи
freqtrade trade --config user_data/config.json --strategy MyStrategy
```

---

## 5. Переопределение параметров через CLI

Многие параметры конфига можно переопределить прямо из командной строки — без редактирования файла:

```bash
freqtrade trade \
    --config user_data/config.json \
    --strategy MyStrategy \
    --dry-run \
    --dry-run-wallet 1000 \
    --stake-amount 50 \
    --max-open-trades 3 \
    --timeframe 15m \
    --logfile user_data/logs/bot.log \
    --db-url sqlite:///user_data/tradesv3.sqlite
```

---

## 6. Чеклист: что пользователь ДОЛЖЕН настроить

| Что                               | Где                          | Обязательно  |
| --------------------------------- | ---------------------------- | ------------ |
| API-ключи биржи                   | `config.json` → `exchange`   | для live     |
| Название стратегии                | `config.json` → `strategy`   | да           |
| `stake_currency` + `stake_amount` | `config.json`                | да           |
| `dry_run: true/false`             | `config.json`                | да           |
| Pairlist (минимум один метод)     | `config.json` → `pairlists`  | да           |
| `populate_indicators()`           | Python-файл стратегии        | да           |
| `populate_entry_trend()`          | Python-файл стратегии        | да           |
| `populate_exit_trend()`           | Python-файл стратегии        | да           |
| `stoploss` + `minimal_roi`        | Стратегия или конфиг         | да           |
| Telegram-токен                    | `config.json` → `telegram`   | опционально  |
| Пароль API Server                 | `config.json` → `api_server` | опционально  |
| FreqAI-конфиг                     | `config.json` → `freqai`     | только с ML  |

---

## Как работает pipeline стратегии

### Общая схема

```
Exchange.fetch_ohlcv()
    → DataFrame [open, high, low, close, volume]
        → populate_indicators()       # добавить колонки индикаторов
            → populate_entry_trend()  # добавить enter_long / enter_short
                → populate_exit_trend()   # добавить exit_long / exit_short
                    → FreqtradeBot.handle_trade()  # прочитать сигналы, выставить ордера
```

Все три метода — **чистые функции над DataFrame**: получают таблицу свечей, возвращают её же с дополнительными колонками. Никакого состояния, никаких побочных эффектов.

---

## Что такое DataFrame в этом контексте

Каждая строка — одна свеча. Изначально 5 колонок (+`date`):

| date             | open    | high    | low     | close   | volume |
| ---------------- | ------- | ------- | ------- | ------- | ------ |
| 2024-01-01 00:00 | 42100.0 | 42350.0 | 41900.0 | 42200.0 | 1523.4 |
| 2024-01-01 00:05 | 42200.0 | …       | …       | …       | …      |

После `populate_indicators()` появляются колонки индикаторов:

| … | rsi  | ema20   | macd  | bb_upper | bb_lower |
| - | ---- | ------- | ----- | -------- | -------- |
| … | 34.2 | 41800.0 | −12.5 | 43100.0  | 41200.0  |

После `populate_entry_trend()` и `populate_exit_trend()` — **сигнальные колонки**:

| … | enter_long | enter_short | exit_long | exit_short |
| - | ---------- | ----------- | --------- | ---------- |
| … | 0          | 0           | 0         | 0          |
| … | **1**      | 0           | 0         | 0          |

---

## populate_indicators — детали

**Задача:** заполнить DataFrame техническими индикаторами, которые понадобятся для сигналов.

```python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # metadata["pair"] == "BTC/USDT" — можно считать разные индикаторы для разных пар

    # ta-lib — обёртка над C-библиотекой, очень быстрая
    dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
    dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)

    # Bollinger Bands возвращают dict
    bb = qtpylib.bollinger_bands(dataframe["close"], window=20, stds=2)
    dataframe["bb_upper"] = bb["upper"]
    dataframe["bb_lower"] = bb["lower"]

    return dataframe  # ОБЯЗАТЕЛЬНО вернуть
```

> **Данные с других пар/таймфреймов.** Если нужны индикаторы по другой паре (например, BTC/USDT как индикатор рынка) или другому таймфрейму, используйте декоратор `@informative` или метод `informative_pairs()` + `merge_informative_pair()`. Прямой доступ через `dp.get_pair_dataframe()` внутри `populate_indicators()` в бэктесте приводит к **look-ahead bias**.

**`startup_candle_count`** — критически важный атрибут. Если RSI требует 14 свечей прогрева, а EMA — 200, ставьте `startup_candle_count = 200`. Фреймворк запрашивает это количество дополнительных свечей при старте и исключает их из торговли и бэктеста.

---

## populate_entry_trend — детали

**Задача:** в строках, где нужен вход, поставить `1` в колонку `enter_long` или `enter_short`.

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (dataframe["rsi"] < 30)                       # RSI перепродан
            & (dataframe["close"] > dataframe["ema20"])   # цена выше EMA
            & (dataframe["volume"] > 0)                   # защита от пустых свечей
        ),
        "enter_long",  # имя колонки зафиксировано фреймворком
    ] = 1

    # Для шорта (если can_short = True):
    dataframe.loc[
        (dataframe["rsi"] > 70) & (dataframe["volume"] > 0),
        "enter_short",
    ] = 1

    return dataframe
```

Дополнительно можно передать **`enter_tag`** — строковый тег, который записывается в БД и помогает анализировать, какой именно сигнал сработал:

```python
dataframe.loc[condition_rsi, "enter_tag"] = "rsi_oversold"
dataframe.loc[condition_macd, "enter_tag"] = "macd_cross"
```

---

## populate_exit_trend — детали

**Задача:** аналогично, но для выхода. Колонки: `exit_long`, `exit_short`.

```python
def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (dataframe["rsi"] > 70) & (dataframe["volume"] > 0),
        "exit_long",
    ] = 1
    return dataframe
```

**Важно:** этот сигнал — лишь один из способов выхода. FreqtradeBot также закроет позицию по:

- **`minimal_roi`** — достигнута целевая прибыль
- **`stoploss`** — достигнут стоп
- **`trailing_stop`** — сработал скользящий стоп
- **`custom_exit()`** — если переопределён в стратегии
- **`force_exit`** — ручная команда через Telegram/API

---

## Как DataFrame связан с реальной торговлей

На каждой итерации цикла FreqtradeBot (упрощённо):

```
1. DataProvider.get_analyzed_dataframe(pair, timeframe)
       Если появилась новая свеча:
           strategy.analyze_pair(pair)
               dp.ohlcv(pair)           → свежие OHLCV с биржи
               populate_indicators()    → твой код
               populate_entry_trend()   → твой код
               populate_exit_trend()    → твой код
               результат → кэш DataProvider

2. handle_trade(trade):
       analyzed_df = dp.get_analyzed_dataframe(pair)
       last_candle = analyzed_df.iloc[-2]
       if last_candle["exit_long"] == 1:
           execute_trade_exit()
           Exchange.create_order("sell", ...)

3. enter_positions():
       last_candle = analyzed_df.iloc[-2]
       if last_candle["enter_long"] == 1:
           execute_entry()
           Exchange.create_order("buy", ...)
           Trade.save()  → SQLite
```

**Почему `iloc[-2]`, а не `iloc[-1]`?**

Последняя свеча (`iloc[-1]`) ещё **не закрыта** — её значения O/H/L/C продолжают меняться. Фреймворк принципиально использует **предпоследнюю (закрытую) свечу** для принятия решений. Это фундаментальное правило предотвращения look-ahead bias, а не просто побочный эффект флага `process_only_new_candles`.

---

## Hyperopt — оптимизация параметров

Параметры стратегии можно сделать **оптимизируемыми**:

```python
class MyStrategy(IStrategy):
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell", optimize=True)
    bb_window = IntParameter(10, 50, default=20, space="buy")
    risk_coeff = DecimalParameter(0.01, 0.1, default=0.05, decimals=3, space="buy")
```

Типы параметров:

| Тип                    | Диапазон          | Пример                            |
| ---------------------- | ----------------- | ---------------------------------- |
| `IntParameter`         | Целые числа       | Период RSI                         |
| `DecimalParameter`     | Дробные           | Коэффициент риска                  |
| `RealParameter`        | Float             | Множитель                          |
| `CategoricalParameter` | Список значений   | `[«ema», «sma», «wma»]`           |
| `BooleanParameter`     | true / false      | Включить/выключить фильтр         |

Внутри метода обращение через `.value`:

```python
def populate_indicators(self, dataframe, metadata):
    dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
    # bb_window.value — при обычной торговле = default (20),
    # при Hyperopt = значение текущей итерации
    bb = qtpylib.bollinger_bands(dataframe["close"], window=self.bb_window.value)
    ...
```

Запуск:

```bash
freqtrade hyperopt --strategy MyStrategy \
    --hyperopt-loss SharpeHyperOptLoss \
    --spaces buy sell \
    --epochs 500
```

Лучший результат сохраняется в `user_data/strategies/.MyStrategy.json` и подгружается автоматически при следующем запуске.

---

## Схема потока данных (полная)

```
Биржа
  ↓ ccxt.fetch_ohlcv()
DataFrame [date, O, H, L, C, V]
  ↓ populate_indicators()        ← ТВОЙ КОД: технический анализ
DataFrame + [rsi, ema, macd, bb, ...]
  ↓ populate_entry_trend()       ← ТВОЙ КОД: логика входа
DataFrame + [enter_long, enter_short, enter_tag]
  ↓ populate_exit_trend()        ← ТВОЙ КОД: логика выхода
DataFrame + [exit_long, exit_short]
  ↓ кэш в DataProvider (in-memory)
  ↓
FreqtradeBot.handle_trade()
  ↓ читает iloc[-2]["enter_long"] == 1?
  ↓
Exchange.create_order()           ← ордер на бирже
  ↓
Trade.save()                      ← запись в SQLite
  ↓
RPCManager.send_msg()             ← уведомление в Telegram / WS
```

---

## Ограничения и подводные камни

| Проблема | Описание |
| -------- | -------- |
| **Look-ahead bias в бэктесте** | Нельзя использовать `dp.get_pair_dataframe()` напрямую внутри `populate_indicators()` для других пар — в бэктесте это даёт доступ к «будущим» данным. Используйте `@informative` или `merge_informative_pair()` |
| **Overfitting при Hyperopt** | Большое число эпох + узкий timerange = параметры, идеально подогнанные под прошлое, но бесполезные в будущем. Всегда проверяйте результат на out-of-sample данных |
| **Ограничения бирж** | Rate limits, разные правила маржи, неполные OHLCV-данные — всё это может вести к расхождению бэктеста и live-торговли |
| **Не все ордеры исполняются** | В бэктесте лимитный ордер всегда исполняется, на live — нет. Параметр `unfilledtimeout` контролирует, через сколько отменять неисполненный ордер |
| **Startup candle count** | Если `startup_candle_count` меньше, чем требует самый «длинный» индикатор, первые сигналы будут основаны на некорректных значениях (NaN) |
```
