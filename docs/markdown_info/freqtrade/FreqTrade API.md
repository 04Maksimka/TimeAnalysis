## Общая картина

В Freqtrade  несколько API для разных задач:
- REST-эндпоинты для UI
- Внутренние Python-интерфейсы между модулями
- Протокол обмена между ботами.

```
┌─────────────────────────────────────────────────────────┐
│                      Freqtrade                          │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  IStrategy  │    │  IFreqaiModel│    │  Exchange  │  │
│  │  (Python    │    │  (Python     │    │  (ccxt     │  │
│  │   API)      │    │   API)       │    │   API)     │  │
│  └─────────────┘    └──────────────┘    └────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              RPCManager                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │   │
│  │  │ Telegram │  │ApiServer │  │ WebhookRPC    │   │   │
│  │  │   RPC    │  │(FastAPI) │  │               │   │   │
│  │  └──────────┘  └──────────┘  └───────────────┘   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Producer/Consumer WebSocket API          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

Итого: **пять разных API**, каждый для своего слоя.

---

## 1. IStrategy API — Python-интерфейс стратегии

**Что это:** абстрактный базовый класс `IStrategy` в `freqtrade/strategy/interface.py`. Контракт между фреймворком и пользовательским кодом.

**Зачем:** пользователь наследуется от `IStrategy` и переопределяет методы. Фреймворк вызывает их в нужный момент — классический паттерн Template Method.

**Кто вызывает:** `FreqtradeBot` и `Backtesting` напрямую через Python.

```python
class IStrategy(ABC):
    # Обязательные методы — фреймворк вызывает их сам
    @abstractmethod
    def populate_indicators(self, dataframe, metadata): ...

    @abstractmethod
    def populate_entry_trend(self, dataframe, metadata): ...

    @abstractmethod
    def populate_exit_trend(self, dataframe, metadata): ...

    # Опциональные callback-и — переопределяются по необходимости
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs): ...
    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs): ...
    def custom_entry_price(self, pair, current_time, proposed_rate, entry_tag, side, **kwargs): ...
    def custom_exit_price(self, pair, trade, current_time, proposed_rate, proposed_profit, **kwargs): ...
    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake, ...): ...
    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, side, **kwargs): ...
    def confirm_trade_exit(self, pair, trade, order_type, amount, rate, time_in_force, ...): ...
    def adjust_trade_position(self, trade, current_time, current_rate, current_profit, ...): ...
    def leverage(self, pair, current_time, current_rate, proposed_leverage, side, **kwargs): ...
    def informative_pairs(self): ...
    def bot_start(self): ...
    def bot_loop_start(self, current_time, **kwargs): ...
```

Это не HTTP и не сокеты — обычный Python. Фреймворк вызывает методы объекта стратегии напрямую в том же процессе.

---

## 2. Exchange API — обёртка над ccxt

**Что это:** модуль `freqtrade/exchange/` — обёртка над библиотекой ccxt, которая унифицирует работу со всеми биржами.

**Зачем:** изолирует остальной код от особенностей конкретных бирж. Стратегия и бот работают с единым интерфейсом `Exchange`.

**Кто вызывает:** `FreqtradeBot`, `DataProvider`, `Backtesting` — через внутренний Python-объект.

```
freqtrade/exchange/
├── exchange.py          ← базовый класс Exchange (fetch_ohlcv, create_order, cancel_order...)
├── binance.py           ← адаптер для Binance (переопределяет специфичные методы)
├── bybit.py             ← адаптер для Bybit
├── kraken.py            ← адаптер для Kraken
├── gate.py
├── okx.py
├── ... (ещё ~15 адаптеров)
└── common.py            ← @retrier декоратор для повтора при сетевых ошибках
```

Класс `Exchange` — фасад над ccxt:
- добавляет retry-логику через декоратор `@retrier`
- нормализует форматы данных (OHLCV, ордера, балансы)
- кэширует markets и ticker-данные
- обрабатывает специфику каждой биржи через адаптеры

```python
# Внутри FreqtradeBot — вызывается напрямую
candles = self.exchange.get_candle_history(pair, timeframe, since_ms)
order = self.exchange.create_order(pair, ordertype, side, amount, rate)
self.exchange.cancel_order(order_id, pair)
```

Снаружи этот слой недоступен.

---

## 3. RPC API — внешние интерфейсы управления

**Что это:** модуль `freqtrade/rpc/` — все каналы, через которые внешний мир общается с ботом.

### 3.1. REST API + WebSocket (ApiServer)

ApiServer — встроенный веб-сервер на FastAPI, который запускается в отдельном потоке рядом с торговым циклом.

**Почему отдельный поток, а не процесс:** ApiServer должен иметь прямой доступ к объектам FreqtradeBot (Trade, Wallets и т.д.) без сериализации. Поэтому он живёт в том же процессе Python, но в отдельном asyncio event loop.

Весь код живёт в `freqtrade/rpc/api_server/`:

```
api_server/
├── api_server.py        ← запуск Uvicorn + FastAPI app, монтирование роутеров
├── api_auth.py          ← JWT аутентификация (/token/login, /token/refresh)
├── api_v1.py            ← информационные эндпоинты (чтение)
├── api_trading.py       ← торговые и управляющие эндпоинты (чтение + запись)
├── api_schemas.py       ← Pydantic-схемы для валидации запросов/ответов
├── api_ws.py            ← WebSocket-эндпоинт (/message/ws)
└── ui/                  ← статика FreqUI (HTML/JS/CSS)
```

Четыре модуля ApiServer решают разные задачи:

#### api_auth.py — кто ты такой?

Отвечает на вопрос: имеет ли право этот клиент делать запросы?

Клиент сначала логинится (`POST /token/login`) с логином и паролем из конфига. В ответ получает два токена — короткоживущий `access_token` (15 минут) и долгоживущий `refresh_token`. Дальше к каждому запросу клиент прикладывает `access_token` в заголовке `Authorization: Bearer ...`. Когда он истекает — обменивает `refresh_token` на новый `access_token` без повторного ввода пароля.

Все остальные три модуля просто говорят FastAPI: «этот эндпоинт требует валидный токен» — через `Depends(get_current_user)`.

Для WebSocket используется отдельный механизм: `validate_ws_token` проверяет либо `ws_token` из конфига, либо валидный JWT.

```
POST /api/v1/token/login     → access_token + refresh_token
POST /api/v1/token/refresh   → новый access_token
```

#### api_v1.py — что сейчас происходит?

Только чтение.

Например, версия бота, текущий конфиг, логи, загрузка CPU/RAM, стратегии, данные для графиков.

```
GET /api/v1/ping              — жив ли бот (без авторизации)
GET /api/v1/version           — версия (без авторизации)
GET /api/v1/show_config       — конфиг + стратегия + State
GET /api/v1/balance           — баланс счёта
GET /api/v1/count             — количество открытых сделок
GET /api/v1/entries           — статистика по enter_tag
GET /api/v1/exits             — статистика по exit_reason
GET /api/v1/logs              — последние строки логов
GET /api/v1/sysinfo           — CPU, RAM
GET /api/v1/health            — last_process timestamp
GET /api/v1/plot_config       — конфиг для графиков
GET /api/v1/strategies        — список доступных стратегий
GET /api/v1/strategy/{name}   — код стратегии
GET /api/v1/available_pairs   — доступные пары по таймфрейму
GET /api/v1/pair_history      — свечи с индикаторами
POST /api/v1/pair_candles     — свечи с фильтрацией колонок (read-only, но с телом)
```

Все эти эндпоинты делегируют в `RPC._rpc_*()` методы.

#### api_trading.py — сделай что-нибудь

Чтение торговых данных, управление ботом. Самый большой модуль ApiServer.

**GET (чтение):**

```
GET /api/v1/status             — открытые позиции
GET /api/v1/trades             — история сделок (limit, offset)
GET /api/v1/trade/{id}         — конкретная сделка
GET /api/v1/profit             — P&L-статистика
GET /api/v1/performance        — прибыль по парам
GET /api/v1/daily              — прибыль по дням
GET /api/v1/weekly             — прибыль по неделям
GET /api/v1/monthly            — прибыль по месяцам
GET /api/v1/historic_balance   — временной ряд баланса (equity)
GET /api/v1/whitelist          — текущий whitelist
GET /api/v1/blacklist          — текущий blacklist
GET /api/v1/locks              — активные блокировки пар
GET /api/v1/pair_candles       — свечи с индикаторами для пары
```

**POST (управление):**

```
POST /api/v1/start             — запустить бота (State → RUNNING)
POST /api/v1/stop              — остановить бота (State → STOPPED)
POST /api/v1/pause             — запретить новые входы
POST /api/v1/stopentry         — то же что pause
POST /api/v1/reload_config     — перечитать конфиг и стратегию
POST /api/v1/forceenter        — принудительно открыть позицию
POST /api/v1/forceexit         — принудительно закрыть позицию
POST /api/v1/blacklist         — добавить пары в blacklist
POST /api/v1/locks             — добавить блокировку пары
POST /api/v1/backtesting       — запустить backtest через API (для FreqUI)
```

**DELETE:**

```
DELETE /api/v1/trades/{id}           — удалить сделку из БД
DELETE /api/v1/trades/{id}/open-order — отменить ордер на бирже
DELETE /api/v1/blacklist             — убрать пары из blacklist
DELETE /api/v1/locks/{id}            — снять блокировку
DELETE /api/v1/backtesting           — прервать backtest
```

#### api_ws.py — говори мне сразу, не жди пока спрошу

WebSocket-соединение. Клиент подключается один раз и подписывается на нужные события. Бот сам присылает сообщения когда что-то происходит: открылась позиция, закрылась, появились новые свечи.

Внутри работает через `MessageStream` — asyncio-очередь. Когда бот что-то делает, `RPCManager.send_msg()` кладёт событие в очередь. `WebSocketChannel` для каждого подключённого клиента читает из очереди и пушит те события, на которые клиент подписан.

```
WS /api/v1/message/ws?token=ws_token
```

**Что клиент может отправить через WebSocket:**

| Тип запроса    | Данные              | Что происходит                                |
| -------------- | ------------------- | --------------------------------------------- |
| `subscribe`    | `[«entry_fill», «exit_fill», ...]` | Зарегистрировать подписки на типы событий |
| `whitelist`    | `{}`                | Запросить текущий whitelist                    |
| `analyzed_df`  | `{pair, limit}`     | Запросить DataFrame с индикаторами для пары   |

**Что клиент получает (push от бота):**

| Тип события    | Когда                                    |
| -------------- | ---------------------------------------- |
| `entry_fill`   | Ордер входа исполнен                     |
| `exit_fill`    | Ордер выхода исполнен                    |
| `entry_cancel` | Ордер входа отменён                      |
| `exit_cancel`  | Ордер выхода отменён                     |
| `analyzed_df`  | Ответ на запрос свечей                   |
| `whitelist`    | Ответ на запрос whitelist                |
| `status`       | Статус бота изменился (start/stop)       |

#### Как модули ApiServer связаны между собой

![[sequence_apiserver_freqtrade.svg]]

Ключевой момент: `api_v1` и `api_trading` не обращаются к `FreqtradeBot` напрямую. Они вызывают методы `RPC._rpc_*()`, а `RPC` уже читает данные из бота. WebSocket (`api_ws`) не вызывает даже `RPC` — он читает из `MessageStream`, куда `RPCManager` кладёт события.

### 3.2. Telegram RPC

**Файл:** `freqtrade/rpc/telegram.py`

Telegram-бот работает параллельно с ApiServer. Он подписывается на те же события через `RPCManager` и принимает команды от пользователя в чате.

```
Пользователь в Telegram         FreqtradeBot
        │                             │
        │  /status                    │
        │ ──────────────────────────► │
        │                    RPC._rpc_trade_status()
        │ ◄────────────────────────── │
        │  "BTC/USDT: +2.3% ..."      │
        │                             │
        │  /forceexit 42              │
        │ ──────────────────────────► │
        │                    RPC._rpc_force_exit(42)
        │ ◄────────────────────────── │
        │  "Trade #42 closed"         │
```

Команды Telegram и REST API вызывают одни и те же методы `RPC` — логика не дублируется. Разница только в транспорте.

**Доступные команды:**

```
/start, /stop, /pause          — управление ботом
/status                        — открытые позиции
/profit                        — статистика P&L
/balance                       — баланс
/trades [N]                    — последние N сделок
/forceexit <id>                — принудительный выход
/forceenter <pair> [side]      — принудительный вход
/blacklist [pair]              — добавить в blacklist
/whitelist                     — показать whitelist
/logs [N]                      — последние N строк логов
/reload_config                 — перечитать конфиг
/performance                   — статистика по парам
```

### 3.3. Webhook RPC

**Файл:** `freqtrade/rpc/webhook.py`

Отправляет HTTP POST-запросы на заданный URL при наступлении событий — push-уведомления для интеграции с внешними системами.

```json
{
    "webhook": {
        "enabled": true,
        "url": "https://your-server.com/hooks/freqtrade",
        "webhookentry": {
            "type": "entry",
            "pair": "{pair}",
            "stake_amount": "{stake_amount}"
        },
        "webhookexitfill": {
            "type": "exit",
            "pair": "{pair}",
            "profit_ratio": "{profit_ratio}"
        }
    }
}
```

В отличие от REST API (где UI опрашивает бота), Webhook — обратная схема: бот сам толкает данные наружу.

### 3.4. RPCManager — диспетчер всех RPC

**Файл:** `freqtrade/rpc/rpc_manager.py`

`RPCManager` — центральный диспетчер:
- хранит список всех активных RPC-обработчиков (ApiServer, Telegram, Webhook)
- при наступлении события вызывает `send_msg()` у каждого
- предоставляет единые методы (`_rpc_trade_status`, `_rpc_force_exit` и т.д.), вызываемые из любого канала

```python
# FreqtradeBot при открытии сделки:
self.rpc.send_msg({
    "type": RPCMessageType.ENTRY,
    "pair": trade.pair,
    "open_rate": trade.open_rate,
    ...
})

# RPCManager рассылает всем:
# → ApiServer → MessageStream → WebSocket-клиенты получают push
# → Telegram отправляет сообщение в чат
# → Webhook делает POST на внешний URL
```

---

## 4. Producer/Consumer WebSocket API

Отдельный механизм, не связанный с FreqUI. Позволяет одному боту (Producer) передавать проанализированные данные другому боту (Consumer) по WebSocket.

**Зачем:** разделить вычисление сигналов и торговлю. Producer запускает тяжёлые вычисления (FreqAI, множество пар), Consumer получает готовые сигналы и торгует.

```
Producer-бот                    Consumer-бот
(считает индикаторы,            (только торгует,
 FreqAI, много пар)              получает сигналы)
        │                              │
        │   WS /api/v1/message/ws      │
        │ ◄──────────────────────────  │
        │  analyzed_df (DataFrame)     │
        │ ──────────────────────────►  │
        │                              │
```

**Настройка Consumer:**

```json
{
    "external_message_consumer": {
        "enabled": true,
        "producers": [
            {
                "name": "main",
                "host": "producer-host",
                "port": 8080,
                "ws_token": "producer_ws_token"
            }
        ]
    }
}
```

**Стратегия Consumer** читает данные через `dp.get_producer_df()`:

```python
def populate_indicators(self, dataframe, metadata):
    producer_df, last_analyzed = self.dp.get_producer_df(
        metadata["pair"],
        self.timeframe,
        producer_name="main"
    )
    if not producer_df.empty:
        dataframe = merge_informative_pair(
            dataframe, producer_df, self.timeframe, self.timeframe
        )
    return dataframe
```

Технически Consumer использует тот же `/api/v1/message/ws`, что и FreqUI, но подписывается на `analyzed_df` и использует данные иначе.

---

## 5. IFreqaiModel API — интерфейс ML-моделей

Абстрактный класс `IFreqaiModel` в `freqtrade/freqai/base_models/`. Контракт для кастомных ML-моделей — аналог IStrategy, но для FreqAI.

```python
class IFreqaiModel(ABC):
    @abstractmethod
    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """Обучить модель на подготовленных данных."""
        ...

    @abstractmethod
    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        """Сделать предсказание на новых данных."""
        ...
```

Встроенные реализации:

```
freqtrade/freqai/prediction_models/
├── LightGBMRegressor.py
├── LightGBMClassifier.py
├── XGBoostRegressor.py
├── XGBoostClassifier.py
├── PyTorchMLPRegressor.py
├── PyTorchTransformerRegressor.py
├── ReinforcementLearner.py
└── ...
```

Кастомная модель — файл в `user_data/freqaimodels/`:

```python
from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

class MyCustomModel(FreqaiMultiOutputRegressor):
    def fit(self, data_dictionary, dk, **kwargs):
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        self.estimator = GradientBoostingRegressor(**self.model_training_parameters)
        self.estimator.fit(X, y.values.ravel())
        return self.estimator

    def predict(self, unfiltered_df, dk, **kwargs):
        # ... предсказание
```

```bash
freqtrade trade \
    --strategy MyStrategy \
    --freqaimodel MyCustomModel \
    --freqaimodel-path user_data/freqaimodels
```

---

## Сводная таблица

| API                      | Тип            | Транспорт            | Кто использует                    | Файлы                       |
| ------------------------ | -------------- | -------------------- | --------------------------------- | --------------------------- |
| **IStrategy**            | Внутренний     | Python (in-process)  | Пользователь (стратегия)          | `strategy/interface.py`     |
| **Exchange (ccxt)**      | Внутренний     | Python → HTTP (ccxt) | FreqtradeBot, DataProvider        | `exchange/`                 |
| **REST API (FastAPI)**   | Внешний        | HTTP + WebSocket     | FreqUI, кастомный UI, скрипты     | `rpc/api_server/`           |
| **Telegram RPC**         | Внешний        | Telegram Bot API     | Пользователь в Telegram           | `rpc/telegram.py`           |
| **Webhook RPC**          | Внешний (push) | HTTP POST            | Внешние системы, алерты           | `rpc/webhook.py`            |
| **Producer/Consumer WS** | Внешний        | WebSocket            | Другой бот (Consumer)             | `rpc/api_server/api_ws.py`  |
| **IFreqaiModel**         | Внутренний     | Python (in-process)  | Пользователь (ML-модель)          | `freqai/base_models/`       |

---

## Как всё связано при работе с UI

Когда FreqUI открыт в браузере и бот торгует — одновременно работают три канала:

```
FreqUI (браузер)
    │
    ├── GET /api/v1/status  (каждые 5 сек)     ← polling через REST API
    ├── GET /api/v1/balance (каждые 5 сек)
    ├── GET /api/v1/profit  (каждые 60 сек)
    │
    └── WS /api/v1/message/ws                  ← постоянное соединение
            │
            ├── ← entry_fill   (мгновенно при открытии)
            ├── ← exit_fill    (мгновенно при закрытии)
            └── ← analyzed_df  (по запросу, для графика)

Параллельно (независимо от UI):
    Telegram ← RPCManager.send_msg() при каждом событии
    Webhook  ← RPCManager.send_msg() при каждом событии
```

Все каналы получают одни и те же данные через `RPCManager` — бот отправляет событие один раз, диспетчер раздаёт его всем подписчикам.