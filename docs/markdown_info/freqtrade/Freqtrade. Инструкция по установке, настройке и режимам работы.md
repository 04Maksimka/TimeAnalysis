## Общий план

```text
1. Установка
   ├── Linux / macOS — через setup.sh (рекомендуется)
   └── Windows — через WSL2

2. Инициализация проекта
   ├── Создание рабочей директории (create-userdir)
   ├── Создание конфигурации (new-config или вручную)
   └── Разделение секретов (два файла или env-переменные)

3. Написание стратегии
   ├── Классическая стратегия — индикаторы и правила
   └── FreqAI-стратегия — ML-модели вместо ручных правил

4. Подготовка данных
   ├── Скачивание исторических свечей (download-data)
   ├── Конвертация форматов (convert-data)
   └── Проверка скачанного (list-data)

5. Тестирование
   ├── Backtesting — проверка на исторических данных
   ├── Анализ look-ahead и recursive bias
   ├── Hyperopt — автоматическая оптимизация параметров
   └── Валидация — проверка на out-of-sample периоде

6. Анализ результатов
   ├── Просмотр сохранённых бэктестов (backtesting-show)
   ├── Визуализация сделок (plot-dataframe, plot-profit)
   └── Интерактивный анализ в Jupyter

7. Paper Trading (Dry Run)
   ├── Запуск бота с виртуальным счётом
   └── Сравнение результатов с backtesting

8. Мониторинг и управление
   ├── Логирование
   ├── FreqUI — веб-интерфейс в браузере
   └── Telegram — уведомления и команды с телефона

9. Live Trading
   ├── Переключение на реальный счёт
   └── Автозапуск и обслуживание
```

## Последовательность с нуля до live-trading на linux

0. **Установить uv**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Обновить shell
   ```bash
   source ~/.bashrc
   ```
   Проверка
   ```bash
   uv --version
   ```
   
1. **Установить Freqtrade**
   ```bash
   git clone https://github.com/freqtrade/freqtrade.git
   cd freqtrade
   git checkout stable
   ./setup.sh -i
   source ./.venv/bin/activate
   ```
   Проверить установку:
   ```bash
   freqtrade --version
   ```

2. **Создать рабочую директорию проекта**
   ```bash
   freqtrade create-userdir --userdir user_data
   ```

3. **Создать основной конфиг**
   ```bash
   freqtrade new-config --config user_data/config.json
   ```
   Если планируется live-trading, сразу удобно завести отдельный приватный файл:
   ```bash
   nano user_data/config-private.json
   ```
   Пример `user_data/config-private.json`:
   ```json
   {
     "exchange": {
       "key": "YOUR_REAL_API_KEY",
       "secret": "YOUR_REAL_API_SECRET"
     },
     "telegram": {
       "enabled": true,
       "token": "YOUR_TELEGRAM_BOT_TOKEN",
       "chat_id": "YOUR_CHAT_ID"
     }
   }
   ```
   Убедиться, что `config-private.json` не попадёт в git.

4. **Создать шаблон стратегии** (опционально)
   ```bash
   freqtrade new-strategy --userdir user_data\
   --strategy MyStrategy --template minimal
   ```

   Команда создаёт только **заготовку** стратегии.  
   После этого нужно открыть файл и реализовать в нём свою торговую логику:
	- добавить индикаторы в `populate_indicators()`
	- задать условия входа в `populate_entry_trend()`
	- задать условия выхода в `populate_exit_trend()`

   Открыть файл можно, например, так:
   ```bash
   nano user_data/strategies/MyStrategy.py
   ```
   **Либо скачать готовые стратегии из repo:**
   ```bash 
   cd user_data/strategies
   git clone https://github.com/freqtrade/freqtrade-strategies.git
   mkdir -p freqtrade_strategies
   cp -r freqtrade-strategies/user_data/strategies/* freqtrade_strategies/
   rm -rf freqtrade-strategies
   cd ../.. #возвращаемся обратно
   ```
   **И подключать их по ключу  `--strategy-path`  (пример бэктеста):**
   ```bash
   freqtrade backtesting \
    --userdir user_data \
    --config user_data/config.json \
    --strategy SampleStrategy \
    --strategy-path freqtrade-strategies/
   ```


5. **Проверить, что стратегия загружается**
   ```bash
   freqtrade list-strategies --userdir user_data
   ```

6. **Скачать исторические данные**
   Пример: скачать данные за 365 дней для таймфреймов `5m` и `1h`.
   ```bash
   freqtrade download-data --userdir user_data --config user_data/config.json --timeframes 5m 1h --days 365
   ```

7. **Проверить скачанные данные**
   ```bash
   freqtrade list-data --userdir user_data --show-timerange
   ```

8. **Запустить первичный backtesting**
   ```bash
   freqtrade backtesting --userdir user_data --config user_data/config.json --strategy MyStrategy --timeframe 5m --logfile user_data/logs/backtesting.log
   ```
   Если стратегия использует конкретный период, лучше сразу задавать его явно:
   ```bash
   freqtrade backtesting --userdir user_data --config user_data/config.json --strategy MyStrategy --timeframe 5m --timerange 20240101-20241231 --logfile user_data/logs/backtesting.log
   ```

9. **Проверить стратегию на look-ahead и recursive bias**
   ```bash
   freqtrade lookahead-analysis --userdir user_data --config user_data/config.json --strategy MyStrategy --logfile user_data/logs/lookahead.log
   ```

   ```bash
   freqtrade recursive-analysis --userdir user_data --config user_data/config.json --strategy MyStrategy --logfile user_data/logs/recursive.log
   ```

10. **Установить зависимости для hyperopt и запустить оптимизацию**
    Если hyperopt-зависимости ещё не стоят:
    ```bash
    uv pip install -r requirements-hyperopt.txt
    ```

    Запуск hyperopt:
    ```bash
    freqtrade hyperopt --userdir user_data --config user_data/config.json --strategy MyStrategy --hyperopt-loss SharpeHyperOptLossDaily --spaces buy sell --epochs 500 --logfile user_data/logs/hyperopt.log
    ```

11. **Повторно запустить backtesting с новыми параметрами**
    ```bash
    freqtrade backtesting --userdir user_data --config user_data/config.json --strategy MyStrategy --cache none --logfile user_data/logs/backtesting-after-hyperopt.log
    ```

12. **Проверить стратегию на out-of-sample периоде**
    То есть на периоде, который не использовался в hyperopt. Пример:
    ```bash
    freqtrade backtesting --userdir user_data --config user_data/config.json --strategy MyStrategy --timerange 20240701-20241231 --cache none --logfile user_data/logs/backtesting-oos.log
    ```

13. **При необходимости визуализировать результаты**
    Если plot-команды не работают — установить зависимости:
    ```bash
    uv pip install -r requirements-plot.txt
    ```

    График свечей и сигналов:
    ```bash
    freqtrade plot-dataframe --userdir user_data --config user_data/config.json --strategy MyStrategy --pair BTC/USDT --timeframe 5m
    ```

    График equity / прибыли:
    ```bash
    freqtrade plot-profit --userdir user_data --config user_data/config.json --strategy MyStrategy
    ```

14. **Установить FreqUI**
    ```bash
    freqtrade install-ui
    ```

15. **Включить API Server и FreqUI в `config.json`**
    Сгенерировать секреты:
    ```bash
    python -c "import secrets; print(secrets.token_hex(32))"
    python -c "import secrets; print(secrets.token_urlsafe(25))"
    ```

    Добавить в `user_data/config.json` секцию:
    ```json
    "api_server": {
      "enabled": true,
      "listen_ip_address": "127.0.0.1",
      "listen_port": 8080,
      "username": "freqtrader",
      "password": "StrongPassword123!",
      "jwt_secret_key": "PASTE_HEX_SECRET_HERE",
      "ws_token": "PASTE_WS_TOKEN_HERE",
      "CORS_origins": [],
      "enable_openapi": false
    }
    ```
    > `CORS_origins` нужен только если вы используете отдельный frontend на другом origin.  
    > Для встроенного FreqUI обычно достаточно пустого списка `[]`.

16. **Запустить dry run**
    Убедиться, что в `config.json`:
    ```json
    "dry_run": true
    ```

    Затем запустить:
    ```bash
    freqtrade trade --userdir user_data --config user_data/config.json --strategy MyStrategy --logfile user_data/logs/freqtrade-dryrun.log
    ```

17. **Настроить Telegram**
    1. Создать бота через **@BotFather**
    2. Получить свой `chat_id` через **@userinfobot**
    3. Добавить в `config.json` или `config-private.json` секцию:
       ```json
       "telegram": {
         "enabled": true,
         "token": "YOUR_TELEGRAM_BOT_TOKEN",
         "chat_id": "YOUR_CHAT_ID",
         "notification_settings": {
           "entry": "on",
           "exit": "on",
           "entry_fill": "on",
           "exit_fill": "on",
           "status": "on",
           "warning": "on"
         }
       }
       ```

18. **Вести dry run не меньше 2 недель**
    Во время dry run:
    - следить за FreqUI: `http://127.0.0.1:8080`
    - смотреть логи:
      ```bash
      tail -f user_data/logs/freqtrade-dryrun.log
      ```
    - использовать Telegram-команды `/status`, `/profit`, `/balance`, `/trades`
    - при необходимости доработать стратегию и повторить backtesting

19. **Сравнить dry run vs backtesting**
    Сравнить:
    - частоту входов
    - среднюю длительность сделок
    - drawdown
    - причины выхода
    - итоговую прибыль
    - поведение на волатильных участках рынка

20. **Подготовить live-конфиг**
    1. В `config.json` переключить:
       ```json
       "dry_run": false
       ```
    2. Добавить реальные API-ключи в `user_data/config-private.json`
    3. Оставить консервативный `stake_amount`
    4. Проверить, что API-ключи **без права вывода средств**

21. **Запустить live trading**
    ```bash
    freqtrade trade --userdir user_data --config user_data/config.json --config user_data/config-private.json --strategy MyStrategy --logfile user_data/logs/freqtrade-live.log
    ```

22. **Настроить автозапуск через systemd**
    Создать unit-файл:
    ```bash
    sudo nano /etc/systemd/system/freqtrade.service
    ```

    Вставить содержимое:
    ```ini
    [Unit]
    Description=Freqtrade trading bot
    After=network.target

    [Service]
    Type=simple
    User=YOUR_USERNAME
    WorkingDirectory=/home/YOUR_USERNAME/freqtrade
    ExecStart=/home/YOUR_USERNAME/freqtrade/.venv/bin/freqtrade trade --userdir /home/YOUR_USERNAME/freqtrade/user_data --config /home/YOUR_USERNAME/freqtrade/user_data/config.json --config /home/YOUR_USERNAME/freqtrade/user_data/config-private.json --strategy MyStrategy --logfile /home/YOUR_USERNAME/freqtrade/user_data/logs/freqtrade-live.log
    Restart=on-failure
    RestartSec=10

    [Install]
    WantedBy=multi-user.target
    ```

    Заменить:
    - `YOUR_USERNAME` на имя вашего Linux-пользователя
    - `MyStrategy` на имя вашей стратегии

    Перечитать конфигурацию systemd:
    ```bash
    sudo systemctl daemon-reload
    ```

    Включить автозапуск:
    ```bash
    sudo systemctl enable freqtrade
    ```

    Запустить сервис:
    ```bash
    sudo systemctl start freqtrade
    ```

    Проверить статус:
    ```bash
    sudo systemctl status freqtrade
    ```

23. **Обслуживать инстанс**
    Смотреть live-логи сервиса:
    ```bash
    journalctl -u freqtrade -f
    ```

    Проверять статус:
    ```bash
    sudo systemctl status freqtrade
    ```

    Обновлять Freqtrade:
    ```bash
    cd freqtrade
    git pull
    ./setup.sh -u
    sudo systemctl restart freqtrade
    ```

24. **Опционально: анализировать результаты после запуска**
    Просмотр сохранённых бэктестов:
    ```bash
    freqtrade backtesting-show --userdir user_data
    ```

    Скопировать Jupyter-шаблон:
    ```bash
    cp freqtrade/templates/strategy_analysis_example.ipynb user_data/notebooks/
    ```

    Запустить Jupyter:
    ```bash
    pip install jupyter
    jupyter notebook --notebook-dir user_data/notebooks/
    ```

---

## 1. Установка

Для работы нужен Python 3.10+, системные библиотеки для компиляции TA-Lib и сам фреймворк.

### Linux / macOS

```bash
# Клонировать репозиторий
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
git checkout stable

# Запустить установщик
./setup.sh -i
```

`setup.sh -i` делает три вещи:

- Создаёт виртуальное окружение `.venv`
- Скачивает и компилирует TA-Lib из исходников
- Устанавливает все Python-зависимости

Активация окружения (нужна при каждом новом терминале):

```bash
source ./.venv/bin/activate
```

> Для удобства добавьте в `~/.bashrc`:
> ```bash
> alias ft='cd ~/freqtrade && source ./.venv/bin/activate'
> ```

### Windows

Freqtrade на Windows работает через WSL2 (Windows Subsystem for Linux):

```powershell
# В PowerShell от администратора
wsl --install
```

После перезагрузки и создания пользователя Linux — все команды те же, что в разделе Linux.

### Проверка

```bash
freqtrade version
# → freqtrade 2024.XX
```

---

## 2. Инициализация проекта

### 2.1. Создание рабочей директории

```bash
freqtrade create-userdir --userdir user_data
```

Результат:

```
user_data/
├── strategies/          ← файлы стратегий (.py)
├── data/                ← исторические данные (по биржам)
├── logs/                ← логи работы бота
├── backtest_results/    ← результаты бэктестов
├── hyperopts/           ← кастомные loss-функции
├── notebooks/           ← Jupyter-ноутбуки для анализа
└── freqaimodels/        ← кастомные ML-модели
```

### 2.2. Создание конфигурации

**Через интерактивный мастер** — задаёт вопросы о бирже, валюте, режиме работы и генерирует рабочий файл:

```bash
freqtrade new-config --config user_data/config.json
```

**Вручную** — создать `user_data/config.json`. Минимальный рабочий конфиг:

```json
{
    "$schema": "https://schema.freqtrade.io/schema.json",
    "trading_mode": "spot",
    "max_open_trades": 3,
    "stake_currency": "USDT",
    "stake_amount": 100,
    "dry_run": true,
    "dry_run_wallet": 1000,
    "exchange": {
        "name": "binance",
        "key": "",
        "secret": "",
        "pair_whitelist": [
            "BTC/USDT",
            "ETH/USDT"
        ]
    },
    "pairlists": [
        {"method": "StaticPairList"}
    ],
    "entry_pricing": {
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1
    },
    "exit_pricing": {
        "price_side": "other",
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

#### Ключевые параметры конфига

**Режим работы:**

| Параметр | Описание |
| -------- | -------- |
| `dry_run` | `true` — виртуальные деньги, `false` — реальная торговля |
| `dry_run_wallet` | Стартовый виртуальный баланс |
| `trading_mode` | `spot`, `futures` или `margin` |
| `margin_mode` | `isolated` или `cross` (только для futures) |

**Капитал и риски:**

| Параметр | Описание |
| -------- | -------- |
| `max_open_trades` | Максимум одновременных позиций; `-1` — без лимита |
| `stake_amount` | Сумма на одну сделку; `"unlimited"` — делить баланс поровну |
| `stake_currency` | Валюта торговли: USDT, BTC и т. д. |
| `tradable_balance_ratio` | Доля баланса для торговли (0.0–1.0, по умолчанию 0.99) |

**Биржа:**

| Параметр | Описание |
| -------- | -------- |
| `exchange.name` | Название биржи: binance, bybit, kraken, okx... |
| `exchange.key` / `secret` | API-ключи. Не нужны для dry run и backtesting |
| `exchange.pair_whitelist` | Список разрешённых пар |
| `exchange.pair_blacklist` | Список запрещённых пар |

**Pairlist — формирование списка пар:**

`StaticPairList` берёт пары из `pair_whitelist`. Для динамического списка используйте цепочку фильтров:

```json
"pairlists": [
    {"method": "VolumePairList", "number_assets": 20, "sort_key": "quoteVolume"},
    {"method": "AgeFilter", "min_days_listed": 10},
    {"method": "PriceFilter", "low_price_ratio": 0.01},
    {"method": "SpreadFilter", "max_spread_ratio": 0.005}
]
```

**Ценообразование ордеров:**

```json
"entry_pricing": {
    "price_side": "same",
    "use_order_book": true,
    "order_book_top": 1
},
"exit_pricing": {
    "price_side": "other",
    "use_order_book": true,
    "order_book_top": 1
}
```

`"same"` — ставим ордер на лучшую цену со своей стороны (пассивный вход). `"other"` — берём цену противоположной стороны (агрессивный выход).

#### Приоритет настроек

```
CLI-аргументы > Переменные окружения > config файлы (последний приоритетнее) > Параметры стратегии
```

### 2.3. Разделение секретов

API-ключи и токены не должны попадать в git. Два подхода:

**Два файла конфига:**

```bash
freqtrade trade --config user_data/config.json --config user_data/config-private.json
```

`config.json` — публичный, в git. `config-private.json` — в `.gitignore`:

```json
{
    "exchange": {
        "key": "REAL_API_KEY",
        "secret": "REAL_SECRET"
    },
    "telegram": {
        "enabled": true,
        "token": "BOT_TOKEN",
        "chat_id": "CHAT_ID"
    }
}
```

**Переменные окружения:**

```bash
export FREQTRADE__EXCHANGE__KEY=yourKey
export FREQTRADE__EXCHANGE__SECRET=yourSecret
export FREQTRADE__TELEGRAM__TOKEN=botToken
export FREQTRADE__TELEGRAM__CHAT_ID=chatId
```

Двойное подчёркивание `__` соответствует вложенности в JSON.

---

## 3. Написание стратегии

Стратегия — Python-класс, который определяет когда входить и когда выходить.

### 3.1. Создание файла

**Через шаблон:**

```bash
freqtrade new-strategy --strategy MyStrategy --template minimal
```

Доступные шаблоны: `minimal` (только обязательные методы), `full` (с основными callbacks), `advanced` (включая custom_stoploss, custom_exit).

**Проверить что стратегия видна:**

```bash
freqtrade list-strategies --userdir user_data
```

### 3.2. Классическая стратегия

```python
from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta


class MyStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '5m'
    stoploss = -0.05
    minimal_roi = {"0": 0.04, "30": 0.02, "60": 0.01}
    startup_candle_count = 200

    buy_rsi = IntParameter(20, 40, default=30, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] < self.buy_rsi.value)
            & (dataframe['close'] > dataframe['ema20'])
            & (dataframe['volume'] > 0),
            'enter_long',
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > 70) & (dataframe['volume'] > 0),
            'exit_long',
        ] = 1
        return dataframe
```

**Обязательные элементы:**

| Элемент | Описание |
| ------- | -------- |
| `INTERFACE_VERSION = 3` | Версия контракта с фреймворком |
| `timeframe` | Период свечи: `1m`, `5m`, `15m`, `1h`, `4h`, `1d` |
| `stoploss` | Стоп-лосс как доля: `-0.05` = −5% |
| `minimal_roi` | Таблица "время в минутах → минимальная прибыль для выхода" |
| `startup_candle_count` | Свечи для прогрева индикаторов (не участвуют в торговле) |
| `populate_indicators()` | Добавить колонки индикаторов в DataFrame |
| `populate_entry_trend()` | Проставить `enter_long = 1` или `enter_short = 1` |
| `populate_exit_trend()` | Проставить `exit_long = 1` или `exit_short = 1` |

**Опциональные callback-методы:**

| Метод | Когда вызывается |
| ----- | ---------------- |
| `custom_stoploss()` | Каждый тик — вернуть динамический стоп |
| `custom_exit()` | Каждый тик — вернуть причину выхода или `None` |
| `custom_stake_amount()` | При входе — переопределить размер позиции |
| `confirm_trade_entry()` | Финальная проверка перед входом (можно отменить) |
| `confirm_trade_exit()` | Финальная проверка перед выходом |
| `adjust_trade_position()` | DCA — докупка или частичный выход |
| `leverage()` | Задать кредитное плечо для фьючерсов |

### 3.3. FreqAI-стратегия — ML вместо ручных правил

FreqAI встраивает обученную ML-модель прямо в `populate_indicators()`. Вместо фиксированных порогов модель предсказывает будущую доходность.

#### Отличия от классической стратегии

| Аспект | Классическая | FreqAI |
| ------ | ------------ | ------ |
| Сигнал входа | Фиксированные правила (`rsi < 30`) | Предсказание модели (`&-s_close > 0.01`) |
| Индикаторы | Считаются один раз | Строятся через `feature_engineering_*()` |
| Обучение | Нет | Walk-forward: модель переобучается периодически |
| Конфиг | Стандартный | Дополнительная секция `freqai` |

#### Структура FreqAI-стратегии

```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class MyFreqAIStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '5m'
    stoploss = -0.05
    minimal_roi = {"0": 0.1}
    startup_candle_count = 40
    can_short = True

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw-volume"] = dataframe["volume"]
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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["do_predict"] == 1) & (dataframe["&-s_close"] > 0.01),
            "enter_long",
        ] = 1
        dataframe.loc[
            (dataframe["do_predict"] == 1) & (dataframe["&-s_close"] < -0.01),
            "enter_short",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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

**Префиксы колонок FreqAI:**

| Префикс | Направление | Смысл |
| -------- | ----------- | ----- |
| `%-` | Стратегия → модель | Входной признак (feature) |
| `&-` | Модель → стратегия | Предсказание (target) |
| `do_predict` | Модель → стратегия | `1` — предсказание надёжно, `0` — аномальные данные |

#### Секция freqai в конфиге

```json
{
    "freqai": {
        "enabled": true,
        "purge_old_models": 2,
        "train_period_days": 30,
        "backtest_period_days": 7,
        "identifier": "my_model_v1",

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
}
```

| Параметр | Зачем |
| -------- | ----- |
| `train_period_days` | Сколько дней истории используется для обучения |
| `backtest_period_days` | Шаг переобучения в бэктесте (каждые N дней — новая модель) |
| `identifier` | Уникальное имя для кэша. При смене — модель обучается заново |
| `indicator_periods_candles` | Периоды для `feature_engineering_expand_all` |
| `include_timeframes` | Доп. таймфреймы: признаки считаются для каждого |
| `include_corr_pairlist` | Пары-корреляторы: признаки от BTC/ETH добавляются как фичи |
| `DI_threshold` | Порог Dissimilarity Index — при превышении `do_predict = 0` |
| `shuffle: false` | **Обязательно false** для временных рядов |

#### Запуск FreqAI

```bash
# Бэктест
freqtrade backtesting --strategy MyFreqAIStrategy --freqaimodel LightGBMRegressor --timerange 20240101-20250101

# Dry run / Live
freqtrade trade --strategy MyFreqAIStrategy --freqaimodel LightGBMRegressor
```

---

## 4. Подготовка данных

### 4.1. Скачивание исторических свечей

Backtesting и Hyperopt работают на исторических данных. API-ключи не нужны.

```bash
freqtrade download-data --config user_data/config.json --timeframes 5m 1h --timerange 20240101-20250101
```

**Параметры:**

| Параметр | Описание |
| -------- | -------- |
| `--timeframes 5m 1h 4h` | Таймфреймы (множественное число) |
| `--timerange YYYYMMDD-YYYYMMDD` | Диапазон дат |
| `--days N` | Последние N дней (альтернатива timerange) |
| `--pairs BTC/USDT ETH/USDT` | Явно указать пары (если не из конфига) |
| `--exchange binance` | Биржа (если не указана в конфиге) |
| `--trading-mode futures` | Для фьючерсных данных (+ mark price, funding rate) |
| `--erase` | Удалить существующие данные перед загрузкой |
| `--prepend` | Дополнить более ранним периодом |
| `--dataformat-ohlcv feather` | Формат хранения: `feather` (по умолчанию), `json`, `jsongz`, `parquet` |

**Примеры:**

```bash
# Спот, несколько пар и таймфреймов, за год
freqtrade download-data --exchange binance --pairs BTC/USDT ETH/USDT SOL/USDT --timeframes 5m 1h --days 365

# Фьючерсы (автоматически скачивает mark price и funding rate)
freqtrade download-data --exchange binance --pairs BTC/USDT:USDT --trading-mode futures
```

**Сколько данных нужно:**

| Цель | Минимум |
| ---- | ------- |
| Первый тест | 1–3 месяца |
| Нормальная оценка | 6–12 месяцев |
| Надёжная валидация | 1–2 года (разные рыночные фазы) |

### 4.2. Конвертация форматов

Если данные уже скачаны в одном формате, а нужен другой (например, из json в feather для ускорения бэктестов):

```bash
freqtrade convert-data --format-from json --format-to feather --datadir user_data/data/binance
```

**Параметры:**

| Параметр | Описание |
| -------- | -------- |
| `--format-from` | Исходный формат: `json`, `jsongz`, `feather`, `parquet` |
| `--format-to` | Целевой формат |
| `--datadir` | Папка с данными |
| `--pairs BTC/USDT` | Конвертировать только указанные пары |
| `--timeframes 5m 1h` | Конвертировать только указанные таймфреймы |
| `--trading-mode futures` | Для фьючерсных данных |
| `--erase` | Удалить исходные файлы после конвертации |

### 4.3. Проверка скачанного

```bash
freqtrade list-data --userdir user_data --show-timerange
```

Показывает все скачанные пары с диапазонами дат и форматами.

---

## 5. Тестирование

### 5.1. Backtesting — проверка на истории

```bash
freqtrade backtesting --config user_data/config.json --strategy MyStrategy --timeframe 5m --timerange 20240101-20250101
```

**Параметры:**

| Параметр | Описание |
| -------- | -------- |
| `--strategy NAME` | Имя класса стратегии (не файла) |
| `--timeframe 5m` | Таймфрейм |
| `--timerange 20240101-20250601` | Период тестирования |
| `--dry-run-wallet 1000` | Начальный баланс |
| `--breakdown month year` | Разбивка результатов по месяцам / годам |
| `--timeframe-detail 1m` | Детальный ТФ для точного SL/ROI внутри свечи |
| `--fee 0.001` | Кастомная комиссия |
| `--cache none` | Пересчитать, игнорируя кэш |
| `--strategy-list S1 S2` | Сравнить несколько стратегий |
| `--export trades` | Сохранить сделки в JSON (по умолчанию) |
| `--export-filename PATH` | Задать явный путь к файлу результатов |

**Чтение результатов:**

| Таблица | Что показывает |
| ------- | -------------- |
| BACKTESTING REPORT | Статистика по парам: средний профит, win rate, длительность |
| LEFT OPEN TRADES | Позиции, незакрытые на конец периода |
| EXIT REASON STATS | Причины закрытия: roi, stop_loss, exit_signal, trailing |
| SUMMARY METRICS | Ключевые метрики: Sharpe, Sortino, Max Drawdown, Profit Factor |

**На что смотреть:**

- `Sharpe > 1` — приемлемое соотношение доходности к риску
- `Max Drawdown < 20%` — терпимая просадка
- `Profit Factor > 1.5` — прибыль значительно превышает убытки
- `Win Rate` — сам по себе мало значит без expectancy

**Ограничения:** ордера всегда исполняются по запрошенной цене, нет slippage. Backtesting не заменяет dry run.

**Повторный просмотр сохранённых результатов:**

Freqtrade сохраняет каждый бэктест в `user_data/backtest_results/`. Просмотреть результаты без повторного запуска:

```bash
freqtrade backtesting-show
```

С фильтром по стратегии или файлу:

```bash
freqtrade backtesting-show --strategy MyStrategy
freqtrade backtesting-show --export-filename user_data/backtest_results/backtest-result-2024-01-01.json
```

Полезно для сравнения нескольких прогонов без перезапуска бэктеста.

### 5.2. Анализ look-ahead и recursive bias

Две распространённые ошибки при написании стратегий, которые делают результаты бэктеста нереалистично хорошими:

**Look-ahead bias** — стратегия случайно использует будущие данные. Например, индикатор считается по незакрытой свече или `shift(-1)` попадает в сигнал.

**Recursive bias** — индикатор использует результаты собственных предыдущих вычислений на этих же данных, что невозможно воспроизвести в live.

Freqtrade предоставляет два инструмента для проверки:

```bash
# Проверка look-ahead bias
freqtrade lookahead-analysis --strategy MyStrategy --timerange 20240101-20250101

# Проверка recursive bias
freqtrade recursive-analysis --strategy MyStrategy --timerange 20240101-20250101
```

**Параметры `lookahead-analysis`:**

| Параметр | Описание |
| -------- | -------- |
| `--strategy NAME` | Стратегия для проверки |
| `--timerange` | Период анализа |
| `--startup-candle-count N` | Переопределить startup_candle_count |

**Параметры `recursive-analysis`:**

| Параметр | Описание |
| -------- | -------- |
| `--strategy NAME` | Стратегия для проверки |
| `--timerange` | Период анализа |
| `--pairs BTC/USDT` | Ограничить конкретными парами |
| `--targeted-trade-amount N` | Количество сделок для анализа |

Оба инструмента выводят список колонок с подозрительным поведением. Если после исправления результаты бэктеста резко ухудшились — стратегия действительно использовала будущие данные.

### 5.3. Hyperopt — автоматическая оптимизация

Автоматически находит лучшие значения параметров стратегии. Требует `IntParameter`, `DecimalParameter` и т. д. в коде стратегии.

**Установка зависимостей:**

```bash
pip install -r requirements-hyperopt.txt
```

**Запуск:**

```bash
freqtrade hyperopt --config user_data/config.json --strategy MyStrategy --hyperopt-loss SharpeHyperOptLossDaily --spaces buy sell --epochs 500 --timerange 20240101-20250101
```

**Параметры:**

| Параметр | Описание |
| -------- | -------- |
| `--hyperopt-loss NAME` | Целевая функция |
| `--spaces` | Что оптимизировать: `buy`, `sell`, `roi`, `stoploss`, `trailing`, `default`, `all` |
| `--epochs N` | Число итераций (500–1000 рекомендуется) |
| `-j N` | Число параллельных процессов (`-1` = все ядра) |
| `--min-trades N` | Минимум сделок для валидной эпохи |
| `--early-stop N` | Остановить после N эпох без улучшения |
| `--random-state INT` | Seed для воспроизводимости результатов |

**Loss-функции:**

| Функция | Оптимизирует |
| ------- | ------------ |
| `SharpeHyperOptLossDaily` | Sharpe ratio по дням — хороший старт |
| `SortinoHyperOptLossDaily` | Только нисходящий риск |
| `CalmarHyperOptLoss` | Доходность / max drawdown |
| `MaxDrawDownHyperOptLoss` | Минимальная просадка |
| `ProfitDrawDownHyperOptLoss` | Баланс прибыли и просадки |

Результаты автоматически сохраняются в `user_data/strategies/.MyStrategy.json` и подгружаются при следующем запуске.

### 5.4. Валидация — проверка на out-of-sample

Hyperopt подгоняет параметры под конкретный период. Чтобы убедиться что стратегия не переобучена:

```bash
# Оптимизация на первой половине данных
freqtrade hyperopt --timerange 20240101-20240701 ...

# Проверка на второй половине (unseen data)
freqtrade backtesting --timerange 20240701-20250101 --cache none
```

Если метрики на out-of-sample существенно хуже — параметры переобучены.

---

## 6. Анализ результатов

### 6.1. Визуализация — plot-dataframe и plot-profit

Freqtrade умеет строить интерактивные графики на Plotly: свечи с индикаторами, точки входа/выхода, equity curve. Требует установки зависимостей:

```bash
pip install -r requirements-plot.txt
```

**График свечей со сделками:**

```bash
freqtrade plot-dataframe --strategy MyStrategy --pair BTC/USDT --timerange 20240101-20240201
```

Открывает HTML-файл в браузере с интерактивным графиком: свечи, все индикаторы из `populate_indicators()`, маркеры входов и выходов, причины выхода.

**Параметры `plot-dataframe`:**

| Параметр | Описание |
| -------- | -------- |
| `--pair BTC/USDT` | Пара для отображения |
| `--timerange` | Период |
| `--indicators1 ema20 ema50` | Индикаторы на основном графике |
| `--indicators2 rsi macd` | Индикаторы на отдельном субграфике |
| `--trade-source file` | Брать сделки из файла бэктеста (по умолчанию) |
| `--trade-source DB` | Брать сделки из живой базы данных |
| `--db-url sqlite:///user_data/tradesv3.sqlite` | Путь к БД (при `--trade-source DB`) |
| `--export-filename` | Конкретный файл результатов бэктеста |

**График equity curve:**

```bash
freqtrade plot-profit --strategy MyStrategy --timerange 20240101-20250101
```

Строит три графика: накопленная прибыль, прибыль по парам, количество одновременных позиций. Полезно чтобы быстро увидеть какие пары тянут вниз и в какие периоды стратегия работала плохо.

**Параметры `plot-profit`:**

| Параметр | Описание |
| -------- | -------- |
| `--strategy NAME` | Стратегия |
| `--timerange` | Период |
| `--pairs BTC/USDT ETH/USDT` | Ограничить конкретными парами |
| `--export-filename` | Конкретный файл результатов |
| `--trade-source DB` | Данные из живой БД вместо файла |

### 6.2. Интерактивный анализ в Jupyter

Для детального анализа — загрузить шаблон ноутбука и установить Jupyter:

```bash
# Скопировать шаблон из репозитория
cp freqtrade/templates/strategy_analysis_example.ipynb user_data/notebooks/

# Установить Jupyter
pip install jupyter

# Запустить
jupyter notebook --notebook-dir user_data/notebooks/
```

**Что доступно в ноутбуке:**

```python
# Загрузить результаты бэктеста
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats

stats = load_backtest_stats("../backtest_results/")
trades = load_backtest_data("../backtest_results/", strategy="MyStrategy")

# Загрузить сделки из живой / dry-run БД
from freqtrade.data.btanalysis import load_trades_from_db

trades_live = load_trades_from_db("sqlite:///../tradesv3.sqlite")

# Загрузить исторические свечи с индикаторами
from freqtrade.data import load_pair_history
from freqtrade.resolvers import StrategyResolver

candles = load_pair_history(
    datadir="../data/binance",
    timeframe="5m",
    pair="BTC/USDT"
)

# Применить стратегию к свечам
from freqtrade.configuration import Configuration
config = Configuration.from_files(["../config.json"])
strategy = StrategyResolver.load_strategy(config)
df = strategy.analyze_ticker(candles, {"pair": "BTC/USDT"})
```

Ноутбук позволяет строить произвольные графики, анализировать сделки по дням недели и часам, сравнивать несколько бэктестов, исследовать распределение прибылей.

> Шаблон не создаётся автоматически командой `create-userdir`. Его нужно скопировать из `freqtrade/templates/` вручную или скачать с GitHub.

---

## 7. Paper Trading (Dry Run)

Бот работает в реальном времени с реальными котировками, но ордера виртуальные. Реалистичнее backtesting: учитывает задержки и неисполнение лимитных ордеров.

### Настройка

```json
{
    "dry_run": true,
    "dry_run_wallet": 1000
}
```

API-ключи биржи не нужны.

### Запуск

```bash
freqtrade trade --config user_data/config.json --strategy MyStrategy --logfile user_data/logs/freqtrade.log
```

Бот работает до ручной остановки (`Ctrl+C` или `/stop` в Telegram).

### Длительность

Минимум 2 недели. Идеально — 1–2 месяца. После окончания — прогнать backtesting за тот же период и сравнить метрики.

---

## 8. Мониторинг и управление

### 8.1. Логирование

По умолчанию Freqtrade пишет логи только в консоль. Чтобы сохранять в файл, добавьте `--logfile` к любой команде:

```bash
freqtrade trade --config user_data/config.json --strategy MyStrategy --logfile user_data/logs/freqtrade.log
```

**Уровни детализации:**

| Флаг | Уровень | Что показывает |
| ---- | ------- | -------------- |
| (без флага) | INFO | Старт/стоп, открытие/закрытие сделок, ошибки |
| `-v` | DEBUG | Все решения стратегии, запросы к DataProvider |
| `-vv` | TRACE | Все запросы к бирже, полные ответы API |

```bash
# Подробный режим для отладки стратегии
freqtrade trade --strategy MyStrategy -v --logfile user_data/logs/debug.log

# Просмотр логов в реальном времени
tail -f user_data/logs/freqtrade.log
```

Флаг `-v` работает с любой командой: `trade`, `backtesting`, `hyperopt`, `download-data`.

### 8.2. FreqUI — веб-интерфейс

Визуальный мониторинг в браузере: открытые позиции, P&L, графики свечей с индикаторами, запуск backtesting.

**Установка:**

```bash
freqtrade install-ui
```

**Настройка в config.json:**

```json
{
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "username": "freqtrader",
        "password": "SuperSecretPassword!",
        "jwt_secret_key": "случайная_строка_32_символов",
        "ws_token": "другая_случайная_строка"
    }
}
```

Генерация секретов:

```python
import secrets
print(secrets.token_hex(32))      # для jwt_secret_key
print(secrets.token_urlsafe(25))  # для ws_token
```

**Доступ:** после запуска бота открыть `http://localhost:8080`, ввести username/password из конфига.

**Swagger API:** при `"enable_openapi": true` — интерактивная документация по адресу `http://localhost:8080/docs`.

### 8.3. Telegram — уведомления и команды

**Шаг 1** — в Telegram открыть **@BotFather**, отправить `/newbot`, указать имя и username (заканчивается на `bot`). Получить token.

**Шаг 2** — открыть **@userinfobot**, нажать Start, получить chat_id.

**Шаг 3** — добавить в конфиг:

```json
{
    "telegram": {
        "enabled": true,
        "token": "1234567890:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "chat_id": "123456789",
        "notification_settings": {
            "entry": "on",
            "exit": "on",
            "entry_fill": "on",
            "exit_fill": "on"
        }
    }
}
```

**Основные команды:**

| Команда | Описание |
| ------- | -------- |
| `/status` | Открытые позиции |
| `/profit` | Статистика P&L |
| `/balance` | Баланс |
| `/start` | Запустить торговлю |
| `/stop` | Остановить |
| `/forceexit <id>` | Принудительно закрыть позицию |
| `/forceenter <pair> long` | Принудительно открыть |
| `/performance` | Результаты по парам |
| `/trades` | История сделок |
| `/reload_config` | Перечитать конфиг |
| `/blacklist <pair>` | Добавить пару в blacklist |
| `/whitelist` | Показать текущий whitelist |
| `/logs` | Последние строки лога |

> Если token утёк — в @BotFather отправьте `/revoke` для пересоздания.

---

## 9. Live Trading

### 9.1. Переключение на реальный счёт

```json
{
    "dry_run": false,
    "exchange": {
        "key": "REAL_KEY",
        "secret": "REAL_SECRET"
    }
}
```

### Чеклист перед live

- [ ] Backtesting: Sharpe > 1, Max Drawdown < 20%, Profit Factor > 1.5
- [ ] Lookahead и recursive bias анализ прошли без замечаний
- [ ] Dry run ≥ 2 недели, результаты близки к backtesting
- [ ] Начинаете с минимального `stake_amount`
- [ ] Telegram уведомления настроены
- [ ] API-ключи с ограниченными правами (только trade, не withdraw)
- [ ] Ключи не в git (отдельный файл или env-переменные)

### Запуск

```bash
freqtrade trade --config user_data/config.json --config user_data/config-private.json --strategy MyStrategy --logfile user_data/logs/freqtrade.log
```

### 9.2. Автозапуск и обслуживание

**systemd-сервис (Linux):**

```bash
sudo nano /etc/systemd/system/freqtrade.service
```

```ini
[Unit]
Description=Freqtrade
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/freqtrade
ExecStart=/home/YOUR_USER/freqtrade/.venv/bin/freqtrade trade \
    --config /home/YOUR_USER/freqtrade/user_data/config.json \
    --config /home/YOUR_USER/freqtrade/user_data/config-private.json \
    --strategy MyStrategy \
    --logfile /home/YOUR_USER/freqtrade/user_data/logs/freqtrade.log
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable freqtrade
sudo systemctl start freqtrade

# Статус и логи:
sudo systemctl status freqtrade
journalctl -u freqtrade -f
```

**Обновление Freqtrade:**

```bash
cd freqtrade
git pull
./setup.sh -u
```

---

## Типичные проблемы

| Проблема | Решение |
| -------- | ------- |
| `freqtrade: command not found` | Активировать venv: `source .venv/bin/activate` |
| `Timestamp outside recvWindow` (Windows/WSL) | Выполнить `wsl --shutdown` и перезапустить |
| `Permission denied` в `user_data` | `sudo chown -R $USER:$USER user_data` |
| Нет данных для backtesting | Выполнить `download-data` с нужным `--timerange` |
| Hyperopt падает | Попробовать с `-j 2` вместо `-j -1` |
| Пустая папка `logs/` | Добавить `--logfile` к команде запуска |
| Нет `strategy_analysis_example.ipynb` | Скопировать из `freqtrade/templates/` |
| Telegram не присылает сообщения | Проверить token, chat_id; нажать Start у бота |
| FreqUI не открывается | Проверить `api_server.enabled: true` и свободен ли порт |
