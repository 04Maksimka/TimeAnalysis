## Обзор

Документ охватывает два сценария установки:
- **Windows** → Docker (единственный рекомендуемый путь)
- **Linux / macOS** → нативная установка через `setup.sh`

Каждый режим работы описан отдельно с пояснениями.

> **Важно.** Официальная документация настоятельно рекомендует Windows-пользователям использовать только Docker. Для production — Linux VPS.

---

## Часть A. Установка на Windows через Docker

### A1. Установить Docker Desktop

Скачать с [docker.com](https://www.docker.com/products/docker-desktop/) и установить.

> ⚠️ **После установки обязательно перезагрузите компьютер.** Без перезагрузки Docker может иметь проблемы с сетевым подключением контейнеров.

### A2. Создать рабочую директорию

```powershell
mkdir ft_userdata
cd ft_userdata
```

### A3. Скачать официальный docker-compose.yml

```powershell
curl https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml -o docker-compose.yml
```

### A4. Подтянуть образ Freqtrade

```bash
docker compose pull
```

### A5. Создать структуру директорий user_data

```bash
docker compose run --rm freqtrade create-userdir --userdir user_data
```

Создаёт папки `data/`, `strategies/`, `logs/`, `backtest_results/`, `hyperopts/` и другие внутри `user_data/`.

### A6. Создать конфиг в интерактивном режиме

```bash
docker compose run --rm freqtrade new-config --config user_data/config.json
```

Мастер задаст ряд вопросов: биржа, торговые пары, Dry Run / Live, FreqUI и т. д. Результат — файл `user_data/config.json`.

> `new-config` генерирует рабочий файл, который затем можно редактировать обычным текстовым редактором. Docker ничего специфичного в `config.json` не требует — параметры те же, что и для нативной установки.

---

## Часть B1. Linux + Docker (рекомендовано для production)

### 1. Установить Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Добавить себя в группу docker (чтобы не писать sudo)
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Создать рабочую директорию и скачать compose

```bash
mkdir ft_userdata && cd ft_userdata

curl https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml \
  -o docker-compose.yml
```

### 3. Подтянуть образ

```bash
docker compose pull
```

### 4. Инициализировать user_data и конфиг

```bash
docker compose run --rm freqtrade create-userdir --userdir user_data
docker compose run --rm freqtrade new-config --config user_data/config.json
```

### 5. Добавить `restart: unless-stopped` (автозапуск после ребута)

Открыть `docker-compose.yml` и добавить строку:

```yaml
services:
  freqtrade:
    image: freqtradeorg/freqtrade:stable
    restart: unless-stopped    # ← добавить
    volumes:
      - "./user_data:/freqtrade/user_data"
    ports:
      - "127.0.0.1:8080:8080"
    command: >
      trade
      --logfile /freqtrade/user_data/logs/freqtrade.log
      --config /freqtrade/user_data/config.json
      --strategy MyStrategy
```

### 6. Запустить

```bash
docker compose up -d

# Проверить:
docker compose ps
docker compose logs -f
curl http://localhost:8080/api/v1/ping   # → {"status":"pong"}
```

---

## Часть B2. Linux нативно (без Docker)

### 1. Установить системные зависимости

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-pip python3-venv python3-dev python3-pandas \
  git curl build-essential
```

**Fedora/RHEL:**

```bash
sudo dnf install python3 python3-pip python3-devel git curl gcc
```

### 2. Клонировать и установить

```bash
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
git checkout stable

# Установщик создаёт venv, ставит TA-Lib и все зависимости
./setup.sh -i
```

### 3. Активировать окружение

```bash
source ./.venv/bin/activate

# Проверить:
freqtrade --version
```

> ⚠️ `source ./.venv/bin/activate` нужно выполнять при каждом новом терминале. Для удобства добавьте алиас в `~/.bashrc`:
```shell
alias ft='cd ~/freqtrade && source ./.venv/bin/activate'
```

### 4. Инициализировать user_data и конфиг

```bash
freqtrade create-userdir --userdir user_data
freqtrade new-config --config user_data/config.json
```

### 5. Автозапуск через systemd (для production)

```bash
sudo nano /etc/systemd/system/freqtrade.service
```

> ⚠️ Заменить YOUR_USERNAME на реального пользователя:

```ini
[Unit]
Description=Freqtrade trading bot
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/freqtrade
ExecStart=/home/YOUR_USERNAME/freqtrade/.venv/bin/freqtrade trade \
  --config /home/YOUR_USERNAME/freqtrade/user_data/config.json \
  --strategy MyStrategy \
  --logfile /home/YOUR_USERNAME/freqtrade/user_data/logs/freqtrade.log
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

---

## Сравнение двух вариантов

|                               | Docker                                | Нативно                   |
| ----------------------------- | ------------------------------------- | ------------------------- |
| Обновление                    | `docker compose pull && up -d`        | `./setup.sh -u` + restart |
| Автоперезапуск                | `restart: unless-stopped`             | systemd-сервис            |
| Изоляция зависимостей         | Полная                                | venv (частичная)          |
| Удобство разработки стратегий | Чуть сложнее (пути внутри контейнера) | Удобнее                   |
| Рекомендация для production   | ✅ Да                                  | ✅ Да (с systemd)          |

---

## Часть C. Конфигурация (config.json)

Независимо от способа установки конфиг-файл одинаковый. Минимальный рабочий шаблон:

```json
{
  "$schema": "https://schema.freqtrade.io/schema.json",
  "max_open_trades": 3,
  "stake_currency": "USDT",
  "stake_amount": 100,
  "dry_run": true,
  "dry_run_wallet": 1000,
  "timeframe": "5m",
  "minimal_roi": {
    "0": 0.01
  },
  "stoploss": -0.05,
  "unfilledtimeout": {
    "entry": 10,
    "exit": 10
  },
  "exchange": {
    "name": "binance",
    "key": "",
    "secret": "",
    "pair_whitelist": ["BTC/USDT", "ETH/USDT"]
  },
  "telegram": {
    "enabled": false,
    "token": "",
    "chat_id": ""
  }
}
```

### Ключевые параметры

| Параметр                  | Тип                   | По умолчанию | Описание                                                       |
| ------------------------- | --------------------- | ------------ | -------------------------------------------------------------- |
| `dry_run`                 | Boolean               | `true`       | `true` — симуляция, `false` — реальная торговля                |
| `dry_run_wallet`          | Float                 | 1000         | Стартовый виртуальный баланс                                   |
| `max_open_trades`         | Int / -1              | —            | Максимум открытых позиций; `-1` — без ограничений              |
| `stake_amount`            | Float / `"unlimited"` | —            | Сумма на сделку; `"unlimited"` — делить баланс поровну         |
| `stake_currency`          | String                | —            | Валюта торговли (`USDT`, `BTC`, …)                             |
| `minimal_roi`             | Dict                  | —            | Порог прибыли для автовыхода (Strategy Override)               |
| `stoploss`                | Float                 | —            | Стоп-лосс как доля: `-0.05` = −5% (Strategy Override)         |
| `exchange.key` / `secret` | String                | `""`         | API-ключи; **не нужны для dry run и backtesting**              |

### Приоритет настроек

```
CLI-аргументы > Переменные окружения > config.json > Параметры стратегии
```

### Секреты через переменные окружения (рекомендуется)

Чтобы не хранить ключи в файле:

```bash
export FREQTRADE__EXCHANGE__KEY=yourExchangeKey
export FREQTRADE__EXCHANGE__SECRET=yourExchangeSecret
export FREQTRADE__TELEGRAM__TOKEN=telegramToken
export FREQTRADE__TELEGRAM__CHAT_ID=chatId
```

### Два конфиг-файла (публичный + приватный)

```bash
freqtrade trade \
  --config user_data/config.json \
  --config user_data/config-private.json
```

Последний файл имеет приоритет — удобно разделить публичный конфиг и файл с ключами. Рекомендуемый подход: `config.example.json` в git как шаблон, реальный `config.json` — в `.gitignore`.

---

## Часть D. Скачивание исторических данных

Необходимый шаг перед backtesting и hyperopt. API-ключи не нужны.

### Команда

**Без Docker:**

```bash
freqtrade download-data \
  --config user_data/config.json \
  --timeframes 5m 1h \
  --timerange 20250101-20260101
```

**Через Docker:**

```bash
docker compose run --rm freqtrade download-data \
  --config /freqtrade/user_data/config.json \
  --timeframes 5m \
  --timerange 20250101-20250201
```

> Внутри контейнера путь начинается с `/freqtrade/user_data/...`, а не с пути на хосте.

### Параметры download-data

| Параметр                       | Описание                                              |
| ------------------------------ | ----------------------------------------------------- |
| `--timeframes 5m 1h`           | Таймфреймы (флаг называется `--timeframes`, не `--timeframe`) |
| `--timerange YYYYMMDD-YYYYMMDD`| Фиксированный диапазон                                |
| `--days N`                     | Последние N дней (альтернатива `--timerange`)         |
| `--pairs BTC/USDT ETH/USDT`    | Явно указать пары                                     |
| `--pairs ".*/USDT"`            | Все USDT-пары биржи (regex)                           |
| `--exchange binance`           | Биржа (если не указана в конфиге)                     |
| `--erase`                      | Удалить существующие данные перед скачиванием         |
| `--prepend`                    | Дополнить данными более раннего периода               |

> ✅ Если данные уже скачаны — Freqtrade автоматически догрузит только недостающий диапазон.

**Просмотр скачанных данных:**

```bash
freqtrade list-data --userdir user_data/ --show-timerange
```

---

## Часть E. Backtesting

Проверка стратегии на исторических данных. Работает полностью офлайн, без API-ключей.

### Запуск

**Без Docker:**

```bash
freqtrade backtesting \
  --config user_data/config.json \
  --strategy MyStrategy \
  --timeframe 5m \
  --timerange 20250101-20260101 \
  --export trades
```

**Через Docker:**

```bash
docker compose run --rm freqtrade backtesting \
  --config /freqtrade/user_data/config.json \
  --strategy MyStrategy \
  --strategy-path /freqtrade/user_data/strategies \
  --timeframe 5m \
  --timerange 20250101-20250201 \
  --export trades
```

### Основные параметры

| Параметр                         | Описание                                                    |
| -------------------------------- | ----------------------------------------------------------- |
| `--strategy NAME`                | Имя класса стратегии (не имя файла)                         |
| `--strategy-path PATH`           | Папка стратегий (если не `user_data/strategies/`)           |
| `--timeframe 5m`                 | Таймфрейм свечей                                            |
| `--timerange 20250101-20250601`  | Период тестирования                                         |
| `--dry-run-wallet 1000`          | Начальный баланс (переопределяет конфиг)                    |
| `--export trades`                | Сохранить сделки в файл (по умолчанию)                      |
| `--export none`                  | Не сохранять                                                |
| `--breakdown month year`         | Разбивка результатов по периодам                            |
| `--timeframe-detail 5m`          | Детальный таймфрейм для внутрисвечного анализа              |
| `--fee 0.001`                    | Задать кастомную комиссию                                   |
| `--cache none`                   | Принудительно пересчитать (игнорировать кэш)                |
| `--strategy-list S1 S2`          | Сравнить несколько стратегий                                |

**Сравнение нескольких стратегий:**

```bash
freqtrade backtesting \
  --strategy-list StrategyA StrategyB \
  --timeframe 5m \
  --timerange 20250101-20260101
```

### Кэширование результатов

По умолчанию Freqtrade переиспользует результаты последнего дня при повторном запуске с теми же параметрами. Чтобы пересчитать принудительно:

```bash
freqtrade backtesting ... --cache none
```

### Детализация внутри свечи (--timeframe-detail)

Backtesting не знает, что было внутри свечи — High наступил раньше Low или наоборот. Флаг `--timeframe-detail` решает это, подгружая более мелкий таймфрейм для проверки:

```bash
freqtrade backtesting \
  --strategy MyStrategy \
  --timeframe 1h \
  --timeframe-detail 5m
```

> Требует предварительно скачанных данных обоих таймфреймов.

### Понимание результатов

**BACKTESTING REPORT** — статистика по парам:

| Колонка       | Описание                                          |
| ------------- | ------------------------------------------------- |
| `Avg Profit %`| Средний профит на сделку                          |
| `Tot Profit %`| `(Финальный − Начальный) / Начальный × 100`       |
| `Avg Duration`| Средняя длительность сделки                       |
| `Win / Loss / Win%` | Статистика побед и проигрышей               |

**LEFT OPEN TRADES** — позиции, незакрытые на конец периода (принудительно закрываются).

**EXIT REASON STATS** — причины выхода: `roi`, `stop_loss`, `exit_signal`, `force_exit`.

**SUMMARY METRICS** — ключевые метрики:

| Метрика              | Что означает                                             |
| -------------------- | -------------------------------------------------------- |
| `CAGR %`             | Годовая доходность с компаундингом                       |
| `Sharpe (closed trades)` | Риск/доходность; >1 — хорошо                         |
| `Sortino`            | Sharpe, учитывающий только нисходящий риск               |
| `Max % underwater`   | Максимальная просадка от пика                            |
| `Profit factor`      | Сумма выигрышей / сумма проигрышей; >1 — прибыльно       |
| `Expectancy (Ratio)` | Средний P&L на сделку; отрицательное — неприбыльно       |

### Допущения backtesting

Backtesting упрощает реальность — это важно понимать:

- Входы исполняются по цене Open следующей свечи после сигнала
- Ордера исполняются **всегда** по запрошенной цене (нет slippage)
- Stoploss срабатывает точно на уровне (в реале может быть хуже)
- **Не заменяет** dry-run в реальном времени

---

## Часть F. Режим Dry Run (Paper Trading)

Dry run — бот работает в реальном времени с реальными данными биржи, но исполняет сделки на **виртуальный счёт**. Реалистичнее backtesting.

### Backtesting vs Dry Run

| Аспект                  | Backtesting                      | Dry Run                           |
| ----------------------- | -------------------------------- | --------------------------------- |
| Данные                  | Исторические                     | Реальное время                    |
| Скорость                | Весь период — за секунды         | Реальное время                    |
| API-ключи               | Не нужны                         | Не нужны                          |
| Заполнение ордеров      | Всегда 100%                      | Зависит от рынка (limit orders)   |
| Slippage                | Нет                              | Есть (задержка несколько секунд)  |
| Реалистичность оценки   | Оптимистичнее                    | Реалистичнее                      |

### Настройка в config.json

```json
{
  "dry_run": true,
  "dry_run_wallet": 1000,
  "exchange": {
    "name": "binance",
    "key": "",
    "secret": ""
  }
}
```

API-ключи для dry run не нужны — бот использует публичные данные биржи.

### Запуск

**Без Docker:**

```bash
freqtrade trade --config user_data/config.json --strategy MyStrategy
```

**Через Docker — через `up`, не через `run`:**

```bash
docker compose up -d
```

> ⚠️ Именно `docker compose up -d`, а не `docker compose run`. Только `up` обеспечивает проброс портов (FreqUI) и автоперезапуск.

Стратегия берётся из секции `command` в `docker-compose.yml` — отредактируйте её перед запуском:

```yaml
command: >
  trade
  --logfile /freqtrade/user_data/logs/freqtrade.log
  --db-url sqlite:////freqtrade/user_data/tradesv3.sqlite
  --config /freqtrade/user_data/config.json
  --strategy MyStrategy
```

### Мониторинг

```bash
# Логи в реальном времени
docker compose logs -f

# Статус контейнера
docker compose ps
```

Логи также пишутся в `user_data/logs/freqtrade.log`.

### FreqUI

Доступен по адресу `http://localhost:8080`, если был включён при `new-config`.

```json
{
  "api_server": {
    "enabled": true,
    "listen_ip_address": "127.0.0.1",
    "listen_port": 8080,
    "username": "myuser",
    "password": "mypassword"
  }
}
```

### Остановка

```bash
docker compose down
# или мягкая остановка через Telegram: /stop
```

---

## Часть G. Live Trading (реальная торговля)

Единственное отличие от Dry Run — `«dry_run»: false` и реальные API-ключи.

### Настройка config.json

```json
{
  "dry_run": false,
  "exchange": {
    "name": "bybit",
    "key": "YOUR_API_KEY",
    "secret": "YOUR_API_SECRET"
  }
}
```

> ⚠️ **Никогда не коммитьте** `config.json` с реальными ключами. Добавьте его в `.gitignore`, а в репозиторий положите `config.example.json` как публичный шаблон.

### Запуск

```bash
docker compose up -d
```

### Рекомендации перед переходом в live

1. Пройти полный цикл: **download-data → backtest → dry run ≥ 2 недели**
2. Убедиться, что результаты dry run близки к backtest
3. Начать с минимального `stake_amount`
4. Настроить Telegram-уведомления
5. Использовать Linux VPS — не Docker on Windows

---

## Часть H. Hyperopt (оптимизация параметров стратегии)

Hyperopt запускает backtesting сотни раз с разными параметрами, используя Optuna для поиска оптимальных значений.

> ⚠️ **Ресурсоёмкая операция.** Занимает все ядра CPU. На долгих прогонах рекомендуется `screen` или `tmux`.

### Установка зависимостей (только нативная установка)

```bash
source .venv/bin/activate
pip install -r requirements-hyperopt.txt
```

В Docker-образе зависимости уже включены.

### Добавление параметров в стратегию

```python
from freqtrade.strategy import (
    IStrategy, IntParameter, DecimalParameter,
    CategoricalParameter, BooleanParameter
)
from functools import reduce

class MyStrategy(IStrategy):
    timeframe = '5m'
    stoploss = -0.05
    minimal_roi = {"0": 0.01}

    buy_rsi = IntParameter(20, 40, default=30, space="buy")
    buy_adx = DecimalParameter(20.0, 40.0, decimals=1, default=25.0, space="buy")
    buy_adx_enabled = BooleanParameter(default=True, space="buy")
    buy_trigger = CategoricalParameter(
        ["bb_lower", "macd_cross"],
        default="bb_lower",
        space="buy"
    )

    def populate_entry_trend(self, dataframe, metadata):
        conditions = []
        if self.buy_adx_enabled.value:
            conditions.append(dataframe['adx'] > self.buy_adx.value)
        conditions.append(dataframe['rsi'] < self.buy_rsi.value)
        if self.buy_trigger.value == 'bb_lower':
            conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
        conditions.append(dataframe['volume'] > 0)
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1
        return dataframe
```

**Типы параметров:**

| Тип                              | Назначение                           |
| -------------------------------- | ------------------------------------ |
| `IntParameter(low, high)`        | Целочисленное значение               |
| `DecimalParameter(low, high, decimals=3)` | Вещественное с фиксированным числом знаков |
| `RealParameter(low, high)`       | Вещественное без ограничений         |
| `CategoricalParameter([...])`    | Перечисление вариантов               |
| `BooleanParameter()`             | Включить / выключить                 |

### Запуск Hyperopt

**Без Docker:**

```bash
freqtrade hyperopt \
  --config user_data/config.json \
  --strategy MyStrategy \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces default \
  -e 500 \
  --timerange 20250101-20260101
```

**Через Docker:**

```bash
docker compose run --rm freqtrade hyperopt \
  --config /freqtrade/user_data/config.json \
  --strategy MyStrategy \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces default \
  -e 500 \
  --timerange 20250101-20260101
```

### Параметры Hyperopt

| Параметр           | Описание                                                    |
| ------------------ | ----------------------------------------------------------- |
| `--spaces`         | Что оптимизировать (см. ниже)                               |
| `-e N`             | Число эпох; рекомендуется 500–1000                          |
| `--hyperopt-loss`  | Целевая функция                                             |
| `-j N`             | Число параллельных процессов (-1 = все ядра)                |
| `--early-stop N`   | Остановить после N эпох без улучшения                       |
| `--random-state INT` | Seed для воспроизводимых результатов                      |
| `--min-trades N`   | Минимальное число сделок для оценки эпохи (по умолчанию 1) |

**Значения `--spaces`:**

| Значение              | Что оптимизируется                        |
| --------------------- | ----------------------------------------- |
| `default`             | Всё кроме `trailing`, `protection`        |
| `all`                 | Всё включая custom                        |
| `buy` / `enter`       | Только сигналы входа                      |
| `sell` / `exit`       | Только сигналы выхода                     |
| `roi`                 | Таблица `minimal_roi`                     |
| `stoploss`            | Значение `stoploss`                       |
| `trailing`            | Параметры trailing stop                   |
| `protection`          | Параметры защиты                          |

### Loss-функции

| Функция                      | Оптимизирует                            |
| ---------------------------- | --------------------------------------- |
| `SharpeHyperOptLossDaily`    | Sharpe ratio (по дням) — хороший старт  |
| `SortinoHyperOptLossDaily`   | Sortino (только downside-риск)          |
| `OnlyProfitHyperOptLoss`     | Максимальная прибыль                    |
| `MaxDrawDownHyperOptLoss`    | Минимальная просадка                    |
| `ProfitDrawDownHyperOptLoss` | Прибыль + просадка                      |
| `CalmarHyperOptLoss`         | Calmar ratio                            |
| `MultiMetricHyperOptLoss`    | Баланс прибыли, просадки, win rate      |

### Применение результатов

Freqtrade автоматически сохраняет лучший результат в `user_data/strategies/.MyStrategy.json`. Можно также перенести значения напрямую в класс:

```python
class MyStrategy(IStrategy):
    buy_params = {
        'buy_rsi': 29,
        'buy_adx': 32.5,
        'buy_adx_enabled': True,
        'buy_trigger': 'bb_lower',
    }
    minimal_roi = {
        "0": 0.106,
        "21": 0.091,
        "78": 0.036,
        "118": 0
    }
    stoploss = -0.279
```

**Приоритет значений:**

```
config.json > parameter file (.json) > strategy *_params > default в параметре
```

---

## Часть I. Полный рабочий цикл

```
1. Написать стратегию
   → user_data/strategies/MyStrategy.py

2. Скачать данные
   → download-data --timeframes 5m --timerange <период>

3. Первичный backtest
   → backtesting --strategy MyStrategy --timeframe 5m

4. Оценить: Sharpe, Drawdown, Win Rate, Profit Factor

5. Гипероптимизация (если нужно)
   → hyperopt --spaces default -e 500

6. Применить параметры → повторить backtest с --cache none

7. Dry Run (≥ 2 недели в реальном времени)
   → docker compose up -d  (dry_run: true)

8. Сравнить dry run vs backtest

9. Live Trading (если dry run подтверждает)
   → docker compose up -d  (dry_run: false + реальные ключи)
```

---

## Типичные проблемы

| Проблема                              | Решение                                                        |
| ------------------------------------- | -------------------------------------------------------------- |
| `freqtrade: command not found`        | Активировать venv: `source ./.venv/bin/activate`               |
| Docker сетевые ошибки (Windows)       | Перезагрузить ПК; выполнить `wsl --shutdown`                   |
| `Timestamp outside recvWindow`        | `wsl --shutdown` → перезапуск Docker                           |
| `Permission denied` в `user_data`    | `sudo chown -R $UID:$GID user_data`                            |
| Hyperopt crash                        | Запустить с `-j 2` или больше (не `-j 1`)                      |
| Нет данных для backtesting            | Выполнить `download-data` с нужным `--timerange`               |
| `Microsoft Visual C++ 14.0 required` | Установить Visual C++ Build Tools или использовать WSL2/Docker |
[[FreqTrade]]