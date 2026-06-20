FreqUI и собственный UI: потоки данных и подключение к Freqtrade
## Общая архитектура

Freqtrade — это backend. Он запускает торговую логику и поднимает встроенный HTTP-сервер на FastAPI. FreqUI — это frontend, который просто делает запросы к этому серверу. Любой другой UI (Python-скрипт, Flask-приложение, React-сайт, Jupyter-ноутбук) может делать то же самое — никакой специальной интеграции не требуется.

```
freqtrade (порт 8080)
│
├── GET  /api/v1/status          → список открытых сделок
├── GET  /api/v1/balance         → баланс счёта
├── POST /api/v1/token/login     → получить JWT-токен
├── POST /api/v1/token/refresh   → обновить токен
├── WS   /api/v1/message/ws      → подписка на события реального времени
└── GET  /docs                   → Swagger UI (если включён)
```

FreqUI грузится браузером как статика из `freqtrade/rpc/api_server/ui/` через FastAPI. Все HTTP-запросы идут на ApiServer, который внутри работает через RPC с FreqtradeBot и его объектами — Trade, Wallets, PairListManager, DataProvider. Сам бот ничего не знает о том, кто именно обращается к API: его собственный UI, ваш кастомный дашборд или curl из терминала.

---

## Часть 1. Настройка Freqtrade

### 1.1. Секция api_server в config.json

```json
{
  "api_server": {
    "enabled": true,
    "listen_ip_address": "0.0.0.0",
    "listen_port": 8080,
    "verbosity": "error",
    "enable_openapi": true,
    "jwt_secret_key": "сгенерируйте_случайную_строку_32+_символа",
    "username": "myuser",
    "password": "MySuperSecretPassword1!",
    "ws_token": "сгенерируйте_отдельный_токен_для_websocket",
    "CORS_origins": ["http://localhost:3000"]
  }
}
```

| Параметр                      | Что делает                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------- |
| `listen_ip_address: "0.0.0.0"` | **Обязательно для Docker.** Без этого API слушает только внутри контейнера                 |
| `listen_port: 8080`            | Порт API-сервера                                                                            |
| `enable_openapi: true`         | Включает Swagger UI по адресу `/docs`                                                       |
| `jwt_secret_key`               | Секрет для подписи JWT-токенов — должен быть случайным и длинным                           |
| `username` / `password`        | Логин/пароль для входа                                                                      |
| `ws_token`                     | Отдельный токен только для WebSocket-подключения                                            |
| `CORS_origins`                 | Список URL, которым браузер разрешает обращаться к API с другого origin                     |

### 1.2. Генерация безопасных токенов

```python
import secrets

print(secrets.token_hex(32))        # для jwt_secret_key
print(secrets.token_urlsafe(25))    # для ws_token
```

### 1.3. Проброс порта в docker-compose.yml

```yaml
services:
  freqtrade:
    image: freqtradeorg/freqtrade:stable
    ports:
      - "127.0.0.1:8080:8080"   # только localhost — безопасно
      # - "8080:8080"            # доступно всем в сети — небезопасно
    volumes:
      - "./user_data:/freqtrade/user_data"
    command: >
      trade
      --logfile /freqtrade/user_data/logs/freqtrade.log
      --config /freqtrade/user_data/config.json
      --strategy MyStrategy
```

### 1.4. Проверка

```bash
docker compose up -d

curl http://localhost:8080/api/v1/ping
# → {"status":"pong"}
```

---

## Часть 2. Аутентификация

### Концепция JWT

После успешного входа Freqtrade выдаёт два токена:

- **access_token** — короткоживущий (15 минут), прикладывается к каждому запросу
- **refresh_token** — долгоживущий (~30 дней), используется для получения нового access_token без повторного ввода пароля

```
Ваш UI                         freqtrade API
  │                                  │
  │  POST /token/login               │
  │  (username + password)           │
  │ ────────────────────────────────►│
  │◄────────────────────────────────-│
  │  {access_token, refresh_token}   │
  │                                  │
  │  GET /status                     │
  │  Authorization: Bearer <access>  │
  │ ────────────────────────────────►│
  │◄─────────────────────────────────│
  │  [{...сделки...}]                │
  │                                  │
  │  (через 15 минут — токен истёк)  │
  │  POST /token/refresh             │
  │  Authorization: Bearer <refresh> │
  │ ────────────────────────────────►│
  │◄─────────────────────────────────│
  │  {новый access_token}            │
```

### Шаг 1: Login

```python
import requests

BASE_URL = "http://localhost:8080/api/v1"

resp = requests.post(
    f"{BASE_URL}/token/login",
    auth=("myuser", "MySuperSecretPassword1!")   # HTTP Basic Auth — только для логина
)
tokens = resp.json()
access_token = tokens["access_token"]
refresh_token = tokens["refresh_token"]
```

FreqUI делает то же самое:

```typescript
// src/composables/loginInfo.ts
const { data } = await axios.post(
  `${auth.url}/api/v1/token/login`,
  {},
  { auth: { username: auth.username, password: auth.password } }
);
// data.access_token и data.refresh_token сохраняются в localStorage
```

### Шаг 2: Использовать access_token

```python
headers = {"Authorization": f"Bearer {access_token}"}

requests.get(f"{BASE_URL}/status", headers=headers).json()
requests.get(f"{BASE_URL}/balance", headers=headers).json()
```

В FreqUI это реализовано через axios interceptor, который автоматически добавляет заголовок ко всем запросам:

```typescript
// src/composables/api.ts
api.interceptors.request.use((request) => {
  const token = userService.accessToken.value;
  if (token) {
    request.headers.set('Authorization', `Bearer ${token}`);
  }
  return request;
});
```

### Шаг 3: Обновить токен

Когда access_token истекает, сервер возвращает `HTTP 401`. Нужно использовать refresh_token:

```python
def refresh_access_token(refresh_token: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/token/refresh",
        headers={"Authorization": f"Bearer {refresh_token}"}
    )
    resp.raise_for_status()
    return resp.json()["access_token"]
```

FreqUI перехватывает 401 и автоматически повторяет исходный запрос с новым токеном:

```typescript
// src/composables/api.ts
api.interceptors.response.use(
  (response) => response,
  (err) => {
    if (err.response?.status === 401) {
      return userService.refreshToken().then((token) => {
        err.config.headers.Authorization = `Bearer ${token}`;
        return axios.request(err.config);
      });
    }
  }
);
```

---

## Часть 3. Потоки данных

Все взаимодействия между UI и Freqtrade делятся на три канала: периодические GET-запросы, управляющие POST/DELETE-запросы и WebSocket для событий реального времени.

### 3.1. GET — чтение состояния

GET-запросы только читают состояние, ничего не меняют. FreqUI опрашивает их периодически: часть эндпоинтов каждые 5 секунд (`refreshFrequent`), часть — каждые 60 секунд (`refreshSlow`).

**Информационные эндпоинты (api_v1.py):**

```python
headers = {"Authorization": f"Bearer {access_token}"}

requests.get(f"{BASE_URL}/ping", headers=headers).json()
# → {"status": "pong"}

requests.get(f"{BASE_URL}/version", headers=headers).json()
# → {"version": "2024.12"}

requests.get(f"{BASE_URL}/show_config", headers=headers).json()
# конфиг + текущий State + версия стратегии

requests.get(f"{BASE_URL}/plot_config", headers=headers).json()
# какие индикаторы и сабплоты рисовать на графике

requests.get(f"{BASE_URL}/sysinfo", headers=headers).json()
# CPU, RAM

requests.get(f"{BASE_URL}/logs?limit=100", headers=headers).json()
```

**Торговые данные (api_trading.py):**

```python
# Открытые позиции
requests.get(f"{BASE_URL}/status", headers=headers).json()

# Агрегированная P&L-статистика
requests.get(f"{BASE_URL}/profit", headers=headers).json()

# Баланс аккаунта
requests.get(f"{BASE_URL}/balance", headers=headers).json()

# История закрытых сделок (с пагинацией)
requests.get(f"{BASE_URL}/trades?limit=50&offset=0", headers=headers).json()

# Конкретная сделка
requests.get(f"{BASE_URL}/trade/42", headers=headers).json()

# Динамика прибыли по дням/неделям/месяцам
requests.get(f"{BASE_URL}/daily?timescale=7", headers=headers).json()
requests.get(f"{BASE_URL}/weekly?timescale=4", headers=headers).json()
requests.get(f"{BASE_URL}/monthly", headers=headers).json()

# График equity (временной ряд баланса)
requests.get(f"{BASE_URL}/historic_balance", headers=headers).json()

# Производительность по парам
requests.get(f"{BASE_URL}/performance", headers=headers).json()

# Разбивка по сигналам входа/выхода
requests.get(f"{BASE_URL}/entries", headers=headers).json()
requests.get(f"{BASE_URL}/exits", headers=headers).json()

# Pairlist
requests.get(f"{BASE_URL}/whitelist", headers=headers).json()
requests.get(f"{BASE_URL}/blacklist", headers=headers).json()

# Активные локи
requests.get(f"{BASE_URL}/locks", headers=headers).json()
```

**Свечи с индикаторами:**

```python
# Свечи + результаты populate_indicators() + сигналы enter_*/exit_*
requests.get(
    f"{BASE_URL}/pair_candles",
    headers=headers,
    params={"pair": "BTC/USDT", "timeframe": "5m", "limit": 100}
).json()

# POST-вариант с фильтрацией колонок (read-only, но тело запроса сложное)
requests.post(
    f"{BASE_URL}/pair_candles",
    headers=headers,
    json={"pair": "BTC/USDT", "timeframe": "5m", "limit": 100, "columns": ["rsi", "ema20"]}
).json()
```

### 3.2. POST / DELETE — управление ботом

POST и DELETE меняют состояние бота или базу данных. Именно поэтому они POST, а не GET — это семантически корректно: запрос производит побочный эффект.

**Управление состоянием бота:**

```python
requests.post(f"{BASE_URL}/start", headers=headers)       # → RUNNING
requests.post(f"{BASE_URL}/stop", headers=headers)        # → STOPPED
requests.post(f"{BASE_URL}/pause", headers=headers)       # запретить новые входы
requests.post(f"{BASE_URL}/reload_config", headers=headers)  # перечитать конфиг
```

**Ручные сделки:**

```python
# Принудительно открыть позицию
requests.post(
    f"{BASE_URL}/forceenter",
    headers=headers,
    json={"pair": "BTC/USDT", "side": "long", "ordertype": "market"}
)

# Принудительно закрыть позицию
requests.post(
    f"{BASE_URL}/forceexit",
    headers=headers,
    json={"tradeid": "42", "ordertype": "market"}
)

# Перечитать сделку с биржи (обновить fill, комиссии)
requests.post(f"{BASE_URL}/trades/42/reload", headers=headers)

# Отменить открытый ордер на бирже
requests.delete(f"{BASE_URL}/trades/42/open-order", headers=headers)

# Удалить сделку из БД (необратимо)
requests.delete(f"{BASE_URL}/trades/42", headers=headers)
```

**Pairlist и локи:**

```python
# Добавить пары в blacklist
requests.post(
    f"{BASE_URL}/blacklist",
    headers=headers,
    json={"blacklist": ["XRP/USDT", "DOGE/USDT"]}
)

# Удалить пару из blacklist
requests.delete(
    f"{BASE_URL}/blacklist",
    headers=headers,
    params={"pairs_to_delete": "XRP/USDT"}
)

# Добавить лок вручную
requests.post(
    f"{BASE_URL}/locks",
    headers=headers,
    json={"locks": [{"pair": "BTC/USDT", "until": "2024-12-31 23:59:59", "reason": "manual"}]}
)

# Удалить лок
requests.delete(f"{BASE_URL}/locks/5", headers=headers)
```

### 3.3. WebSocket — события реального времени

WebSocket используется для получения событий без постоянного опроса. Подключение — отдельный эндпоинт, авторизация через `ws_token` (не JWT access_token).

```python
import asyncio
import websockets
import json

WS_URL = "ws://localhost:8080/api/v1/message/ws"
WS_TOKEN = "ваш_ws_token_из_config"

async def listen():
    async with websockets.connect(f"{WS_URL}?token={WS_TOKEN}") as ws:
        # Подписаться на нужные типы событий
        await ws.send(json.dumps({
            "type": "subscribe",
            "data": ["entry_fill", "exit_fill", "status", "analyzed_df"]
        }))

        async for message in ws:
            event = json.loads(message)
            print(f"type: {event['type']}, data: {event['data']}")

asyncio.run(listen())
```

**Запросы от клиента через WebSocket:**

Помимо подписки, клиент может запрашивать данные прямо через WS-соединение — это аналог POST, но по WebSocket:

```python
# Запросить текущий whitelist
await ws.send(json.dumps({"type": "whitelist", "data": {}}))

# Запросить проанализированные свечи для конкретной пары
await ws.send(json.dumps({
    "type": "analyzed_df",
    "data": {"pair": "BTC/USDT", "limit": 100}
}))
```

**Типы push-событий (сервер → клиент):**

| Тип              | Что означает                                              |
| ---------------- | --------------------------------------------------------- |
| `entry_fill`     | Позиция открыта (ордер исполнен)                          |
| `exit_fill`      | Позиция закрыта                                           |
| `entry_cancel`   | Ордер на вход отменён                                     |
| `exit_cancel`    | Ордер на выход отменён                                    |
| `analyzed_df`    | Ответ на запрос свечей с индикаторами                     |
| `whitelist`      | Ответ на запрос whitelist / изменился список пар          |
| `status`         | Изменился статус бота (start/stop)                        |

**Пример: мониторинг сделок:**

```python
async def trade_monitor():
    async with websockets.connect(f"{WS_URL}?token={WS_TOKEN}") as ws:
        await ws.send(json.dumps({
            "type": "subscribe",
            "data": ["entry_fill", "exit_fill"]
        }))
        async for raw in ws:
            event = json.loads(raw)
            if event["type"] == "entry_fill":
                d = event["data"]
                print(f"🟢 ОТКРЫТА: {d['pair']} @ {d['open_rate']}")
            elif event["type"] == "exit_fill":
                d = event["data"]
                print(f"🔴 ЗАКРЫТА: {d['pair']} прибыль: {d['profit_ratio']:.2%}")

asyncio.run(trade_monitor())
```

### 3.4. Пример end-to-end: Force Exit

Чтобы показать, как все три канала работают вместе — разберём один сценарий: пользователь нажимает «Force exit» по сделке #42.

```
1. UI отправляет:
   POST /api/v1/forceexit
   Authorization: Bearer <jwt>
   {"tradeid": 42, "ordertype": "market"}

2. api_trading.py валидирует payload (ForceExitPayload)
   → вызывает rpc._rpc_force_exit("42", ordertype, ...)

3. RPC обращается к FreqtradeBot
   → создаётся market-sell ордер через Exchange
   → сделка закрывается, записывается в БД

4. RPCManager генерирует событие exit_fill в MessageStream

5. WebSocket-канал Dashboard, подписанный на exit_fill,
   получает событие и отправляет его клиенту:
   {"type": "exit_fill", "data": {"pair": "BTC/USDT", "profit_ratio": 0.032, ...}}

6. UI обновляет таблицу сделок без дополнительного GET
   (или дополнительно перезапрашивает GET /status для надёжности)
```

---

## Часть 4. Семантика GET / POST / WebSocket

| Канал          | Когда использовать                                                                                  |
| -------------- | --------------------------------------------------------------------------------------------------- |
| **GET**        | Только чтение. Не меняет Trade, Wallets, PairListManager. «Дай мне текущее состояние»               |
| **POST/DELETE**| Меняет состояние бота или БД. Команды (start/stop, forceenter/forceexit, blacklist, locks)          |
| **POST** (read-only с телом) | Запросы с богатым payload (фильтрация свечей) — read-only по смыслу, но GET с query-строкой неудобен |
| **WebSocket**  | Push-события реального времени + тяжёлые запросы данных (analyzed_df) без polling                   |

---

## Часть 5. Структура FreqUI как образец

FreqUI написан на Vue 3 + TypeScript. Это хорошая точка отсчёта при построении собственного UI — все паттерны работы с API уже реализованы.

```
src/
├── composables/
│   ├── loginInfo.ts      ← логин, сохранение токенов, refresh (изучить первым)
│   ├── api.ts            ← axios с interceptors для автообновления токена
│   └── backgroundJob.ts  ← фоновые задачи (polling)
├── stores/
│   ├── ftbot.ts          ← все вызовы API: getStatus(), getBalance(), getTrades()...
│   ├── ftbotwrapper.ts   ← мульти-бот, авто-refresh (refreshFrequent / refreshSlow)
│   └── btStore.ts        ← данные для Backtesting UI
├── views/                ← страницы (Trade, Dashboard, Backtesting...)
├── components/           ← UI-компоненты
└── types/                ← TypeScript-типы для всех API-ответов
```

Что изучить по порядку:

1. **`loginInfo.ts`** — полная реализация логина, хранения токенов в localStorage, refresh-логики
2. **`api.ts`** — axios instance с двумя interceptors: добавление `Authorization: Bearer` и перехват 401
3. **`ftbot.ts`** — фактически SDK для работы с API Freqtrade (47 KB, все вызовы в одном месте)
4. **`ftbotwrapper.ts`** — управление несколькими ботами, периодическое обновление данных

Запустить FreqUI рядом с ботом:

```bash
git clone https://github.com/freqtrade/frequi.git
cd frequi
pnpm install
pnpm run dev
# → http://localhost:3000
# Войти: URL = http://localhost:8080, username/password из конфига
```

---

## Часть 6. CORS

Браузер блокирует запросы с одного origin к другому. Если UI живёт на `localhost:3000`, а API — на `localhost:8080`, это разные origin.

**Симптом:**

```
Access to fetch at 'http://localhost:8080/api/v1/status' from origin
'http://localhost:3000' has been blocked by CORS policy
```

**Решение** — добавить URL UI в `CORS_origins`:

```json
"CORS_origins": ["http://localhost:3000"]
```

> ⚠️ Без trailing slash: `"http://localhost:3000"` — работает, `"http://localhost:3000/"` — нет.

| Сценарий                                  | Значение CORS_origins                                    |
| ----------------------------------------- | -------------------------------------------------------- |
| UI на том же порту, что API               | Не нужен                                                 |
| UI на другом порту (`localhost:3000`)     | `[«http://localhost:3000»]`                              |
| UI на другом домене                       | `[«https://myui.example.com»]`                           |
| Несколько UI                              | `[«http://localhost:3000», «http://localhost:4000»]`      |

CORS касается только браузерных клиентов. Python-скрипты и curl обращаются напрямую без ограничений.

---

## Часть 7. Swagger / OpenAPI

Если `enable_openapi: true` — после запуска бота доступен интерактивный explorer:

```
http://localhost:8080/docs
```

Там можно увидеть все эндпоинты с описанием параметров и схемами ответов, авторизоваться через кнопку **Authorize** и вызывать любой эндпоинт кнопкой **Try it out**. Полезно на этапе исследования API перед написанием своего UI.

---

## Часть 8. Минимальный Python-клиент

```python
import requests


class FreqtradeClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/") + "/api/v1"
        self.username = username
        self.password = password
        self.access_token: str | None = None
        self.refresh_token: str | None = None

    def login(self) -> None:
        resp = requests.post(
            f"{self.base_url}/token/login",
            auth=(self.username, self.password),
        )
        resp.raise_for_status()
        data = resp.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]

    def _refresh(self) -> None:
        resp = requests.post(
            f"{self.base_url}/token/refresh",
            headers={"Authorization": f"Bearer {self.refresh_token}"},
        )
        resp.raise_for_status()
        self.access_token = resp.json()["access_token"]

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        headers = {"Authorization": f"Bearer {self.access_token}"}
        resp = requests.request(method, f"{self.base_url}/{endpoint}", headers=headers, **kwargs)
        if resp.status_code == 401:
            self._refresh()
            headers["Authorization"] = f"Bearer {self.access_token}"
            resp = requests.request(method, f"{self.base_url}/{endpoint}", headers=headers, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _get(self, endpoint: str, **kwargs) -> dict:
        return self._request("GET", endpoint, **kwargs)

    def _post(self, endpoint: str, **kwargs) -> dict:
        return self._request("POST", endpoint, **kwargs)

    def _delete(self, endpoint: str, **kwargs) -> dict:
        return self._request("DELETE", endpoint, **kwargs)

    # Чтение
    def ping(self):           return self._get("ping")
    def status(self):         return self._get("status")
    def balance(self):        return self._get("balance")
    def profit(self):         return self._get("profit")
    def daily(self, days=7):  return self._get("daily", params={"timescale": days})
    def trades(self, limit=50, offset=0):
        return self._get("trades", params={"limit": limit, "offset": offset})
    def pair_candles(self, pair: str, timeframe: str, limit: int = 100):
        return self._get("pair_candles", params={"pair": pair, "timeframe": timeframe, "limit": limit})

    # Управление
    def start(self):          return self._post("start")
    def stop(self):           return self._post("stop")
    def reload_config(self):  return self._post("reload_config")
    def force_exit(self, trade_id: int):
        return self._post("forceexit", json={"tradeid": str(trade_id), "ordertype": "market"})
    def force_enter(self, pair: str, side: str = "long"):
        return self._post("forceenter", json={"pair": pair, "side": side})
    def delete_trade(self, trade_id: int):
        return self._delete(f"trades/{trade_id}")


# Использование
client = FreqtradeClient("http://localhost:8080", "myuser", "MyPassword1!")
client.login()

print(client.ping())
print(client.status())
print(client.balance())
print(client.daily(7))
candles = client.pair_candles("BTC/USDT", "5m", limit=50)
```

---

## Часть 9. Готовый Python-клиент от Freqtrade

```bash
pip install freqtrade-client
```

```python
from freqtrade_client import FtRestClient

client = FtRestClient("http://localhost:8080", "myuser", "MyPassword1!")

print(client.ping())
print(client.status())
print(client.balance())
client.forceexit(42)
```

Не требует установки самого Freqtrade — только `requests`. Удобен для скриптов, Jupyter-ноутбуков и интеграций.