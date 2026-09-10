# RoadAi

RoadAi — сервис для автоматического обнаружения дефектов дорожного покрытия на изображениях, видео и RTSP-потоках.

Проект использует модель **YOLO** для обнаружения объектов и предоставляет REST API на **FastAPI**. Для работы с потоковым видео также используется WebSocket.

## Возможности

* обнаружение дефектов на изображениях;
* обработка видеофайлов;
* обработка RTSP-потоков;
* получение результатов в реальном времени через WebSocket;
* API-документация через Swagger UI;
* API Key для защиты эндпоинтов;
* запуск в Docker.

## Стек

* Python 3.11
* FastAPI
* Uvicorn
* Ultralytics YOLO
* OpenCV
* WebSocket
* Docker / Docker Compose

## Запуск

### 1. Клонирование проекта

```bash
git clone <repository-url>
cd RoadAi
```

### 2. Настройка переменных окружения

Создайте файл `.env` и укажите необходимые параметры, например API-ключ:

```env
INTERNAL_API_KEY=your_api_key
```

### 3. Запуск через Docker Compose

Соберите образ и запустите сервис:

```bash
docker compose up --build
```

После первой сборки для повторного запуска:

```bash
docker compose up
```

Для остановки:

```bash
docker compose down
```

## API

После запуска API доступно по адресу:

```text
http://localhost:8080
```

Swagger UI:

```text
http://localhost:8080/docs
```

Основные эндпоинты:

```text
POST /detect/image
POST /detect/video
POST /rtsp/start
POST /rtsp/stop
WS   /ws/live
```

Для запросов используется заголовок:

```text
X-API-Key: your_api_key
```

## Примечание

Сервис предназначен для обработки запросов на обнаружение дорожных дефектов и может использоваться как отдельный worker в составе более крупной системы.
А узнать больше обо всем проекте сможете здесь в документации: https://github.com/SuleimanovErik/RoadVision-AI