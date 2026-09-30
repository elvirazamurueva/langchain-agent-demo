"""Конфигурация проекта: переменные окружения и создание модели.

Модель подключается не к OpenAI напрямую, а к OpenAI-совместимому
эндпоинту GenAPI. Все настройки берутся из файла .env.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Папка проекта (там же лежит .env).
PROJECT_DIR = Path(__file__).resolve().parent

# Явно указываем путь к .env, чтобы настройки подхватывались
# независимо от текущей рабочей директории.
load_dotenv(PROJECT_DIR / ".env")

DEFAULT_GENAPI_URL = "https://proxy.gen-api.ru/v1/chat/completions"
DEFAULT_MODEL_ID = "gpt-4o-mini"


def normalize_base_url(raw_url: str) -> str:
    """Преобразует полный URL эндпоинта в base_url для OpenAI-клиента.

    ChatOpenAI (через официальный OpenAI SDK) ожидает БАЗОВЫЙ адрес API,
    к которому сам добавляет путь /chat/completions. Поэтому, если в .env
    указан полный URL вида .../v1/chat/completions, нужно оставить только
    .../v1.

    Примеры:
        https://proxy.gen-api.ru/v1/chat/completions -> https://proxy.gen-api.ru/v1
        https://proxy.gen-api.ru/v1/               -> https://proxy.gen-api.ru/v1
    """
    url = (raw_url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


def get_api_key() -> str:
    """Возвращает API-ключ GenAPI. Значение ключа нигде не логируется."""
    return os.getenv("GENAPI_API_KEY", "").strip()


def get_base_url() -> str:
    """Возвращает base_url для OpenAI-совместимого клиента."""
    return normalize_base_url(os.getenv("GENAPI_URL", DEFAULT_GENAPI_URL))


def get_model_id() -> str:
    """Возвращает идентификатор модели (например, gpt-4o-mini)."""
    return os.getenv("MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID


def create_model() -> ChatOpenAI:
    """Создаёт LangChain-модель, настроенную на GenAPI.

    Возвращает ChatOpenAI, но с base_url, указывающим на GenAPI, и
    api_key из переменной окружения GENAPI_API_KEY.
    """
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "Не задан GENAPI_API_KEY. Укажите ключ в файле .env "
            "(образец — .env.example)."
        )

    return ChatOpenAI(
        model=get_model_id(),
        api_key=api_key,
        base_url=get_base_url(),
        temperature=0,
    )
