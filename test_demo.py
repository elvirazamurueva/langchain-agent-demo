"""Небольшие проверки проекта (без обращения к сети).

Запуск:
    python test_demo.py

Проверяется:
  * проект импортируется без ошибок;
  * конфигурация правильно преобразует GENAPI_URL в base_url;
  * пользовательский tool get_cat_fact работает;
  * память (checkpointer) сохраняет состояние в рамках одного thread
    и изолирует разные thread'ы.

Для проверки памяти используется встроенная тестовая модель
(GenericFakeChatModel), поэтому реальный API-ключ не нужен.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import config
import tools
from tools import get_cat_fact


class ToolCapableFakeModel(GenericFakeChatModel):
    """Тестовая модель, которая умеет принимать привязку инструментов.

    Встроенная GenericFakeChatModel не реализует bind_tools, поэтому
    агент с инструментами на ней не запускается. Здесь мы просто
    игнорируем привязку — инструменты в проверке памяти не вызываются.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


def test_imports_and_config() -> None:
    """Проект импортируется, а конфигурация нормализует base_url."""
    assert tools.TOOLS, "Список инструментов пуст"

    assert (
        config.normalize_base_url("https://proxy.gen-api.ru/v1/chat/completions")
        == "https://proxy.gen-api.ru/v1"
    )
    assert (
        config.normalize_base_url("https://proxy.gen-api.ru/v1/")
        == "https://proxy.gen-api.ru/v1"
    )
    assert (
        config.normalize_base_url("https://api.example.com/v1")
        == "https://api.example.com/v1"
    )
    print("[OK] импорты и конфигурация (normalize_base_url)")


def test_tool() -> None:
    """Инструмент возвращает факт о породе и корректно обрабатывает неизвестную."""
    known = get_cat_fact.invoke({"breed": "мейн-кун"})
    assert "Мейн-кун" in known, known

    alias = get_cat_fact.invoke({"breed": "британец"})
    assert "британская короткошёрстная" in alias, alias

    unknown = get_cat_fact.invoke({"breed": "тигр"})
    assert "не найден" in unknown, unknown

    print("[OK] tool get_cat_fact")


def test_memory() -> None:
    """Память хранит историю в рамках thread и изолирует разные thread'ы."""
    model = ToolCapableFakeModel(
        messages=iter(
            [
                AIMessage(content="Привет, Елена!"),
                AIMessage(content="Тебя зовут Елена."),
                AIMessage(content="Это отдельный разговор."),
            ]
        )
    )
    agent = create_agent(
        model=model,
        tools=tools.TOOLS,
        system_prompt="тестовая инструкция",
        checkpointer=InMemorySaver(),
    )

    thread_a = {"configurable": {"thread_id": "thread-a"}}
    thread_b = {"configurable": {"thread_id": "thread-b"}}

    # Два сообщения подряд в ОДНОМ thread.
    agent.invoke({"messages": [{"role": "user", "content": "Меня зовут Елена."}]}, thread_a)
    state_a = agent.invoke(
        {"messages": [{"role": "user", "content": "Как меня зовут?"}]}, thread_a
    )

    contents_a = [m.content for m in state_a["messages"]]
    assert "Меня зовут Елена." in contents_a, contents_a
    assert "Как меня зовут?" in contents_a, contents_a
    assert len(state_a["messages"]) == 4, contents_a  # 2 сообщения user + 2 ответа AI

    # Другой thread — отдельная история.
    state_b = agent.invoke({"messages": [{"role": "user", "content": "Привет!"}]}, thread_b)
    contents_b = [m.content for m in state_b["messages"]]
    assert "Меня зовут Елена." not in contents_b, contents_b
    assert len(state_b["messages"]) == 2, contents_b

    print("[OK] память (checkpointer) и изоляция thread'ов")


def test_live_agent_optional() -> None:
    """Живая проверка GenAPI — только если задан GENAPI_API_KEY."""
    if not config.get_api_key():
        print("[SKIP] живая проверка GenAPI (GENAPI_API_KEY не задан)")
        return

    from agent import build_agent

    agent = build_agent()
    config_ = {"configurable": {"thread_id": "live-check"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Привет! Ответь одним словом."}]},
        config_,
    )
    assert result["messages"][-1].content
    print("[OK] живая проверка GenAPI")


def main() -> None:
    test_imports_and_config()
    test_tool()
    test_memory()
    test_live_agent_optional()
    print("\nВсе проверки пройдены.")


if __name__ == "__main__":
    main()
