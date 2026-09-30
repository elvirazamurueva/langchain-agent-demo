"""Консольный диалог с AI-агентом.

Запуск:
    python main.py

Команды:
    exit — завершить программу;
    new  — начать новый разговор (новый thread_id).

Любое другое сообщение отправляется агенту. История диалога хранится
в памяти агента (checkpointer) и привязана к текущему thread_id.
"""

from __future__ import annotations

import uuid

from agent import build_agent


def extract_text(message) -> str:
    """Аккуратно достаёт текст ответа из сообщения модели."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def ask(agent, thread_id: str, text: str) -> str:
    """Отправляет сообщение агенту в рамках указанного thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config,
    )
    return extract_text(result["messages"][-1])


def main() -> None:
    print("AI-агент LangChain (GenAPI).")
    print("Команды: 'exit' — выход, 'new' — начать новый разговор.")

    try:
        agent = build_agent()
    except RuntimeError as exc:
        # Например, не задан GENAPI_API_KEY. Значение ключа не печатаем.
        print(f"Ошибка конфигурации: {exc}")
        return

    thread_id = str(uuid.uuid4())
    print(f"Текущий thread_id: {thread_id}\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command in {"exit", "quit", "выход"}:
            print("Пока!")
            break

        if command == "new":
            thread_id = str(uuid.uuid4())
            print(f"Начат новый разговор. thread_id: {thread_id}\n")
            continue

        try:
            answer = ask(agent, thread_id, user_input)
        except Exception as exc:  # noqa: BLE001 — показываем пользователю текст ошибки
            print(f"Ошибка при обращении к модели: {exc}\n")
            continue

        print(f"Агент: {answer}\n")


if __name__ == "__main__":
    main()
