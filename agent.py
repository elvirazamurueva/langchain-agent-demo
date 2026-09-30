"""Создание AI-агента на актуальном API LangChain.

Используется современный ``create_agent`` из ``langchain.agents``
(вместо устаревших ``initialize_agent`` / ``AgentExecutor``).

Память (история диалога) реализована через checkpointer LangGraph —
``InMemorySaver``.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from config import create_model
from tools import TOOLS

# Системная инструкция агенту.
SYSTEM_PROMPT = (
    "Ты дружелюбный ассистент. Всегда отвечай на русском языке.\n"
    "Если вопрос касается информации о породах кошек (факты, особенности), "
    "используй инструмент get_cat_fact и опирайся ТОЛЬКО на его результат.\n"
    "Никогда не выдумывай результат работы инструмента.\n"
    "Если инструмент не нужен — отвечай самостоятельно, используя свои знания."
)


def build_agent():
    """Создаёт агента с памятью для хранения истории диалога.

    Возвращает скомпилированный граф агента. История разговора хранится
    в checkpointer и привязана к ``thread_id`` в config.
    """
    model = create_model()
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent
