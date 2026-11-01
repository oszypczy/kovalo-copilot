---
name: langgraph-patterns
description: Konwencje LangChain 1.x i LangGraph 1.x w projekcie Kovalo Decision Copilot. Używaj przy pisaniu lub testowaniu grafu copilota (StateGraph, stan, reducery, routing, interrupt/Command resume, checkpointer Postgres), structured output, agenta create_agent z narzędziami, fabryki modeli (init_chat_model, rate limiter, fallback) i atrap modeli w testach.
---

# Wzorce LangChain / LangGraph w projekcie

Zawsze weryfikuj sygnatury w MCP `langchain-docs` (API 1.x zmienia się szybko). Poniższe to konwencje projektu, nie dokumentacja.

## Modele
- Jedyny punkt tworzenia modeli: `kovalo.llm.get_chat_model(purpose=...)`, w środku `init_chat_model(model, model_provider=..., rate_limiter=InMemoryRateLimiter(...))` + `.with_fallbacks([...])`.
- Structured output: `model.with_structured_output(PydanticSchema, include_raw=True)`. Błąd parsowania → jedno ponowienie, potem status `error` z surowym tekstem w rekordzie (nigdy cicha odmowa).
- Tokeny: `AIMessage.usage_metadata` → licznik kosztów `kovalo.cost`.

## Graf copilota
- Stan: `TypedDict` z `messages: Annotated[list[AnyMessage], add_messages]`; reszta pól bez reducera (nadpisywane). Wniosek o rabat jako model Pydantic z `revision`.
- Fabryka: `build_copilot_graph(*, understand_model, document_answerer, checkpointer)`. `document_answerer` to callable/Runnable (RAG w MP3, agent przez MCP w MP5), żeby podmiana węzła nie zmieniała grafu.
- Routing: `add_conditional_edges("understand", route_by_kind, {...})`; funkcje routujące są czyste i mają testy.
- Pauzy: w węźle `value = interrupt({"kind": "approval", "role": role, "request": req.model_dump()})`. Wznowienie: `graph.invoke(Command(resume=value), config={"configurable": {"thread_id": tid}})`. Kod przed `interrupt()` w węźle wykona się ponownie przy wznowieniu, więc przed pauzą nie umieszczaj efektów ubocznych.
- Strażnik pauzy: przed nową wiadomością sprawdź `graph.get_state(config).interrupts`; jeśli jest aktywna pauza, odrzuć wiadomość.
- Checkpointer: `PostgresSaver` w aplikacji (`setup()` przy starcie), `InMemorySaver` w testach.

## Agent
- `create_agent(model, tools=[...], system_prompt=agent_system)`; limit kroków przez `recursion_limit` (24), timeout 180 s sprawdzany przed każdym krokiem.
- Narzędzia: `@tool` z `args_schema` (Pydantic), zwracają dict z `status`. Trace zbieramy z `ToolMessage` + `AIMessage.tool_calls` i zapisujemy z rodzajem `data`/`calculation`.

## Testy
- `GenericFakeChatModel(messages=iter([...]))` dla prostych odpowiedzi; do structured output i tool calling skryptowany model zwracający `AIMessage(tool_calls=[...])` (wzorzec w `reference/course/mini_project_4/tests/conftest.py` i `mini_project_5/tests/conftest.py`).
- Dialogi: odtworzenie z idealnym `understand` (atrapa zwraca oczekiwane pola z YAML-a), żeby testować workflow niezależnie od LLM-a.
