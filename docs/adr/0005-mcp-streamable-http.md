# ADR-0005: Serwer MCP jako osobny kontener (streamable HTTP) + stdio

- Status: przyjęty
- Data: 2026-11-01

## Kontekst
W kursie każdy klient uruchamia własną kopię serwera MCP przez stdio. W docker-compose serwer jest usługą, a klient (`api`) działa w innym kontenerze.

## Decyzja
Serwer FastMCP obsługuje oba transporty: `streamable-http` (kontener `mcp-server`, port 8765, dla `api`) i `stdio` (MCP Inspector, Claude Desktop, Claude Code). Klient w `api` korzysta bezpośrednio z MCP SDK (`list_tools`, `call_tool`), zachowuje `structuredContent` i waliduje go z kontraktem narzędzia. Limity czasu i 4 klasy błędów jak w kursie.

## Konsekwencje
- Serwer musi być bezstanowy względem sesji (narzędzia tylko do odczytu, więc to naturalne).
- Klasa błędu `transport` obejmuje też błędy HTTP i zerwane połączenie.
- Testy: serwer w pamięci (jak w kursie) + test integracyjny przez HTTP w compose.
