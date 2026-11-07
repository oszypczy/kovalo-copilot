# Serwery-zabawki MCP (testy stdio)

Port `reference/course/mini_project_5/tests/toy_servers/` (F5, T1). Każdy zapisuje swój PID do pliku
z `argv[1]`, żeby test sprawdził, że proces naprawdę się zakończył. Nie są pakietem; uruchamia je
`kovalo.mcp_client.connection.stdio_connection([plik, pid_file], cwd, ...)`.

- `exits_at_start.py`: kończy działanie przed odpowiedzią na `initialize`.
- `silent.py`: nigdy nie odpowiada na `initialize`.
- `stubborn.py`: odpowiada ręcznie, potem ignoruje EOF na stdin i SIGTERM.
- `fragile.py`: prawdziwy FastMCP; `crash` zabija proces, `hang` się zawiesza, `ping` odpowiada.
- `env_report.py`: NASZ; zapisuje nazwy zmiennych środowiska procesu (test braku kluczy, NFR-7).
