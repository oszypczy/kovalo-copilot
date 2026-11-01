# ADR-0001: Jedna zintegrowana aplikacja zamiast pięciu miniprojektów

- Status: przyjęty
- Data: 2026-11-01

## Kontekst
Kurs składa się z pięciu miniprojektów, a każdy korzysta z kodu poprzednich. Prowadzący dopuszcza połączenie ich w jeden większy projekt.

## Decyzja
Budujemy jedną aplikację `kovalo` (pakiet `src/kovalo/`) z warstwami odpowiadającymi miniprojektom (mapowanie w [01-overview.md](../01-overview.md)). Aplikacja działa w docker-compose. Kod kursu leży w `reference/course/` tylko jako specyfikacja.

## Konsekwencje
- Nie korzystamy z `check.py` kursu. Zastępują go nasze testy (port testów kursu) i ewaluacje.
- Wartości `PINNED` zostają, więc wyniki da się porównać z przebiegami referencyjnymi.
- Quizy kursu dalej przygotowujemy z `docs/course/mpN_CONCEPTS.md`.

## Rozważane alternatywy
- Pięć osobnych folderów jak w kursie: prostsze sprawdzenie zgodności, ale bez aplikacji i Dockera.
