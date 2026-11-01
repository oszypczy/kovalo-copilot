---
name: course-step
description: Procedura przeniesienia kroku miniprojektu kursu TEG (mini_project_N/src/stepK_*.py) do zintegrowanej aplikacji Kovalo Decision Copilot. Używaj, gdy zadanie dotyczy implementacji funkcji odpowiadającej krokowi kursu (MP1–MP5) albo gdy trzeba sprawdzić, co dokładnie kurs wymaga od danego elementu.
---

# Przeniesienie kroku kursu do aplikacji

1. **Znajdź krok.** Mapowanie MP → moduł: `docs/01-overview.md`. Wymaganie: `docs/02-requirements.md` (FR-N.x).
2. **Przeczytaj specyfikację kursu:**
   - `reference/course/mini_project_N/src/stepK_*.py`: docstring CEL / WYNIK / WYMAGANIA i sygnatura,
   - `reference/course/mini_project_N/tests/test_stepK_*.py`: oczekiwane zachowanie i przypadki brzegowe,
   - `docs/course/mpN_README.md` (sekcje „Przybornik”, „Jeśli utkniesz”, przebieg referencyjny) i `docs/course/mpN_CONCEPTS.md`,
   - `reference/course/helpers/`: czy jest gotowa logika do przeniesienia.
3. **Zachowaj** wartości PINNED, nazwy pól wyników, statusy i formaty identyfikatorów (`POL-001#002`, `POL-001#ctx002`).
4. **Dostosuj** tylko to, co wynika z ADR-ów: LLM przez `kovalo.llm` (Gemini), embeddingi fastembed, PGVector, MCP przez HTTP.
5. **Testy:** przenieś testy kursu do `tests/unit/<moduł>/` (test-engineer), potem implementacja (ai-engineer).
6. **Udokumentuj** odstępstwo od kursu (jeśli jest) w `docs/01-overview.md` → „Czym różnimy się od wersji kursowej” albo w ADR.
7. **Porównaj** z przebiegiem referencyjnym z README kursu i zapisz wynik w raporcie ewaluacji.
