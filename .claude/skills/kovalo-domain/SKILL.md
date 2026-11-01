---
name: kovalo-domain
description: Wiedza domenowa Kovalo Distribution dla projektu Kovalo Decision Copilot. Używaj przy każdej pracy dotyczącej polityki rabatowej (POL-001, course-v1, osoby zatwierdzające, marża minimalna, wyjątki), kar OTIF i umów z dostawcami (SUP-114, SUP-207, CON-101/102/103), narzędzi agenta i ich statusów, statusów tury workflow, dokumentów korpusu (doc_id) lub formatowania liczb po polsku.
---

# Domena Kovalo

Pełny opis: `docs/05-domain.md`. Słownik: `docs/course/slownik.md`. Teksty źródłowe: `data/corpus/*.md`.

## Polityka rabatowa course-v1 (z `config.yaml` → `policy`)
- Przedziały (rabat do limitu włącznie): ≤5 Przedstawiciel handlowy; ≤12 Regionalny kierownik sprzedaży; ≤20 Dyrektor handlowy; ≤100 CEO **i** CFO.
- Wyjątki obniżają poziom, tylko do limitu: `clearance` cap 35 → Regionalny kierownik; `strategic_account` cap 25 → Dyrektor handlowy; `competitor_offer` cap 15 → Regionalny kierownik. Nigdy nie podnoszą poziomu.
- Wybierany jest najniższy uprawniony poziom.
- Marża po rabacie < 9 (ściśle mniej) dodaje CFO, chyba że `clearance` (zwolnienie obowiązuje także powyżej 35%).
- Przypadki kontrolne: `docs/05-domain.md` (tabela). Muszą przechodzić w 100%.

## Kary OTIF
- SUP-114 Mleczna Dolina (CON-101): cel 97%, 2% faktury za każdy **pełny** punkt poniżej celu, limit 10%, rozliczenie notą uznaniową.
- SUP-207 Aqua Kryształ (CON-102): cel 96%, **brak kar** (`penalty_clause: false`, stawka `null`, nigdy 0).
- CON-103 TransBałt: 98% palet w oknie czasowym to **nie** miesięczny OTIF → `not_applicable`; nigdy nie liczy się z tego kary.
- Kontrola: lipiec 2026 → 729 600 groszy (7296,00 PLN); czerwiec → 362 400; maj → 0.
- Rozwiązanie umowy CON-101: OTIF **< 94%** przez 3 kolejne miesiące. 94,0 nie jest „mniej niż 94”. Na koniec 09.2026 warunek NIE jest spełniony (lipiec 94,6%).

## Narzędzia i statusy
- Wynik narzędzia: zawsze `status` ∈ {ok, no_data, unknown_supplier, not_applicable, out_of_scope, invalid_request}. Brak danych za 2026-10 to `no_data`, nie wyjątek.
- Kwoty w **całkowitych groszach** (`*_grosze`), procenty `*_pct`, punkty procentowe `*_points`.
- Narzędzia nie zatwierdzają, nie zapisują, nie obiecują.
- Statusy tury: answered, pending_clarification, pending_approval, approved, rejected, cancelled, error (+ limit dla agenta).

## Liczby i język
- Odpowiedzi po polsku: przecinek dziesiętny i spacja tysięcy (94,6%; 7296,00 PLN; 18 240 000 groszy).
- „7,296” jest niejednoznaczne: nie produkuj takich zapisów.
- Odpowiedzi człowieka w pauzach: zatwierdź/odrzuć/anuluj (oraz approve/reject/cancel), marża „7,5” lub „7.5”.
- Identyfikatory, klucze i statusy po angielsku. PL-601 jest celowo po angielsku.
