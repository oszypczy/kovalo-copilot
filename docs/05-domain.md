# 05. Domena Kovalo

Pełny opis: [course/business_scenario.md](course/business_scenario.md), słownik pojęć: [course/slownik.md](course/slownik.md).

Kovalo Distribution sp. z o.o. to fikcyjny dystrybutor żywności z Łodzi. Ma trzy magazyny (Łódź, Poznań, Gdańsk), ok. 140 dostawców, ok. 1400 klientów, 180 mln PLN sprzedaży i docelową marżę brutto 12,5%.

## Korpus (`data/corpus/`, 18 dokumentów)

| doc_id | Tytuł | Typ |
|---|---|---|
| POL-001 | Polityka akceptacji rabatów | policy |
| POL-002 | Polityka limitów kredytowych i terminów płatności | policy |
| POL-003 | Polityka zwrotów i reklamacji towarów uszkodzonych | policy |
| POL-004 | Polityka podróży służbowych i rozliczania wydatków | policy |
| POL-005 | Polityka wyboru i wdrażania dostawców | policy |
| CON-101 | Streszczenie umowy z dostawcą: Mleczna Dolina (SUP-114) | contract |
| CON-102 | Streszczenie umowy z dostawcą: Aqua Kryształ (SUP-207) | contract |
| CON-103 | Umowa z podwykonawcą logistycznym: TransBałt | contract |
| PRO-201 | Procedura realizacji zamówienia: od przyjęcia do dostawy | procedure |
| PRO-202 | Procedura inwentaryzacji i spisywania strat magazynowych | procedure |
| PRO-203 | Procedura postępowania przy incydentach w łańcuchu chłodniczym | procedure |
| PL-601 | Regulamin magazynu w Gdańsku (**po angielsku**) | procedure |
| REP-301 | Przegląd operacyjny za II kwartał 2026 r. | report |
| REP-302 | Analiza odejść klientów w I półroczu 2026 r. | report |
| MTG-401 | Notatka z posiedzenia zarządu: 14 lipca 2026 r. | meeting_notes |
| MTG-402 | Notatka z posiedzenia komitetu cenowego: 5 sierpnia 2026 r. | meeting_notes |
| MTG-403 | Spotkanie przeglądowe z dostawcą: Mleczna Dolina, 20 sierpnia 2026 r. | meeting_notes |
| FAQ-501 | Definicje wskaźników KPI | faq |

## Dane firmy (`data/company/`, stan na 2026-10-01)

| Plik | Zawartość |
|---|---|
| `suppliers.csv` | `SUP-114` Mleczna Dolina (CON-101), `SUP-207` Aqua Kryształ (CON-102) |
| `supplier_otif.csv` | miesięczny OTIF w %; za 2026-10 brak danych (to zwykły wynik `no_data`, nie błąd) |
| `supplier_invoices.csv` | faktury w **całkowitych groszach** |
| `contract_terms.yaml` | warunki OTIF: SUP-114 cel 97%, kara 2% za każdy **pełny** punkt poniżej celu, limit 10%; SUP-207 cel 96%, **bez kar**; CON-103 (TransBałt) to **nie** miesięczny OTIF (`not_applicable`) |

Kara (CON-101), ręcznie: lipiec 94,6% → 2,4 pp → 2 pełne → 4% × 182 400,00 PLN = **7296,00 PLN**; czerwiec 95,2% → 1 pełny → 2% × 181 200,00 = **3624,00 PLN**; maj 96,4% → 0 pełnych → **brak kary**.

Rozwiązanie umowy (CON-101): OTIF **mniej niż 94%** przez trzy kolejne miesiące. Lipiec 94,6% przerywa serię (sierpień 93,8%, wrzesień 93,5%), więc na koniec września warunek **nie** jest spełniony. Agent kursu mylił się tu w każdym z 6 przebiegów. Dlatego dodajemy narzędzie liczące to w kodzie (FR-4.7).

## Polityka rabatowa `course-v1` (POL-001 jako kod)

1. **Zwykły przedział:** ≤5% przedstawiciel handlowy; >5–12% regionalny kierownik sprzedaży; >12–20% dyrektor handlowy; >20% **CEO i CFO**.
2. **Wyjątki**, każdy tylko do swojego limitu (`cap`): wyprzedaż końcówek (`clearance`) → regionalny kierownik do 35%; klient strategiczny (`strategic_account`) → dyrektor handlowy do 25%; udokumentowana oferta konkurencji (`competitor_offer`) → regionalny kierownik do 15%. Wyjątek może obniżyć poziom uprawnień, nigdy go nie podnosi.
3. Wybierany jest **najniższy** uprawniony poziom.
4. **Minimalna marża:** marża po rabacie **poniżej 9%** dodaje CFO, chyba że to wyprzedaż końcówek (zwolniona także powyżej 35%).

Przypadki testowe (muszą przejść w 100%):

| Wniosek | Zatwierdzają |
|---|---|
| 3%, klient strategiczny, marża 10% | Przedstawiciel handlowy |
| 15%, zwykły, marża 10% | Dyrektor handlowy |
| 22%, zwykły, marża 10% | CEO i CFO |
| 22%, klient strategiczny, marża 7% | Dyrektor handlowy i CFO |
| 14%, klient strategiczny + oferta konkurencji, marża 7% | Regionalny kierownik sprzedaży i CFO |
| 30%, wyprzedaż końcówek, marża 4% | Regionalny kierownik sprzedaży |
| 36%, wyprzedaż końcówek, marża 4% | CEO i CFO (limit wyprzedaży przekroczony; zwolnienie z minimalnej marży zostaje) |

## Narzędzia (MP4, udostępniane przez MCP w MP5)

| Narzędzie | Zwraca | Rodzaj w trace |
|---|---|---|
| `search_documents(question)` | surowe chunki z `doc_id` i `chunk_id` | data |
| `get_contract_terms(supplier)` | cel OTIF, klauzula kary, stawka, limit, `source_doc_id` | data |
| `get_supplier_otif(supplier, from_month, to_month)` | obserwacje + lista brakujących miesięcy + `complete` | data |
| `get_invoice_value(supplier, month)` | `invoiced_grosze` | data |
| `calculate_otif_penalty(otif_pct, target_pct, rate_per_full_point_pct, cap_pct, invoice_grosze)` | wszystkie wartości pośrednie + `penalty_grosze` | calculation |
| `check_discount_policy(discount_pct, margin_after_pct, clearance, strategic_account, competitor_offer, contractual_volume_rebate)` | wymagane role + uzasadnienie (tylko informacja) | calculation |
| `check_termination_condition(supplier, as_of_month)` *(nasze)* | trzy miesiące, próg, wynik porównania | calculation |

Każdy wynik ma `status`: `ok`, `no_data`, `unknown_supplier`, `not_applicable`, `out_of_scope`, `invalid_request`. **Żadne** narzędzie nie zatwierdza, nie zapisuje i nie obiecuje.

## Statusy tury

`answered`, `pending_clarification`, `pending_approval`, `approved` (symulowany), `rejected`, `cancelled`, `error` (+ `limit` dla agenta).

## Prompty

Prompty `system`, `answer_system`, `understand_system`, `agent_system`, `assistant_understand_system` przenosimy dosłownie z [`reference/course/config.yaml`](../reference/course/config.yaml) (`PINNED`). Zmiany promptów zapisujemy jako nowe wersje (`prompts.<nazwa>_v2`) z wynikiem ewaluacji przed i po zmianie.
