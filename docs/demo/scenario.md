# Scenariusz demo: pokaz aplikacji na żywo

Scenariusz pokazu aplikacji na żywo (około 5–7 minut, bez slajdów i bez nagrania w zapasie: decyzja użytkownika z 2026-11-07). Rdzeń to rozmowa **C1** z ewaluacji E5 ([data/eval/mini_project_5_conversations.yaml](../../data/eval/mini_project_5_conversations.yaml)): cztery tury, które w przebiegu na żywo z 2026-11-04 przeszły w 100% (ścieżka, status, pauzy, fakty). Kontrakt: [contracts/f6.md §10](../contracts/f6.md).

Akceptacja jest symulowana: przyciski Zatwierdź / Odrzuć / Anuluj zastępują osoby zatwierdzające, nic nie trafia do CRM-u.

## 1. Przygotowanie (dzień wcześniej i 10 minut przed)

**Dzień wcześniej (próba):**
1. `make up`, potem `make ingest` (jeśli baza była czyszczona). `curl http://localhost:8080/health` ma zwrócić `"status": "ok"`.
2. Jedno przejście scenariusza z części 2 (około 10 wywołań Gemini). Sprawdź liczby z tabeli w części 3.

**10 minut przed pokazem:**
1. Docker Desktop działa, `make up` (przy aktualnych obrazach budowanie bierze warstwy z cache), `curl http://localhost:8080/health` → `ok`.
2. Przeglądarka: okno około 1280×800, zoom 100%, jedna karta z http://localhost:8000, bez paska zakładek (macOS Chrome: Cmd+Shift+B), tryb gościa albo osobny profil.
3. Nowa rozmowa: odśwież stronę (każde odświeżenie to nowy wątek i zerowa suma kosztu).
4. W drugiej karcie (do pokazania na końcu): [reports/dashboard.md](../../reports/dashboard.md) na GitHubie albo w podglądzie Markdownu w edytorze (działa bez sieci).
5. Zapas: w trzeciej karcie rozmowa z porannej próby (tury 1–2 z części 2, około 6 wywołań), **nieodświeżana**. Przy błędzie limitu albo sieci pokazujesz w niej kroki „(przez MCP)”, ścieżkę dowodową i linię kosztu.

**Prywatność (sprawdź przed udostępnieniem ekranu):**
- [ ] żadnego okna z `.env`, terminala z `env`/`printenv` ani AI Studio z kluczem w kadrze,
- [ ] pasek zakładek i inne karty ukryte, powiadomienia wyciszone (Centrum sterowania → Skupienie),
- [ ] udostępniasz tylko okno przeglądarki, nie cały ekran.

## 2. Przebieg

Każdą wiadomość wklejaj w pole czatu w całości i wysyłaj Enterem. Czekaj na koniec tury (zniknięcie wskaźnika pracy) przed następną wiadomością.

### Tura 1: kara umowna przez MCP (około 1,5 min na żywo)

```
Ile wynosi kara, którą Mleczna Dolina jest winna Kovalo za lipiec 2026 r.? Podaj kwotę w PLN.
```

Co widać i co pokazać kursorem:
- Kroki agenta pojawiają się na żywo, każdy z dopiskiem „(przez MCP)”: warunki umowy (`get_contract_terms`), OTIF (`get_supplier_otif`), wartość faktur (`get_invoice_value`), kalkulator kary (`calculate_otif_penalty`). Rozwiń jeden krok: argumenty (`SUP-114`, `2026-07`) i wynik z linią „Wywołanie MCP: …”.
- Odpowiedź: **7296,00 PLN** (model może zapisać „7 296,00 PLN”; obie formy są poprawne).
- Kliknij „Szczegóły: Ścieżka dowodowa”: panel boczny pokazuje, z którego narzędzia pochodzi każda liczba (OTIF 94,6% wobec celu 97%, 2 pełne punkty, 4% × 182 400,00 PLN).
- Pod odpowiedzią linia kursywą: „Tura: N wywołań (gemini-3.5-flash-lite), … tokenów wejścia, … wyjścia. Koszt: 0 USD w darmowym tierze; po cenach płatnych … centa (Gemini) / … centa (cennik kursu). Rozmowa: 1 tura, …”.

Co powiedzieć: liczby liczy kod (narzędzia i kalkulator kary), nie model; model tylko wybiera narzędzia i składa odpowiedź. Narzędzia działają w osobnym kontenerze `mcp-server`, ten sam serwer działa w Claude Code i MCP Inspector.

### Tura 2: wniosek o rabat i pauza na akceptację (około 1 min)

```
Teraz prześlij do akceptacji rabat 14% na wodę niegazowaną dla Biedronki z marżą po rabacie 10%.
```

Co widać:
- Odpowiedź bez kroków agenta: „Wniosek: …”, „Wymagana akceptacja: Dyrektor handlowy (symulacja)” i uzasadnienie z polityki `course-v1` (14% mieści się w przedziale > 12–20%, marża 10% nie dodaje CFO, bo próg to 9%).
- Pauza z przyciskami **Zatwierdź**, **Odrzuć**, **Anuluj**; pod nią linia kosztu (1 wywołanie: rozpoznanie wiadomości).
- Kliknij **Zatwierdź** → „Wniosek zatwierdzony (wersja 1): Dyrektor handlowy. To symulacja: nic nie trafia do CRM-u.” Ta tura nie ma linii kosztu (wznowienie pauzy nie wywołuje LLM-a).

Co powiedzieć: kto zatwierdza, decyduje kod polityki, nie prompt; workflow LangGraph zatrzymuje się w `interrupt()`, a pauza jest w checkpoincie Postgresa, więc przetrwa restart `api`.

### Tura 3: pytanie o dane w trakcie sprawy (około 1 min)

```
A jaki był OTIF dostawcy Aqua Kryształ w lipcu 2026 r.?
```

Co widać: jeden krok `get_supplier_otif (przez MCP)` (`SUP-207`), odpowiedź **95,1%**, ścieżka dowodowa, linia kosztu. Zatwierdzony wniosek się nie zmienia.

### Tura 4: ponowne przesłanie bez zmian (około 30 s)

```
Prześlij jeszcze raz ten sam rabat dla Biedronki.
```

Co widać: „Już zatwierdzone (wersja 1), akceptacja: Dyrektor handlowy. Wniosek się nie zmienił, więc nie trzeba go akceptować ponownie. …”, **bez nowej pauzy**. Linia kosztu z sumą „Rozmowa: 4 tury, … / … centa” (kliknięcie Zatwierdź nie jest liczone jako tura z LLM-em).

### Opcjonalnie, jeśli zostaje czas

Pytanie o politykę bez składania wniosku (1 wywołanie, rozmowa C2 z E5):

```
Kto musi zatwierdzić rabat 22% dla naszego klienta strategicznego Carrefour Express przy marży po rabacie 7%?
```

Oczekiwane: „Rabat 22% (klient strategiczny, marża po rabacie 7%) zatwierdza: Dyrektor handlowy i CFO.” z uzasadnieniem i zdaniem „To tylko informacja: żaden wniosek nie został złożony.”

Na koniec (0 wywołań): karta z [reports/dashboard.md](../../reports/dashboard.md): wyniki E1–E5 wobec kursu, koszty przebiegów i zużycie darmowego limitu.

## 3. Oczekiwane wyniki i budżet

| Tura | Oczekiwany wynik | Pauza | Wywołania Gemini (szacunek) |
|---|---|---|---|
| 1 | 7296,00 PLN; 4 kroki „(przez MCP)” | brak | ≈ 5 (rozpoznanie 1 + agent ≈ 4) |
| 2 | Dyrektor handlowy; po Zatwierdź: zatwierdzony (wersja 1) | Dyrektor handlowy | 1 (wznowienie: 0) |
| 3 | 95,1% | brak | ≈ 3 (rozpoznanie 1 + agent ≈ 2) |
| 4 | „Już zatwierdzone”, bez pauzy | brak | 1 |
| opcjonalnie | Dyrektor handlowy i CFO | brak | 1 |
| **Razem** | | | **≈ 10** (z pytaniem opcjonalnym ≈ 11) |

Budżet w dniu pokazu: poranna próba ≈ 6 (karta zapasowa) + pokaz ≈ 10–11, razem około 17 wywołań, czyli w limicie Flash-Lite. Model główny `gemini-3.5-flash-lite` (w F4–F5 około 50 wywołań dziennie bez błędów 429); fallback `gemini-3.6-flash` ma tylko około 20 zapytań dziennie, więc nie rób w dniu pokazu dodatkowych prób.

Dokładne liczby tokenów i centów w linii kosztu zależą od długości odpowiedzi modelu i nie są stałe; sprawdź tylko, że linia jest, że suma rośnie z każdą turą z LLM-em i że po kliknięciu Zatwierdź jej nie ma. Jeśli w linii pojawi się drugi model (`gemini-3.6-flash`), tura poszła przez fallback: to też jest poprawne.

## 4. Plan B

| Sytuacja | Co zrobić |
|---|---|
| Pojedynczy błąd 429 albo przeciążenie modelu głównego | Nic nie rób: aplikacja sama przełącza się na model zapasowy `gemini-3.6-flash` (w linii kosztu pojawi się drugi model). To dobry moment, żeby pokazać fallback. |
| Odpowiedź „Nie udało się odpowiedzieć na to pytanie…” albo długie czekanie (429 na obu modelach, limit dzienny) | Odczekaj około minuty i wyślij wiadomość jeszcze raz, ale tylko raz (każda próba zużywa limit). Jeśli znowu błąd: przejdź do karty zapasowej z porannej rozmowy i omów kroki, ścieżkę dowodową i linię kosztu na niej. Tury 2 i 4 (wniosek, „Już zatwierdzone”) wymagają tylko jednego wywołania, więc zwykle przechodzą nawet przy prawie wyczerpanym limicie. |
| Brak sieci | Karta zapasowa (już wyrenderowana rozmowa), dashboard w podglądzie Markdownu i README z lokalnego klona. Kliknięcie Zatwierdź na pauzie nie wywołuje LLM-a, więc wznowienie pauzy działa bez sieci. Ścieżkę dowodową P1 z zapisanego przebiegu E4 wypisuje też `uv run kovalo show-trace P1` (0 wywołań; wymaga lokalnych JSON-ów raportów w `reports/mp4/`). |
| `/health` = `degraded` (brak klucza albo indeksu) | Brak klucza: uzupełnij `GOOGLE_API_KEY` w `.env`, potem `make down` i `make up`. Brak indeksu: `make ingest`. |
| Odpowiedź z inną kwotą niż 7296,00 PLN lub 95,1% | Rozwiń ścieżkę dowodową i kroki: liczby z narzędzi są deterministyczne, a różnica byłaby w składaniu odpowiedzi; zgłoś po pokazie, nie poprawiaj na żywo. |
| „Poprzednia odpowiedź nie została dokończona…” | Po restarcie `api` w trakcie tury: serwis zamknął przerwaną turę, wyślij wiadomość jeszcze raz. |
