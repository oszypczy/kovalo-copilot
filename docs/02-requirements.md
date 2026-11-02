# 02. Wymagania

Źródło wymagań: README i CONCEPTS miniprojektów kursu ([course/](course/)). Każde wymaganie ma identyfikator `FR-x.y`, gdzie `x` to numer miniprojektu, z którego pochodzi. Kryteria akceptacji odwołują się do przebiegów referencyjnych kursu.

## Wymagania funkcjonalne

### FR-1: fundament wiedzy (MP1)

| ID | Wymaganie | Kryterium akceptacji |
|---|---|---|
| FR-1.1 | Wczytaj 18 dokumentów `data/corpus/*.md` jako `Document` LangChaina z metadanymi z `catalog.csv` (`doc_id, title, doc_type, department, owner, effective_date, language`) | 18 dokumentów, każde pole metadanych wypełnione; brak pliku z katalogu = błąd |
| FR-1.2 | Policz tokeny tokenizerem `o200k_base` i oszacuj miesięczny koszt (400 pytań/dzień, 21 dni, 350 tokenów wyjścia, ceny z `config.yaml`) dla dwóch wariantów: retrieval i cały korpus w prompcie. Dwa cenniki obok siebie: referencyjny kursu (`gpt-5.4-mini`, PINNED, do porównania z przebiegiem wzorcowym) i nasz (płatny tier Gemini, `cost.our_model` i `cost.our_prices_per_1m_tokens`, [ADR-0007](adr/0007-knowledge-index-f1.md)) | po cenach kursu: 8762 tokeny; $17,78 wobec $69,86 (x3,9); po cenach `gemini-3.6-flash` (0,75 / 3,75 USD, stan 2026-11-02) przy wyszukiwaniu referencyjnym: $15,58 wobec $67,65 |
| FR-1.3 | Podziel dokumenty na chunki (`RecursiveCharacterTextSplitter`, 800/120, separatory z configu) z identyfikatorem `<doc_id>#<nnn>` | 47 chunków, najdłuższy ≤ 800 znaków |
| FR-1.4 | Zbuduj embeddingi z cache (`CacheBackedEmbeddings`) i indeks wektorowy | ponowne indeksowanie nie liczy embeddingów od nowa |
| FR-1.5 | Wyszukuj `top_k=3` chunki po podobieństwie kosinusowym; policz podobieństwo także samodzielnie | wynik własnego cosine = wynik vector store (±1e-6 w pamięci, ±1e-5 w pgvector, który trzyma `float4`; wynik zawsze jako podobieństwo, [ADR-0007](adr/0007-knowledge-index-f1.md)) |
| FR-1.6 | Fabryka LLM-a: jeden punkt konfiguracji dostawcy (Gemini; opcjonalnie OpenAI w trybie zgodnym z kursem) | zmiana `llm.provider` w configu przełącza model bez zmian w kodzie |

### FR-2: odpowiedzi ze źródłami (MP2)

| ID | Wymaganie | Kryterium akceptacji |
|---|---|---|
| FR-2.1 | Chunki z kontekstem: najpierw podział według nagłówków Markdown, potem według rozmiaru; każdy chunk zaczyna się od `<tytuł> > <sekcja>`, id `<doc_id>#ctxNNN` | ok. 87 chunków; każdy ma nagłówek z tytułem dokumentu |
| FR-2.2 | Odpowiedź w ustalonym schemacie (structured output): `answer`, `answerable`, `citations[]` + rekord przebiegu (pytanie, chunki, prompt, tokeny, odpowiedź) | 100% odpowiedzi parsuje się do schematu albo kończy się statusem `error` (nigdy cichą odmową) |
| FR-2.3 | Ocena rekordu bez LLM-a: frazy dowodowe, pokrycie kluczowych faktów, poprawność odwołań, trafność odmów | deterministyczna, pokryta testami jednostkowymi |
| FR-2.4 | Sędzia LLM (LLM-as-judge) w schemacie: `correct`, `grounded`, uzasadnienie | błąd sędziego liczony osobno, nie jako ocena |
| FR-2.5 | *(rozszerzenie)* Wyszukiwanie hybrydowe (BM25 + wektory) dla dokładnych identyfikatorów (`SUP-114`) | M03 („Co SUP-114 mówi o karach?”) znajduje CON-101 w top-3 |

### FR-3: workflow decyzyjny (MP3)

| ID | Wymaganie | Kryterium akceptacji |
|---|---|---|
| FR-3.1 | Silnik polityki rabatowej `course-v1` jako czysty kod (przedziały, wyjątki z limitem, najniższy uprawniony poziom, minimalna marża 9% → CFO, wyprzedaż końcówek zwolniona) | tabela przypadków z [05-domain.md](05-domain.md) przechodzi w 100% |
| FR-3.2 | Węzeł `understand`: LLM klasyfikuje wiadomość (`document_question / policy_question / approval_request / other`), `request_action` (`new / update / none`), `standalone`, pola rabatu i flagi (`null`, gdy niewspomniane) | na dialogach kursu: kierowanie 13/13, pola 11/11 (cel) |
| FR-3.3 | Graf LangGraph: `understand → route → answer_documents / answer_policy / reply_other / update_request → clarify ⟲ → decide → approve ⟲` | odtworzenie dialogów z idealnym `understand` (bez LLM-a) przechodzi 100% |
| FR-3.4 | Pauzy przez `interrupt()` i wznowienie `Command(resume=…)`: pytanie o marżę (liczba albo `cancel`) i akceptacja każdej roli po kolei (`approved / rejected / cancel`) | nieprawidłowa odpowiedź → ta sama rola pytana ponownie |
| FR-3.5 | Wersjonowanie wniosku: zmiana warunków → nowa wersja, wcześniejsze akceptacje przestają się liczyć; te same warunki po pełnej akceptacji → „Już zatwierdzone” | testy stanu wniosku |
| FR-3.6 | Strażnik: w trakcie pauzy przyjmujemy tylko odpowiedź na bieżące pytanie | nowa wiadomość w pauzie → odrzucona z komunikatem |
| FR-3.7 | Polskie odpowiedzi człowieka: „zatwierdź / odrzuć / anuluj”, liczby „7,5” i „7.5” | testy parsera |
| FR-3.8 | Statusy tury: `answered, pending_clarification, pending_approval, approved, rejected, cancelled, error` | każda tura kończy się jednym z nich |

### FR-4: agent z narzędziami (MP4)

| ID | Wymaganie | Kryterium akceptacji |
|---|---|---|
| FR-4.1 | Dostęp do danych firmy (`data/company/`): dostawcy, miesięczny OTIF (z brakującymi miesiącami), faktury w **całkowitych groszach**, warunki OTIF z umów | `no_data`, `unknown_supplier`, `not_applicable` to zwykłe wyniki; uszkodzony plik to wyjątek |
| FR-4.2 | Kalkulator kary OTIF z wartościami pośrednimi (`shortfall_points`, `full_points`, `penalty_pct_uncapped`, `penalty_pct`, `penalty_grosze`) na liczbach całkowitych i `Decimal` | P1 = 729 600 groszy, P2 = 362 400, P3 = 0 |
| FR-4.3 | Sześć narzędzi tylko do odczytu: `search_documents, get_contract_terms, get_supplier_otif, get_invoice_value, calculate_otif_penalty, check_discount_policy`; każdy wynik ma pole `status` | żadne narzędzie nie może niczego zatwierdzić ani zapisać (test możliwości) |
| FR-4.4 | Agent ReAct (`create_agent` z LangChain) z promptem `agent_system`, limitem 24 kroków i 180 s | przekroczenie limitu → status `limit`, nie wyjątek |
| FR-4.5 | Zapis przebiegu (trace): każde wywołanie narzędzia z argumentami, wynikiem, rodzajem (`data / calculation`), tokenami i czasem | `show_trace` wypisuje ścieżkę dowodową jak w README MP4 |
| FR-4.6 | *(opcjonalne)* Wariant supervisora z trzema agentami specjalistycznymi (dokumenty, dane, polityka) do porównania | ten sam program oceniający dla obu wariantów |
| FR-4.7 | *(rozszerzenie)* Narzędzie `check_termination_condition` (trzy kolejne miesiące OTIF < 94%) liczone w kodzie | T1 poprawne (lekcja z przebiegów referencyjnych) |

### FR-5: MCP i końcowy asystent (MP5)

| ID | Wymaganie | Kryterium akceptacji |
|---|---|---|
| FR-5.1 | Serwer FastMCP „kovalo”: 6 narzędzi (cienkie nakładki na FR-4), dokumenty jako zasoby, prompt; transporty `stdio` i `streamable-http` | MCP Inspector widzi narzędzia, zasoby i prompt |
| FR-5.2 | Klient MCP w asystencie: narzędzia z `list_tools()`, wywołania `call_tool()` z zachowaniem `structuredContent` i walidacją zgodności z rekordem z FR-4 | 4 klasy błędów z warstwą: `tool / protocol / transport / validation` |
| FR-5.3 | Limity: start 60 s, wywołanie 120 s, zamknięcie 10 s; połączenie po przekroczeniu limitu nie jest używane ponownie | testy z serwerami symulującymi błędy |
| FR-5.4 | Asystent = graf z FR-3 z jednym podmienionym węzłem: `document_question → answer_with_tools` (agent z FR-4 przez MCP); prompt `assistant_understand_system` | ewaluacja asystenta: ścieżka 25/25 (cel) |
| FR-5.5 | Akceptacje nie są narzędziem serwera: zostają w asystencie | inny klient MCP nie widzi workflow akceptacji |

### FR-6: aplikacja (nasze rozszerzenie)

| ID | Wymaganie | Kryterium akceptacji |
|---|---|---|
| FR-6.1 | API (FastAPI): `POST /threads`, `POST /threads/{id}/messages`, `POST /threads/{id}/resume`, `GET /threads/{id}` (stan, pauza), `GET /threads/{id}/trace`, `GET /health` | kontrakt OpenAPI, testy API |
| FR-6.2 | UI (Chainlit): czat, strumieniowanie odpowiedzi, kroki agenta (tool calls) jako rozwijane `Step`, pauzy jako przyciski akceptacji (`AskActionMessage`) i pole marży, źródła jako elementy z treścią chunku | ręczny scenariusz C1–C5 z MP5 przechodzi w UI |
| FR-6.3 | Indeksowanie jako polecenie `kovalo ingest` (`make ingest`), idempotentne | ponowne uruchomienie nie duplikuje chunków |
| FR-6.4 | Ewaluacja jako zadanie (`eval`) z raportem Markdown/JSON w `reports/` | raport porównuje wynik z referencyjnym |
| FR-6.5 | Licznik kosztów i tokenów per tura i per przebieg | widoczny w UI i raporcie ewaluacji |

## Wymagania niefunkcjonalne

| ID | Wymaganie |
|---|---|
| NFR-1 | **Darmowe działanie:** domyślna konfiguracja nie wymaga płatnego klucza (Gemini free tier + lokalne embeddingi). |
| NFR-2 | **Uruchomienie jednym poleceniem:** `docker compose up` (+ `make ingest` przy pierwszym uruchomieniu). |
| NFR-3 | **Testy bez LLM-a:** `pytest -m "not live"` nie wywołuje żadnego API (atrapy modeli `GenericFakeChatModel` / skryptowane scenariusze). |
| NFR-4 | **Odporność na limity darmowego tieru:** `InMemoryRateLimiter`, ponowienia z backoffem, `with_fallbacks` (Gemini Flash → Gemini Flash-Lite, osobny limit na model). |
| NFR-5 | **Audytowalność:** każda liczba w odpowiedzi agenta pochodzi z wyniku narzędzia; trace zapisany per tura. |
| NFR-6 | **Język:** UI, dokumentacja i odpowiedzi po polsku; identyfikatory, klucze i statusy po angielsku. Liczby w zapisie polskim (94,6%; 7296,00 PLN). |
| NFR-7 | **Sekrety:** klucze tylko w `.env` (poza gitem); nigdy w argumentach ani wynikach narzędzi MCP. |
| NFR-8 | **Porównywalność z kursem:** wartości `PINNED` z `config.yaml` kursu zostają bez zmian. |
| NFR-9 | **Jakość kodu:** Python 3.12, `uv`, `ruff`, typy (`mypy` w trybie podstawowym), CI na GitHub Actions. |
