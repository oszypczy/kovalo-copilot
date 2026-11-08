# Scenariusz demo

Prompty do wklejenia w czat (http://localhost:8000) i ścieżka, którą każdy uruchamia w grafie copilota ([diagram, strona 2](../architecture/kovalo-architecture-2-graph.png)). Wszystkie pochodzą z rozmów ewaluacji E5 ([data/eval/mini_project_5_conversations.yaml](../../data/eval/mini_project_5_conversations.yaml)). Akceptacja jest symulowana: przyciski zastępują osoby zatwierdzające, nic nie trafia do CRM-u.

## answer_documents

```
Ile wynosi kara, którą Mleczna Dolina jest winna Kovalo za lipiec 2026 r.? Podaj kwotę w PLN.
```

understand → answer_documents (agent, narzędzia przez MCP) → koniec tury. Odpowiedź: 7296,00 PLN.

## answer_policy

```
Kto musi zatwierdzić rabat 22% dla naszego klienta strategicznego Carrefour Express przy marży po rabacie 7%?
```

understand → answer_policy → koniec tury. Odpowiedź: Dyrektor handlowy i CFO, bez składania wniosku.

## reply_other

```
Dzięki!
```

understand → reply_other → koniec tury. Krótka odpowiedź z listą tego, co copilot potrafi.

## update_request → clarify → decide → approve

```
Prześlij do akceptacji rabat 8% na chipsy dla Delikatesów Pod Lipami.
```

1. understand → update_request: wniosek bez marży, więc clarify (pauza z pytaniem o marżę po rabacie).
2. Wpisz w czat `7` → decide: wymagana akceptacja Regionalnego kierownika sprzedaży i CFO.
3. approve: pauza na pierwszą rolę → **Zatwierdź**, potem pauza na drugą rolę → **Zatwierdź**. Status: zatwierdzony.

## update_request → decide (już zatwierdzone)

W tej samej rozmowie, po zatwierdzeniu:

```
Prześlij jeszcze raz ten sam rabat dla Delikatesów Pod Lipami.
```

understand → update_request → decide: wszystkie role już zatwierdziły, więc „Już zatwierdzone”, bez nowej pauzy.
