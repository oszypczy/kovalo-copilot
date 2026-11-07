"""Odtwarza wycinki raportów E1-E5 w tym katalogu z prawdziwych `reports/mp*/*.json` (poza gitem).

Wycinek = ten sam kształt JSON-a co raport, ale bez ciężkich rekordów pytanie po pytaniu (teksty
odpowiedzi, rekordy RAG, transkrypty, ścieżki dowodowe) i z przyciętymi długimi napisami. Adaptery
dashboardu mają czytać tylko pola zbiorcze, więc test na wycinku jest tak samo mocny jak na całości.

    uv run python tests/fixtures/dashboard/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = {
    "e1": "reports/mp1/2026-11-02-e1.json",
    "e2": "reports/mp2/2026-11-02-e2.json",
    "e3": "reports/mp3/2026-11-03-e3.json",
    "e4": "reports/mp4/2026-11-03-e4.json",
    "e5": "reports/mp5/2026-11-04-e5.json",
}
DROP = {
    "transcript", "record", "verdict", "events", "trace", "context", "prompt", "output",
    "top_scores", "top_ids", "top_doc_ids", "top_chunk_ids", "answer", "context_chunks",
    "evidence_path", "steps", "tool_calls_detail", "messages", "replies",
}  # fmt: skip
HEAVY = 400  # bajtów: tylko duże wartości z DROP są wycinane (małe pary [trafione, razem] zostają)


def prune(value: Any, key: str | None = None) -> Any:
    if key == "trace" and isinstance(value, dict):  # zostaje tożsamość pytania i wynik
        return {
            k: prune(v, k) for k, v in value.items() if k in ("question_id", "question", "status")
        }
    if key in DROP and len(json.dumps(value)) > HEAVY:
        return None
    if isinstance(value, dict):
        return {k: prune(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [prune(v) for v in value]
    if isinstance(value, str) and len(value) > 80:
        return value[:80]
    return value


def main() -> None:
    for stage, rel in SOURCES.items():
        data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        text = json.dumps(prune(data), ensure_ascii=False, indent=1) + "\n"
        (OUT / f"{stage}.json").write_text(text, encoding="utf-8")
        print(stage, len(text) // 1024, "KB")


if __name__ == "__main__":
    main()
