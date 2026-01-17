from __future__ import annotations

import csv
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable, List, Optional

@dataclass(frozen=True)
class TickerItem:
    ticker: str
    name: str = ""
    exchange: str = ""

def load_tickers(csv_path: str | Path) -> List[TickerItem]:
    path = Path(csv_path)
    if not path.exists():
        return []
    items: List[TickerItem] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("ticker") or "").strip().upper()
            if not t:
                continue
            items.append(
                TickerItem(
                    ticker=t,
                    name=(row.get("name") or "").strip(),
                    exchange=(row.get("exchange") or "").strip(),
                )
            )
    return items

def search(items: Iterable[TickerItem], query: str, limit: int = 10) -> List[TickerItem]:
    q = (query or "").strip().lower()
    items = list(items)
    if not q:
        return items[:limit]

    # rank: exact ticker > substring in ticker/name > fuzzy ticker
    exact = [x for x in items if x.ticker.lower() == q]
    if exact:
        return exact

    contains = [
        x for x in items
        if q in x.ticker.lower() or q in x.name.lower()
    ]
    if contains:
        return contains[:limit]

    tickers = [x.ticker for x in items]
    fuzzy = set(get_close_matches(q.upper(), tickers, n=limit, cutoff=0.4))
    out = [x for x in items if x.ticker in fuzzy]
    return out[:limit]

def pick_interactive(items: List[TickerItem]) -> str:
    if not items:
        raise SystemExit("Brak listy tickerów (configs/tickers.csv). Podaj --ticker ręcznie.")

    while True:
        q = input("Szukaj (ticker/nazwa), Enter = pokaż listę: ").strip()
        matches = search(items, q, limit=12)

        if not matches:
            print("Brak wyników. Spróbuj inaczej.")
            continue

        for i, it in enumerate(matches, start=1):
            suffix = f" | {it.exchange}" if it.exchange else ""
            name = f" - {it.name}" if it.name else ""
            print(f"{i:>2}. {it.ticker}{name}{suffix}")

        choice = input("Wybierz numer (1..), albo wpisz ticker: ").strip().upper()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(matches):
                return matches[idx - 1].ticker
            print("Niepoprawny numer.")
            continue

        if choice:
            return choice
