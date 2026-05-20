"""Fetch ITTF World Team Championships London 2026 (event_id=2712, tournament_id=3216).

Stand-alone — paginates ITTF matches endpoint for this single tournament and
accepts MT/WT (team) categories. Inside team rubbers, individual singles
matches still appear with one player on each side (player_a vs player_x), so
we use those for 1v1 Elo validation.

Output: experiments/london_2026_matches.parquet (raw)
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.config import ITTF_BASE_URL, get_headers
from pipeline.http import fetch_json

TOURNAMENT_ID = 3216
EVENT_ID = 2712
EVENT_NAME = "ITTF World Team Table Tennis Championships Finals London 2026"
EVENT_DATE = date(2026, 5, 10)
PAGE_SIZE = 100
ACCEPTED_CATEGORIES = {"MT", "WT", "MS", "WS"}
OUTPUT = Path(__file__).resolve().parent / "london_2026_matches.parquet"

_WO_RE = re.compile(r"W/?O", re.IGNORECASE)
_RET_RE = re.compile(r"RET", re.IGNORECASE)
_RESULT_RE = re.compile(r"^\s*\d+\s*:\s*\d+\s*$")


def _safe_int(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _safe_str(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s else None


def _parse_result(result_raw, pa, pb):
    if pa is None or pb is None:
        return None, "missing_player_id"
    if not result_raw:
        return None, "missing_result"
    s = result_raw.strip()
    if _WO_RE.search(s):
        return None, "walkover"
    if _RET_RE.search(s):
        return None, "retirement"
    if not _RESULT_RE.match(s):
        return None, "non_standard_result"
    a, b = [int(x.strip()) for x in s.split(":")]
    if a == b:
        return None, "non_standard_result"
    return (pa if a > b else pb), None


def _match_key(eid, dt, pa, pb, result, games):
    return hashlib.sha256(
        f"{eid}|{dt}|{pa}|{pb}|{result}|{games}".encode("utf-8")
    ).hexdigest()


def fetch_all() -> pd.DataFrame:
    headers = get_headers()
    matches = []
    offset = 0
    date_str = EVENT_DATE.isoformat()

    while True:
        url = (
            f"{ITTF_BASE_URL}index.php?"
            f"option=com_fabrik&view=list&listid=31&Itemid=250"
            f"&resetfilters=1&format=json"
            f"&vw_matches___tournament_id_raw[value][]={TOURNAMENT_ID}"
            f"&limit31={PAGE_SIZE}&limitstart31={offset}"
        )
        print(f"  offset {offset}...", end=" ", flush=True)
        data = fetch_json(url, headers)
        if data is None:
            print("FAIL")
            break

        if isinstance(data, list):
            rows = data[0] if data and isinstance(data[0], list) else data
        elif isinstance(data, dict):
            rows = data.get("data", data.get("rows", []))
            if isinstance(rows, dict):
                rows = list(rows.values())
        else:
            rows = []

        print(f"got {len(rows)} rows")
        if not rows:
            break

        for row in rows:
            cat = _safe_str(row.get("vw_matches___event_raw"))
            if cat not in ACCEPTED_CATEGORIES:
                continue

            src_id = _safe_int(row.get("vw_matches___id_raw") or row.get("__pk_val"))
            pa = _safe_int(row.get("vw_matches___player_a_id_raw"))
            pa_name = _safe_str(row.get("vw_matches___name_a_raw"))
            pb_a = _safe_int(row.get("vw_matches___player_b_id_raw"))  # doubles partner of A
            px = _safe_int(row.get("vw_matches___player_x_id_raw"))
            px_name = _safe_str(row.get("vw_matches___name_x_raw"))
            py = _safe_int(row.get("vw_matches___player_y_id_raw"))  # doubles partner of X

            result_raw = _safe_str(row.get("vw_matches___res_raw"))
            games_raw = _safe_str(row.get("vw_matches___games_raw"))
            if result_raw:
                result_raw = re.sub(r"\s*-\s*", ":", result_raw)

            is_doubles = (pb_a is not None) or (py is not None)
            winner_id, drop_reason = _parse_result(result_raw, pa, px)

            matches.append({
                "match_key": _match_key(EVENT_ID, date_str, pa or 0, px or 0, result_raw or "", games_raw or ""),
                "source_match_id": src_id,
                "match_date": EVENT_DATE,
                "event_id": EVENT_ID,
                "event_name": EVENT_NAME,
                "player_a_id": pa,
                "player_a_name": pa_name or "",
                "player_b_id": px,
                "player_b_name": px_name or "",
                "winner_id": winner_id,
                "result": result_raw or "",
                "games": games_raw or "",
                "category": cat,
                "is_doubles": is_doubles,
                "drop_reason": "doubles" if is_doubles else drop_reason,
            })

        total = data.get("total") if isinstance(data, dict) else None
        if len(rows) < PAGE_SIZE:
            break
        if total is not None:
            try:
                if offset + len(rows) >= int(total):
                    break
            except (ValueError, TypeError):
                pass
        offset += PAGE_SIZE
        time.sleep(0.5)

    return pd.DataFrame(matches)


def main() -> None:
    print(f"Fetching tournament {TOURNAMENT_ID}: {EVENT_NAME}")
    df = fetch_all()
    print(f"\nTotal records: {len(df)}")
    if df.empty:
        return

    print(f"  Categories: {df['category'].value_counts().to_dict()}")
    print(f"  Doubles: {df['is_doubles'].sum()}")
    print(f"  Drop reasons: {df['drop_reason'].value_counts(dropna=False).to_dict()}")
    usable = df[df["drop_reason"].isna()]
    print(f"  Usable singles (winner known): {len(usable)}")
    if not usable.empty:
        print(f"  Unique players in usable matches: {pd.concat([usable['player_a_id'], usable['player_b_id']]).nunique()}")

    df.to_parquet(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
