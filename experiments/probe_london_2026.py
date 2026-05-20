"""Probe ITTF API for London 2026 raw rows — inspect categories before filtering."""

from pipeline.config import get_headers, ITTF_BASE_URL
from pipeline.http import fetch_json
import pandas as pd

TOURNAMENT_ID = 3216
PAGE_SIZE = 100


def main() -> None:
    headers = get_headers()
    url = (
        f"{ITTF_BASE_URL}index.php?"
        f"option=com_fabrik&view=list&listid=31&Itemid=250"
        f"&resetfilters=1&format=json"
        f"&vw_matches___tournament_id_raw[value][]={TOURNAMENT_ID}"
        f"&limit31={PAGE_SIZE}&limitstart31=0"
    )
    print(f"URL: {url}")
    data = fetch_json(url, headers)
    if data is None:
        print("Failed to fetch")
        return

    if isinstance(data, list):
        rows = data[0] if data and isinstance(data[0], list) else data
    elif isinstance(data, dict):
        rows = data.get("data", data.get("rows", []))
        if isinstance(rows, dict):
            rows = list(rows.values())
    else:
        rows = []

    print(f"Returned rows: {len(rows)}")
    if not rows:
        print("Empty response. Full payload:")
        print(repr(data)[:500])
        return

    print("\nFirst row keys:")
    print(list(rows[0].keys())[:30])

    print("\nFirst row sample:")
    for k, v in list(rows[0].items())[:25]:
        print(f"  {k}: {repr(v)[:80]}")

    # Category distribution
    cats = pd.Series([r.get("vw_matches___event_raw") for r in rows]).value_counts(dropna=False)
    print(f"\nEvent (vw_matches___event_raw) values: {cats.to_dict()}")

    # Total per API
    if isinstance(data, dict) and "total" in data:
        print(f"\nTotal per API: {data['total']}")


if __name__ == "__main__":
    main()
