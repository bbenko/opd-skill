# Open Payments Operator Skill

This workspace contains an operator-focused skill and CLI for querying CMS Open Payments data.

Canonical skill name: `opd`.

## Agent System Compatibility

This skill can be used in agentic systems such as Codex, Claude, and OpenClaw.

- Use `SKILL.md` as the behavior/instruction contract.
- Use `scripts/opd.py` as the executable query interface.
- Any agent runtime that can run Python and allow network access to CMS can use this skill.

## Repository layout

- `SKILL.md`
  - Skill instructions and usage workflow.
- `scripts/opd.py`
  - Main CLI tool for operator queries.
- `references/query-patterns.md`
  - Prompt-to-command mappings.
- `references/year-distribution-map.json`
  - Local year-to-distribution cache used to skip metastore lookups.

## Requirements

- Python 3.10+ (tested with Python 3.13.5)
- No third-party package is required for:
  - `scripts/opd.py`

## Quick start

Run from the repo root:

```bash
python3 scripts/opd.py --help
```

### 1) Individual Provider payments

```bash
python3 scripts/opd.py provider --first "Jane" --last "Smith" --year 2024
```

Filter to a Nature of Payment:

```bash
python3 scripts/opd.py provider --first "Jane" --last "Smith" --nature "Food and Beverage" --year 2024
```

### 2) Individual Providers paid by a company

```bash
python3 scripts/opd.py company "Pfizer" --year 2024
```

Faster large-company query:

```bash
python3 scripts/opd.py company "Medtronic" --nature "Food and Beverage" --year 2024
```

### 3) Individual Providers paid for a product/drug

```bash
python3 scripts/opd.py product "Eliquis" --year 2024
```

## Common flags

- `--year 2024`
  - Required. Selects the reporting year dataset.
- `--nature "Food and Beverage"`
  - Optional filter by Nature of Payment or Other Transfer of Value.
- `--timeout 180`
  - Optional HTTP timeout in seconds for slower API responses.
- `--output json`
  - Compact JSON output (default is formatted JSON).
- `--product-fields "1,2"` (product mode)
  - Optional product columns to search. Default is `1`.

## Fast defaults

This CLI now uses fixed defaults to stay simple and fast:

- Exact matching for provider, company, and nature filters.
- Product mode defaults to exact search on product field 1 only.
- When field 1 returns no rows, output includes a message suggesting rerun with broader `--product-fields`.
- Datastore queries fetch one page (`100` rows) per request.
- Server-side count aggregation is disabled for speed.
- Timeout retries are disabled; timeout errors return a rerun hint with `--timeout`.
- Year-to-distribution mapping is auto-cached in `references/year-distribution-map.json`.

## Slow API guidance

The OPD API can be slow. Use this retry order:

1. Start with default parameters first.
2. If the command times out, rerun with higher `--timeout` (for example `--timeout 180`).
3. If product search returns zero results on default field 1, rerun with broader fields such as `--product-fields "1,2"`.

### Output totals semantics

- `totals.matched_rows`
  - Number of matched rows returned in the fetched page(s).
- `totals.fetched_rows`
  - Number of rows actually fetched into the response payload before output-row truncation.
- `totals.pagination_capped`
  - `true` when the request hits the configured page cap.

### Output links

- Provider payment rows include OPD profile links when IDs are available: `payment_url`, `physician_url`, and `company_url`.
- Aggregated provider rows include URL arrays when available: `physician_urls` and `company_urls`.

## Notes

- `references/year-distribution-map.json` is used as an automatic local cache.
- The first query for a year may include metastore lookup; later queries for the same year reuse cached distribution IDs.

- In restricted sandboxes, binding local ports may require elevated permissions.
- Live CMS calls require network access to `https://openpaymentsdata.cms.gov`.
