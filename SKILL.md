---
name: opd
description: >-
  Query and analyze CMS Open Payments data using Advanced Search language for
  Individual Providers and Companies. Use when users ask which payments or
  other transfers of value an Individual Provider received, which Individual
  Providers received payments from a company, or which Individual Providers are
  tied to a product or drug, including Nature of Payment filters such as Food
  and Beverage. Trigger phrases: opd, open payments, cms open payments.
---

# Open Payments Operator

Use this skill to answer operational questions against CMS Open Payments data.

## Use Workflow

1. Identify the intent:
- `provider`: list payments for an Individual Provider.
- `company`: list Individual Providers paid by a company.
- `product`: list Individual Providers paid in records tied to a product or drug.

2. Run the query script with the matching mode:

```bash
python3 scripts/opd.py provider --first "first_name" --last "last_name" --year 2024
python3 scripts/opd.py company "company_name" --year 2024
python3 scripts/opd.py product "product_name" --year 2024
```

3. Add filters when requested:
- `--nature "Food and Beverage"` for Nature of Payment filtering.
- `--year 2024` is required to target a reporting year.
- Optional `--output json` returns compact JSON instead of pretty JSON.
- Use `--timeout` to increase HTTP timeout if the API times out.
- Product mode defaults to product field 1; use `--product-fields "1,2"` to broaden search when needed.
- Internal defaults are intentionally fixed for simplicity and speed (exact matching, one-page fetch, no server-side count, no timeout retries).
- OPD API latency can be high; always try defaults first, then increase `--timeout` on timeout, then broaden product fields only when results are zero.

4. Summarize results for the user:
- For Individual Provider lookups, report payments with company, amount, date, and payment nature.
- For company/product lookups, report providers with payment counts and total amounts.

## Output Contract

Treat script output as canonical structured data:
- `payments`: row-level payment details for `provider`.
- `providers`: provider-level aggregates for `company` and `product`.
- `totals`: top-level counts and total dollars.

Use default output for formatted JSON, or `--output json` for compact JSON.

## Field and Prompt Guidance

Use [references/query-patterns.md](references/query-patterns.md) for intent mapping and prompt-to-command patterns.
