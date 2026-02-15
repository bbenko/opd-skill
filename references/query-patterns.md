# Query Patterns

Use these mappings to convert natural-language questions into CLI calls.

## Pattern 1: Payments for an Individual Provider

Intent:
- "Show payments for provider Jane Smith."
- "Show only Food and Beverage payments for provider Jane Smith."

Command template:

```bash
python3 scripts/opd.py provider --first "<FIRST_NAME>" --last "<LAST_NAME>" --nature "<OPTIONAL_PAYMENT_NATURE>" --year <YEAR>
```

Result focus:
- `payments[*].company_name`
- `payments[*].amount_usd`
- `payments[*].payment_nature`
- `payments[*].date_of_payment`

## Pattern 2: Individual Providers Paid by a Company

Intent:
- "List individual providers paid by company X."
- "Who did company X pay for Food and Beverage?"

Command template:

```bash
python3 scripts/opd.py company "<COMPANY_NAME>" --nature "<OPTIONAL_PAYMENT_NATURE>" --year <YEAR>
```

Result focus:
- `providers[*].provider_name`
- `providers[*].payment_count`
- `providers[*].total_amount_usd`

## Pattern 3: Individual Providers Paid for a Product/Drug

Intent:
- "List providers paid related to product Y."
- "Who received payments tied to drug Y?"

Command template:

```bash
python3 scripts/opd.py product "<PRODUCT_OR_DRUG_NAME>" --nature "<OPTIONAL_PAYMENT_NATURE>" --year <YEAR>
```

Result focus:
- `providers[*].provider_name`
- `providers[*].payment_count`
- `providers[*].total_amount_usd`

## Resolution Notes

- OPD API can be slow; follow this order: defaults first, then higher `--timeout` if timed out, then broader `--product-fields` only when product results are zero.
- Keep commands simple: use `--year`, optional `--nature`, optional `--timeout`, and optional `--output json`.
- Matching behavior is fixed to exact-match defaults for speed and consistency.
- Product mode defaults to exact search on field 1; use `--product-fields "1,2"` if field 1 returns no rows.
- Queries fetch a single page of results and skip server-side count aggregation for faster response.
- Timeout retries are disabled; on timeout, rerun with higher `--timeout`.
- Year-to-distribution mapping is cached automatically in `references/year-distribution-map.json`.
