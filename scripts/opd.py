#!/usr/bin/env python3
"""
Query CMS Open Payments DKAN datastore for operator-facing workflows.

Supported modes:
1) provider
2) company
3) product
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://openpaymentsdata.cms.gov"
DEFAULT_TITLE_CONTAINS = "General Payment Data"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
DEFAULT_DISTRIBUTION_MAP_PATH = str(Path(__file__).resolve().parents[1] / "references" / "year-distribution-map.json")
DEFAULT_OPERATOR = "="
DEFAULT_NATURE_OPERATOR = "="
DEFAULT_COMPANY_OPERATOR = "="
DEFAULT_PRODUCT_FIELDS = "1"
DEFAULT_LIMIT_PER_PAGE = 100
DEFAULT_MAX_PAGES = 1
DEFAULT_MAX_OUTPUT_ROWS = 300
DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 2
DEFAULT_INCLUDE_COUNT = False

PHYSICIAN_FIRST_FIELD = "covered_recipient_first_name"
PHYSICIAN_LAST_FIELD = "covered_recipient_last_name"
PAYMENT_AMOUNT_FIELD = "total_amount_of_payment_usdollars"
PAYMENT_DATE_FIELD = "date_of_payment"
PAYMENT_NATURE_FIELD = "nature_of_payment_or_transfer_of_value"
PAYMENT_FORM_FIELD = "form_of_payment_or_transfer_of_value"
PAYMENT_RECORD_ID_FIELD = "record_id"
PHYSICIAN_PROFILE_ID_FIELD = "covered_recipient_profile_id"
COMPANY_PROFILE_ID_FIELD = "applicable_manufacturer_or_applicable_gpo_making_payment_id"
PAYMENT_ROUTE_TYPE_GENERAL = "General"

COMPANY_FIELDS = [
    "applicable_manufacturer_or_applicable_gpo_making_payment_name",
    "submitting_applicable_manufacturer_or_applicable_gpo_name",
]

PRODUCT_FIELDS = [
    f"name_of_drug_or_biological_or_device_or_medical_supply_{i}" for i in range(1, 6)
]

DEFAULT_PROPERTIES = [
    PAYMENT_RECORD_ID_FIELD,
    PHYSICIAN_PROFILE_ID_FIELD,
    COMPANY_PROFILE_ID_FIELD,
    PHYSICIAN_FIRST_FIELD,
    PHYSICIAN_LAST_FIELD,
    PAYMENT_AMOUNT_FIELD,
    PAYMENT_DATE_FIELD,
    PAYMENT_NATURE_FIELD,
    PAYMENT_FORM_FIELD,
    *COMPANY_FIELDS,
    *PRODUCT_FIELDS,
]

PROVIDER_MODE_PROPERTIES = DEFAULT_PROPERTIES
COMPANY_MODE_PROPERTIES = [
    PAYMENT_RECORD_ID_FIELD,
    PHYSICIAN_PROFILE_ID_FIELD,
    COMPANY_PROFILE_ID_FIELD,
    PHYSICIAN_FIRST_FIELD,
    PHYSICIAN_LAST_FIELD,
    PAYMENT_AMOUNT_FIELD,
    PAYMENT_NATURE_FIELD,
    *COMPANY_FIELDS,
]
PRODUCT_MODE_PROPERTIES = [
    PAYMENT_RECORD_ID_FIELD,
    PHYSICIAN_PROFILE_ID_FIELD,
    COMPANY_PROFILE_ID_FIELD,
    PHYSICIAN_FIRST_FIELD,
    PHYSICIAN_LAST_FIELD,
    PAYMENT_AMOUNT_FIELD,
    PAYMENT_NATURE_FIELD,
    *COMPANY_FIELDS,
]


def add_query_params(url: str, params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return url
    return f"{url}?{urlencode(params)}"


def timeout_rerun_hint(timeout: int) -> str:
    return (
        f"Request timed out after {timeout}s. "
        f"Rerun with a higher timeout, for example: --timeout {max(timeout * 2, timeout + 60)}"
    )


def is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return True
        return "timed out" in str(reason).lower()
    return "timed out" in str(exc).lower()


def http_json(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 180,
    retries: int = 2,
) -> Any:
    target_url = add_query_params(url, params)
    request_body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if request_body is not None:
        headers["Content-Type"] = "application/json"

    attempts = max(0, retries) + 1
    for attempt in range(attempts):
        req = Request(target_url, data=request_body, headers=headers, method=method.upper())
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if exc.code == 504:
                raise RuntimeError(f"HTTP 504 for {target_url}: {timeout_rerun_hint(timeout)}") from exc
            if exc.code in RETRYABLE_STATUS and attempt < attempts - 1:
                time.sleep(0.6 * (2**attempt))
                continue
            raise RuntimeError(f"HTTP {exc.code} for {target_url}: {detail[:500]}") from exc
        except (URLError, socket.timeout, TimeoutError) as exc:
            if is_timeout_error(exc):
                raise RuntimeError(f"Network timeout for {target_url}: {timeout_rerun_hint(timeout)}") from exc
            if attempt < attempts - 1:
                time.sleep(0.6 * (2**attempt))
                continue
            raise RuntimeError(f"Network error for {target_url}: {exc}") from exc

    raise RuntimeError("Unexpected request failure")


def extract_year(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"\b(20\d{2})\b", text)
    return int(match.group(1)) if match else None


def pick_dataset(
    datasets: List[Dict[str, Any]],
    *,
    year: Optional[int],
    title_contains: Optional[str],
    title_exact: Optional[str],
) -> Optional[Dict[str, Any]]:
    candidates = datasets

    if title_exact:
        match_title = title_exact.strip().lower()
        candidates = [d for d in candidates if d.get("title", "").strip().lower() == match_title]
    elif title_contains:
        match_text = title_contains.strip().lower()
        candidates = [d for d in candidates if match_text in d.get("title", "").lower()]

    themed = [d for d in candidates if "General Payments" in (d.get("theme") or [])]
    if themed:
        candidates = themed

    # Exact title selection is treated as fully explicit and skips year filtering.
    if year is not None and not title_exact:
        year_text = str(year)
        candidates = [
            d
            for d in candidates
            if year_text in d.get("title", "") or d.get("temporal", "").startswith(f"{year_text}-")
        ]

    def sort_key(dataset: Dict[str, Any]) -> Tuple[int, str, str]:
        issued = dataset.get("issued") or ""
        modified = dataset.get("modified") or ""
        title_year = extract_year(dataset.get("title", "")) or 0
        temporal_year = extract_year(dataset.get("temporal", "")) or 0
        return (max(title_year, temporal_year), issued, modified)

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0] if candidates else None


def pick_distribution_id(dataset: Dict[str, Any]) -> Optional[str]:
    refs = dataset.get("%Ref:distribution")
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, dict) and ref.get("identifier"):
                return str(ref["identifier"])

    distributions = dataset.get("distribution") or []
    if isinstance(distributions, list):
        for dist in distributions:
            if isinstance(dist, dict) and dist.get("identifier"):
                return str(dist["identifier"])
    return None


def load_year_distribution_map(path_text: Optional[str]) -> Dict[int, Dict[str, Any]]:
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Unable to read distribution map file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in distribution map file {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError("Distribution map JSON must be an object mapping year to distribution information.")

    parsed: Dict[int, Dict[str, Any]] = {}
    for year_key, value in raw.items():
        try:
            year_int = int(str(year_key).strip())
        except ValueError as exc:
            raise RuntimeError(f"Invalid year key in distribution map: {year_key}") from exc

        if isinstance(value, str):
            distribution_id = value.strip()
            dataset_title = None
            dataset_id = None
        elif isinstance(value, dict):
            distribution_id = str(value.get("distribution_id", "")).strip()
            dataset_title = value.get("dataset_title")
            dataset_id = value.get("dataset_id")
        else:
            raise RuntimeError(
                f"Invalid value for year {year_int} in distribution map. "
                "Use a distribution-id string or an object with distribution_id."
            )

        if not distribution_id:
            raise RuntimeError(f"Distribution map entry for year {year_int} is missing distribution_id.")

        parsed[year_int] = {
            "distribution_id": distribution_id,
            "dataset_title": dataset_title,
            "dataset_id": dataset_id,
        }

    return parsed


def save_year_distribution_map(path_text: str, mapping: Dict[int, Dict[str, Any]]) -> None:
    path = Path(path_text).expanduser()
    serialized = {str(year): value for year, value in sorted(mapping.items())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialized, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def condition(field: str, value: str, operator: str) -> Dict[str, Any]:
    return {"property": field, "value": value, "operator": operator}


def unique_preserve(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def fetch_records(
    *,
    distribution_id: str,
    conditions: List[Dict[str, Any]],
    properties: Sequence[str],
    limit_per_page: int,
    max_pages: int,
    timeout: int,
    retries: int,
    include_count: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], Optional[int], bool]:
    query_url = f"{BASE_URL}/api/1/datastore/query/{distribution_id}"
    all_rows: List[Dict[str, Any]] = []
    total_count: Optional[int] = None
    pagination_capped = False
    page_limit = max(1, limit_per_page)
    page_cap = max(1, max_pages)
    properties_list = list(properties)

    for page_idx in range(page_cap):
        offset = page_idx * page_limit
        payload: Dict[str, Any] = {
            "conditions": conditions,
            "properties": properties_list,
            "limit": page_limit,
            "offset": offset,
            "keys": True,
            "schema": False,
            "format": "json",
        }
        if include_count is not None:
            payload["count"] = bool(include_count)
        response = http_json("POST", query_url, payload=payload, timeout=timeout, retries=retries)
        rows = response.get("results", [])
        if not isinstance(rows, list):
            raise RuntimeError("Datastore response missing list-valued 'results'")
        total_count = response.get("count")
        all_rows.extend(rows)
        if len(rows) < page_limit:
            break
        if page_idx == page_cap - 1:
            pagination_capped = True

    return all_rows, total_count, pagination_capped


def fetch_records_with_company_field_fallback(
    *,
    distribution_id: str,
    conditions: List[Dict[str, Any]],
    properties: Sequence[str],
    company_filter_value: str,
    company_operator: str,
    explicit_company_field: Optional[str],
    limit_per_page: int,
    max_pages: int,
    timeout: int,
    retries: int,
    include_count: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], Optional[int], bool, str]:
    candidates = [explicit_company_field] if explicit_company_field else COMPANY_FIELDS
    candidates = [field for field in candidates if field]
    if not candidates:
        raise RuntimeError("No company-field candidates were available")

    errors: List[str] = []
    first_success: Optional[Tuple[List[Dict[str, Any]], Optional[int], bool, str]] = None

    if len(candidates) > 1:
        grouped_conditions = conditions + [
            {"groupOperator": "or", "conditions": [condition(field, company_filter_value, company_operator) for field in candidates]}
        ]
        try:
            rows, count, pagination_capped = fetch_records(
                distribution_id=distribution_id,
                conditions=grouped_conditions,
                properties=properties,
                limit_per_page=limit_per_page,
                max_pages=max_pages,
                timeout=timeout,
                retries=retries,
                include_count=include_count,
            )
            return rows, count, pagination_capped, f"or_group({'|'.join(candidates)})"
        except RuntimeError as exc:
            errors.append(f"or_group: {exc}")

    for field in candidates:
        try:
            rows, count, pagination_capped = fetch_records(
                distribution_id=distribution_id,
                conditions=conditions + [condition(field, company_filter_value, company_operator)],
                properties=properties,
                limit_per_page=limit_per_page,
                max_pages=max_pages,
                timeout=timeout,
                retries=retries,
                include_count=include_count,
            )
            if first_success is None:
                first_success = (rows, count, pagination_capped, field)
            if rows:
                return rows, count, pagination_capped, field
        except RuntimeError as exc:
            errors.append(f"{field}: {exc}")

    if first_success is not None:
        return first_success

    joined = " | ".join(errors) if errors else "no company-field candidates were available"
    raise RuntimeError(f"Company filter failed for all field candidates: {joined}")


def parse_amount(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def normalize_count(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized >= 0 else None


def first_non_empty(row: Dict[str, Any], fields: Sequence[str]) -> Optional[str]:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def extract_products(row: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for field in PRODUCT_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            names.append(text)
    return unique_preserve(names)


def payment_detail_url(record_id: Optional[str], year: Optional[int], payment_type: str = PAYMENT_ROUTE_TYPE_GENERAL) -> Optional[str]:
    text = (record_id or "").strip()
    if not text or year is None:
        return None
    return f"{BASE_URL}/payment/{int(year)}/{payment_type}/{text}"


def physician_profile_url(profile_id: Optional[str]) -> Optional[str]:
    text = (profile_id or "").strip()
    if not text:
        return None
    return f"{BASE_URL}/physician/{text}"


def company_profile_url(company_id: Optional[str]) -> Optional[str]:
    text = (company_id or "").strip()
    if not text:
        return None
    return f"{BASE_URL}/company/{text}"


def normalize_payment_row(row: Dict[str, Any], *, year: Optional[int] = None) -> Dict[str, Any]:
    first = first_non_empty(row, [PHYSICIAN_FIRST_FIELD]) or ""
    last = first_non_empty(row, [PHYSICIAN_LAST_FIELD]) or ""
    provider_name = " ".join(p for p in [first, last] if p).strip()
    record_id = first_non_empty(row, [PAYMENT_RECORD_ID_FIELD])
    physician_profile_id = first_non_empty(row, [PHYSICIAN_PROFILE_ID_FIELD])
    company_id = first_non_empty(row, [COMPANY_PROFILE_ID_FIELD])

    return {
        "record_id": record_id,
        "payment_url": payment_detail_url(record_id, year),
        "physician_profile_id": physician_profile_id,
        "physician_url": physician_profile_url(physician_profile_id),
        "company_id": company_id,
        "company_url": company_profile_url(company_id),
        "provider_first_name": first,
        "provider_last_name": last,
        "provider_name": provider_name or None,
        "company_name": first_non_empty(row, COMPANY_FIELDS),
        "amount_usd": round(parse_amount(row.get(PAYMENT_AMOUNT_FIELD)), 2),
        "date_of_payment": row.get(PAYMENT_DATE_FIELD),
        "payment_nature": row.get(PAYMENT_NATURE_FIELD),
        "payment_form": row.get(PAYMENT_FORM_FIELD),
        "products": extract_products(row),
    }


def dedupe_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        marker = json.dumps(row, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(row)
    return out


def summarize_providers(rows: Sequence[Dict[str, Any]], *, year: Optional[int] = None) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        payment = normalize_payment_row(row, year=year)
        first = payment["provider_first_name"] or ""
        last = payment["provider_last_name"] or ""
        key = f"{first.lower()}::{last.lower()}"

        if key not in grouped:
            grouped[key] = {
                "provider_first_name": first,
                "provider_last_name": last,
                "provider_name": payment["provider_name"],
                "payment_count": 0,
                "total_amount_usd": 0.0,
                "payment_urls": set(),
                "physician_profile_ids": set(),
                "physician_urls": set(),
                "company_names": set(),
                "company_ids": set(),
                "company_urls": set(),
                "payment_natures": set(),
            }

        target = grouped[key]
        target["payment_count"] += 1
        target["total_amount_usd"] += float(payment["amount_usd"] or 0.0)
        if payment["payment_url"]:
            target["payment_urls"].add(payment["payment_url"])
        if payment["physician_profile_id"]:
            target["physician_profile_ids"].add(payment["physician_profile_id"])
        if payment["physician_url"]:
            target["physician_urls"].add(payment["physician_url"])
        if payment["company_name"]:
            target["company_names"].add(payment["company_name"])
        if payment["company_id"]:
            target["company_ids"].add(payment["company_id"])
        if payment["company_url"]:
            target["company_urls"].add(payment["company_url"])
        if payment["payment_nature"]:
            target["payment_natures"].add(str(payment["payment_nature"]))

    result: List[Dict[str, Any]] = []
    for value in grouped.values():
        result.append(
            {
                "provider_first_name": value["provider_first_name"],
                "provider_last_name": value["provider_last_name"],
                "provider_name": value["provider_name"],
                "payment_urls": sorted(value["payment_urls"]),
                "physician_profile_ids": sorted(value["physician_profile_ids"]),
                "physician_urls": sorted(value["physician_urls"]),
                "payment_count": value["payment_count"],
                "total_amount_usd": round(value["total_amount_usd"], 2),
                "company_names": sorted(value["company_names"]),
                "company_ids": sorted(value["company_ids"]),
                "company_urls": sorted(value["company_urls"]),
                "payment_natures": sorted(value["payment_natures"]),
            }
        )

    result.sort(
        key=lambda item: (float(item["total_amount_usd"]), int(item["payment_count"])),
        reverse=True,
    )
    return result


def apply_truncation(rows: List[Dict[str, Any]], max_rows: int) -> Tuple[List[Dict[str, Any]], bool]:
    if max_rows < 1 or len(rows) <= max_rows:
        return rows, False
    return rows[:max_rows], True


def parse_product_fields(csv_text: str) -> List[str]:
    indexes = []
    for part in csv_text.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            idx = int(text)
        except ValueError as exc:
            raise ValueError(f"Invalid product field index: {text}") from exc
        if idx < 1 or idx > 5:
            raise ValueError("Product field indexes must be between 1 and 5")
        indexes.append(idx)
    if not indexes:
        raise ValueError("At least one product field index is required")
    return [f"name_of_drug_or_biological_or_device_or_medical_supply_{idx}" for idx in unique_preserve(indexes)]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--year", type=int, required=True, help="Program year to target (required, example: 2024).")
    parser.add_argument(
        "--nature",
        dest="payment_type",
        default=None,
        help="Filter Nature of Payment or Other Transfer of Value (example: Food and Beverage).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument("--output", choices=["pretty", "json"], default="pretty", help="Output formatting.")


def load_distribution_and_dataset(year: int, timeout: int) -> Tuple[str, Optional[Dict[str, Any]]]:
    try:
        distribution_map = load_year_distribution_map(DEFAULT_DISTRIBUTION_MAP_PATH)
    except RuntimeError:
        distribution_map = {}

    mapped = distribution_map.get(year)
    if mapped:
        distribution_id = str(mapped["distribution_id"])
        dataset_title = mapped.get("dataset_title")
        dataset_id = mapped.get("dataset_id")
        mapped_dataset: Optional[Dict[str, Any]]
        if dataset_title is None and dataset_id is None:
            mapped_dataset = None
        else:
            mapped_dataset = {
                "title": dataset_title,
                "identifier": dataset_id,
                "_selection_source": "distribution_map",
            }
        return distribution_id, mapped_dataset

    datasets = http_json(
        "GET",
        f"{BASE_URL}/api/1/metastore/schemas/dataset/items",
        params={"show-reference-ids": "true"},
        timeout=timeout,
        retries=DEFAULT_RETRIES,
    )
    if not isinstance(datasets, list):
        raise RuntimeError("Dataset listing did not return an array")

    dataset = pick_dataset(
        datasets,
        year=year,
        title_contains=DEFAULT_TITLE_CONTAINS,
        title_exact=None,
    )
    if not dataset:
        raise RuntimeError("No matching dataset found for requested year.")

    distribution_id = pick_distribution_id(dataset)
    if not distribution_id:
        raise RuntimeError("Matching dataset has no distribution identifier.")

    distribution_map[year] = {
        "distribution_id": distribution_id,
        "dataset_title": dataset.get("title"),
        "dataset_id": dataset.get("identifier"),
    }
    try:
        save_year_distribution_map(DEFAULT_DISTRIBUTION_MAP_PATH, distribution_map)
    except OSError:
        pass

    return distribution_id, dataset


def dataset_meta(dataset: Optional[Dict[str, Any]], distribution_id: str) -> Dict[str, Any]:
    if dataset is None:
        return {"distribution_id": distribution_id}
    meta = {
        "distribution_id": distribution_id,
        "dataset_title": dataset.get("title"),
        "dataset_id": dataset.get("identifier"),
    }
    if dataset.get("_selection_source"):
        meta["selection_source"] = dataset.get("_selection_source")
    return meta


def execute_product_query(
    *,
    distribution_id: str,
    product_value: str,
    product_fields: Sequence[str],
    operator: str,
    payment_type: Optional[str],
    payment_type_operator: str,
    limit_per_page: int,
    max_pages: int,
    timeout: int,
    retries: int,
    include_count: bool,
) -> Tuple[List[Dict[str, Any]], Optional[int], bool, str]:
    rows: List[Dict[str, Any]] = []
    server_match_sum: Optional[int] = None
    pagination_capped = False
    query_strategy = "or_group"
    base_conditions: List[Dict[str, Any]] = []

    if payment_type:
        base_conditions.append(condition(PAYMENT_NATURE_FIELD, payment_type, payment_type_operator))

    grouped_conditions = base_conditions + [
        {"groupOperator": "or", "conditions": [condition(field, product_value, operator) for field in product_fields]}
    ]

    try:
        rows, query_count, query_pagination_capped = fetch_records(
            distribution_id=distribution_id,
            conditions=grouped_conditions,
            properties=PRODUCT_MODE_PROPERTIES,
            limit_per_page=limit_per_page,
            max_pages=max_pages,
            timeout=timeout,
            retries=retries,
            include_count=include_count,
        )
        server_match_sum = normalize_count(query_count)
        pagination_capped = query_pagination_capped
        return rows, server_match_sum, pagination_capped, query_strategy
    except RuntimeError:
        # Fallback for DKAN instances that do not support grouped OR conditions.
        query_strategy = "per_field_fallback"
        rows = []
        server_match_sum = 0
        for field in product_fields:
            field_conditions = base_conditions + [condition(field, product_value, operator)]
            page_rows, query_count, query_pagination_capped = fetch_records(
                distribution_id=distribution_id,
                conditions=field_conditions,
                properties=PRODUCT_MODE_PROPERTIES,
                limit_per_page=limit_per_page,
                max_pages=max_pages,
                timeout=timeout,
                retries=retries,
                include_count=include_count,
            )
            rows.extend(page_rows)
            pagination_capped = pagination_capped or query_pagination_capped
            normalized_count = normalize_count(query_count)
            if normalized_count is None:
                server_match_sum = None
            elif server_match_sum is not None:
                server_match_sum += normalized_count
                if normalized_count > len(page_rows):
                    pagination_capped = True

        return rows, server_match_sum, pagination_capped, query_strategy


def run_physician_payments(args: argparse.Namespace, distribution_id: str) -> Dict[str, Any]:
    if not args.first and not args.last:
        raise RuntimeError("Provide --first and/or --last for provider.")

    conditions: List[Dict[str, Any]] = []
    if args.first:
        conditions.append(condition(PHYSICIAN_FIRST_FIELD, args.first, DEFAULT_OPERATOR))
    if args.last:
        conditions.append(condition(PHYSICIAN_LAST_FIELD, args.last, DEFAULT_OPERATOR))
    if args.payment_type:
        conditions.append(condition(PAYMENT_NATURE_FIELD, args.payment_type, DEFAULT_NATURE_OPERATOR))

    rows, count, pagination_capped = fetch_records(
        distribution_id=distribution_id,
        conditions=conditions,
        properties=PROVIDER_MODE_PROPERTIES,
        limit_per_page=DEFAULT_LIMIT_PER_PAGE,
        max_pages=DEFAULT_MAX_PAGES,
        timeout=args.timeout,
        retries=DEFAULT_RETRIES,
        include_count=DEFAULT_INCLUDE_COUNT,
    )

    all_payments = [normalize_payment_row(row, year=args.year) for row in rows]
    all_payments.sort(
        key=lambda item: (float(item["amount_usd"]), str(item["date_of_payment"] or "")),
        reverse=True,
    )
    payments, truncated = apply_truncation(all_payments, DEFAULT_MAX_OUTPUT_ROWS)

    total_amount = round(sum(float(item["amount_usd"]) for item in all_payments), 2)
    distinct_companies = sorted({item["company_name"] for item in all_payments if item["company_name"]})
    payment_natures = sorted({str(item["payment_nature"]) for item in all_payments if item["payment_nature"]})
    server_matched_rows = normalize_count(count)
    if server_matched_rows is None:
        server_matched_rows = len(all_payments)
    else:
        pagination_capped = server_matched_rows > len(all_payments)

    return {
        "mode": getattr(args, "mode_name", "provider"),
        "filters": {
            "first": args.first,
            "last": args.last,
            "payment_type": args.payment_type,
            "year": args.year,
        },
        "totals": {
            "matched_rows": server_matched_rows,
            "fetched_rows": len(all_payments),
            "returned_rows": len(payments),
            "total_amount_usd": total_amount,
            "distinct_companies": len(distinct_companies),
            "distinct_payment_natures": len(payment_natures),
            "pagination_capped": pagination_capped,
            "truncated": truncated,
        },
        "payments": payments,
    }


def run_physicians_by_company(args: argparse.Namespace, distribution_id: str) -> Dict[str, Any]:
    company_value = args.company_name

    conditions: List[Dict[str, Any]] = []
    if args.payment_type:
        conditions.append(condition(PAYMENT_NATURE_FIELD, args.payment_type, DEFAULT_NATURE_OPERATOR))

    rows, count, pagination_capped, _ = fetch_records_with_company_field_fallback(
        distribution_id=distribution_id,
        conditions=conditions,
        properties=COMPANY_MODE_PROPERTIES,
        company_filter_value=company_value,
        company_operator=DEFAULT_COMPANY_OPERATOR,
        explicit_company_field=None,
        limit_per_page=DEFAULT_LIMIT_PER_PAGE,
        max_pages=DEFAULT_MAX_PAGES,
        timeout=args.timeout,
        retries=DEFAULT_RETRIES,
        include_count=DEFAULT_INCLUDE_COUNT,
    )

    all_providers = summarize_providers(rows, year=args.year)
    providers, truncated = apply_truncation(all_providers, DEFAULT_MAX_OUTPUT_ROWS)
    total_amount = round(sum(float(item["total_amount_usd"]) for item in all_providers), 2)
    server_matched_rows = normalize_count(count)
    if server_matched_rows is None:
        server_matched_rows = len(rows)
    else:
        pagination_capped = server_matched_rows > len(rows)

    return {
        "mode": getattr(args, "mode_name", "company"),
        "filters": {
            "company": company_value,
            "payment_type": args.payment_type,
            "year": args.year,
        },
        "totals": {
            "matched_rows": server_matched_rows,
            "fetched_rows": len(rows),
            "distinct_providers": len(all_providers),
            "returned_providers": len(providers),
            "total_amount_usd": total_amount,
            "pagination_capped": pagination_capped,
            "truncated": truncated,
        },
        "providers": providers,
    }


def run_physicians_by_product(args: argparse.Namespace, distribution_id: str) -> Dict[str, Any]:
    product_value = args.product_name
    product_fields = parse_product_fields(args.product_fields)
    rows, _, pagination_capped, _ = execute_product_query(
        distribution_id=distribution_id,
        product_value=product_value,
        product_fields=product_fields,
        operator=DEFAULT_OPERATOR,
        payment_type=args.payment_type,
        payment_type_operator=DEFAULT_NATURE_OPERATOR,
        limit_per_page=DEFAULT_LIMIT_PER_PAGE,
        max_pages=DEFAULT_MAX_PAGES,
        timeout=args.timeout,
        retries=DEFAULT_RETRIES,
        include_count=DEFAULT_INCLUDE_COUNT,
    )

    deduped_rows = dedupe_rows(rows)
    all_providers = summarize_providers(deduped_rows, year=args.year)
    providers, truncated = apply_truncation(all_providers, DEFAULT_MAX_OUTPUT_ROWS)
    total_amount = round(sum(float(item["total_amount_usd"]) for item in all_providers), 2)
    matched_rows = len(deduped_rows)
    message = None
    using_default_field_1 = len(product_fields) == 1 and product_fields[0] == PRODUCT_FIELDS[0]
    if matched_rows == 0 and using_default_field_1:
        message = (
            f'No results found in product field 1 for "{product_value}". '
            f'Try again with broader fields, for example: product "{product_value}" --year {args.year} --product-fields 1,2'
        )

    payload = {
        "mode": getattr(args, "mode_name", "product"),
        "filters": {
            "product": product_value,
            "payment_type": args.payment_type,
            "year": args.year,
        },
        "totals": {
            "matched_rows": matched_rows,
            "fetched_rows": len(deduped_rows),
            "distinct_providers": len(all_providers),
            "returned_providers": len(providers),
            "total_amount_usd": total_amount,
            "pagination_capped": pagination_capped,
            "truncated": truncated,
        },
        "providers": providers,
    }
    if message:
        payload["message"] = message
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operator workflows for querying CMS Open Payments datastore."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    provider_parser = subparsers.add_parser(
        "provider",
        help="List payment records for an Individual Provider.",
    )
    provider_parser.set_defaults(run_mode=run_physician_payments, mode_name="provider")
    add_common_args(provider_parser)
    provider_parser.add_argument("--first", default=None, help="Individual Provider first name.")
    provider_parser.add_argument("--last", default=None, help="Individual Provider last name.")

    company_parser = subparsers.add_parser(
        "company",
        help="List Individual Providers that received payments from a specific company.",
    )
    company_parser.set_defaults(run_mode=run_physicians_by_company, mode_name="company")
    add_common_args(company_parser)
    company_parser.add_argument("company_name", help="Drug or medical device company name.")

    product_parser = subparsers.add_parser(
        "product",
        help="List Individual Providers tied to payment records for a product or drug.",
    )
    product_parser.set_defaults(run_mode=run_physicians_by_product, mode_name="product")
    add_common_args(product_parser)
    product_parser.add_argument("product_name", help="Product or drug name.")
    product_parser.add_argument(
        "--product-fields",
        default=DEFAULT_PRODUCT_FIELDS,
        help="Comma-separated product field indexes to search (1-5). Default: 1.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        distribution_id, dataset = load_distribution_and_dataset(args.year, args.timeout)
        run_mode = getattr(args, "run_mode", None)
        if run_mode is None:
            raise RuntimeError(f"Unsupported mode: {args.mode}")
        payload = run_mode(args, distribution_id)

        result = {"dataset": dataset_meta(dataset, distribution_id), **payload}
        if args.output == "json":
            print(json.dumps(result, separators=(",", ":"), sort_keys=True))
        else:
            print(json.dumps(result, indent=2))
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
