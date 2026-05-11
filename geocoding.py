import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Optional

import requests


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CACHE_PATH = Path("geocode_cache.json")
USER_AGENT = "streamlit-incident-heatmap/1.0 (contact: local-app)"
MIN_REQUEST_INTERVAL_SECONDS = 1.1
DEFAULT_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
LOG_PATH = Path("nominatim.log")
LOCATION_PREFIXES = ("गांव", "गाव", "वासी", "निवासी", "vill", "village", "ward", "वार्ड")
TOKEN_ALIASES = {
    "तराव़डी": "तरावडी",
    "तरावड़ी": "तरावडी",
    "तरोड़ी": "तरावडी",
    "तरावडी": "तरावडी",
    "taroari": "taraori",
    "tarawadi": "taraori",
    "taraori": "taraori",
    "nilokhri": "nilokheri",
    "nilokhedi": "nilokheri",
}


logger = logging.getLogger("nominatim_geocoder")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False


def normalize_address(
    address: str,
    district: str = "Karnal",
    state: str = "Haryana",
    country: str = "India",
) -> str:
    cleaned = clean_location_text(address)
    if not cleaned:
        return ""
    scope_parts = [
        " ".join(str(district).strip().split()),
        " ".join(str(state).strip().split()),
        " ".join(str(country).strip().split()),
    ]
    scope_parts = [part for part in scope_parts if part]
    if not scope_parts:
        return cleaned
    return f"{cleaned}, {', '.join(scope_parts)}"


def clean_location_text(address: str) -> str:
    cleaned = " ".join(str(address).strip().split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"[|;]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    tokens = cleaned.split(" ")
    if tokens and tokens[0].lower() in LOCATION_PREFIXES:
        tokens = tokens[1:]
    normalized_tokens = []
    for token in tokens:
        key = token.lower()
        normalized_tokens.append(TOKEN_ALIASES.get(key, token))
    return " ".join(normalized_tokens).strip()


def load_cache(cache_path: Path = CACHE_PATH) -> Dict[str, Dict[str, object]]:
    if not cache_path.exists():
        # If the cache file does not exist, create an empty JSON file so later code can safely read/write it
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({}), encoding="utf-8")
        return {}
        
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        return {}
    return {}


def save_cache(cache: Dict[str, Dict[str, object]], cache_path: Path = CACHE_PATH) -> None:
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)


def geocode_address(
    address: str,
    district: str = "Karnal",
    state: str = "Haryana",
    country: str = "India",
    session: Optional[requests.Session] = None,
    pause_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
) -> Optional[Dict[str, object]]:
    cleaned = clean_location_text(address)
    if not cleaned:
        logger.info("SKIP_EMPTY_ADDRESS")
        return None

    http = session or requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "hi,en"}
    pass_queries = [
        ("pass_1", normalize_address(cleaned, district=district, state=state, country=country)),
        ("pass_2", normalize_address(cleaned, district=district, state="", country=country)),
        ("pass_3", normalize_address(cleaned, district="", state=state, country=country)),
        ("pass_4", normalize_address(cleaned, district="", state="", country=country)),
    ]

    tried_queries = []
    for pass_name, normalized in pass_queries:
        if not normalized or normalized in tried_queries:
            continue
        tried_queries.append(normalized)
        params = {
            "q": normalized,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "in",
            "addressdetails": 0,
        }
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info("REQUEST %s attempt=%s query=%s", pass_name, attempt, normalized)
                response = http.get(
                    NOMINATIM_URL,
                    params=params,
                    headers=headers,
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                )
                if response.status_code == 429 and attempt < MAX_RETRIES:
                    logger.warning("RATE_LIMIT_429 %s attempt=%s query=%s", pass_name, attempt, normalized)
                    time.sleep(pause_seconds * attempt)
                    continue
                response.raise_for_status()
                payload = response.json()
                if payload:
                    item = payload[0]
                    lat = float(item["lat"])
                    lon = float(item["lon"])
                    logger.info("SUCCESS %s query=%s lat=%s lon=%s", pass_name, normalized, lat, lon)
                    return {
                        "lat": lat,
                        "lon": lon,
                        "normalized": normalize_address(cleaned, district=district, state=state, country=country),
                        "status": "success",
                        "matched_query": normalized,
                        "matched_pass": pass_name,
                        "cleaned_address": cleaned,
                    }
                logger.info("NO_RESULT %s query=%s", pass_name, normalized)
                break
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                logger.error("ERROR %s attempt=%s query=%s error=%s", pass_name, attempt, normalized, exc)
                if attempt == MAX_RETRIES:
                    logger.error("FAILED %s query=%s", pass_name, normalized)
                    break
                time.sleep(pause_seconds * attempt)
            finally:
                time.sleep(pause_seconds)
    return {
        "lat": None,
        "lon": None,
        "normalized": normalize_address(cleaned, district=district, state=state, country=country),
        "status": "no_result",
        "matched_query": "",
        "matched_pass": "",
        "cleaned_address": cleaned,
    }


def geocode_google_address(
    address: str,
    api_key: str,
    district: str = "Karnal",
    state: str = "Haryana",
    country: str = "India",
    session: Optional[requests.Session] = None,
) -> Optional[Dict[str, object]]:
    cleaned = clean_location_text(address)
    if not cleaned:
        logger.info("SKIP_EMPTY_ADDRESS_GOOGLE")
        return None

    http = session or requests.Session()
    normalized = normalize_address(cleaned, district=district, state=state, country=country)
    
    params = {
        "address": normalized,
        "key": api_key
    }
    
    try:
        logger.info("REQUEST_GOOGLE query=%s", normalized)
        response = http.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params=params,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        
        if payload.get("status") == "OK" and payload.get("results"):
            item = payload["results"][0]
            lat = item["geometry"]["location"]["lat"]
            lon = item["geometry"]["location"]["lng"]
            logger.info("SUCCESS_GOOGLE query=%s lat=%s lon=%s", normalized, lat, lon)
            return {
                "lat": lat,
                "lon": lon,
                "normalized": normalized,
                "status": "success",
                "matched_query": normalized,
                "matched_pass": "google_pass_1",
                "cleaned_address": cleaned,
            }
        else:
            logger.info("NO_RESULT_GOOGLE query=%s status=%s", normalized, payload.get("status"))
    except Exception as exc:
        logger.error("ERROR_GOOGLE query=%s error=%s", normalized, exc)
        
    return {
        "lat": None,
        "lon": None,
        "normalized": normalized,
        "status": "no_result",
        "matched_query": "",
        "matched_pass": "",
        "cleaned_address": cleaned,
    }

