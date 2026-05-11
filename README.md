# Streamlit Incident Heatmap (India)

Simple one-page Streamlit app to upload complaint data, geocode short village/locality addresses using OpenStreetMap Nominatim with fallback search passes, and visualize a heatmap by location/class.

## Features

- Upload Excel (`.xlsx`) complaint file
- Upload starts in the main view (top of page)
- Select geocoding scope using:
  - `District` (default: `Karnal`)
  - `State` (default: `Haryana`)
  - `Country` (default: `India`)
- Required columns:
  - `Complainent Address`
  - `Class of Incident`
- Geocode with Nominatim and persist local cache in `geocode_cache.json`
- Multi-pass geocoding fallback to improve hit rate for short Hindi/English addresses
- Filter map by `Class of Incident`
- View heatmap + class-colored points + optional on-map class labels
- Default map view focuses on Karnal, with optional `Fit all points`
- Track totals, geocoded records, and unresolved addresses

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Input file notes

- The app reads the first sheet of uploaded Excel.
- It expects the required column names exactly as listed above.
- Rows with blank addresses are skipped for geocoding.

## Scope behavior

- Each geocode query is normalized as:
  - `<address>, <district>, <state>, <country>`
- With defaults, this becomes:
  - `<address>, Karnal, Haryana, India`
- Changing scope changes query normalization and cache keys.

## Geocoding quality behavior

- Addresses are lightly cleaned before geocoding (spacing/noise reduction and common variant normalization).
- Queries are attempted in fallback order:
  - pass 1: `<address>, <district>, <state>, <country>`
  - pass 2: `<address>, <district>, <country>`
  - pass 3: `<address>, <state>, <country>`
  - pass 4: `<address>, <country>`
- Cache stores status metadata (`success` / `no_result`) and matched pass details to reduce repeated failures.
- App shows a geocode quality summary:
  - unique addresses
  - resolved in pass 1
  - resolved via fallback
  - unresolved

## Nominatim usage guidance

- This prototype uses the public Nominatim endpoint (free).
- Keep request volume low and rely on cache reuse.
- Avoid repeated geocoding for the same addresses.
- Respect fair use and acceptable usage limits.

## Cache behavior

- Geocode cache is stored in `geocode_cache.json`.
- Cache key is the full normalized scoped query string.
- To clear cache, delete `geocode_cache.json`.
