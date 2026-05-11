from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from geocoding import geocode_address, geocode_google_address, load_cache, normalize_address, save_cache


REQUIRED_COLUMNS = ["Complainent Address", "Class of Incident"]
LOG_PATH = Path("nominatim.log")


def ensure_required_columns(dataframe: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    return (len(missing) == 0, missing)


def class_color_palette(classes: List[str]) -> Dict[str, List[int]]:
    palette = [
        [230, 25, 75],    # Red
        [60, 180, 75],    # Green
        [255, 225, 25],   # Yellow
        [0, 130, 200],    # Blue
        [245, 130, 48],   # Orange
        [145, 30, 180],   # Purple
        [70, 240, 240],   # Cyan
        [240, 50, 230],   # Magenta
        [210, 245, 60],   # Lime
        [250, 190, 212],  # Pink
        [0, 128, 128],    # Teal
        [220, 190, 255],  # Lavender
        [170, 110, 40],   # Brown
        [255, 250, 200],  # Beige
        [128, 0, 0],      # Maroon
        [170, 255, 195],  # Mint
        [128, 128, 0],    # Olive
        [255, 215, 180],  # Apricot
        [0, 0, 128],      # Navy
        [128, 128, 128],  # Grey
    ]
    mapping: Dict[str, List[int]] = {}
    for idx, class_name in enumerate(classes):
        mapping[class_name] = palette[idx % len(palette)]
    return mapping


def geocode_dataframe(
    dataframe: pd.DataFrame,
    district: str,
    state: str,
    country: str,
    api_key: str = "",
) -> Tuple[pd.DataFrame, int, Dict[str, int]]:
    cache = load_cache()
    session = requests.Session()
    geocoded_rows = 0
    diagnostics = {
        "total_unique": 0,
        "resolved_pass_1": 0,
        "resolved_fallback": 0,
        "unresolved": 0,
        "cache_hits": 0,
    }
    locations = dataframe["Complainent Address"].fillna("").astype(str).tolist()
    unique_addresses = sorted(set(" ".join(addr.split()) for addr in locations if addr.strip()))
    diagnostics["total_unique"] = len(unique_addresses)

    for address in unique_addresses:
        base_norm = normalize_address(address, district=district, state=state, country=country)
        cache_key = f"{base_norm}_google" if api_key else base_norm
        
        if cache_key in cache:
            diagnostics["cache_hits"] += 1
            with LOG_PATH.open("a", encoding="utf-8") as log_file:
                log_file.write(f"CACHE_HIT | {cache_key}\n")
            continue
            
        if api_key:
            result = geocode_google_address(
                address,
                api_key=api_key,
                district=district,
                state=state,
                country=country,
                session=session,
            )
        else:
            result = geocode_address(
                address,
                district=district,
                state=state,
                country=country,
                session=session,
            )
            
        if result is None or result.get("status") != "success":
            cleaned_address = result.get("cleaned_address", address) if result else address
            cache[cache_key] = {
                "lat": None,
                "lon": None,
                "status": "no_result",
                "matched_query": "",
                "matched_pass": "",
                "cleaned_address": cleaned_address,
            }
            diagnostics["unresolved"] += 1
            with LOG_PATH.open("a", encoding="utf-8") as log_file:
                log_file.write(f"CACHE_STORE_NULL | {cache_key}\n")
            continue
            
        lat = result["lat"]
        lon = result["lon"]
        normalized = result["normalized"]
        save_key = f"{normalized}_google" if api_key else normalized
        
        cache[save_key] = {
            "lat": lat,
            "lon": lon,
            "status": result.get("status", "success"),
            "matched_query": result.get("matched_query", ""),
            "matched_pass": result.get("matched_pass", ""),
            "cleaned_address": result.get("cleaned_address", address),
        }
        if result.get("matched_pass") in ("pass_1", "google_pass_1"):
            diagnostics["resolved_pass_1"] += 1
        else:
            diagnostics["resolved_fallback"] += 1
        geocoded_rows += 1
        with LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"CACHE_STORE_SUCCESS | {save_key} | {lat},{lon} | {result.get('matched_pass','')}\n"
            )

    save_cache(cache)

    latitudes: List[float] = []
    longitudes: List[float] = []
    for address in locations:
        cleaned = " ".join(address.split())
        if not cleaned:
            latitudes.append(None)
            longitudes.append(None)
            continue
        base_norm = normalize_address(cleaned, district=district, state=state, country=country)
        cache_key = f"{base_norm}_google" if api_key else base_norm
        entry = cache.get(cache_key)
        if not entry:
            latitudes.append(None)
            longitudes.append(None)
            continue
        latitudes.append(entry.get("lat"))
        longitudes.append(entry.get("lon"))

    output = dataframe.copy()
    output["latitude"] = latitudes
    output["longitude"] = longitudes
    output["cleaned_query"] = output["Complainent Address"].fillna("").astype(str).map(
        lambda value: normalize_address(value, district=district, state=state, country=country)
    )
    return output, geocoded_rows, diagnostics


def build_map(dataframe: pd.DataFrame, show_labels: bool, map_style: str) -> Optional[Tuple[pdk.Deck, List[str], Dict[str, List[int]]]]:
    valid_points = dataframe.dropna(subset=["latitude", "longitude"]).copy()
    if valid_points.empty:
        return None

    classes = sorted(valid_points["Class of Incident"].dropna().astype(str).unique().tolist())
    color_map = class_color_palette(classes)
    valid_points["color"] = valid_points["Class of Incident"].astype(str).map(color_map)

    # Aggregate counts by latitude and longitude to show on the map
    coords_count = valid_points.groupby(["latitude", "longitude"]).size().reset_index(name="case_count")
    coords_count["case_count_str"] = coords_count["case_count"].astype(str)
    
    # Merge count back into valid_points so scatter radius and tooltip can use it
    valid_points = valid_points.merge(coords_count, on=["latitude", "longitude"])

    # Locked to Karnal district and 2D view
    center_lat = 29.6857
    center_lon = 76.9905
    zoom = 10.5

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=valid_points,
        get_position="[longitude, latitude]",
        get_weight=1,
        aggregation="SUM",
        radiusPixels=20,
    )
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=valid_points,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=150,
        pickable=True,
        opacity=0.7,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )
    text_layer = pdk.Layer(
        "TextLayer",
        data=valid_points,
        get_position="[longitude, latitude]",
        get_text="Class of Incident",
        get_size=12,
        get_color=[20, 20, 20, 220],
        get_angle=0,
        get_text_anchor="'start'",
        get_alignment_baseline="'center'",
        get_pixel_offset=[8, 0],
        pickable=False,
    )

    count_layer = pdk.Layer(
        "TextLayer",
        data=coords_count,
        get_position="[longitude, latitude]",
        get_text="case_count_str",
        get_size=18,
        get_color=[255, 255, 255, 255],
        get_alignment_baseline="'center'",
        get_text_anchor="'middle'",
        font_weight="bold",
        pickable=False,
    )

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0, bearing=0)
    layers = [heatmap_layer, scatter_layer, count_layer]
    if show_labels:
        layers.append(text_layer)
    pydeck_style = "dark"
    if map_style == "Light":
        pydeck_style = "light"
    elif map_style == "Satellite":
        pydeck_style = "satellite"
    elif map_style == "Road":
        pydeck_style = "road"

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "Class: {Class of Incident}\nLocation: {Complainent Address}"},
        map_style=pydeck_style,
    )
    return deck, classes, color_map


def main() -> None:
    st.set_page_config(page_title="Incident Heatmap", layout="wide")
    st.title("Incident Heatmap")
    st.caption("Upload complaint data, set location scope, geocode with Nominatim, and inspect incident density.")

    with st.sidebar:
        st.header("1) Upload Data")
        uploaded_file = st.file_uploader("Upload complaint Excel file", type=["xlsx"])
        
        st.header("2) Geocoding Scope")
        district = st.text_input("District", value="Karnal")
        state = st.text_input("State", value="Haryana")
        country = st.text_input("Country", value="India")
        st.info(f"Scope: `{district.strip() or '-'}, {state.strip() or '-'}, {country.strip() or '-'}`")

        # Get Google API Key directly from secrets (no UI)
        google_api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")

        st.header("3) Actions")
        geocode_now = st.button("Geocode missing addresses", type="primary", use_container_width=True)
        clear_geocode = st.button("Reset session result", use_container_width=True)
        
        st.header("Map Controls")
        map_style = st.selectbox("Base Map Style", ["Dark", "Light", "Satellite", "Road"], index=0)

    if uploaded_file is None:
        st.info("Upload an `.xlsx` file to begin.")
        return

    try:
        raw_df = pd.read_excel(uploaded_file)
    except Exception as exc:  # broad exception to show user-friendly error
        st.error(f"Could not read Excel file: {exc}")
        return

    has_columns, missing_columns = ensure_required_columns(raw_df)
    if not has_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return

    if "geocoded_df" not in st.session_state:
        st.session_state["geocoded_df"] = None
    if "geocode_diagnostics" not in st.session_state:
        st.session_state["geocode_diagnostics"] = {}
    if "geocode_scope_key" not in st.session_state:
        st.session_state["geocode_scope_key"] = None

    scope_key = f"{district.strip()}|{state.strip()}|{country.strip()}"
    if st.session_state["geocode_scope_key"] != scope_key:
        st.session_state["geocoded_df"] = None
        st.session_state["geocode_scope_key"] = scope_key

    if clear_geocode:
        st.session_state["geocoded_df"] = None
        st.sidebar.success("Session geocoded result cleared.")

    if geocode_now or st.session_state["geocoded_df"] is None:
        with st.spinner("Geocoding addresses (with cache)..."):
            geocoded_df, new_geocodes, diagnostics = geocode_dataframe(
                raw_df,
                district=district,
                state=state,
                country=country,
                api_key=google_api_key,
            )
        st.session_state["geocoded_df"] = geocoded_df
        st.session_state["geocode_diagnostics"] = diagnostics
        st.success(f"Geocoding finished. New cache hits added: {new_geocodes}")

    result_df = st.session_state["geocoded_df"].copy()

    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        available_classes = sorted(result_df["Class of Incident"].dropna().astype(str).unique().tolist())
        selected_classes = st.multiselect(
            "Filter by Class of Incident",
            available_classes,
            default=available_classes,
        )
    with filter_col2:
        st.write("") # padding
        st.write("")
        show_labels = st.toggle("Show class labels on map", value=True)

    if selected_classes:
        result_df = result_df[result_df["Class of Incident"].astype(str).isin(selected_classes)]
    else:
        result_df = result_df.iloc[0:0]

    total_count = len(result_df)
    geocoded_count = int(result_df["latitude"].notna().sum()) if "latitude" in result_df.columns else 0
    failed_count = total_count - geocoded_count

    st.markdown("---")
    metrics = st.columns(3)
    metrics[0].metric("Total Records", total_count)
    metrics[1].metric("Geocoded Records", geocoded_count)
    metrics[2].metric("Unresolved Records", failed_count)
    st.markdown("---")

    map_col, legend_col = st.columns([4, 1])
    map_data = build_map(result_df, show_labels=show_labels, map_style=map_style)
    
    if map_data:
        deck, classes, color_map = map_data
        with map_col:
            st.pydeck_chart(deck, use_container_width=True)
        with legend_col:
            st.markdown("#### Legend")
            for c in classes:
                r, g, b = color_map[c]
                swatch = f"<div style='width: 16px; height: 16px; background-color: rgb({r},{g},{b}); border-radius: 4px; margin-right: 8px; flex-shrink: 0;'></div>"
                st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 8px;'>{swatch}<span style='font-size: 14px; line-height: 1.2;'>{c}</span></div>", unsafe_allow_html=True)
    else:
        st.warning("No valid coordinates found to render map.")


    unresolved = result_df[result_df["latitude"].isna() | result_df["longitude"].isna()]
    diagnostics = st.session_state.get("geocode_diagnostics", {})
    if diagnostics:
        st.markdown("### Geocode quality summary")
        diag_cols = st.columns(4)
        diag_cols[0].metric("Unique addresses", diagnostics.get("total_unique", 0))
        diag_cols[1].metric("Resolved pass 1", diagnostics.get("resolved_pass_1", 0))
        diag_cols[2].metric("Resolved fallback", diagnostics.get("resolved_fallback", 0))
        diag_cols[3].metric("Unresolved", diagnostics.get("unresolved", 0))
    with st.expander("Show unresolved addresses"):
        st.dataframe(
            unresolved[["Complainent Address", "Class of Incident", "cleaned_query"]].drop_duplicates(),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("### Filtered Data View")
    st.dataframe(result_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
