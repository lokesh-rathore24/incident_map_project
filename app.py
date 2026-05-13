from __future__ import annotations
from typing import Optional
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from geocoding import geocode_address, geocode_google_address, load_cache, normalize_address, save_cache


REQUIRED_COLUMNS = ["Complainent Address", "Class of Incident"]
LOG_PATH = Path("nominatim.log")

ICON_MAPPING = {
    "Cyber Crime (other than financial fraud)": "https://img.icons8.com/color/48/000000/cyber-security.png",
    "Cyber Financial Fraud": "https://img.icons8.com/color/48/000000/bank-cards.png",
    "Other IPC/BNS Crimes": "https://img.icons8.com/color/48/000000/handcuffs.png",
    "Miscellaneous": "https://img.icons8.com/color/48/000000/box-important--v1.png",
    "Crime Against SC/ST": "https://img.icons8.com/color/48/000000/scales.png",
    "Crime against Children": "https://img.icons8.com/color/48/000000/children.png",
    "Matrimonial Dispute": "https://img.icons8.com/color/48/000000/wedding-rings.png",
    "Illegal Immigration": "https://img.icons8.com/color/48/000000/passport.png",
    "Job Related Fraud": "https://img.icons8.com/color/48/000000/briefcase.png",
    "Property/Land Dispute": "https://img.icons8.com/color/48/000000/home.png",
    "Other Economic Offence": "https://img.icons8.com/color/48/000000/money-bag.png",
    "Noise Pollution": "https://img.icons8.com/color/48/000000/speaker.png",
    "Runaway Couples": "https://img.icons8.com/color/48/000000/running.png",
    "Security Threat": "https://img.icons8.com/color/48/000000/warning-shield.png",
    "Minor Accident": "https://img.icons8.com/color/48/000000/car-crash.png",
    "Crime Against Women": "https://img.icons8.com/color/48/000000/female.png",
    "Corruption/Demand of Bribe": "https://img.icons8.com/color/48/000000/cash.png",
    "Lost Property": "https://img.icons8.com/color/48/000000/lost-and-found.png",
    "Hurt": "https://img.icons8.com/color/48/000000/bandage.png",
    "Intimidation": "https://img.icons8.com/color/48/000000/angry.png",
}
DEFAULT_ICON = "https://img.icons8.com/color/48/000000/marker.png"


def ensure_required_columns(dataframe: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    return (len(missing) == 0, missing)

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


def build_map(dataframe: pd.DataFrame, show_labels: bool, map_style: str) -> Optional[Tuple[pdk.Deck, List[str]]]:
    valid_points = dataframe.dropna(subset=["latitude", "longitude"]).copy()
    if valid_points.empty:
        return None

    classes = sorted(valid_points["Class of Incident"].dropna().astype(str).unique().tolist())
    
    def get_icon_data(class_name: str) -> dict:
        url = ICON_MAPPING.get(class_name, DEFAULT_ICON)
        return {
            "url": url,
            "width": 128,
            "height": 128,
            "anchorY": 128
        }
        
    valid_points["icon_data"] = valid_points["Class of Incident"].astype(str).map(get_icon_data)

    # Aggregate counts by latitude and longitude to show on the map
    coords_count = valid_points.groupby(["latitude", "longitude"]).size().reset_index(name="case_count")
    coords_count["case_count_str"] = coords_count["case_count"].astype(str)
    
    # Merge count back into valid_points so scatter radius and tooltip can use it
    valid_points = valid_points.merge(coords_count, on=["latitude", "longitude"])
    
    # Calculate dynamic properties for high-incident highlighting
    # Cap the max icon size growth to prevent gigantic icons
    valid_points["icon_size"] = 3 + valid_points["case_count"].apply(lambda c: min((c - 1) * 0.1, 1.5))
    valid_points["elevation"] = valid_points["case_count"] * 30  # 30 meters per case
    valid_points["count_color"] = valid_points["case_count"].apply(lambda c: [255, 50, 50, 255] if c >= 3 else [255, 255, 255, 255])
    
    # Apply to coords_count as well for count text layer
    coords_count["elevation"] = coords_count["case_count"] * 30
    coords_count["count_color"] = coords_count["case_count"].apply(lambda c: [255, 50, 50, 255] if c >= 3 else [255, 255, 255, 255])

    # Locked to Karnal district and 3D view
    center_lat = 29.6857
    center_lon = 76.9905
    zoom = 10.5

    column_layer = pdk.Layer(
        "ColumnLayer",
        data=valid_points,
        get_position="[longitude, latitude]",
        get_elevation="elevation",
        radius=60,
        get_fill_color="[255, 50, 50, 150]",
        pickable=True,
        extruded=True,
    )

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=valid_points,
        get_position="[longitude, latitude]",
        get_weight=1,
        aggregation="SUM",
        radiusPixels=20,
    )
    icon_layer = pdk.Layer(
        "IconLayer",
        data=valid_points,
        get_icon="icon_data",
        get_size="icon_size",
        size_scale=10,
        get_position="[longitude, latitude, elevation]",
        pickable=True,
    )
    text_layer = pdk.Layer(
        "TextLayer",
        data=valid_points,
        get_position="[longitude, latitude, elevation]",
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
        get_position="[longitude, latitude, elevation]",
        get_text="case_count_str",
        get_size=18,
        get_color="count_color",
        get_alignment_baseline="'center'",
        get_text_anchor="'middle'",
        font_weight="bold",
        pickable=False,
    )

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=45, bearing=0)
    layers = [heatmap_layer, column_layer, icon_layer, count_layer]
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
        tooltip={"text": "Class: {Class of Incident}\nLocation: {Complainent Address}\nIncidents at this location: {case_count}"},
        map_style=pydeck_style,
    )
    return deck, classes


def render_map_view(view_id: int, df: pd.DataFrame, available_classes: List[str], map_style: str, datetime_cols: List[str]) -> None:
    with st.container(border=True):
        col_title, col_del = st.columns([0.85, 0.15])
        with col_title:
            st.text_input(
                "View Name", 
                value=f"Map View {view_id}", 
                label_visibility="collapsed", 
                key=f"name_{view_id}"
            )
        with col_del:
            if st.button("🗑️", key=f"del_{view_id}", help="Delete View", use_container_width=True):
                st.session_state["map_view_ids"].remove(view_id)
                st.rerun()
        
        view_df = df.copy()

        if datetime_cols:
            col_date1, col_date2 = st.columns([1, 2])
            with col_date1:
                date_col = st.selectbox(
                    "Date Column", 
                    options=datetime_cols, 
                    key=f"date_col_{view_id}",
                    label_visibility="collapsed"
                )
            
            view_df[date_col] = pd.to_datetime(view_df[date_col], errors="coerce")
            min_date = view_df[date_col].min()
            max_date = view_df[date_col].max()
            
            if pd.notna(min_date) and pd.notna(max_date):
                min_date_val = min_date.date()
                max_date_val = max_date.date()
                with col_date2:
                    selected_dates = st.date_input(
                        "Date Range",
                        value=(min_date_val, max_date_val),
                        min_value=min_date_val,
                        max_value=max_date_val,
                        key=f"date_input_{view_id}",
                        label_visibility="collapsed"
                    )
                
                if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                    start_dt = pd.to_datetime(selected_dates[0])
                    end_dt = pd.to_datetime(selected_dates[1]) + pd.Timedelta(days=1, seconds=-1)
                    view_df = view_df[(view_df[date_col] >= start_dt) & (view_df[date_col] <= end_dt)]

        with st.popover("⚙️ Filter & Legend"):
            show_labels = st.toggle("Show class labels", value=True, key=f"show_labels_{view_id}")
            
            filter_df_initial = pd.DataFrame({
                "Show": [True] * len(available_classes),
                "Icon": [ICON_MAPPING.get(c, DEFAULT_ICON) for c in available_classes],
                "Class": available_classes
            })

            edited_filter_df = st.data_editor(
                filter_df_initial,
                column_config={
                    "Show": st.column_config.CheckboxColumn("Show", default=True),
                    "Icon": st.column_config.ImageColumn("Icon"),
                    "Class": st.column_config.TextColumn("Class", disabled=True),
                },
                hide_index=True,
                use_container_width=True,
                key=f"data_editor_{view_id}"
            )
            selected_classes = edited_filter_df[edited_filter_df["Show"]]["Class"].tolist()

        if selected_classes:
            view_df = view_df[view_df["Class of Incident"].astype(str).isin(selected_classes)]
        else:
            view_df = view_df.iloc[0:0]

        total_count = len(view_df)
        geocoded_count = int(view_df["latitude"].notna().sum()) if "latitude" in view_df.columns else 0
        failed_count = total_count - geocoded_count

        metrics = st.columns(3)
        metrics[0].metric("Total Records", total_count)
        metrics[1].metric("Geocoded Records", geocoded_count)
        metrics[2].metric("Unresolved Records", failed_count)
        
        map_data = build_map(view_df, show_labels=show_labels, map_style=map_style)
        if map_data:
            deck, _ = map_data
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.warning("No valid coordinates found to render map.")


def main() -> None:
    st.set_page_config(page_title="Incident Heatmap", layout="wide")
    st.title("Incident Heatmap")
    st.caption("Upload complaint data, set location scope, geocode with Nominatim/Google, and inspect incident density.")

    col_h, col_pop = st.columns([0.95, 0.05])
    with col_h:
        st.subheader("📂 Data Upload")
    with col_pop:
        with st.popover("ℹ️"):
            st.markdown("**How to use this app:**")
            st.markdown("1. **Upload** an `.xlsx` file containing incident data.")
            st.markdown("2. **Ensure** it has `Complainent Address` and `Class of Incident` columns.")
            st.markdown("3. **After upload**, the data will be validated. Then, you can **set** your geocoding scope below.")
            st.markdown("4. **Click** *Geocode missing addresses* to resolve locations and plot them on the map.")
            st.markdown("5. **Use** the *⚙️ Filter & Legend* to show/hide specific incident types and toggle labels.")

    uploaded_file = st.file_uploader("Upload complaint Excel file", type=["xlsx"])

    with st.container(border=True):
        st.subheader("⚙️ Configuration & Actions")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            district = st.text_input("District", value="Karnal")
        with col2:
            state = st.text_input("State", value="Haryana")
        with col3:
            country = st.text_input("Country", value="India")
        with col4:
            map_style = st.selectbox("Base Map Style", ["Light", "Dark", "Satellite", "Road"], index=0)

        # Get Google API Key directly from secrets (no UI)
        google_api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")

        col_btn1, col_btn2, col_info = st.columns([1, 1, 2])
        with col_btn1:
            geocode_now = st.button("Geocode missing addresses", type="primary", use_container_width=True)
        with col_btn2:
            clear_geocode = st.button("Reset session result", use_container_width=True)
        with col_info:
            st.info(f"Scope: `{district.strip() or '-'}, {state.strip() or '-'}, {country.strip() or '-'}`")

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
        st.success("Session geocoded result cleared.")

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

    full_df = st.session_state["geocoded_df"]
    available_classes = sorted(full_df["Class of Incident"].dropna().astype(str).unique().tolist())

    # Identify potential datetime columns for view-level filtering
    datetime_cols = []
    for col in result_df.columns:
        if pd.api.types.is_datetime64_any_dtype(result_df[col]):
            datetime_cols.append(col)
        elif result_df[col].dtype == object:
            if any(x in col.lower() for x in ["date", "time", "timestamp"]):
                try:
                    sample = result_df[col].dropna().head(10)
                    if not sample.empty:
                        pd.to_datetime(sample, errors="raise")
                        datetime_cols.append(col)
                except:
                    pass

    st.markdown("---")

    if "map_view_ids" not in st.session_state:
        st.session_state["map_view_ids"] = [1]
    if "next_view_id" not in st.session_state:
        st.session_state["next_view_id"] = 2

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Add New Map View", type="primary", use_container_width=True):
            st.session_state["map_view_ids"].append(st.session_state["next_view_id"])
            st.session_state["next_view_id"] += 1
            st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

    view_ids = st.session_state["map_view_ids"]
    
    for i in range(0, len(view_ids), 2):
        cols = st.columns(2)
        with cols[0]:
            render_map_view(view_ids[i], result_df, available_classes, map_style, datetime_cols)
        if i + 1 < len(view_ids):
            with cols[1]:
                render_map_view(view_ids[i+1], result_df, available_classes, map_style, datetime_cols)

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
    st.markdown("### Full Data View")
    st.dataframe(result_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
