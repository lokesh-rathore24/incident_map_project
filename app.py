from __future__ import annotations
from typing import Optional
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import base64
import hashlib
import inspect
import io

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from classifier import classify_dataframe
from geocoding import geocode_address, geocode_google_address, load_cache, normalize_address, save_cache


REQUIRED_COLUMNS = ["Complainent Address", "Class of Incident"]
LOG_PATH = Path("nominatim.log")

ICONS_DIR = Path("police_icons")
ICON_FILES = {
    "Cyber Crime (other than financial fraud)": "01_cyber_crime.png",
    "Cyber Financial Fraud": "02_cyber_financial_fraud.png",
    "Other IPC/BNS Crimes": "03_other_ipc_bns.png",
    "Miscellaneous": "04_miscellaneous.png",
    "Crime Against SC/ST": "05_sc_st_crime.png",
    "Crime against Children": "06_crime_against_children.png",
    "Matrimonial Dispute": "07_matrimonial_dispute.png",
    "Illegal Immigration": "08_illegal_immigration.png",
    "Job Related Fraud": "09_job_related_fraud.png",
    "Property/Land Dispute": "10_property_land_dispute.png",
    "Other Economic Offence": "11_other_economic_offence.png",
    "Noise Pollution": "12_noise_pollution.png",
    "Runaway Couples": "13_runaway_couples.png",
    "Security Threat": "14_security_threat.png",
    "Minor Accident": "15_minor_accident.png",
    "Crime Against Women": "16_crime_against_women.png",
    "Corruption/Demand of Bribe": "17_corruption_bribe.png",
    "Lost Property": "18_lost_property.png",
    "Hurt": "19_hurt.png",
    "Intimidation": "20_intimidation.png",
}

DEFAULT_MARKER_ICON = "https://img.icons8.com/fluency/48/000000/marker.png"


def load_local_icons() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for category, filename in ICON_FILES.items():
        file_path = ICONS_DIR / filename
        if file_path.exists():
            try:
                encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
                mapping[category] = f"data:image/png;base64,{encoded}"
            except Exception:
                mapping[category] = DEFAULT_MARKER_ICON
        else:
            mapping[category] = DEFAULT_MARKER_ICON
    return mapping

ICON_MAPPING = load_local_icons()

CATEGORY_COLOR_MAPPING: Dict[str, List[int]] = {
    "Cyber Crime (other than financial fraud)": [35, 122, 255, 180],
    "Cyber Financial Fraud": [123, 31, 162, 180],
    "Other IPC/BNS Crimes": [255, 153, 51, 180],
    "Miscellaneous": [96, 125, 139, 180],
    "Crime Against SC/ST": [244, 67, 54, 180],
    "Crime against Children": [255, 193, 7, 180],
    "Matrimonial Dispute": [233, 30, 99, 180],
    "Illegal Immigration": [0, 150, 136, 180],
    "Job Related Fraud": [121, 85, 72, 180],
    "Property/Land Dispute": [76, 175, 80, 180],
    "Other Economic Offence": [255, 87, 34, 180],
    "Noise Pollution": [63, 81, 181, 180],
    "Runaway Couples": [156, 39, 176, 180],
    "Security Threat": [244, 67, 54, 180],
    "Minor Accident": [255, 235, 59, 180],
    "Crime Against Women": [233, 30, 99, 180],
    "Corruption/Demand of Bribe": [255, 152, 0, 180],
    "Lost Property": [121, 85, 72, 180],
    "Hurt": [244, 67, 54, 180],
    "Intimidation": [63, 81, 181, 180],
}


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


@st.cache_data
def build_map(dataframe: pd.DataFrame, show_labels: bool, map_style: str) -> Optional[Tuple[pdk.Deck, List[str]]]:
    valid_points = dataframe.dropna(subset=["latitude", "longitude"]).copy()
    if valid_points.empty:
        return None

    classes = sorted(valid_points["Class of Incident"].dropna().astype(str).unique().tolist())

    # Aggregate by location and incident class to avoid rendering duplicate markers
    # for multiple records at the exact same lat/lon.
    agg_points = (
        valid_points
        .groupby(["latitude", "longitude", "Class of Incident"], as_index=False)
        .agg(
            case_count=("Class of Incident", "size"),
            Complainent_Address=("Complainent Address", "first"),
        )
    )
    agg_points = agg_points.rename(columns={"Complainent_Address": "Complainent Address"})
    agg_points["case_count_str"] = agg_points["case_count"].astype(str)
    agg_points["fill_color"] = agg_points["Class of Incident"].astype(str).map(
        lambda class_name: CATEGORY_COLOR_MAPPING.get(class_name, [96, 125, 139, 180])
    )
    agg_points["elevation"] = agg_points["case_count"] * 40
    agg_points["count_color"] = agg_points["case_count"].apply(
        lambda c: [255, 50, 50, 255] if c >= 3 else [255, 255, 255, 255]
    )
    agg_points["icon_data"] = agg_points["Class of Incident"].astype(str).map(
        lambda class_name: {
            "url": ICON_MAPPING.get(class_name, DEFAULT_MARKER_ICON),
            "width": 64,
            "height": 64,
            "anchorY": 64,
        }
    )

    center_lat = 29.6857
    center_lon = 76.9905
    zoom = 10.5

    column_layer = pdk.Layer(
        "ColumnLayer",
        data=agg_points,
        get_position="[longitude, latitude]",
        get_elevation="elevation",
        radius=30,
        get_fill_color="fill_color",
        pickable=True,
        extruded=True,
        opacity=0.8,
    )

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=agg_points,
        get_position="[longitude, latitude]",
        get_weight="case_count",
        aggregation="SUM",
        radiusPixels=20,
    )

    icon_layer = pdk.Layer(
        "IconLayer",
        data=agg_points,
        get_icon="icon_data",
        get_size=24,
        size_scale=1.5,
        get_position="[longitude, latitude, elevation + 45]",
        pickable=False,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=agg_points,
        get_position="[longitude, latitude, elevation + 55]",
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
        data=agg_points,
        get_position="[longitude, latitude, elevation + 22]",
        get_text="case_count_str",
        get_size=18,
        get_color="count_color",
        get_alignment_baseline="'center'",
        get_text_anchor="'middle'",
        font_weight="'bold'",
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


def _render_pydeck_chart(deck: pdk.Deck, key: Optional[str] = None):
    signature = inspect.signature(st.pydeck_chart)
    params = signature.parameters
    if "use_container_width" in params:
        return st.pydeck_chart(deck, use_container_width=True, key=key)
    if "width" in params:
        return st.pydeck_chart(deck, width="stretch", key=key)
    return st.pydeck_chart(deck, key=key)


def _init_view_state(view_id: int, available_classes: List[str], default_date_range=None) -> None:
    """Pre-initialise session-state entries for every widget in a map view.

    Streamlit resets widgets whose ``value`` parameter is supplied alongside a
    ``key`` whenever the widget tree changes (e.g. a new map view is added and
    ``st.rerun()`` is called).  By writing defaults into ``session_state``
    *before* widget creation and **never** passing ``value`` to the widget, we
    let Streamlit read from its own state store, which survives reruns.
    """
    _ss = st.session_state

    # View name
    name_key = f"name_{view_id}"
    if name_key not in _ss:
        _ss[name_key] = f"Map View {view_id}"

    # Show-labels toggle
    label_key = f"show_labels_{view_id}"
    if label_key not in _ss:
        _ss[label_key] = True

    # Date range (tuple of two datetime.date)
    if default_date_range is not None:
        date_key = f"date_input_{view_id}"
        if date_key not in _ss:
            _ss[date_key] = default_date_range

    # Class-filter state: stored as a plain dict {class_name: bool} so it
    # is independent of the data_editor's internal delta format.
    filter_key = f"_filter_state_{view_id}"
    if filter_key not in _ss:
        _ss[filter_key] = {c: True for c in available_classes}


def render_map_view(view_id: int, df: pd.DataFrame, available_classes: List[str], map_style: str, datetime_cols: List[str]) -> None:
    with st.container(border=True):
        # --- Compute date bounds early so we can pre-init state --------
        date_range_defaults = None
        if datetime_cols:
            # Peek at the first datetime col to figure out min/max
            _tmp = df.copy()
            _peek_col = st.session_state.get(f"date_col_{view_id}", datetime_cols[0])
            if _peek_col not in datetime_cols:
                _peek_col = datetime_cols[0]
            _tmp[_peek_col] = pd.to_datetime(_tmp[_peek_col], errors="coerce")
            _min = _tmp[_peek_col].min()
            _max = _tmp[_peek_col].max()
            if pd.notna(_min) and pd.notna(_max):
                date_range_defaults = (_min.date(), _max.date())

        # Pre-initialise ALL widget states before any widget is created
        _init_view_state(view_id, available_classes, default_date_range=date_range_defaults)

        # --- Title row -------------------------------------------------
        col_title, col_del = st.columns([0.86, 0.14])
        with col_title:
            st.text_input(
                "View Name",
                label_visibility="collapsed",
                key=f"name_{view_id}"
            )
        with col_del:
            if st.button("🗑️", key=f"del_{view_id}", help="Delete this map view", width="stretch"):
                st.session_state["map_view_ids"].remove(view_id)
                st.rerun()

        view_df = df.copy()

        # --- Date-range filter -----------------------------------------
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
                date_key = f"date_input_{view_id}"

                with col_date2:
                    selected_dates = st.date_input(
                        "Date Range",
                        min_value=min_date_val,
                        max_value=max_date_val,
                        key=date_key,
                        label_visibility="collapsed",
                    )

                if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                    start_dt = pd.to_datetime(selected_dates[0])
                    end_dt = pd.to_datetime(selected_dates[1]) + pd.Timedelta(days=1, seconds=-1)
                    view_df = view_df[(view_df[date_col] >= start_dt) & (view_df[date_col] <= end_dt)]

        # --- Class filter & legend -------------------------------------
        filter_state_key = f"_filter_state_{view_id}"
        filter_state = st.session_state[filter_state_key]

        with st.popover("⚙️ Filter & Legend"):
            show_labels = st.toggle("Show class labels", key=f"show_labels_{view_id}")

            # Build base dataframe from persisted filter state so edits
            # survive reruns even if data_editor deltas are lost.
            filter_df = pd.DataFrame({
                "Show": [filter_state.get(c, True) for c in available_classes],
                "Icon": [ICON_MAPPING.get(c, DEFAULT_MARKER_ICON) for c in available_classes],
                "Class": available_classes
            })
            filter_df["Show"] = filter_df["Show"].astype(bool)

            edited_filter_df = st.data_editor(
                filter_df,
                column_config={
                    "Show": st.column_config.CheckboxColumn("Show", default=True),
                    "Icon": st.column_config.ImageColumn("Icon"),
                    "Class": st.column_config.TextColumn("Class", disabled=True),
                },
                hide_index=True,
                width="stretch",
                key=f"data_editor_{view_id}"
            )

            # Persist the edited Show values back into our own state dict
            for _, row in edited_filter_df.iterrows():
                filter_state[row["Class"]] = row["Show"]

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

        reset_count = st.session_state.get(f"reset_count_{view_id}", 0)
        with st.spinner("Rendering incidents on map..."):
            map_data = build_map(view_df, show_labels=show_labels, map_style=map_style)
            if map_data:
                deck, _ = map_data
                # Changing the key forces Streamlit to fully remount the PyDeck component,
                # which re-applies initial_view_state (zoom + position reset)
                _render_pydeck_chart(deck, key=f"map_{view_id}_{reset_count}")
            else:
                st.warning("No valid coordinates found to render map.")


def main() -> None:
    st.set_page_config(page_title="CrimeLens", layout="wide")
    st.title("CrimeLens")
    st.caption("Upload complaint data, set location scope, geocode with Nominatim/Google, and inspect incident density.")

    col_h, col_pop = st.columns([0.95, 0.05])
    with col_h:
        st.subheader("📂 Data Upload")
    with col_pop:
        with st.popover("ℹ️"):
            st.markdown("**How to use this app:**")
            st.markdown("1. **Upload** an `.xlsx` file containing incident data.")
            st.markdown("2. **Ensure** it has `Complainent Address` and `Class of Incident` columns. A `Date` column is also recommended.")
            st.markdown("3. **Geocode:** Set your scope and click *Geocode missing addresses* to resolve locations.")
            st.markdown("4. **Multiple Views:** Click *✨ Add New Map View* at the bottom to compare different maps side-by-side.")
            st.markdown("5. **Customize:** Click the Map View title to rename it. Use the *⚙️ Filter & Legend* to show/hide specific incident types.")
            st.markdown("6. **Time Travel:** Select a Date Range just below the Map Name to filter incidents by time for that specific view.")

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload complaint Excel file", type=["xlsx"])
    with col_sample:
        sample_columns = [
            "Sr.No",
            "Date of Submission",
            "Class of Incident",
            "Complainent Address",
            "Complaint Description",
        ]
        sample_df = pd.DataFrame(columns=sample_columns)
        buf = io.BytesIO()
        sample_df.to_excel(buf, index=False)
        buf.seek(0)
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Sample Format",
            data=buf,
            file_name="crimelens_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            help="Download a blank Excel template with all required columns."
        )

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
            geocode_now = st.button("Geocode missing addresses", type="primary", width="stretch")
        with col_btn2:
            clear_geocode = st.button("Reset session result", width="stretch")
        with col_info:
            st.info(f"Scope: `{district.strip() or '-'}, {state.strip() or '-'}, {country.strip() or '-'}`")

    if uploaded_file is None:
        if not st.session_state.get("_ready_logged"):
            print("📡 CrimeLens app is ready! Open http://localhost:8501 and upload your .xlsx file to trigger classification.", flush=True)
            st.session_state["_ready_logged"] = True
        st.info("Upload an `.xlsx` file to begin.")
        return

    try:
        raw_df = pd.read_excel(uploaded_file)
        
        # Normalize column names (case-insensitive & strip whitespace)
        col_mapping = {}
        for col in raw_df.columns:
            norm = str(col).strip().lower()
            if norm == "complaint description":
                col_mapping[col] = "Complaint Description"
            elif norm == "class of incident":
                col_mapping[col] = "Class of Incident"
            elif norm == "complainent address":
                col_mapping[col] = "Complainent Address"
            elif norm in ("incident date", "date"):
                col_mapping[col] = "Incident Date"
            else:
                col_mapping[col] = str(col).strip()
        raw_df = raw_df.rename(columns=col_mapping)
        
    except Exception as exc:  # broad exception to show user-friendly error
        st.error(f"Could not read Excel file: {exc}")
        return

    # --- AI Classification Step ---
    openai_api_key = st.secrets.get("OPEN_AI_API_KEY", "")
    
    # Diagnostic prints to terminal
    print(f"📊 File uploaded! Row count: {len(raw_df)}", flush=True)
    print(f"   Available columns: {list(raw_df.columns)}", flush=True)
    print(f"   OpenAI API Key detected: {bool(openai_api_key)}", flush=True)
    
    if openai_api_key and "Complaint Description" in raw_df.columns:
        # Include API key in hash so changing the key forces re-classification
        hash_input = uploaded_file.getvalue() + openai_api_key.encode("utf-8")
        file_hash = hashlib.md5(hash_input).hexdigest()

        if st.session_state.get("_ai_classified_hash") != file_hash:
            placeholder = st.empty()
            with placeholder.container():
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 16px;
                        padding: 2rem;
                        text-align: center;
                        border: 1px solid rgba(255,255,255,0.1);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                    ">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🤖</div>
                        <h3 style="color: #e0e0e0; margin: 0 0 0.5rem 0;">Cleaning file with AI</h3>
                        <p style="color: #9e9e9e; font-size: 0.9rem;">Classifying complaint descriptions into incident categories using OpenAI</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                progress_bar = st.progress(0, text="Initialising...")

                def _update_progress(done: int, total: int) -> None:
                    progress_bar.progress(
                        done / total,
                        text=f"Processing batch {done} / {total}  ({done * 100 // total}%)",
                    )

                raw_df = classify_dataframe(
                    raw_df, openai_api_key, batch_size=50, progress_callback=_update_progress
                )

            placeholder.empty()
            st.session_state["_ai_classified_hash"] = file_hash
            st.session_state["_ai_classified_df"] = raw_df
            st.toast("✅ AI classification complete!", icon="🤖")
        else:
            raw_df = st.session_state["_ai_classified_df"]

    # --- Safety net: ensure Class of Incident is never None/NaN/empty ---
    if "Class of Incident" in raw_df.columns:
        raw_df["Class of Incident"] = (
            raw_df["Class of Incident"]
            .fillna("Miscellaneous")
            .astype(str)
            .replace({"None": "Miscellaneous", "nan": "Miscellaneous", "null": "Miscellaneous", "": "Miscellaneous"})
        )

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
        if st.button("✨ Add New Map View", type="primary", width="stretch"):
            st.session_state["map_view_ids"].append(st.session_state["next_view_id"])
            st.session_state["next_view_id"] += 1
            st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

    view_ids = st.session_state["map_view_ids"]
    
    i = 0
    while i < len(view_ids):
        # If there's only 1 view remaining, render it full-width
        if i == len(view_ids) - 1:
            render_map_view(view_ids[i], result_df, available_classes, map_style, datetime_cols)
            i += 1
        else:
            cols = st.columns(2)
            with cols[0]:
                render_map_view(view_ids[i], result_df, available_classes, map_style, datetime_cols)
            with cols[1]:
                render_map_view(view_ids[i+1], result_df, available_classes, map_style, datetime_cols)
            i += 2

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
            width="stretch",
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("### Full Data View")
    st.dataframe(result_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
