# -*- coding: utf-8 -*-
"""
Created on Fri Aug 8 2025
@author: SuPAR Group ~ Shaffie & Reza (updated)
Merged with Robust Normalizer logic
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import io
import pathlib
import csv

st.set_page_config(page_title="Paving Temperature Visualizer", layout="wide")
st.title("SuPave Dashboard (Robust)")

st.caption(
    "Upload your Paving Data. Supports: \n"
    "1. **Filled Excel Template** (.xlsx) \n"
    "2. **Raw Scanner CSVs** (e.g., with [DATA] tags and Scanner[i] columns)"
)

# --- Downloadable Excel template section ---
TEMPLATE_PATH = pathlib.Path("Paver_Upload_Template.xlsx")

with st.expander("📄 Download input template (for manual data)"):
    st.markdown(
        """
        **Required columns for Excel:**
        - **Time** → timestamp.
        - **Moving distance** → cumulative distance (m).
        - **Coordinates** → lat/lon or easting/northing.
        - **Temperature columns** → headers like `6.25 m [°C] R`.
        """
    )
    if TEMPLATE_PATH.exists():
        with open(TEMPLATE_PATH, "rb") as f:
            st.download_button(
                label="📥 Download Excel template",
                data=f.read(),
                file_name="SuPave_Input_Template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

# ==========================================
# 1. ROBUST PARSING HELPERS (From Normalizer)
# ==========================================

def clean_header(header: str) -> str:
    """Fixes encoding mojibake (Â° -> °) and normalizes."""
    h = str(header)
    h = h.replace("Â°", "°").replace("â°", "°")
    h = h.replace("(", "[").replace(")", "]")
    return h.strip()

WIDTH_COL_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*m\s*\[\s*°C\s*\]\s*([LR])?\s*$', re.I)
SCANNER_RE = re.compile(r"scanner\s*\[?_?(\d+)\]?.*\[(?:°|deg\s*)?c\]", re.I)

def col_to_width(col: str):
    """Parses headers like '6.25 m [°C] R' into float 6.25"""
    # Clean header first to ensure ° matches
    s = clean_header(col)
    m = WIDTH_COL_RE.match(s)
    if not m:
        return None
    v = float(m.group(1))
    side = m.group(2)
    if side:
        return v if side.upper() == "R" else -v
    return 0.0

def build_template_positions(max_m=6.25, step=0.25):
    r = np.arange(step, max_m + 1e-9, step)
    l = -r[::-1]
    return np.concatenate([l, [0.0], r])

def template_col_from_x(x: float) -> str:
    if abs(x) < 1e-12: return "0.00 m [°C]"
    return f"{abs(x):.2f} m [°C] {'R' if x > 0 else 'L'}"

def resample_scanners(df: pd.DataFrame, width: float) -> pd.DataFrame:
    """Interpolates Scanner[i] columns into the grid format used by the visualizer."""
    scanners = []
    for c in df.columns:
        m = SCANNER_RE.search(clean_header(c))
        if m:
            scanners.append((int(m.group(1)), c))

    if not scanners:
        return df

    scanners.sort(key=lambda t: t[0])
    cols = [c for _, c in scanners]
    
    # Grid Setup
    x_raw = np.linspace(-float(width)/2.0, float(width)/2.0, len(cols))
    x_tgt = build_template_positions()
    inside = np.abs(x_tgt) <= (float(width)/2.0 + 1e-9)

    # Extract numeric data
    raw_vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    tgt = np.full((raw_vals.shape[0], len(x_tgt)), np.nan, dtype=float)

    # Interpolate row-by-row
    for i in range(raw_vals.shape[0]):
        y = raw_vals[i, :]
        valid = np.isfinite(y)
        if valid.sum() >= 2:
            tgt[i, inside] = np.interp(x_tgt[inside], x_raw[valid], y[valid])

    out = df.copy()
    for j, x in enumerate(x_tgt):
        out[template_col_from_x(x)] = tgt[:, j]
    
    return out

def load_robust_data(uploaded_file):
    """Handles Excel OR Raw CSVs with [DATA] tags and PavingWidth headers."""
    filename = uploaded_file.name.lower()
    
    # A. EXCEL
    if filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    
    # B. CSV (The tricky part)
    # We must read the file into text first to parse the header
    content_bytes = uploaded_file.getvalue()
    
    # Detect encoding
    encoding = 'utf-8'
    try:
        content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        encoding = 'cp1252'
    
    text_content = content_bytes.decode(encoding, errors='replace')
    lines = text_content.splitlines()
    
    # 1. Extract PavingWidth
    paving_width = None
    data_start_idx = None
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if line_clean.upper().startswith("[DATA]"):
            data_start_idx = i
            break
        
        if line_clean.lower().startswith("pavingwidth"):
            # Split by ; : or ,
            parts = re.split(r"[;:,]", line_clean)
            for p in parts[1:]:
                if p.strip():
                    try:
                        paving_width = float(p.strip())
                        break
                    except ValueError:
                        continue

    # 2. Load Data Frame
    if data_start_idx is not None:
        # Data starts AFTER [DATA] tag
        # Determine separator
        sep = ';' if lines[data_start_idx].count(';') > lines[data_start_idx].count(',') else ','
        
        # Use StringIO to feed the data part to pandas
        data_str = "\n".join(lines[data_start_idx+1:])
        df = pd.read_csv(io.StringIO(data_str), sep=sep)
    else:
        # Fallback: No [DATA] tag, assume standard CSV
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip')

    # 3. Resample if we found a width
    if paving_width:
        st.success(f"Detected Raw CSV. Resampling scanners using PavingWidth: {paving_width}m")
        df = resample_scanners(df, paving_width)
    
    return df

# ==========================================
# 2. ANALYSIS HELPERS (Kinematic / Projection)
# ==========================================

def robust_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=False, dayfirst=False)
    if dt.isna().all():
        dt = pd.to_datetime(series, errors="coerce", utc=False, dayfirst=True)
    return dt

def manual_lambert72_to_wgs84(easting, northing):
    """Fallback projection for Belgium"""
    try:
        lon_approx = 4.35 + (easting - 149000) * 0.000014
        lat_approx = 50.85 + (northing - 170000) * 0.000009
        return lon_approx, lat_approx
    except:
        return np.nan, np.nan

def detect_stops_kinematic(df: pd.DataFrame, time_col: str, dist_col: str,
                           stop_speed_threshold_m_m: float = 1.0,
                           min_stop_s: float = 10.0) -> pd.DataFrame:
    t = df[time_col]
    dist = df[dist_col]
    
    if t.isna().all() or dist.isna().all():
        return pd.DataFrame()

    dt_s = t.diff().dt.total_seconds().fillna(1.0).replace(0.0, 1.0)
    d_dist = dist.diff().fillna(0.0)
    
    velocity_m_m = (d_dist / dt_s) * 60.0
    velocity_m_m = velocity_m_m.clip(lower=0.0)
    velocity_smooth = velocity_m_m.rolling(window=3, center=True, min_periods=1).median()

    is_stopped = velocity_smooth < stop_speed_threshold_m_m
    groups = (is_stopped != is_stopped.shift()).cumsum()
    
    rows = []
    for _, group_df in df[is_stopped].groupby(groups):
        if group_df.empty: continue
        start_time = group_df[time_col].iloc[0]
        end_time = group_df[time_col].iloc[-1]
        duration = (end_time - start_time).total_seconds()
        
        if duration >= min_stop_s:
            rows.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration_s": float(duration),
                "moving_dist_start": float(group_df[dist_col].iloc[0]),
                "moving_dist_end": float(group_df[dist_col].iloc[-1]),
                "moving_dist_mid": float((group_df[dist_col].iloc[0] + group_df[dist_col].iloc[-1]) / 2.0),
            })
    return pd.DataFrame(rows)


# ==========================================
# 3. MAIN APP UI
# ==========================================

uploaded_file = st.file_uploader("Upload your file", type=["xlsx", "xls", "csv"])

if uploaded_file:
    # --- LOAD DATA (ROBUST) ---
    try:
        df = load_robust_data(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # --- NORMALIZE HEADERS ---
    # Fix the Â° symbol in headers so mapping works
    df.columns = [clean_header(c) for c in df.columns]

    # --- COLUMN MAPPING ---
    # Smart-map variations of "Moving distance" etc.
    CORE_MAP = {
        "Time": ["time", "timestamp", "datetime", "date time"],
        "Moving distance": ["moving distance", "distance", "dist", "chainage", "station", "distance [m]"],
        "latitude": ["latitude", "lat", "projectlatitude"],
        "longitude": ["longitude", "lon", "lng", "projectlongitude"],
        "easting": ["easting", "utm_e", "x"],
        "northing": ["northing", "utm_n", "y"],
    }
    
    # Rename columns to standard names
    for col in df.columns:
        c_lower = col.strip().lower()
        for std, variants in CORE_MAP.items():
            if c_lower in variants:
                df.rename(columns={col: std}, inplace=True)

    # --- COORD CHECK ---
    has_latlon = "latitude" in df.columns and "longitude" in df.columns
    has_en = "easting" in df.columns and "northing" in df.columns

    # --- TEMP COLUMN CHECK ---
    temp_cols = [c for c in df.columns if "°C" in c or "[C]" in c]
    width_map = {c: col_to_width(c) for c in temp_cols}
    # Only keep columns that successfully mapped to a width
    temp_cols = [c for c, w in width_map.items() if w is not None]

    if not temp_cols:
        st.error("No valid temperature columns found. (Expected format: '6.25 m [°C] R')")
        if "Scanner[1]" in str(df.columns):
            st.warning("Scanner columns detected but PavingWidth was missing in the header, so they couldn't be resampled.")
        st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Paving Settings")
        width_min, width_max = st.slider("Paving width range (m)", -6.25, 6.25, (-2.0, 2.0), 0.25)
        st.divider()
        st.subheader("Stop Detection")
        stop_speed_thresh = st.number_input("Stop Velocity (m/min)", 0.1, 10.0, 1.0)
        min_stop_dur = st.number_input("Min Stop Duration (s)", 1.0, 300.0, 15.0)

    # --- DATA CLEANING ---
    needed = ["Time", "Moving distance"]
    if has_latlon: needed += ["latitude", "longitude"]
    if has_en: needed += ["easting", "northing"]

    for col in needed:
        if col not in df.columns: df[col] = np.nan

    # Time & Distance Fixes
    df["Time"] = robust_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)

    # Monotonic Distance Force
    if "Moving distance" in df.columns:
        df["Moving distance"] = pd.to_numeric(df["Moving distance"], errors='coerce').ffill()
        df["Moving distance"] = df["Moving distance"].cummax()

    # Filter by width
    selected_cols = [c for c in temp_cols if width_min <= width_map[c] <= width_max]
    if not selected_cols:
        st.warning(f"No columns found in width range {width_min}m to {width_max}m.")
        st.stop()

    # --- STOP DETECTION ---
    stops = detect_stops_kinematic(df, "Time", "Moving distance", stop_speed_thresh, min_stop_dur)
    stop_count = len(stops)

    # --- HEATMAP GENERATION ---
    bin_size = 5.0
    temp_long = df.melt(id_vars=needed, value_vars=selected_cols, var_name="WidthCol", value_name="Temperature")
    temp_long["Width_m"] = temp_long["WidthCol"].map(width_map)
    temp_long["Temperature"] = pd.to_numeric(temp_long["Temperature"], errors="coerce")
    
    # Filter noise
    temp_long.loc[temp_long["Temperature"] <= 10.0, "Temperature"] = np.nan 

    temp_long["dist_bin"] = (
        np.round(pd.to_numeric(temp_long["Moving distance"], errors="coerce") / bin_size) * bin_size
    ).astype(float)

    # Pivot & Interpolate
    grid = temp_long.pivot_table(
        index="Width_m", columns="dist_bin", values="Temperature", aggfunc="mean"
    ).sort_index().sort_index(axis=1)
    
    grid = grid.interpolate(method='linear', axis=1, limit=1)

    # --- PLOTLY HEATMAP ---
    st.subheader(f"Paving Temperature Heatmap (Stops: {stop_count})")
    
    fig = px.imshow(
        grid.values,
        x=grid.columns,
        y=grid.index,
        origin="lower",
        aspect="auto",
        color_continuous_scale="Turbo",
        labels={"x": "Distance (m)", "y": "Width (m)", "color": "Temp (°C)"}
    )
    
    if not stops.empty:
        for _, r in stops.iterrows():
            dmid = r["moving_dist_mid"]
            if pd.notna(dmid):
                fig.add_vline(x=dmid, line_width=2, line_color="black", line_dash="dash", opacity=0.5)
                fig.add_annotation(x=dmid, y=grid.index.max(), text=f"{r['duration_s']:.0f}s", showarrow=False, yshift=10)

    st.plotly_chart(fig, use_container_width=True)

    # --- MAP ---
    df["avg_temp"] = df[selected_cols].mean(axis=1)
    map_df = pd.DataFrame()

    if has_latlon:
        map_df = df[["latitude", "longitude", "avg_temp"]].dropna().rename(columns={"latitude":"lat", "longitude":"lon"})
    elif has_en:
        # Projection Fallback
        lons, lats = [], []
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
            lons, lats = transformer.transform(df["easting"].values, df["northing"].values)
        except ImportError:
            st.warning("Pyproj not installed. Using manual approximation.")
            for e, n in zip(df["easting"], df["northing"]):
                lo, la = manual_lambert72_to_wgs84(e, n)
                lons.append(lo)
                lats.append(la)
        
        df["lon"], df["lat"] = lons, lats
        map_df = df[["lat", "lon", "avg_temp"]].dropna()

    if not map_df.empty:
        st.subheader("GPS Map")
        deck = pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=map_df["lat"].mean(), 
                longitude=map_df["lon"].mean(), 
                zoom=16
            ),
            layers=[
                pdk.Layer(
                    "HeatmapLayer",
                    data=map_df,
                    get_position='[lon, lat]',
                    get_weight="avg_temp",
                    radiusPixels=30,
                    intensity=1,
                    threshold=0.1
                )
            ]
        )
        st.pydeck_chart(deck)
    else:
        st.info("No valid coordinates available for map.")

else:
    st.info("Please upload an Excel file or Pave CSV.")
