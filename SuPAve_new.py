# -*- coding: utf-8 -*-
"""
Created on Fri Aug 8 2025
@author: SuPAR Group ~ Shaffie & Reza (updated)
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

st.set_page_config(page_title="Paving Temperature Visualizer", layout="wide")
st.title("SuPave Dashboard")

st.caption(
    "Upload your Excel file in the provided structure. "
    "If latitude/longitude is present, easting/northing will be ignored."
)

#  glossary for the page
with st.expander("Pave Quality Performance Indicators"):
    st.markdown(
        "- **TSI (Temperature Segregation Index):** Max Temp – Mean Temp per distance bin. High TSI ⇒ higher risk of density variation.\n"
        "- **DRS (Differential Range Statistics):** T98.5 − T1 across width (robust range). High DRS ⇒ strong temperature spread/segregation.\n"
        "- **Cold Risk:** Cells below a fixed threshold (e.g., 120 °C). Indicates potential insufficient compaction and poorer bonding.\n"
        "- **Risk Area (Relative):** Cells below a chosen % of the overall average temperature (default 90%). Highlights *relative* cold spots."
    )

# ---- Helpers ----
WIDTH_COL_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*m\s*\[\s*°C\s*\]\s*([LR])?\s*$', re.I)

def col_to_width(col: str):
    """
    '6.25 m [°C] R' -> +6.25
    '6.25 m [°C] L' -> -6.25
    '0.00 m [°C]'   -> 0.0
    """
    s = str(col)
    m = WIDTH_COL_RE.match(s)
    if not m:
        return None
    v = float(m.group(1))
    side = m.group(2)
    if side:
        return v if side.upper() == "R" else -v
    return 0.0

def robust_datetime(series: pd.Series) -> pd.Series:
    """Coerce Time to datetime; tries best-effort parsing."""
    dt = pd.to_datetime(series, errors="coerce", utc=False, dayfirst=False)
    if dt.isna().all():
        dt = pd.to_datetime(series, errors="coerce", utc=False, dayfirst=True)
    return dt

def detect_stops_by_temperature(df: pd.DataFrame,
                                time_col: str,
                                dist_col: str,
                                temp_cols: list,
                                bin_size: float) -> pd.DataFrame:
    """
    Detect idle periods using width-wise temperature cooling pattern.
    Returns a DataFrame with one row per detected stop.
    """
    # Fixed thresholds
    rate_thr = -0.02   # °C/s (≈ -1.2 °C/min)
    frac_thr = 0.70    # fraction of width columns meeting rate_thr
    min_stop_s = 8.0   # seconds
    max_dist_delta_during_stop = bin_size  # meters

    # Ensure time & distance
    t = robust_datetime(df[time_col])
    if t.isna().all():
        return pd.DataFrame(columns=[
            "start_time","end_time","duration_s","moving_dist_mid","moving_dist_start","moving_dist_end"
        ])

    order = np.argsort(t.values)
    df = df.iloc[order].reset_index(drop=True)
    t = robust_datetime(df[time_col])
    dist = pd.to_numeric(df[dist_col], errors="coerce")

    # dt (seconds)
    dt_s = t.diff().dt.total_seconds()
    med_dt = float(np.nanmedian(dt_s)) if np.isfinite(np.nanmedian(dt_s)) else 1.0
    dt_s = dt_s.fillna(med_dt).replace(0.0, med_dt)

    # numeric temperatures
    temps = df[temp_cols].apply(pd.to_numeric, errors="coerce")

    # °C/s per column
    dT = temps.diff()
    rate = dT.divide(dt_s, axis=0)

    # Fraction of columns cooling at or below threshold
    frac_cooling = (rate <= rate_thr).sum(axis=1) / max(1, len(temp_cols))
    cool_mask = frac_cooling >= frac_thr

    # Consecutive True segments
    segments = []
    in_seg = False
    seg_start = None
    for i, flag in enumerate(cool_mask.to_numpy()):
        if flag and not in_seg:
            in_seg = True
            seg_start = i
        elif not flag and in_seg:
            segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(cool_mask) - 1))

    rows = []
    for a, b in segments:
        start_time = t.iloc[a]
        end_time = t.iloc[b]
        if pd.isna(start_time) or pd.isna(end_time):
            continue
        duration = (end_time - start_time).total_seconds()
        if duration < min_stop_s:
            continue

        d_start = dist.iloc[a] if pd.notna(dist.iloc[a]) else np.nan
        d_end = dist.iloc[b] if pd.notna(dist.iloc[b]) else np.nan
        pass_dist = True
        if pd.notna(d_start) and pd.notna(d_end):
            pass_dist = abs(float(d_end) - float(d_start)) <= max_dist_delta_during_stop
        if not pass_dist:
            continue

        mid_idx = a + (b - a) // 2
        d_mid = dist.iloc[mid_idx] if pd.notna(dist.iloc[mid_idx]) else np.nan

        rows.append({
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": float(duration),
            "moving_dist_start": float(d_start) if pd.notna(d_start) else np.nan,
            "moving_dist_end": float(d_end) if pd.notna(d_end) else np.nan,
            "moving_dist_mid": float(d_mid) if pd.notna(d_mid) else np.nan,
        })

    return pd.DataFrame(rows)

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload your paving temperature file", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Coordinate type (optional for heatmap)
    has_latlon = all(c in df.columns for c in ["latitude", "longitude"]) and \
                  df["latitude"].notna().any() and df["longitude"].notna().any()
    has_en = all(c in df.columns for c in ["easting", "northing"]) and \
              df["easting"].notna().any() and df["northing"].notna().any()

    if has_latlon:
        coord_type = "latlon"
    elif has_en:
        coord_type = "eastnorth"
    else:
        coord_type = None

    # Temperature columns
    temp_cols = [c for c in df.columns if "°C" in str(c)]
    width_map = {c: col_to_width(c) for c in temp_cols}
    temp_cols = [c for c, w in width_map.items() if w is not None]

    if not temp_cols:
        st.error("No temperature columns found (expected headers like '6.25 m [°C] R').")
        st.stop()

    # Sidebar
    width_min, width_max = st.sidebar.slider(
        "Select paving width range (m)",
        -6.25, 6.25, (-2.0, 2.0), 0.25
    )
    bin_size = 5.0

    # Cold-risk slicer
    cold_thr = st.sidebar.slider(
        "Cold-risk threshold (°C)",
        min_value=60, max_value=200, value=120, step=1,
        help="Cells below this temperature are considered 'cold risk'."
    )

    # NEW: Risk Area % slicer (80%–100%)
    risk_pct = st.sidebar.slider(
        "Risk area threshold (% of overall average)",
        min_value=80, max_value=100, value=90, step=1,
        help="Cells below this % of the overall average temperature are flagged as risk areas."
    )

    # Filter temperature columns by selected width
    selected_cols = [c for c in temp_cols if width_min <= width_map[c] <= width_max]
    if not selected_cols:
        st.warning("No temperature columns within the selected width range.")
        st.stop()

    # Ensure columns exist for melt
    needed = ["Time", "Moving distance", "latitude", "longitude", "easting", "northing"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    # Parse time early
    df["Time"] = robust_datetime(df["Time"])

    # ---- Detect stops
    stops = detect_stops_by_temperature(
        df=df,
        time_col="Time",
        dist_col="Moving distance",
        temp_cols=selected_cols,
        bin_size=bin_size
    )
    stop_count = int(len(stops))
    total_idle = float(stops["duration_s"].sum()) if stop_count > 0 else 0.0
    avg_idle = float(stops["duration_s"].mean()) if stop_count > 0 else 0.0

    # ---- Heatmap data
    temp_long = df.melt(
        id_vars=needed,
        value_vars=selected_cols,
        var_name="WidthCol",
        value_name="Temperature"
    )
    temp_long["Width_m"] = temp_long["WidthCol"].map(width_map)

    if "Moving distance" not in df.columns:
        st.error("Missing 'Moving distance' column.")
        st.stop()

    # Treat 0 °C as NaN before binning/aggregating
    temp_long["Temperature"] = pd.to_numeric(temp_long["Temperature"], errors="coerce")
    temp_long.loc[temp_long["Temperature"] == 0.0, "Temperature"] = np.nan

    temp_long["dist_bin"] = (
        np.round(pd.to_numeric(temp_long["Moving distance"], errors="coerce") / bin_size) * bin_size
    ).astype(float)

    grid = temp_long.pivot_table(
        index="Width_m", columns="dist_bin", values="Temperature", aggfunc="mean"
    ).sort_index(axis=0).sort_index(axis=1)

    # ===== per-bin coordinate/time metadata (for hover) =====
    bins = grid.columns.to_list()

    def _nearest_meta_for_bins(df_src: pd.DataFrame, bins_list):
        dist_series = pd.to_numeric(df_src["Moving distance"], errors="coerce")
        lat = pd.to_numeric(df_src.get("latitude"), errors="coerce")
        lon = pd.to_numeric(df_src.get("longitude"), errors="coerce")
        east = pd.to_numeric(df_src.get("easting"), errors="coerce")
        north = pd.to_numeric(df_src.get("northing"), errors="coerce")
        t = robust_datetime(df_src.get("Time"))

        rows = []
        for d in bins_list:
            idx = (dist_series - float(d)).abs().idxmin()
            rows.append({
                "dist_bin": float(d),
                "lat": (lat.iloc[idx] if lat is not None and pd.notna(lat.iloc[idx]) else np.nan),
                "lon": (lon.iloc[idx] if lon is not None and pd.notna(lon.iloc[idx]) else np.nan),
                "easting": (east.iloc[idx] if east is not None and pd.notna(east.iloc[idx]) else np.nan),
                "northing": (north.iloc[idx] if north is not None and pd.notna(north.iloc[idx]) else np.nan),
                "timestr": (t.iloc[idx].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(t.iloc[idx]) else "")
            })
        return pd.DataFrame(rows).set_index("dist_bin")

    bin_meta = _nearest_meta_for_bins(df, bins)

    if coord_type == "latlon":
        coord_label_short = "lat/lon"
        label1, label2 = "Latitude", "Longitude"
        coord1_vec = bin_meta["lat"].to_numpy()
        coord2_vec = bin_meta["lon"].to_numpy()
    elif coord_type == "eastnorth":
        coord_label_short = "easting/northing"
        label1, label2 = "Easting", "Northing"
        coord1_vec = bin_meta["easting"].to_numpy()
        coord2_vec = bin_meta["northing"].to_numpy()
    else:
        coord_label_short = "no coordinates"
        label1, label2 = "Coord1", "Coord2"
        coord1_vec = np.full(len(bins), np.nan)
        coord2_vec = np.full(len(bins), np.nan)

    times_vec = bin_meta["timestr"].astype(str).to_numpy()

    # Broadcast to grid shape
    n_rows, n_cols = grid.shape
    coord1_2d = np.tile(coord1_vec, (n_rows, 1))
    coord2_2d = np.tile(coord2_vec, (n_rows, 1))
    times_2d  = np.tile(times_vec,  (n_rows, 1))
    customdata_full = np.dstack([coord1_2d, coord2_2d, times_2d])   # for the main heatmap (with time)
    customdata_coords_only = np.dstack([coord1_2d, coord2_2d])      # for cold/risk maps (no time)

    hover_tpl_main = (
        "Distance: %{x:.2f} m<br>"
        "Width: %{y:.2f} m<br>"
        "Temp: %{z:.1f} °C<br>"
        f"{label1}: %{{customdata[0]}}<br>"
        f"{label2}: %{{customdata[1]}}<br>"
        "Time: %{customdata[2]}<extra></extra>"
    )
    hover_tpl_no_time = (
        "Distance: %{x:.2f} m<br>"
        "Width: %{y:.2f} m<br>"
        "Temp: %{z:.1f} °C<br>"
        f"{label1}: %{{customdata[0]}}<br>"
        f"{label2}: %{{customdata[1]}}<extra></extra>"
    )
    # ===== END metadata/hover =====

    # ---- Plotly heatmap (full temperatures)
    title = f"Paving Temperature Heatmap — Stops: {stop_count} | Total idle: {total_idle:.1f}s"
    fig = px.imshow(
        grid.values,
        x=grid.columns,
        y=grid.index,
        origin="lower",
        aspect="auto",
        color_continuous_scale="Turbo",
        labels={"x": "Moving distance (m)", "y": "Width (m)", "color": "Temperature (°C)"},
        title=title
    )
    # main heatmap keeps time in hover
    fig.update_traces(customdata=customdata_full, hovertemplate=hover_tpl_main)
    fig.update_layout(title_text=title + f" — Coords: {coord_label_short}")

    # Shade idle intervals
    if stop_count > 0 and len(grid.columns) > 0:
        col_vals = np.array(grid.columns, dtype=float)
        y_max = float(np.max(grid.index))
        for _, r in stops.iterrows():
            d0, d1 = r["moving_dist_start"], r["moving_dist_end"]
            dmid = r["moving_dist_mid"]
            dur_s = r["duration_s"]
            if not pd.notna(d0) or not pd.notna(d1):
                if pd.notna(dmid):
                    d0 = dmid - 0.5 * bin_size
                    d1 = dmid + 0.5 * bin_size
                else:
                    continue
            if d1 < d0:
                d0, d1 = d1, d0
            x0 = float(col_vals[np.abs(col_vals - float(d0)).argmin()])
            x1 = float(col_vals[np.abs(col_vals - float(d1)).argmin()])
            if x1 == x0:
                x1 = x0 + bin_size * 0.5
            fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="black", opacity=0.18)
            x_anno = (x0 + x1) / 2.0
            fig.add_annotation(x=x_anno, y=y_max, text=f"{dur_s:.0f}s", showarrow=False, yshift=10, xref="x", yref="y")

    st.plotly_chart(fig, use_container_width=True)

    # ---- Cold-risk Heatmap (T < cold_thr) ----
    st.subheader("Cold-risk Heatmap")
    st.caption(
       f"Cells below the threshold (**{cold_thr} °C**) are highlighted. "
       "Cold zones cool too quickly and may not achieve target density."
    )
    cold_only = grid.where(grid < float(cold_thr), np.nan)

    if np.isnan(cold_only.to_numpy()).all():
        st.info("No cells fall below the selected cold-risk threshold.")
    else:
        fig_cold = px.imshow(
            cold_only.values,
            x=cold_only.columns,
            y=cold_only.index,
            origin="lower",
            aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": "Moving distance (m)", "y": "Width (m)", "color": "Temperature (°C)"},
            title=f"Cold-risk Heatmap (T < {cold_thr} °C)"
        )
        # No time shown in hover for cold-risk
        fig_cold.update_traces(customdata=customdata_coords_only, hovertemplate=hover_tpl_no_time)
        fig_cold.update_layout(title_text=f"Cold-risk Heatmap (T < {cold_thr} °C) — Coords: {coord_label_short}")

        # Optional: show stop bands
        if stop_count > 0 and len(cold_only.columns) > 0:
            col_vals = np.array(cold_only.columns, dtype=float)
            for _, r in stops.iterrows():
                d0, d1 = r["moving_dist_start"], r["moving_dist_end"]
                dmid = r["moving_dist_mid"]
                if not pd.notna(d0) or not pd.notna(d1):
                    if pd.notna(dmid):
                        d0 = dmid - 0.5 * bin_size
                        d1 = dmid + 0.5 * bin_size
                    else:
                        continue
                if d1 < d0:
                    d0, d1 = d1, d0
                x0 = float(col_vals[np.abs(col_vals - float(d0)).argmin()])
                x1 = float(col_vals[np.abs(col_vals - float(d1)).argmin()])
                if x1 == x0:
                    x1 = x0 + bin_size * 0.5
                fig_cold.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="black", opacity=0.15)
        st.plotly_chart(fig_cold, use_container_width=True)

    # ==== Risk Area Heatmap (T < risk_pct% of overall average) ====
    st.subheader("Risk Area Heatmap (Relative to Average)")

    risk_only = pd.DataFrame(index=grid.index, columns=grid.columns, data=np.nan)
    risk_threshold = np.nan

    overall_mean_temp = float(np.nanmean(grid.values)) if grid.size > 0 else np.nan
    if np.isnan(overall_mean_temp):
        st.info("Risk map unavailable (no valid temperatures to compute the average).")
    else:
        risk_threshold = (risk_pct / 100.0) * overall_mean_temp
        st.caption(
            f"Highlights areas where the temperature is **< {risk_pct}% of the overall average**. "
            f"Average: {overall_mean_temp:.1f} °C ⇒ Threshold: < {risk_threshold:.1f} °C."
        )
        risk_only = grid.where(grid < risk_threshold, np.nan)
        if np.isnan(risk_only.to_numpy()).all():
            st.info("No risk areas detected (no cells below the selected percentage of the average temperature).")
        else:
            fig_risk = px.imshow(
                risk_only.values,
                x=risk_only.columns,
                y=risk_only.index,
                origin="lower",
                aspect="auto",
                color_continuous_scale="Reds",
                labels={"x": "Moving distance (m)", "y": "Width (m)", "color": "Temperature (°C)"},
                title=f"Risk Area Heatmap (T < {risk_pct}% of Avg ≈ {risk_threshold:.1f} °C)"
            )
            # No time shown in hover for risk map
            fig_risk.update_traces(customdata=customdata_coords_only, hovertemplate=hover_tpl_no_time)
            fig_risk.update_layout(title_text=f"Risk Area Heatmap (T < {risk_pct}% of Avg ≈ {risk_threshold:.1f} °C) — Coords: {coord_label_short}")

            # Optional: stop bands
            if stop_count > 0 and len(risk_only.columns) > 0:
                col_vals = np.array(risk_only.columns, dtype=float)
                for _, r in stops.iterrows():
                    d0, d1 = r["moving_dist_start"], r["moving_dist_end"]
                    dmid = r["moving_dist_mid"]
                    if not pd.notna(d0) or not pd.notna(d1):
                        if pd.notna(dmid):
                            d0 = dmid - 0.5 * bin_size
                            d1 = dmid + 0.5 * bin_size
                        else:
                            continue
                    if d1 < d0:
                        d0, d1 = d1, d0
                    x0 = float(col_vals[np.abs(col_vals - float(d0)).argmin()])
                    x1 = float(col_vals[np.abs(col_vals - float(d1)).argmin()])
                    if x1 == x0:
                        x1 = x0 + bin_size * 0.5
                    fig_risk.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="black", opacity=0.15)
            st.plotly_chart(fig_risk, use_container_width=True)
    # === END Risk Area Heatmap ====

    # ================== TSI CALCULATION ONLY ===================
    df_trimmed = pd.DataFrame(
        grid.T.values,
        columns=grid.index.astype(float),
        index=grid.columns.astype(float)
    ).copy()
    df_trimmed["Moving distance"] = df_trimmed.index.values

    widths_cols = [c for c in df_trimmed.columns if c != "Moving distance"]
    temps = df_trimmed[widths_cols].values

    # Calculate TSI_C = Max Temp - Mean Temp (per moving distance row)
    df_trimmed['TSI_C'] = temps.max(axis=1) - temps.mean(axis=1)

    # Calculate overall average TSI_C
    avg_tsi = df_trimmed['TSI_C'].mean()

    # Show just the numeric result
    st.subheader("Temperature Segregation Index (TSI)")
    st.caption(
        "TSI measures **temperature segregation across the mat at each distance bin**. "
        "Computed as *(Max − Mean)* temperature across width for that section."
    )
    st.write(f"Average TSI (Max Temp - Mean Temp): **{avg_tsi:.2f} °C**")

    # ================== FILTERED DRS (T98.5 − T1) ===================
    min_valid_per_row = 10

    def drs_row_percentiles(arr_1d: np.ndarray) -> float:
        arr = arr_1d.astype(float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.nan
        p_low = np.percentile(arr, 1.0)
        p_high = np.percentile(arr, 98.5)
        return float(p_high - p_low)

    row_valid_counts = df_trimmed[widths_cols].count(axis=1)
    reliable_mask = row_valid_counts >= min_valid_per_row

    row_drs_all = df_trimmed[widths_cols].apply(lambda r: drs_row_percentiles(r.to_numpy()), axis=1)
    row_drs_reliable = row_drs_all[reliable_mask]

    temps_reliable = df_trimmed.loc[reliable_mask, widths_cols].to_numpy().astype(float)
    temps_reliable = temps_reliable[~np.isnan(temps_reliable)]
    if temps_reliable.size > 0:
        T_low = np.percentile(temps_reliable, 1.0)
        T_high = np.percentile(temps_reliable, 98.5)
        DRS_C = float(T_high - T_low)
        DRS_F = DRS_C * 9.0 / 5.0
    else:
        T_low = T_high = DRS_C = DRS_F = np.nan

    summary = pd.DataFrame({
        'Statistic': ['T1 (°C)', 'T98.5 (°C)', 'DRS (°C)', 'DRS (°F)'],
        'Value': [T_low, T_high, DRS_C, DRS_F]
    })
    st.subheader("Differential Range Statistics (Filtered to Reliable Rows)")
    st.caption(
        "DRS is a robust spread metric across the paving width, computed as T98.5 − T1. "
        "Higher DRS indicates stronger temperature segregation."
    )
    st.text(summary.to_string(index=False))

    if row_drs_reliable.dropna().empty:
        st.info("No reliable rows available for DRS histogram.")
    else:
        fig_drs_hist, ax_drs_hist = plt.subplots(figsize=(10, 5))
        counts, bins_hist, _ = ax_drs_hist.hist(
            row_drs_reliable.dropna(),
            bins=30, density=True, alpha=0.6,
            color='skyblue', edgecolor='black'
        )
        try:
            kde = gaussian_kde(row_drs_reliable.dropna())
            x_vals = np.linspace(float(row_drs_reliable.min()), float(row_drs_reliable.max()), 500)
            ax_drs_hist.plot(x_vals, kde(x_vals), linewidth=2, color='darkred', label='Density Curve')
        except Exception:
            pass
        ax_drs_hist.set_title("Histogram of Row-Wise DRS (T98.5 − T1)")
        ax_drs_hist.set_xlabel("Row-wise DRS (°C)")
        ax_drs_hist.set_ylabel("Density")
        ax_drs_hist.legend(loc="upper right")
        st.pyplot(fig_drs_hist)

    # ---- Summary of stops
    if stop_count > 0:
        st.success(f"Detected **{stop_count}** paver stop(s). Total idle time: **{total_idle:.1f} s**, average: **{avg_idle:.1f} s**.")
        with st.expander("Show detected stops table"):
            st.dataframe(
                stops[["start_time","end_time","duration_s","moving_dist_start","moving_dist_end","moving_dist_mid"]]
                .rename(columns={
                    "duration_s":"duration (s)",
                    "moving_dist_start":"dist start (m)",
                    "moving_dist_end":"dist end (m)",
                    "moving_dist_mid":"dist mid (m)",
                })
            )
    # ---- Map Visualization (free OSM version) ----
    df["avg_temp_sel"] = df[selected_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    
    map_df = None
    if has_latlon:
        map_df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    elif has_en:
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(
                pd.to_numeric(df["easting"], errors="coerce").to_numpy(),
                pd.to_numeric(df["northing"], errors="coerce").to_numpy(),
            )
            df["lon"], df["lat"] = lon, lat
            map_df = df
        except Exception as e:
            st.warning(f"Coordinate conversion failed: {e}")
    
    if map_df is not None:
        map_df = map_df.loc[map_df[["lat", "lon"]].notna().all(axis=1), ["lat", "lon", "avg_temp_sel"]].copy()
    
        if not map_df.empty:
            midpoint = (float(np.average(map_df["lat"])), float(np.average(map_df["lon"])))
    
            start_point = map_df.iloc[[0]].copy()
            end_point = map_df.iloc[[-1]].copy()
            start_point["type"] = "Start"
            end_point["type"] = "End"
            start_point["r"], start_point["g"], start_point["b"], start_point["a"] = 0, 255, 0, 220
            end_point["r"], end_point["g"], end_point["b"], end_point["a"] = 255, 0, 0, 220
    
            # Stop markers
            stop_markers = pd.DataFrame(columns=["lat","lon","type","r","g","b","a"])
            if "lat" in df.columns and "lon" in df.columns and stop_count > 0:
                points = []
                for _, row in stops.iterrows():
                    if pd.notna(row["moving_dist_mid"]):
                        idx = (pd.to_numeric(df["Moving distance"], errors="coerce") - row["moving_dist_mid"]).abs().idxmin()
                    else:
                        idx = (df["Time"] - row["start_time"]).abs().idxmin()
                    if pd.notna(df.at[idx, "lat"]) and pd.notna(df.at[idx, "lon"]):
                        points.append({
                            "lat": float(df.at[idx, "lat"]), "lon": float(df.at[idx, "lon"]),
                            "type": "Stop", "r": 0, "g": 0, "b": 255, "a": 220
                        })
                if points:
                    stop_markers = pd.DataFrame(points)
    
            markers_df = pd.concat(
                [start_point.assign(type="Start"), end_point.assign(type="End"), stop_markers],
                ignore_index=True
            )
    
            # Base map = OpenStreetMap (free)
            osm_tiles = pdk.Layer(
                "TileLayer",
                data="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
                min_zoom=0,
                max_zoom=19,
                tile_size=256,
            )
    
            # Temperature heat layer
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_weight="avg_temp_sel",
                radiusPixels=30,
                intensity=1.0,
                aggregation="SUM",
            )
    
            marker_layer = pdk.Layer(
                "ScatterplotLayer",
                data=markers_df,
                get_position='[lon, lat]',
                get_fill_color='[r, g, b, a]',
                get_radius=6,
                pickable=True
            )
    
            deck = pdk.Deck(
                map_style=None,  # no Mapbox
                initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=16, pitch=0),
                layers=[osm_tiles, heatmap_layer, marker_layer],
                tooltip={"html": "<b>{type}</b>", "style": {"backgroundColor": "white", "color": "black"}},
            )
    
            st.subheader("Temperature Heatmap on Map (OpenStreetMap)")
            st.caption("Markers: Start = green, End = red, Stops = blue. Basemap: OpenStreetMap (free).")
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.info("No valid coordinate rows to show on the map.")

    
    
    # # ---- Map Visualization
    # df["avg_temp_sel"] = df[selected_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    # map_df = None
    # if has_latlon:
    #     map_df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    # elif has_en:
    #     try:
    #         from pyproj import Transformer
    #         transformer = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
    #         lon, lat = transformer.transform(
    #             pd.to_numeric(df["easting"], errors="coerce").to_numpy(),
    #             pd.to_numeric(df["northing"], errors="coerce").to_numpy(),
    #         )
    #         df["lon"], df["lat"] = lon, lat
    #         map_df = df
    #     except Exception as e:
    #         st.warning(f"Coordinate conversion failed: {e}")

    # if map_df is not None:
    #     map_df = map_df.loc[map_df[["lat", "lon"]].notna().all(axis=1), ["lat", "lon", "avg_temp_sel"]].copy()
    #     if not map_df.empty:
    #         midpoint = (float(np.average(map_df["lat"])), float(np.average(map_df["lon"])))
    #         start_point = map_df.iloc[[0]].copy()
    #         end_point = map_df.iloc[[-1]].copy()
    #         start_point["type"] = "Start"
    #         end_point["type"] = "End"
    #         start_point["r"], start_point["g"], start_point["b"], start_point["a"] = 0, 255, 0, 220
    #         end_point["r"], end_point["g"], end_point["b"], end_point["a"] = 255, 0, 0, 220

    #         stop_markers = pd.DataFrame(columns=["lat","lon","type","r","g","b","a"])
    #         if "lat" in df.columns and "lon" in df.columns and stop_count > 0:
    #             points = []
    #             for _, row in stops.iterrows():
    #                 if pd.notna(row["moving_dist_mid"]):
    #                     idx = (pd.to_numeric(df["Moving distance"], errors="coerce") - row["moving_dist_mid"]).abs().idxmin()
    #                 else:
    #                     idx = (df["Time"] - row["start_time"]).abs().idxmin()
    #                 if pd.notna(df.at[idx, "lat"]) and pd.notna(df.at[idx, "lon"]):
    #                     points.append({"lat": float(df.at[idx, "lat"]), "lon": float(df.at[idx, "lon"]),
    #                                     "type": "Stop", "r": 0, "g": 0, "b": 255, "a": 220})
    #             if points:
    #                 stop_markers = pd.DataFrame(points)

    #         markers_df = pd.concat([start_point.assign(type="Start"), end_point.assign(type="End"), stop_markers], ignore_index=True)

    #         heatmap_layer = pdk.Layer(
    #             "HeatmapLayer",
    #             data=map_df,
    #             get_position='[lon, lat]',
    #             get_weight="avg_temp_sel",
    #             radiusPixels=30,
    #             intensity=1.0,
    #             aggregation=pdk.types.String("SUM"),
    #         )
    #         marker_layer = pdk.Layer(
    #             "ScatterplotLayer",
    #             data=markers_df,
    #             get_position='[lon, lat]',
    #             get_fill_color='[r, g, b, a]',
    #             get_radius=6,
    #             pickable=True
    #         )
    #         deck = pdk.Deck(
    #             map_style="mapbox://styles/mapbox/satellite-streets-v11",
    #             initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=17, pitch=0),
    #             layers=[heatmap_layer, marker_layer],
    #             tooltip={"html": "<b>{type}</b>", "style": {"backgroundColor": "white", "color": "black"}},
    #         )
    #         st.subheader("Temperature Heatmap on Map (with Stop Markers)")
    #         st.caption("Markers: Start = green, End = red, Stops = blue.")
    #         st.pydeck_chart(deck)
    #     else:
    #         st.info("No valid coordinate rows to show on the map.")
            # ====== EXPORT TO EXCEL ======
    st.subheader("Export Coldspots & Risk Areas")
    st.caption("Download an Excel listing each flagged cell with GPS coordinates and timestamps.")

    # Map a (width, distance bin) cell back to the nearest original row to fetch time/coords
    def _map_cell_to_row(df_src: pd.DataFrame, dist_value: float) -> dict:
        idx = (pd.to_numeric(df_src["Moving distance"], errors="coerce") - dist_value).abs().idxmin()
        row = df_src.loc[idx]
        return {
            "Time": row.get("Time", pd.NaT),
            "Latitude": row.get("latitude", np.nan),
            "Longitude": row.get("longitude", np.nan),
            "Easting": row.get("easting", np.nan),
            "Northing": row.get("northing", np.nan),
        }

    # Flatten masked heatmap cells (non-NaN) into a table
    def _flagged_cells_to_df(masked_grid: pd.DataFrame, label: str) -> pd.DataFrame:
        if not isinstance(masked_grid, pd.DataFrame) or masked_grid.empty:
            return pd.DataFrame(columns=[
                "Type","Width (m)","Distance bin (m)","Temperature (°C)",
                "Time","Latitude","Longitude","Easting","Northing"
            ])
        records = []
        for width, r in masked_grid.iterrows():
            for dist, val in r.items():
                if pd.notna(val):
                    meta = _map_cell_to_row(df, float(dist))
                    records.append({
                        "Type": label,
                        "Width (m)": float(width),
                        "Distance bin (m)": float(dist),
                        "Temperature (°C)": float(val),
                        **meta
                    })
        out = pd.DataFrame(records)
        if not out.empty:
            out = out.sort_values(["Distance bin (m)", "Width (m)"]).reset_index(drop=True)
        return out

    # Build the two tables (safe even if nothing is flagged)
    coldspots_df = _flagged_cells_to_df(cold_only, "Coldspot")
    riskareas_df = _flagged_cells_to_df(risk_only, "Risk area")

    # Create an XLSX in-memory using openpyxl (bundled with Anaconda)
    def _make_excel(cold_df, risk_df, cold_thr_val, risk_thr_val) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            readme = pd.DataFrame({
                "Field": [
                    "Type","Width (m)","Distance bin (m)","Temperature (°C)",
                    "Time","Latitude","Longitude","Easting","Northing",
                    "Cold threshold (°C)","Risk threshold (°C)"
                ],
                "Description": [
                    "Coldspot or Risk area row",
                    "Transverse offset from centerline (m); negative = Left, positive = Right",
                    "Longitudinal bin along paving direction (m)",
                    "Averaged cell temperature falling under the criterion",
                    "Nearest timestamp for the cell",
                    "WGS84 latitude (deg)","WGS84 longitude (deg)",
                    "Projected Easting (if available)","Projected Northing (if available)",
                    f"Cells with T < {cold_thr_val:.1f} °C",
                    (f"Cells with T < {risk_pct}% of overall mean ({risk_thr_val:.1f} °C)"
                     if np.isfinite(risk_thr_val) else "N/A")
                ]
            })
            readme.to_excel(writer, index=False, sheet_name="README")
            (cold_df if not cold_df.empty else pd.DataFrame(columns=readme.columns[:0])
             ).to_excel(writer, index=False, sheet_name="Coldspots")
            (risk_df if not risk_df.empty else pd.DataFrame(columns=readme.columns[:0])
             ).to_excel(writer, index=False, sheet_name="RiskAreas")
        buf.seek(0)
        return buf.read()

    excel_bytes = _make_excel(coldspots_df, riskareas_df, cold_thr, risk_threshold)
    st.download_button(
        label="📥 Export Coldspots & Risk Areas (Excel)",
        data=excel_bytes,
        file_name="Paver_ColdRisk_Export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Please upload an Excel file to proceed.")
    
    
