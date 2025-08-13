
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
st.set_page_config(page_title="Paving Temperature Visualizer", layout="wide")
st.title("SuPave Dashboard")

st.caption(
    "Upload your Excel file in the provided structure. "
    "If latitude/longitude is present, easting/northing will be ignored."
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
    
    # NEW: Cold-risk slicer
    cold_thr = st.sidebar.slider(
        "Cold-risk threshold (°C)",
        min_value=60, max_value=200, value=120, step=1,
        help="Cells below this temperature are considered 'cold risk'."
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

    temp_long["dist_bin"] = (
        np.round(pd.to_numeric(temp_long["Moving distance"], errors="coerce") / bin_size) * bin_size
    ).astype(float)

    grid = temp_long.pivot_table(
        index="Width_m", columns="dist_bin", values="Temperature", aggfunc="mean"
    ).sort_index(axis=0).sort_index(axis=1)

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
    
    # Shade each idle interval (vrect) and annotate with duration
    if stop_count > 0 and len(grid.columns) > 0:
        col_vals = np.array(grid.columns, dtype=float)
        y_min = float(np.min(grid.index))
        y_max = float(np.max(grid.index))

        for _, r in stops.iterrows():
            d0, d1 = r["moving_dist_start"], r["moving_dist_end"]
            dmid = r["moving_dist_mid"]
            dur_s = r["duration_s"]

            # If no start/end, approximate around mid
            if not pd.notna(d0) or not pd.notna(d1):
                if pd.notna(dmid):
                    d0 = dmid - 0.5 * bin_size
                    d1 = dmid + 0.5 * bin_size
                else:
                    continue

            if d1 < d0:
                d0, d1 = d1, d0

            # snap to nearest available bins for cleaner alignment
            x0 = float(col_vals[np.abs(col_vals - float(d0)).argmin()])
            x1 = float(col_vals[np.abs(col_vals - float(d1)).argmin()])
            if x1 == x0:
                x1 = x0 + bin_size * 0.5

            fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="black", opacity=0.18)
            # annotate at mid of band, near the top
            x_anno = (x0 + x1) / 2.0
            fig.add_annotation(
                x=x_anno, y=y_max,
                text=f"{dur_s:.0f}s",
                showarrow=False,
                yshift=10,
                xref="x", yref="y"
            )

    st.plotly_chart(fig, use_container_width=True)

    # ---- NEW: Cold-risk Heatmap (filters cells < cold_thr)
    st.subheader("Cold-risk Heatmap")
    st.caption(f"Cells shown are **below {cold_thr} °C** (based on binned moving distance).")

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
        # Optional: show the same stop bands on the cold map
        if stop_count > 0 and len(cold_only.columns) > 0:
            col_vals = np.array(cold_only.columns, dtype=float)
            y_max = float(np.max(cold_only.index))
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

    # ==== NEW: Risk Area Heatmap (T < 90% of overall average) ====
    # Uses the already-computed 'grid' to derive a global mean temperature,
    # then highlights cells below 90% of that mean.
    st.subheader("Risk Area Heatmap (Relative to Average)")
    overall_mean_temp = float(np.nanmean(grid.values)) if grid.size > 0 else np.nan
    if np.isnan(overall_mean_temp):
        st.info("Risk map unavailable (no valid temperatures to compute the average).")
    else:
        risk_threshold = 0.9 * overall_mean_temp
        st.caption(
            f"Risk = cells where **Temperature < 90% of average**. "
            f"Average: {overall_mean_temp:.1f} °C ⇒ Threshold: < {risk_threshold:.1f} °C."
        )
        risk_only = grid.where(grid < risk_threshold, np.nan)
        if np.isnan(risk_only.to_numpy()).all():
            st.info("No risk areas detected (no cells below 90% of the average temperature).")
        else:
            fig_risk = px.imshow(
                risk_only.values,
                x=risk_only.columns,
                y=risk_only.index,
                origin="lower",
                aspect="auto",
                color_continuous_scale="Reds",
                labels={"x": "Moving distance (m)", "y": "Width (m)", "color": "Temperature (°C)"},
                title=f"Risk Area Heatmap (T < {risk_threshold:.1f} °C)"
            )
            # Optionally overlay stop bands here as well for context
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
    # ==== END Risk Area Heatmap ====
    # ================== Temperature & TSI / DRS PROFILES ===================
    # Build a row-wise view for profiles (rows = Moving distance bins, columns = widths)
    df_trimmed = pd.DataFrame(
        grid.T.values,
        columns=grid.index.astype(float),
        index=grid.columns.astype(float)
    ).copy()
    df_trimmed["Moving distance"] = df_trimmed.index.values

    # X (paving width) and Y (moving distance) axes for pcolormesh
    widths_cols = [c for c in df_trimmed.columns if c != "Moving distance"]
    widths_2 = np.array(widths_cols, dtype=float)
    Y2 = df_trimmed["Moving distance"].to_numpy()

    st.subheader("Temperature & TSI Profiles")
    temps = df_trimmed[widths_cols].values
    df_trimmed['TSI_C'] = temps.max(axis=1) - temps.mean(axis=1)
    avg_tsi = df_trimmed['TSI_C'].mean()
    tsi_cat = ('Low' if avg_tsi <= 5 else
               'Moderate' if avg_tsi <= 20 else 'High')
    fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    pcm4 = ax4.pcolormesh(widths_2, Y2, temps, shading='auto')
    ax4.set_title("Temperature [°C]")
    ax4.set_xlabel("Paving width (m)")
    ax4.set_ylabel("Moving distance (m)")
    fig4.colorbar(pcm4, ax=ax4)
    Z_tsi = np.tile(df_trimmed['TSI_C'].values[:, None], (1, len(widths_2)))
    pcm5 = ax5.pcolormesh(widths_2, Y2, Z_tsi, shading='auto', cmap='inferno')
    ax5.set_title("TSI Profile [°C]")
    ax5.set_xlabel("Paving width (m)")
    fig4.suptitle(f"Average TSI: {avg_tsi:.2f}°C → {tsi_cat} Segregation")
    fig4.colorbar(pcm5, ax=ax5)
    st.pyplot(fig4)

    st.subheader("Differential Range Statistics (DRS)")
    temps_all = temps.flatten()
    T10_C = np.percentile(temps_all, 10)
    T90_C = np.percentile(temps_all, 90)
    DRS_C = T90_C - T10_C
    DRS_F = DRS_C * 9/5 + 32
    drs_severity = ('Low' if DRS_C <= 5 else
                    'Moderate' if DRS_C <= 10 else 'High')
    summary = pd.DataFrame({
        'Statistic': ['T10 (°C)', 'T90 (°C)', 'DRS (°C)', 'DRS (°F)', 'Severity'],
        'Value': [T10_C, T90_C, DRS_C, DRS_F, drs_severity]
    })
    st.text(summary.to_string(index=False))

    st.subheader("Distribution of Differential Range Statistics (Row-Wise)")
    row_drs = df_trimmed[widths_cols].apply(
        lambda row: np.percentile(row, 90) - np.percentile(row, 10), axis=1
    )
    df_trimmed['DRS_row'] = row_drs

    fig_drs_hist, ax_drs_hist = plt.subplots(figsize=(10, 5))
    counts, bins, _ = ax_drs_hist.hist(row_drs, bins=30, color='skyblue', edgecolor='black', density=True, alpha=0.6)
    kde = gaussian_kde(row_drs)
    x_vals = np.linspace(float(row_drs.min()), float(row_drs.max()), 500)
    ax_drs_hist.plot(x_vals, kde(x_vals), color='darkred', linewidth=2, label='Density Curve')
    ax_drs_hist.set_title("Histogram of Row-Wise DRS Values with KDE")
    ax_drs_hist.set_xlabel("DRS per Row (°C)")
    ax_drs_hist.set_ylabel("Density")
    ax_drs_hist.legend()
    st.pyplot(fig_drs_hist)

    st.subheader("Average Temperature Along the Paving Width per Moving Distance")
    df_trimmed["Avg_Temp_Row"] = df_trimmed[widths_cols].mean(axis=1)
    avg_temp_moving = df_trimmed[["Moving distance", "Avg_Temp_Row"]].copy()
    avg_temp_moving = avg_temp_moving.sort_values("Moving distance")
    fig_avg_moving = px.line(
        avg_temp_moving,
        x="Moving distance",
        y="Avg_Temp_Row",
        title="Average Temperature Along Paving Width (per Moving Distance)",
        labels={"Moving distance": "Moving Distance (m)", "Avg_Temp_Row": "Average Temperature (°C)"},
    )
    st.plotly_chart(fig_avg_moving, use_container_width=True)
    # ---- Summary
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
    # else:
    #     st.info("No idle periods detected by the temperature pattern criteria.")

    # ---- Map Visualization 
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

            stop_markers = pd.DataFrame(columns=["lat","lon","type","r","g","b","a"])
            if "lat" in df.columns and "lon" in df.columns and stop_count > 0:
                points = []
                for _, row in stops.iterrows():
                    if pd.notna(row["moving_dist_mid"]):
                        idx = (pd.to_numeric(df["Moving distance"], errors="coerce") - row["moving_dist_mid"]).abs().idxmin()
                    else:
                        idx = (df["Time"] - row["start_time"]).abs().idxmin()
                    if pd.notna(df.at[idx, "lat"]) and pd.notna(df.at[idx, "lon"]):
                        points.append({"lat": float(df.at[idx, "lat"]), "lon": float(df.at[idx, "lon"]),
                                        "type": "Stop", "r": 0, "g": 0, "b": 255, "a": 220})
                if points:
                    stop_markers = pd.DataFrame(points)

            markers_df = pd.concat([start_point.assign(type="Start"), end_point.assign(type="End"), stop_markers], ignore_index=True)

            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_weight="avg_temp_sel",
                radiusPixels=30,
                intensity=1.0,
                aggregation=pdk.types.String("SUM"),
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
                map_style="mapbox://styles/mapbox/satellite-streets-v11",
                initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=17, pitch=0),
                layers=[heatmap_layer, marker_layer],
                tooltip={"html": "<b>{type}</b>", "style": {"backgroundColor": "white", "color": "black"}},
            )
            st.subheader("Temperature Heatmap on Map (with Stop Markers)")
            st.caption("Markers: Start = green, End = red, Stops = blue.")
            st.pydeck_chart(deck)
        else:
            st.info("No valid coordinate rows to show on the map.")
else:
    st.info("Please upload an Excel file to proceed.")
