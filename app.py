import io
import os
import csv
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# DOKŁADNA GEOMETRIA LASERÓW RSP [m] (-1.610m do +1.618m)
# ==========================================
DEFAULT_RSP_X_STR = (
    "-1.610, -1.444, -1.285, -1.145, -1.007, -0.850, -0.700, -0.550, -0.400, "
    "-0.200, 0.000, 0.200, 0.400, 0.550, 0.700, 0.850, 1.003, 1.140, 1.280, "
    "1.440, 1.618"
)

DEFAULT_RSP_X_LIST = [
    -1.610, -1.444, -1.285, -1.145, -1.007, -0.850, -0.700, -0.550, -0.400,
    -0.200, 0.000, 0.200, 0.400, 0.550, 0.700, 0.850, 1.003, 1.140, 1.280,
    1.440, 1.618
]

# ==========================================
# 1. FUNKCJE POMOCNICZE DSN 2019 I KLASYFIKACJI
# ==========================================
def get_dsn_class(h_val):
    if h_val < 11.0: return "A"
    elif h_val < 21.0: return "B"
    elif h_val < 31.0: return "C"
    else: return "D"

def get_dsn_color(klasa):
    if klasa == "A": return "#28a745"
    elif klasa == "B": return "#ffc107"
    elif klasa == "C": return "#fd7e14"
    else: return "#dc3545"

# ==========================================
# 2. PARSERY ZOPTYMALIZOWANE (SUROWE PLIKI, .2025, GPX, CSV, XLSX)
# ==========================================
def parse_native_rsp_file(file_bytes):
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.strip().split("\n")
    sensors, profiles_data, gps_data = [], [], []

    for line in lines:
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts or not parts[0]: continue
        code = parts[0]

        if code.isdigit() and 5201 <= int(code) <= 5221:
            try: sensors.append((int(code), float(parts[-1]) / 1000.0))
            except ValueError: pass
        elif code == "5401":
            try: profiles_data.append([len(profiles_data) + 1] + [float(x) for x in parts[3:24]])
            except (ValueError, IndexError): pass
        elif code == "5280":
            try: gps_data.append({"Km": float(parts[1]), "LAT": float(parts[5]), "LON": float(parts[6]), "ELE": float(parts[7]), "Profil_idx": len(gps_data) + 1})
            except (ValueError, IndexError): pass

    df_prof = None
    if profiles_data:
        cols = ["Profil"] + [f"L{i+1}" for i in range(21)]
        df_prof = pd.DataFrame(profiles_data, columns=cols)

    df_gps = pd.DataFrame(gps_data) if gps_data else None
    x_positions = [s[1] for s in sorted(sensors, key=lambda x: x[0])] if len(sensors) == 21 else None
    return df_prof, df_gps, x_positions

def parse_gpx_cached(file_bytes):
    try:
        tree = ET.ElementTree(ET.fromstring(file_bytes))
        root = tree.getroot()
        for elem in root.iter():
            if "}" in elem.tag: elem.tag = elem.tag.split("}", 1)[1]

        records = []
        for tag in ["rtept", "trkpt", "wpt"]:
            for pt in root.iter(tag):
                try:
                    lat, lon = float(pt.attrib.get("lat")), float(pt.attrib.get("lon"))
                    ele_str = pt.findtext("ele")
                    ele = float(ele_str) if ele_str else 0.0
                    name = pt.findtext("name")
                    clean_km = float(name) if name and name.replace(".", "", 1).replace("-", "").isdigit() else len(records) + 1
                    records.append({"LAT": lat, "LON": lon, "ELE": ele, "Km": clean_km, "Profil_idx": len(records) + 1})
                except (ValueError, TypeError): continue
        return pd.DataFrame(records) if records else None
    except Exception: return None

def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))

def get_file_bytes_and_name(uploaded_file, default_paths):
    if uploaded_file is not None: return uploaded_file.getvalue(), uploaded_file.name
    for path in default_paths:
        if os.path.exists(path):
            with open(path, "rb") as f: return f.read(), os.path.basename(path)
    return None, None

@st.cache_data
def load_rsp_data(file_bytes, file_name):
    ext = file_name.lower().split('.')[-1]
    if ext in ["rsp", "txt"]:
        df_prof, _, _ = parse_native_rsp_file(file_bytes)
        if df_prof is not None: return df_prof

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=";", decimal=",")
        if df.shape[1] < 5:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", decimal=",")
    except Exception:
        df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = df.columns.astype(str).str.strip()
    return df

@st.cache_data
def parse_rsp_gps_cached(file_bytes, file_name):
    ext = file_name.lower().split('.')[-1]
    if ext == "gpx":
        df_gpx = parse_gpx_cached(file_bytes)
        if df_gpx is not None: return df_gpx
    if ext in ["rsp", "txt"]:
        _, df_gps, _ = parse_native_rsp_file(file_bytes)
        if df_gps is not None: return df_gps

    if ext in ["csv", "2025", "2026", "2024", "txt"]:
        try: df = pd.read_csv(io.BytesIO(file_bytes), header=None, sep=None, engine="python")
        except Exception: df = pd.read_csv(io.BytesIO(file_bytes), header=None, sep=";")
    else: df = pd.read_excel(io.BytesIO(file_bytes), header=None)

    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(",", "."), errors="coerce")

    lat_col, lon_col = None, None
    for col in df_clean.columns:
        vals = df_clean[col].dropna()
        if len(vals) > 0:
            m = vals.mean()
            if 48.0 <= m <= 56.0 and lat_col is None: lat_col = col
            elif 13.0 <= m <= 25.0 and lon_col is None: lon_col = col

    if lat_col is None: lat_col = 5
    if lon_col is None: lon_col = 6

    res = pd.DataFrame({"LAT": df_clean[lat_col], "LON": df_clean[lon_col], "Profil_idx": np.arange(1, len(df_clean) + 1)})
    if 1 in df_clean.columns: res["Km"] = df_clean[1]
    return res.dropna(subset=["LAT", "LON"]).reset_index(drop=True)

@st.cache_data
def parse_lcms_raw_cached(file_bytes, file_name):
    ext = file_name.lower().split('.')[-1]

    if b"<KOLC" in file_bytes[:1000] or b"<ODCINEK>" in file_bytes[:1000]:
        tree = ET.ElementTree(ET.fromstring(file_bytes.decode("utf-8", errors="ignore")))
        root = tree.getroot()
        profiles = {}
        prof_idx = 1
        for odc in root.findall(".//ODCINEK"):
            try:
                l_kol_str = odc.findtext("L_KOL")
                p_kol_str = odc.findtext("P_KOL")
                profiles[prof_idx] = {
                    "x": np.array([]), "z": np.array([]),
                    "rut_l": float(l_kol_str) if l_kol_str else 0.0,
                    "rut_r": float(p_kol_str) if p_kol_str else 0.0,
                }
                prof_idx += 1
            except Exception:
                continue
        return profiles

    raw_rows = []
    if ext in ["xlsx", "xls"]:
        df_excel = pd.read_excel(io.BytesIO(file_bytes), header=None)
        raw_rows = [[str(v).replace(",", ".").strip() if pd.notna(v) else "" for v in r] for r in df_excel.values]
    else:
        text_stream = io.StringIO(file_bytes.decode("utf-8", errors="ignore"))
        first_line = text_stream.readline()
        text_stream.seek(0)
        sep = ";" if ";" in first_line else ("," if "," in first_line else "\t")
        reader = csv.reader(text_stream, delimiter=sep)
        for row in reader:
            raw_rows.append([p.strip().replace(",", ".") for p in row])

    data_row_indices = []
    for idx, row in enumerate(raw_rows):
        if len(row) > 1:
            val_a = row[0].replace(".0", "").strip()
            if val_a.isdigit() and len(val_a) > 0:
                try:
                    float(row[1].replace(",", "."))
                    data_row_indices.append(idx)
                except ValueError: pass

    profiles = {}
    prof_idx = 1
    for k in range(0, len(data_row_indices) - 1, 2):
        r_left_h, r_left_d = data_row_indices[k], data_row_indices[k] + 1
        r_right_h, r_right_d = data_row_indices[k + 1], data_row_indices[k + 1] + 1
        if r_right_d >= len(raw_rows): break

        try:
            row_lh, row_ld = raw_rows[r_left_h], raw_rows[r_left_d]
            row_rh, row_rd = raw_rows[r_right_h], raw_rows[r_right_d]

            rut_l = float(row_lh[1]) if len(row_lh) > 1 and row_lh[1] != "" else 0.0
            rut_r = float(row_lh[2]) if len(row_lh) > 2 and row_lh[2] != "" else 0.0

            x_left = [float(v) for v in row_lh[3:] if v != ""]
            z_left = [float(v) for v in row_ld[3:] if v != ""]
            x_right = [float(v) for v in row_rh[3:] if v != ""]
            z_right = [float(v) for v in row_rd[3:] if v != ""]

            min_len_l, min_len_r = min(len(x_left), len(z_left)), min(len(x_right), len(z_right))
            x_full = np.concatenate([np.array(x_left[:min_len_l]), np.array(x_right[:min_len_r])])
            if len(x_full) > 0 and np.max(np.abs(x_full)) > 20: x_full = x_full / 1000.0

            z_full = np.concatenate([np.array(z_left[:min_len_l]), np.array(z_right[:min_len_r])])
            sort_order = np.argsort(x_full)

            profiles[prof_idx] = {"x": x_full[sort_order], "z": z_full[sort_order], "rut_l": rut_l, "rut_r": rut_r}
            prof_idx += 1
        except Exception: continue
    return profiles

@st.cache_data
def parse_lcms_xml(file_bytes):
    tree = ET.ElementTree(ET.fromstring(file_bytes))
    root = tree.getroot()
    records = []
    for elem in root.iter():
        if "}" in elem.tag: elem.tag = elem.tag.split("}", 1)[1]
    for pic in root.findall("PicIndex"):
        station, lat, lon = pic.findtext("Station"), pic.findtext("LAT"), pic.findtext("LON")
        if lat and lon:
            records.append({"Station_m": float(station) if station else 0.0, "LAT": float(lat), "LON": float(lon), "Filename": pic.findtext("Filename") or "", "IDDROGI": pic.findtext("IDDROGI") or ""})
    df_xml = pd.DataFrame(records)
    if not df_xml.empty: df_xml = df_xml.sort_values(by="Station_m").drop_duplicates(subset=["Station_m"]).reset_index(drop=True)
    return df_xml

def smooth_profile(z_vals, window_size=31):
    if len(z_vals) < window_size: return z_vals
    return pd.Series(z_vals).rolling(window=window_size, center=True, min_periods=1).mean().values

def get_upper_hull_pts(x_sec, y_sec):
    pts = sorted(list(zip(x_sec, y_sec)), key=lambda p: p[0])
    upper = []
    for p in pts:
        while len(upper) >= 2:
            o, a, b = upper[-2], upper[-1], p
            if (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]) >= 0: upper.pop()
            else: break
        upper.append(p)
    return upper

def get_upper_hull(x_sec, y_sec):
    upper = get_upper_hull_pts(x_sec, y_sec)
    return np.interp(x_sec, [p[0] for p in upper], [p[1] for p in upper])

@st.cache_data
def compute_all_rsp_ruts_cached(df_rsp, x_vals_tuple):
    col_profil = "Profil" if "Profil" in df_rsp.columns else df_rsp.columns[0]
    laser_cols = [c for c in df_rsp.columns if c != col_profil]

    for col in laser_cols:
        df_rsp[col] = pd.to_numeric(df_rsp[col].astype(str).str.replace(",", "."), errors="coerce")

    prof_ids = df_rsp[col_profil].values
    data_mat = df_rsp[laser_cols].fillna(0.0).values.astype(float)
    x_arr = np.array(x_vals_tuple)

    ruts_l, ruts_r = [], []
    for row in -data_mat:
        min_len = min(len(x_arr), len(row))
        if min_len < 3:
            ruts_l.append(0.0)
            ruts_r.append(0.0)
            continue

        row_trunc, x_trunc = row[:min_len], x_arr[:min_len]
        i_mid_adj, i_max_adj = int(min_len * 0.3), int(min_len * 0.7)
        idx_crown = min_len // 2 if i_mid_adj >= i_max_adj else i_mid_adj + np.argmax(row_trunc[i_mid_adj : i_max_adj + 1])

        x_left, y_left = x_trunc[: idx_crown + 1], row_trunc[: idx_crown + 1]
        x_right, y_right = x_trunc[idx_crown:], row_trunc[idx_crown:]

        ruts_l.append(float(np.max(get_upper_hull(x_left, y_left) - y_left)) if len(y_left) > 0 else 0.0)
        ruts_r.append(float(np.max(get_upper_hull(x_right, y_right) - y_right)) if len(y_right) > 0 else 0.0)

    return pd.DataFrame({"Profil": prof_ids, "RSP_Rut_L": ruts_l, "RSP_Rut_R": ruts_r, "RSP_Rut_Max": np.maximum(ruts_l, ruts_r)})

@st.cache_data
def compute_all_lcms_calc_ruts_cached(lcms_profiles):
    ruts_calc_l, ruts_calc_r = [], []
    prof_list = sorted(lcms_profiles.keys())
    for pid in prof_list:
        pdata = lcms_profiles[pid]
        x_pts, z_pts = pdata["x"], pdata["z"]
        if len(x_pts) < 10:
            ruts_calc_l.append(0.0)
            ruts_calc_r.append(0.0)
            continue

        z_smooth = smooth_profile(z_pts, window_size=31)
        mid_mask = (x_pts >= -0.5) & (x_pts <= 0.5)
        idx_crown = np.where(mid_mask)[0][np.argmax(z_smooth[mid_mask])] if np.any(mid_mask) else np.argmin(np.abs(x_pts))

        x_l, z_l = x_pts[: idx_crown + 1], z_smooth[: idx_crown + 1]
        x_r, z_r = x_pts[idx_crown:], z_smooth[idx_crown:]

        ruts_calc_l.append(float(np.max(get_upper_hull(x_l, z_l) - z_l)) if len(z_l) > 0 else 0.0)
        ruts_calc_r.append(float(np.max(get_upper_hull(x_r, z_r) - z_r)) if len(z_r) > 0 else 0.0)

    return pd.DataFrame({"Profil": prof_list, "LCMS_Calc_Rut_L": ruts_calc_l, "LCMS_Calc_Rut_R": ruts_calc_r, "LCMS_Calc_Rut_Max": np.maximum(ruts_calc_l, ruts_calc_r)})

def auto_align_offset_clean(df_rsp_ruts, lcms_profiles, max_search=150):
    if df_rsp_ruts is None or lcms_profiles is None or len(lcms_profiles) == 0: return 0.0
    rsp_map = dict(zip(df_rsp_ruts["Profil"], df_rsp_ruts["RSP_Rut_Max"]))
    lcms_map = {p: max(data.get("rut_l", 0.0), data.get("rut_r", 0.0)) for p, data in lcms_profiles.items()}
    common_profs = np.arange(int(df_rsp_ruts["Profil"].min()), int(df_rsp_ruts["Profil"].max()) + 1)
    rsp_vec = np.array([rsp_map.get(p, np.nan) for p in common_profs])

    best_offset, max_corr = 0, -1e9
    for delta in range(-max_search, max_search + 1):
        lcms_vec = np.array([lcms_map.get(p + delta, np.nan) for p in common_profs])
        valid = ~np.isnan(rsp_vec) & ~np.isnan(lcms_vec)
        if np.sum(valid) > 10:
            r_sub, l_sub = rsp_vec[valid] - np.mean(rsp_vec[valid]), lcms_vec[valid] - np.mean(lcms_vec[valid])
            denom = np.linalg.norm(r_sub) * np.linalg.norm(l_sub)
            if denom > 1e-9:
                corr = np.sum(r_sub * l_sub) / denom
                if corr > max_corr: max_corr, best_offset = corr, delta
    return best_offset

def auto_align_section(df_rsp_ruts, lcms_profiles, p_start, p_end, max_search=50):
    sub_rsp = df_rsp_ruts[(df_rsp_ruts["Profil"] >= p_start) & (df_rsp_ruts["Profil"] <= p_end)]
    if len(sub_rsp) < 10: return 0.0
    return auto_align_offset_clean(sub_rsp, lcms_profiles, max_search=max_search)

def make_map_trace(lat, lon, mode="lines", marker=None, line=None, name="", text=None):
    kwargs = {"lat": lat, "lon": lon, "mode": mode, "name": name}
    if marker is not None: kwargs["marker"] = marker
    if line is not None:
        clean_line = line.copy()
        clean_line.pop("dash", None)
        kwargs["line"] = clean_line
    if text is not None:
        kwargs["text"] = text
        kwargs["hoverinfo"] = "text"
    if hasattr(go, "Scattermap"): return go.Scattermap(**kwargs)
    elif hasattr(go, "Scattermapbox"): return go.Scattermapbox(**kwargs)
    else: return go.Scattergeo(**kwargs)

# ==========================================
# 3. KREATOR WYKRESÓW BLANDA-ALTMANA I TABEL RPD (GDDKiA)
# ==========================================
def build_bland_altman_fig(rsp_v, lcms_v, title="PRAWA KOLEINA"):
    means, diffs = (rsp_v + lcms_v) / 2.0, lcms_v - rsp_v
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    loa_upper, loa_lower = mean_diff + 1.96 * sd_diff, mean_diff - 1.96 * sd_diff

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=means, y=diffs, mode="markers", name="Pomiary", marker=dict(size=7, color="#ffffcc", line=dict(width=1, color="#333333")), hovertemplate="Średnia: %{x:.2f} mm<br>Różnica (LCMS-RSP): %{y:.2f} mm<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="lightgray")
    fig.add_hline(y=mean_diff, line_width=3, line_color="#0066cc", annotation_text=f"Średnia: {mean_diff:+.2f} mm", annotation_position="right", annotation_font=dict(color="#0066cc", size=13))
    fig.add_hline(y=loa_upper, line_width=2, line_dash="dash", line_color="red", annotation_text=f"+1.96 SD: {loa_upper:+.2f} mm", annotation_position="right", annotation_font=dict(color="red", size=12))
    fig.add_hline(y=loa_lower, line_width=2, line_dash="dash", line_color="red", annotation_text=f"-1.96 SD: {loa_lower:+.2f} mm", annotation_position="right", annotation_font=dict(color="red", size=12))
    fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=18)), xaxis_title="ŚREDNIA (LCMS, RSP) [mm]", yaxis_title="RÓŻNICA (LCMS - RSP) [mm]", height=480, template="plotly_white", hovermode="closest")
    return fig, mean_diff, sd_diff, loa_upper, loa_lower

def build_rpd_table_df(rsp_v, lcms_v):
    mean_v = (rsp_v + lcms_v) / 2.0
    diff_abs = np.abs(lcms_v - rsp_v)
    rpd = np.where(mean_v > 0, (diff_abs / mean_v) * 100.0, 0.0)
    m_low, m_high, n_tot = rpd <= 30.0, rpd > 30.0, len(rsp_v)

    def sub_stat(m_grp, label):
        n_grp = int(np.sum(m_grp))
        if n_grp == 0: return {"zakres RPD": label, "n": 0, "%": "0%", "RSP > LCMS": 0, "Wyniki równe": 0, "LCMS > RSP": 0, "%RSP > LCMS": "0%", "%Wyniki równe": "0%", "%LCMS > RSP": "0%"}
        r_sub, l_sub = np.round(rsp_v[m_grp], 1), np.round(lcms_v[m_grp], 1)
        rsp_gt, eq, lcms_gt = int(np.sum(r_sub > l_sub)), int(np.sum(r_sub == l_sub)), int(np.sum(l_sub > r_sub))
        return {"zakres RPD": label, "n": n_grp, "%": f"{(n_grp / n_tot)*100:.0f}%", "RSP > LCMS": rsp_gt, "Wyniki równe": eq, "LCMS > RSP": lcms_gt, "%RSP > LCMS": f"{(rsp_gt / n_grp)*100:.0f}%", "%Wyniki równe": f"{(eq / n_grp)*100:.0f}%", "%LCMS > RSP": f"{(lcms_gt / n_grp)*100:.0f}%"}

    return pd.DataFrame([sub_stat(m_low, "0-30%"), sub_stat(m_high, ">30%"), {"zakres RPD": "suma", "n": n_tot, "%": "100%", "RSP > LCMS": "-", "Wyniki równe": "-", "LCMS > RSP": "-", "%RSP > LCMS": "-", "%Wyniki równe": "-", "%LCMS > RSP": "-"}])

def compute_50m_dsn_sections_full(df_ruts, step_m=1.0, section_m=50.0):
    profs_per_sec = max(1, int(round(section_m / step_m)))
    results = []
    num_sections = int(np.ceil(len(df_ruts) / profs_per_sec))
    for i in range(num_sections):
        sub = df_ruts.iloc[i * profs_per_sec : (i + 1) * profs_per_sec]
        if len(sub) == 0: continue
        st_start, st_end = sub["Profil"].iloc[0], sub["Profil"].iloc[-1]
        rsp_l, rsp_r = sub["RSP_Rut_L"].values, sub["RSP_Rut_R"].values
        rsp_max = np.maximum(rsp_l, rsp_r)

        e_l_rsp, d_l_rsp = np.mean(rsp_l), np.std(rsp_l, ddof=0)
        hm_l_rsp = int(np.round(e_l_rsp + 2 * d_l_rsp))
        e_r_rsp, d_r_rsp = np.mean(rsp_r), np.std(rsp_r, ddof=0)
        hm_r_rsp = int(np.round(e_r_rsp + 2 * d_r_rsp))
        e_m_rsp, d_m_rsp = np.mean(rsp_max), np.std(rsp_max, ddof=0)
        hm_m_rsp = int(np.round(e_m_rsp + 2 * d_m_rsp))

        rec = {
            "Odcinek": f"{st_start} - {st_end} m",
            "RSP L_KOL E(h)": round(e_l_rsp, 2), "RSP L_KOL Dh": round(d_l_rsp, 2), "RSP L_KOL Hm": hm_l_rsp, "RSP L_KOL Klasa": get_dsn_class(hm_l_rsp),
            "RSP P_KOL E(h)": round(e_r_rsp, 2), "RSP P_KOL Dh": round(d_r_rsp, 2), "RSP P_KOL Hm": hm_r_rsp, "RSP P_KOL Klasa": get_dsn_class(hm_r_rsp),
            "RSP MAX_KOL E(h)": round(e_m_rsp, 2), "RSP MAX_KOL Dh": round(d_m_rsp, 2), "RSP MAX_KOL Hm": hm_m_rsp, "RSP MAX_KOL Klasa": get_dsn_class(hm_m_rsp),
        }

        if "LCMS_Rut_L" in sub.columns:
            lcms_l, lcms_r, lcms_max = sub["LCMS_Rut_L"].values, sub["LCMS_Rut_R"].values, sub["LCMS_Rut_Max"].values
            e_l_lcms = np.nanmean(lcms_l) if not np.all(np.isnan(lcms_l)) else 0.0
            d_l_lcms = np.nanstd(lcms_l, ddof=0) if not np.all(np.isnan(lcms_l)) else 0.0
            hm_l_lcms = int(np.round(e_l_lcms + 2 * d_l_lcms))
            e_r_lcms = np.nanmean(lcms_r) if not np.all(np.isnan(lcms_r)) else 0.0
            d_r_lcms = np.nanstd(lcms_r, ddof=0) if not np.all(np.isnan(lcms_r)) else 0.0
            hm_r_lcms = int(np.round(e_r_lcms + 2 * d_r_lcms))
            e_m_lcms = np.nanmean(lcms_max) if not np.all(np.isnan(lcms_max)) else 0.0
            d_m_lcms = np.nanstd(lcms_max, ddof=0) if not np.all(np.isnan(lcms_max)) else 0.0
            hm_m_lcms = int(np.round(e_m_lcms + 2 * d_m_lcms))

            rec.update({
                "LCMS L_KOL E(h)": round(e_l_lcms, 2), "LCMS L_KOL Dh": round(d_l_lcms, 2), "LCMS L_KOL Hm": hm_l_lcms, "LCMS L_KOL Klasa": get_dsn_class(hm_l_lcms),
                "LCMS P_KOL E(h)": round(e_r_lcms, 2), "LCMS P_KOL Dh": round(d_r_lcms, 2), "LCMS P_KOL Hm": hm_r_lcms, "LCMS P_KOL Klasa": get_dsn_class(hm_r_lcms),
                "LCMS MAX_KOL E(h)": round(e_m_lcms, 2), "LCMS MAX_KOL Dh": round(d_m_lcms, 2), "LCMS MAX_KOL Hm": hm_m_lcms, "LCMS MAX_KOL Klasa": get_dsn_class(hm_m_lcms),
            })
        results.append(rec)
    return pd.DataFrame(results)

def make_50m_bar_chart(df_sec_dsn, param_prefix="P_KOL", title="Prawa Koleina (P_KOL)"):
    fig = go.Figure()
    col_rsp_hm, col_rsp_klasa = f"RSP {param_prefix} Hm", f"RSP {param_prefix} Klasa"
    colors_rsp = [get_dsn_color(k) for k in df_sec_dsn[col_rsp_klasa]]
    fig.add_trace(go.Bar(x=df_sec_dsn["Odcinek"], y=df_sec_dsn[col_rsp_hm], name=f"RSP {title} (Hm)", marker_color=colors_rsp, text=[f"{hm} mm ({k})" for hm, k in zip(df_sec_dsn[col_rsp_hm], df_sec_dsn[col_rsp_klasa])], textposition="auto"))

    col_lcms_hm, col_lcms_klasa = f"LCMS {param_prefix} Hm", f"LCMS {param_prefix} Klasa"
    if col_lcms_hm in df_sec_dsn.columns:
        colors_lcms = [get_dsn_color(k) for k in df_sec_dsn[col_lcms_klasa]]
        fig.add_trace(go.Bar(x=df_sec_dsn["Odcinek"], y=df_sec_dsn[col_lcms_hm], name=f"LCMS {title} (Hm)", marker_color=colors_lcms, opacity=0.7, text=[f"{hm} mm ({k})" for hm, k in zip(df_sec_dsn[col_lcms_hm], df_sec_dsn[col_lcms_klasa])], textposition="auto"))

    fig.add_hline(y=11, line_dash="dash", line_color="#ffc107", annotation_text="Próg B (11mm)")
    fig.add_hline(y=21, line_dash="dash", line_color="#fd7e14", annotation_text="Próg C (21mm)")
    fig.add_hline(y=31, line_dash="dash", line_color="#dc3545", annotation_text="Próg D (31mm)")
    fig.update_layout(title=f"Ocena Hm dla Odcinków 50m – {title}", xaxis_title="Odcinek Pikietażu", yaxis_title="Ocena Hm [mm]", barmode="group", height=480)
    return fig


# ==========================================
# 4. KONFIGURACJA STRONY STREAMLIT
# ==========================================
st.set_page_config(page_title="RSP vs LCMS – System Analizy Kolein DSN 2019", layout="wide")
st.title("⚡ System Analizy RSP + LCMS (DSN 2019 & GDDKiA)")

if "offset_pikitaz_input" not in st.session_state: st.session_state["offset_pikitaz_input"] = 0.0

# ==========================================
# 5. PANEL BOCZNY
# ==========================================
st.sidebar.header("1. Dane RSP (Profilograf / Surowe)")
uploaded_rsp = st.sidebar.file_uploader("Wgraj plik RSP (.rsp / .2025 / .csv / .txt)", type=["rsp", "2025", "2026", "2024", "xlsx", "csv", "txt"], key="rsp_input")
uploaded_rsp_gps = st.sidebar.file_uploader("Wgraj plik GPS RSP (.gpx / .rsp / .2025 / .csv / .txt)", type=["gpx", "rsp", "2025", "2026", "2024", "xlsx", "csv", "txt"], key="rsp_gps_input")

st.sidebar.header("2. Dane LCMS (Surowe CSV / Excel / .2025)")
uploaded_lcms = st.sidebar.file_uploader("Wgraj plik LCMS (.csv / .2025 / .xlsx)", type=["csv", "2025", "2026", "2024", "xlsx"], key="lcms_input")
uploaded_xml = st.sidebar.file_uploader("Wgraj plik XML z LCMS (GPS)", type=["xml"], key="xml_input")

bytes_rsp, name_rsp = get_file_bytes_and_name(uploaded_rsp, ["data/rsp.rsp", "data/rsp.xlsx", "data/rsp.csv", "rsp.rsp", "rsp.xlsx", "rsp.csv"])
bytes_rsp_gps, name_rsp_gps = get_file_bytes_and_name(uploaded_rsp_gps, ["data/rsp_gps.gpx", "data/rsp_gps.csv", "data/rsp_gps.xlsx", "rsp_gps.gpx", "rsp_gps.csv"])
bytes_lcms, name_lcms = get_file_bytes_and_name(uploaded_lcms, ["data/lcms.csv", "data/lcms.xlsx", "lcms.csv", "lcms.xlsx"])
bytes_xml, name_xml = get_file_bytes_and_name(uploaded_xml, ["data/lcms.xml", "lcms.xml"])

auto_x_from_rsp = None
if bytes_rsp is not None and (name_rsp.lower().endswith(".rsp") or name_rsp.lower().endswith(".txt")):
    _, _, auto_x_from_rsp = parse_native_rsp_file(bytes_rsp)

st.sidebar.header("3. Synchronizacja i Pikietaż")

def sync_offset_from_slider(): st.session_state["offset_pikitaz_input"] = float(st.session_state["offset_slider_key"])
def sync_offset_from_input(): st.session_state["offset_slider_key"] = int(round(st.session_state["offset_pikitaz_input"]))

if "offset_slider_key" not in st.session_state: st.session_state["offset_slider_key"] = int(round(st.session_state["offset_pikitaz_input"]))

st.sidebar.slider("🎚️ Główny suwak przesunięcia [profili]:", min_value=-200, max_value=200, value=int(round(st.session_state["offset_pikitaz_input"])), step=1, key="offset_slider_key", on_change=sync_offset_from_slider)
glowny_offset = st.sidebar.number_input("✍️ Wpisz dokładnie [m / profili]:", min_value=-500.0, max_value=500.0, value=float(st.session_state["offset_pikitaz_input"]), step=0.1, format="%.1f", key="offset_pikitaz_input", on_change=sync_offset_from_input)

podziel_na_sekcje = st.sidebar.checkbox("Dziel trasę na sekcje (osobne przesunięcia dla odcinków)", value=False)
sekcja_m = 500
sekcje_offsets = {}

if podziel_na_sekcje:
    sekcja_m = st.sidebar.select_slider("Długość pojedynczej sekcji pikietażu [m]:", options=[250, 500, 1000, 2000, 5000], value=500)
    st.sidebar.markdown("##### Ręczne offsety dla poszczególnych sekcji:")

st.sidebar.header("4. Geometria Belki RSP")
typ_geometrii = st.sidebar.radio("Typ konfiguracji czujników RSP:", ["Wprowadzona w pliku / Domyślna RSP", "Stały odstęp (równomierny)"])

if typ_geometrii == "Wprowadzona w pliku / Domyślna RSP":
    default_x_val = ", ".join(f"{x:.3f}" for x in auto_x_from_rsp) if auto_x_from_rsp else DEFAULT_RSP_X_STR
    x_input_str = st.sidebar.text_area("Pozycje X [m] dla 21 laserów:", value=default_x_val, height=90)
    try: pozycja_x = [float(val.strip().replace(",", ".")) for val in x_input_str.split(",") if val.strip() != ""]
    except Exception: pozycja_x = auto_x_from_rsp or DEFAULT_RSP_X_LIST
else:
    krok_laserow = st.sidebar.number_input("Odstęp [m]:", min_value=0.05, max_value=0.50, value=0.165, step=0.005, format="%.3f")
    pozycja_x = [round((i - 10) * krok_laserow, 3) for i in range(21)]

st.sidebar.header("5. Wybór Źródła Kolein LCMS")
zrodlo_lcms = st.sidebar.radio("Wykresy podłużne i kafelki LCMS bazują na:", ["Z pliku LCMS (surowe B2/C2)", "Wyliczone z profilu (Wirtualna Łata)", "Oba źródła jednocześnie"])

st.sidebar.header("6. Geometria i Dopasowanie Poprzeczne LCMS vs RSP")
zaczep_prawa_krawedz = st.sidebar.checkbox("🔗 Zaczep prawą krawędź LCMS do krawędzi RSP", value=True)
skaluj_szerokosc = st.sidebar.checkbox("📐 Skaluj szerokość LCMS do szerokości belki RSP", value=False)
manual_x_shift_lcms = st.sidebar.number_input("Ręczne przesunięcie poprzeczne X LCMS [m]:", min_value=-2.0, max_value=2.0, value=0.0, step=0.01, format="%.2f")

test_stacjonarny = st.sidebar.checkbox("Tryb testu stacjonarnego (Łata 0)", value=False)
tryb_zero_l11 = st.sidebar.checkbox("Ustaw L11 jako Zero (wysokość względna RSP)", value=True)
pokaz_late_rsp = st.sidebar.checkbox("Pokaż łatę i koleiny RSP", value=True)
wyrownaj_spadek_lcms = st.sidebar.checkbox("Wyzeruj spadek poprzeczny LCMS do wyświetlania", value=True)
pokaz_late_lcms = st.sidebar.checkbox("Pokaż łatę i koleiny LCMS", value=True)

df_rsp_gps, lcms_profiles, df_xml_gps = None, None, None
if bytes_rsp_gps is not None:
    try: df_rsp_gps = parse_rsp_gps_cached(bytes_rsp_gps, name_rsp_gps); st.sidebar.success(f"GPS RSP: {len(df_rsp_gps)} pkt.")
    except Exception as e: st.sidebar.error(f"Błąd GPS RSP: {e}")
if bytes_lcms is not None:
    try: lcms_profiles = parse_lcms_raw_cached(bytes_lcms, name_lcms); st.sidebar.success(f"LCMS: {len(lcms_profiles)} profili.")
    except Exception as e: st.sidebar.error(f"Błąd LCMS: {e}")
if bytes_xml is not None:
    try: df_xml_gps = parse_lcms_xml(bytes_xml); st.sidebar.success(f"GPS LCMS: {len(df_xml_gps)} pkt.")
    except Exception as e: st.sidebar.error(f"Błąd XML: {e}")

# ==========================================
# 6. GŁÓWNA LOGIKA APLIKACJI Z TABAMI
# ==========================================
if bytes_rsp is not None:
    df_rsp = load_rsp_data(bytes_rsp, name_rsp)
    col_profil = "Profil" if "Profil" in df_rsp.columns else df_rsp.columns[0]

    df_rsp[col_profil] = pd.to_numeric(df_rsp[col_profil].astype(str).str.replace(",", "."), errors="coerce")
    df_rsp = df_rsp.dropna(subset=[col_profil])

    if df_rsp.empty:
        st.error("❌ Wczytany plik RSP nie zawiera prawidłowych numerów profili.")
        st.stop()

    min_p, max_p = int(df_rsp[col_profil].min()), int(df_rsp[col_profil].max())

    if podziel_na_sekcje:
        num_sekcji = int(np.ceil((max_p - min_p + 1) / sekcja_m))
        for s_i in range(num_sekcji):
            p_start_s = min_p + s_i * sekcja_m
            p_end_s = min(max_p, p_start_s + sekcja_m - 1)
            key_sec = f"sec_offset_{s_i}_{sekcja_m}"
            if key_sec not in st.session_state: st.session_state[key_sec] = 0.0
            sekcje_offsets[s_i] = st.sidebar.number_input(f"Sekcja {s_i+1} ({p_start_s}-{p_end_s}m) [m]:", min_value=-200.0, max_value=200.0, value=float(st.session_state[key_sec]), step=0.1, format="%.1f", key=key_sec)

    def get_offset_for_profil(p_rsp):
        if not podziel_na_sekcje: return glowny_offset
        return glowny_offset + sekcje_offsets.get(int((p_rsp - min_p) // sekcja_m), 0.0)

    st.sidebar.header("7. Wybór profilu")
    wybrany_profil = st.sidebar.number_input(f"Numer profilu RSP [{min_p} - {max_p}]:", min_value=min_p, max_value=max_p, value=min_p, step=1)

    # Mechanizm przyciągania do najbliższego dostępnego profilu!
    wiersz_rsp = df_rsp[df_rsp[col_profil] == wybrany_profil]
    if wiersz_rsp.empty and not df_rsp.empty:
        closest_idx = (df_rsp[col_profil] - wybrany_profil).abs().idxmin()
        wiersz_rsp = df_rsp.loc[[closest_idx]]
        wybrany_profil = int(wiersz_rsp[col_profil].values[0])
        st.warning(f"⚠️ Zmieniono suwak na najbliższy dostępny wpis w pliku: profil nr {wybrany_profil}.")

    całkowity_offset = get_offset_for_profil(wybrany_profil)
    wybrany_profil_lcms = int(round(wybrany_profil + całkowity_offset))

    df_all_rsp_ruts = compute_all_rsp_ruts_cached(df_rsp, tuple(pozycja_x))

    if lcms_profiles is not None and len(lcms_profiles) > 0:
        df_lcms_calc_all = compute_all_lcms_calc_ruts_cached(lcms_profiles)
        calc_dict_l = dict(zip(df_lcms_calc_all["Profil"], df_lcms_calc_all["LCMS_Calc_Rut_L"]))
        calc_dict_r = dict(zip(df_lcms_calc_all["Profil"], df_lcms_calc_all["LCMS_Calc_Rut_R"]))

        lcms_l_mapped, lcms_r_mapped, lcms_max_mapped, lcms_target_p = [], [], [], []
        for p_rsp in df_all_rsp_ruts["Profil"].values:
            p_lcms = int(round(p_rsp + get_offset_for_profil(p_rsp)))
            lcms_target_p.append(p_lcms)
            if p_lcms in lcms_profiles:
                if zrodlo_lcms == "Z pliku LCMS (surowe B2/C2)": rl, rr = lcms_profiles[p_lcms]["rut_l"], lcms_profiles[p_lcms]["rut_r"]
                elif zrodlo_lcms == "Wyliczone z profilu (Wirtualna Łata)": rl, rr = calc_dict_l.get(p_lcms, np.nan), calc_dict_r.get(p_lcms, np.nan)
                else: rl, rr = lcms_profiles[p_lcms]["rut_l"], lcms_profiles[p_lcms]["rut_r"]
                rmax = max(rl, rr) if pd.notna(rl) and pd.notna(rr) else np.nan
            else: rl, rr, rmax = np.nan, np.nan, np.nan
            lcms_l_mapped.append(rl); lcms_r_mapped.append(rr); lcms_max_mapped.append(rmax)

        df_all_rsp_ruts["LCMS_Rut_L"] = lcms_l_mapped
        df_all_rsp_ruts["LCMS_Rut_R"] = lcms_r_mapped
        df_all_rsp_ruts["LCMS_Rut_Max"] = lcms_max_mapped
        df_all_rsp_ruts["LCMS_Target_Profil"] = lcms_target_p
        df_merged_ruts = df_all_rsp_ruts.dropna(subset=["LCMS_Rut_L", "LCMS_Rut_R"]).copy()
    else:
        df_merged_ruts = df_all_rsp_ruts.copy()

    if lcms_profiles is not None:
        if st.sidebar.button("⚡ Automatyczna Synchronizacja Pikietażu"):
            if podziel_na_sekcje:
                num_sekcji = int(np.ceil((max_p - min_p + 1) / sekcja_m))
                for s_i in range(num_sekcji):
                    p_start_s = min_p + s_i * sekcja_m
                    p_end_s = min(max_p, p_start_s + sekcja_m - 1)
                    sug_sec_off = auto_align_section(df_all_rsp_ruts, lcms_profiles, p_start_s, p_end_s, max_search=100)
                    st.session_state[f"sec_offset_{s_i}_{sekcja_m}"] = float(sug_sec_off)
                st.sidebar.success("✅ Dopasowano przesunięcia indywidualnie dla wszystkich sekcji!")
            else:
                sug_offset = auto_align_offset_clean(df_all_rsp_ruts, lcms_profiles, max_search=150)
                st.session_state["offset_pikitaz_input"], st.session_state["offset_slider_key"] = float(sug_offset), int(round(sug_offset))
                st.sidebar.success(f"✅ Optymalne przesunięcie pikietażu wynosi: {sug_offset:+} m")
            st.rerun()

    tab_main, tab_dsn50, tab_ba_gddkia = st.tabs(["📐 Przekrój Poprzeczny & Cała Trasa", "📊 Ocena 50m DSN 2019 (Klasy A-D)", "📈 Analiza Porównawcza GDDKiA (Bland-Altman & RPD)"])

    if not wiersz_rsp.empty:
        kolumny_laserow = [c for c in df_rsp.columns if c != col_profil]
        # Bezpieczne parsowanie danych wybranego profilu
        odczyty_raw = pd.to_numeric(wiersz_rsp.iloc[0][kolumny_laserow], errors='coerce').fillna(0).values.astype(float)
        ile_laserow = len(odczyty_raw)

        x_vals = pozycja_x if len(pozycja_x) == ile_laserow else [round((i - (ile_laserow - 1) / 2) * 0.165, 3) for i in range(ile_laserow)]

        if tryb_zero_l11:
            idx_srodek = ile_laserow // 2
            y_vals = odczyty_raw[idx_srodek] - odczyty_raw
            y_label = "Wysokość względna do L11 [mm]"
        else:
            y_vals = odczyty_raw
            y_label = "Odczyt surowy czujnika [mm]"

        i_min_mid, i_max_mid = int(ile_laserow * 0.3), int(ile_laserow * 0.7)
        idx_crown = i_min_mid + np.argmax(y_vals[i_min_mid : i_max_mid + 1])

        x_left, y_left = np.array(x_vals[: idx_crown + 1]), np.array(y_vals[: idx_crown + 1])
        x_right, y_right = np.array(x_vals[idx_crown:]), np.array(y_vals[idx_crown:])

        lata_lewa_y, lata_prawa_y = get_upper_hull(x_left, y_left), get_upper_hull(x_right, y_right)

        diff_l = lata_lewa_y - y_left
        idx_kol_l = np.argmax(diff_l)
        koleina_lewa_mm = diff_l[idx_kol_l]

        diff_p = lata_prawa_y - y_right
        idx_kol_p_rel = np.argmax(diff_p)
        idx_kol_p = idx_crown + idx_kol_p_rel
        koleina_prawa_mm = diff_p[idx_kol_p_rel]

        max_koleina_rsp = max(koleina_lewa_mm, koleina_prawa_mm)

        kol_lcms_l, kol_lcms_p, prof_lcms_data = None, None, None
        if lcms_profiles and wybrany_profil_lcms in lcms_profiles:
            prof_lcms_data = lcms_profiles[wybrany_profil_lcms]
            kol_lcms_l, kol_lcms_p = prof_lcms_data.get("rut_l", None), prof_lcms_data.get("rut_r", None)

        calc_lcms_rut_l, calc_lcms_rut_r = None, None
        if prof_lcms_data is not None:
            x_lcms_orig, z_lcms_raw = prof_lcms_data["x"].copy(), prof_lcms_data["z"]
            z_lcms_smooth = smooth_profile(z_lcms_raw, window_size=31)

            mid_mask_raw = (x_lcms_orig >= -0.5) & (x_lcms_orig <= 0.5)
            idx_c_lcms = np.where(mid_mask_raw)[0][np.argmax(z_lcms_smooth[mid_mask_raw])] if np.any(mid_mask_raw) else np.argmin(np.abs(x_lcms_orig))

            x_lcms_l_raw, z_lcms_l_raw = x_lcms_orig[: idx_c_lcms + 1], z_lcms_smooth[: idx_c_lcms + 1]
            x_lcms_r_raw, z_lcms_r_raw = x_lcms_orig[idx_c_lcms:], z_lcms_smooth[idx_c_lcms:]

            lata_raw_l, lata_raw_r = get_upper_hull(x_lcms_l_raw, z_lcms_l_raw), get_upper_hull(x_lcms_r_raw, z_lcms_r_raw)
            calc_lcms_rut_l = float(np.max(lata_raw_l - z_lcms_l_raw)) if len(z_lcms_l_raw) > 0 else 0.0
            calc_lcms_rut_r = float(np.max(lata_raw_r - z_lcms_r_raw)) if len(z_lcms_r_raw) > 0 else 0.0

        # --- ZAKŁADKA 1: PRZEKRÓJ POPRZECZNY I CAŁA TRASA ---
        with tab_main:
            if test_stacjonarny:
                st.subheader("🧪 Weryfikacja Testu Stacjonarnego (DSN 2019)")
                if max_koleina_rsp <= 0.5: st.success(f"✅ **TEST POZYTYWNY**: Odchyłka wynosi **{max_koleina_rsp:.2f} mm** (≤ 0,5 mm).")
                else: st.error(f"❌ **TEST NEGATYWNY**: Odchyłka wynosi **{max_koleina_rsp:.2f} mm** i przekracza limit 0,5 mm dla łaty!")

            st.subheader(f"Profil RSP nr {wybrany_profil} (LCMS nr {wybrany_profil_lcms}, przesunięcie: {całkowity_offset:+.1f} m)")

            st.markdown("##### 🟦 Dane z Profilografu RSP")
            rc1, rc2, rc3 = st.columns(3)
            klasa_rsp_l, klasa_rsp_p, klasa_rsp_m = get_dsn_class(koleina_lewa_mm), get_dsn_class(koleina_prawa_mm), get_dsn_class(max_koleina_rsp)
            rc1.metric("RSP Koleina Lewa", f"{koleina_lewa_mm:.1f} mm (Klasa {klasa_rsp_l})", f"Laser L{idx_kol_l+1} ({x_vals[idx_kol_l]:.2f}m)")
            rc2.metric("RSP Koleina Prawa", f"{koleina_prawa_mm:.1f} mm (Klasa {klasa_rsp_p})", f"Laser L{idx_kol_p+1} ({x_vals[idx_kol_p]:.2f}m)")
            rc3.metric("RSP Koleina Maksymalna", f"{max_koleina_rsp:.1f} mm (Klasa {klasa_rsp_m})")

            st.markdown("##### 🟩 Dane z Systemu LCMS")
            lc1, lc2, lc3 = st.columns(3)
            if kol_lcms_l is not None and kol_lcms_p is not None:
                if zrodlo_lcms == "Z pliku LCMS (surowe B2/C2)": disp_l, disp_p, label_sub = kol_lcms_l, kol_lcms_p, "(z pliku B2/C2)"
                elif zrodlo_lcms == "Wyliczone z profilu (Wirtualna Łata)": disp_l, disp_p, label_sub = calc_lcms_rut_l, calc_lcms_rut_r, "(z Łaty DSN)"
                else: disp_l, disp_p, label_sub = kol_lcms_l, kol_lcms_p, f"(Plik: {kol_lcms_l:.1f} / Łata: {calc_lcms_rut_l:.1f})"

                max_disp = max(disp_l, disp_p)
                klasa_lcms_l, klasa_lcms_p, klasa_lcms_m = get_dsn_class(disp_l), get_dsn_class(disp_p), get_dsn_class(max_disp)

                lc1.metric(f"LCMS Koleina Lewa {label_sub}", f"{disp_l:.1f} mm (Klasa {klasa_lcms_l})", delta=f"{koleina_lewa_mm - disp_l:+.1f} mm vs RSP")
                lc2.metric(f"LCMS Koleina Prawa {label_sub}", f"{disp_p:.1f} mm (Klasa {klasa_lcms_p})", delta=f"{koleina_prawa_mm - disp_p:+.1f} mm vs RSP")
                lc3.metric("LCMS Koleina Maksymalna", f"{max_disp:.1f} mm (Klasa {klasa_lcms_m})", delta=f"{max_koleina_rsp - max_disp:+.1f} mm vs RSP")
            else:
                lc1.info("Brak profilu LCMS dla wybranego pikietażu.")

            # WYKRES PRZEKROJU POPRZECZNEGO
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name="Profil RSP (Spline)", line=dict(shape="spline", smoothing=1.2, color="royalblue", width=3), marker=dict(size=7, color="navy"), hovertemplate=("Poz X: %{x} m<br>Wys Y: %{y:.1f} mm<extra></extra>")))

            if pokaz_late_rsp:
                fig.add_trace(go.Scatter(x=x_left, y=lata_lewa_y, mode="lines", name="Łata RSP Lewa", line=dict(color="crimson", width=2)))
                fig.add_trace(go.Scatter(x=x_right, y=lata_prawa_y, mode="lines", name="Łata RSP Prawa", line=dict(color="darkorange", width=2)))
                x_kl, y_kl_prof, y_kl_lata = x_vals[idx_kol_l], y_vals[idx_kol_l], lata_lewa_y[idx_kol_l]
                fig.add_trace(go.Scatter(x=[x_kl, x_kl], y=[y_kl_lata, y_kl_prof], mode="lines+markers", name=f"Głębokość RSP Lewa ({koleina_lewa_mm:.1f} mm)", line=dict(color="crimson", width=3), marker=dict(size=6, symbol="triangle-down")))
                fig.add_annotation(x=x_kl, y=(y_kl_lata + y_kl_prof) / 2, text=f" <b>RSP: {koleina_lewa_mm:.1f} mm</b>", showarrow=False, font=dict(color="crimson", size=12), bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="crimson", borderwidth=1, xanchor="right")
                x_kp, y_kp_prof, y_kp_lata = x_vals[idx_kol_p], y_vals[idx_kol_p], lata_prawa_y[idx_kol_p_rel]
                fig.add_trace(go.Scatter(x=[x_kp, x_kp], y=[y_kp_lata, y_kp_prof], mode="lines+markers", name=f"Głębokość RSP Prawa ({koleina_prawa_mm:.1f} mm)", line=dict(color="darkorange", width=3), marker=dict(size=6, symbol="triangle-down")))
                fig.add_annotation(x=x_kp, y=(y_kp_prof + y_kp_lata) / 2, text=f" <b>RSP: {koleina_prawa_mm:.1f} mm</b>", showarrow=False, font=dict(color="darkorange", size=12), bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="darkorange", borderwidth=1, xanchor="right")

            if prof_lcms_data is not None:
                x_lcms_orig, z_lcms_raw = prof_lcms_data["x"].copy(), prof_lcms_data["z"]
                z_lcms_smooth = smooth_profile(z_lcms_raw, window_size=31)
                rsp_x_max, rsp_x_min, x_lcms = np.max(x_vals), np.min(x_vals), x_lcms_orig.copy()

                if skaluj_szerokosc and len(x_lcms) > 1:
                    lcms_min, lcms_max = np.min(x_lcms), np.max(x_lcms)
                    x_lcms = rsp_x_min + (x_lcms - lcms_min) * ((rsp_x_max - rsp_x_min) / (lcms_max - lcms_min))
                elif zaczep_prawa_krawedz and len(x_lcms) > 0:
                    x_lcms = x_lcms + (rsp_x_max - np.max(x_lcms))

                x_lcms = x_lcms + manual_x_shift_lcms
                mid_mask_raw = (x_lcms >= -0.5) & (x_lcms <= 0.5)
                idx_crown_lcms = np.where(mid_mask_raw)[0][np.argmax(z_lcms_smooth[mid_mask_raw])] if np.any(mid_mask_raw) else np.argmin(np.abs(x_lcms))

                x_lcms_l, z_lcms_l_raw = x_lcms[: idx_crown_lcms + 1], z_lcms_smooth[: idx_crown_lcms + 1]
                x_lcms_r, z_lcms_r_raw = x_lcms[idx_crown_lcms:], z_lcms_smooth[idx_crown_lcms:]
                lata_raw_l, lata_raw_r = get_upper_hull(x_lcms_l, z_lcms_l_raw), get_upper_hull(x_lcms_r, z_lcms_r_raw)

                if wyrownaj_spadek_lcms and len(x_lcms) > 10:
                    p_trend = np.polyfit(x_lcms, z_lcms_smooth, 1)
                    trend_line = p_trend[0] * x_lcms
                    z_lcms_leveled = z_lcms_smooth - trend_line
                    lata_plot_l, lata_plot_r = lata_raw_l - trend_line[: idx_crown_lcms + 1], lata_raw_r - trend_line[idx_crown_lcms:]
                else:
                    z_lcms_leveled, lata_plot_l, lata_plot_r = z_lcms_smooth, lata_raw_l, lata_raw_r

                z_ref_lcms = z_lcms_leveled[np.argmin(np.abs(x_lcms))] if len(x_lcms) > 0 else 0
                z_lcms_plot = z_lcms_leveled - z_ref_lcms
                lata_plot_l_zero, lata_plot_r_zero = lata_plot_l - z_ref_lcms, lata_plot_r - z_ref_lcms

                fig.add_trace(go.Scatter(x=x_lcms, y=z_lcms_plot, mode="lines", name=f"Profil LCMS nr {wybrany_profil_lcms}", line=dict(color="forestgreen", width=2)))

                if pokaz_late_lcms and len(x_lcms) > 0:
                    fig.add_trace(go.Scatter(x=x_lcms_l, y=lata_plot_l_zero, mode="lines", name="Łata LCMS Lewa", line=dict(color="limegreen", width=2)))
                    fig.add_trace(go.Scatter(x=x_lcms_r, y=lata_plot_r_zero, mode="lines", name="Łata LCMS Prawa", line=dict(color="lime", width=2)))

                    val_l_show = kol_lcms_l if zrodlo_lcms == "Z pliku LCMS (surowe B2/C2)" and kol_lcms_l is not None else calc_lcms_rut_l
                    val_r_show = kol_lcms_p if zrodlo_lcms == "Z pliku LCMS (surowe B2/C2)" and kol_lcms_p is not None else calc_lcms_rut_r

                    diff_lcms_l = lata_plot_l_zero - z_lcms_plot[: idx_crown_lcms + 1]
                    if len(diff_lcms_l) > 0:
                        idx_k_l_c = np.argmax(diff_lcms_l)
                        x_kl_c, y_kl_c_prof, y_kl_c_lata = x_lcms_l[idx_k_l_c], z_lcms_plot[idx_k_l_c], lata_plot_l_zero[idx_k_l_c]
                        fig.add_trace(go.Scatter(x=[x_kl_c, x_kl_c], y=[y_kl_c_lata, y_kl_c_prof], mode="lines+markers", name=f"Głębokość LCMS Lewa ({val_l_show:.1f} mm)", line=dict(color="forestgreen", width=2.5), marker=dict(size=5, symbol="triangle-down")))
                        fig.add_annotation(x=x_kl_c, y=(y_kl_c_lata + y_kl_c_prof) / 2, text=f"<b>LCMS: {val_l_show:.1f} mm</b> ", showarrow=False, font=dict(color="forestgreen", size=12), bgcolor="rgba(235, 255, 235, 0.95)", bordercolor="forestgreen", borderwidth=1, xanchor="left")

                    diff_lcms_r = lata_plot_r_zero - z_lcms_plot[idx_crown_lcms:]
                    if len(diff_lcms_r) > 0:
                        idx_k_r_c_rel = np.argmax(diff_lcms_r)
                        idx_k_r_c = idx_crown_lcms + idx_k_r_c_rel
                        x_kp_c, y_kp_c_prof, y_kp_c_lata = x_lcms[idx_k_r_c], z_lcms_plot[idx_k_r_c], lata_plot_r_zero[idx_k_r_c_rel]
                        fig.add_trace(go.Scatter(x=[x_kp_c, x_kp_c], y=[y_kp_c_lata, y_kp_c_prof], mode="lines+markers", name=f"Głębokość LCMS Prawa ({val_r_show:.1f} mm)", line=dict(color="forestgreen", width=2.5), marker=dict(size=5, symbol="triangle-down")))
                        fig.add_annotation(x=x_kp_c, y=(y_kp_c_lata + y_kp_c_prof) / 2, text=f"<b>LCMS: {val_r_show:.1f} mm</b> ", showarrow=False, font=dict(color="forestgreen", size=12), bgcolor="rgba(235, 255, 235, 0.95)", bordercolor="forestgreen", borderwidth=1, xanchor="left")

            fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray", annotation_text="Oś L11 (0.0 m)")
            fig.update_layout(title=f"Przekrój Poprzeczny – Profil RSP nr {wybrany_profil}", xaxis_title="Pozycja czujnika na belce [m]", yaxis_title=y_label, height=580, hovermode="x unified", xaxis=dict(range=[-1.85, 1.85], dtick=0.2), yaxis=dict(autorange=True if tryb_zero_l11 else "reversed"))
            st.plotly_chart(fig, use_container_width=True)

            # WYKRESY PODŁUŻNE NA CAŁEJ TRASIE Z DECYMACJĄ WIDOKU
            st.markdown("---")
            st.subheader("📈 Przebieg Kolein na Całej Trasie (Zoptymalizowane dla długich odcinków)")
            
            plot_step = max(1, len(df_all_rsp_ruts) // 10000)
            df_plot = df_all_rsp_ruts.iloc[::plot_step]
            if plot_step > 1: st.info(f"💡 Ze względu na długość trasy ({len(df_all_rsp_ruts)} profili), wykresy poniżej wyświetlają co {plot_step} profil dla płynności. Obliczenia w tle (DSN/GDDKiA) zawsze wykonują się na 100% profili.")

            has_lcms = lcms_profiles is not None and len(lcms_profiles) > 0
            sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs(["👈 Koleina Lewa", "👉 Koleina Prawa", "🔝 Koleina Maksymalna", "📊 Zbiorcze"])

            with sub_t1:
                fig_l = go.Figure()
                fig_l.add_trace(go.Scatter(x=df_plot["Profil"], y=df_plot["RSP_Rut_L"], mode="lines", name="RSP Koleina Lewa (Łata DSN)", line=dict(color="royalblue", width=2)))
                if has_lcms: fig_l.add_trace(go.Scatter(x=df_plot["Profil"], y=df_plot["LCMS_Rut_L"], mode="lines", name="LCMS Lewa (Dopasowana)", line=dict(color="forestgreen", width=2), connectgaps=False))
                fig_l.add_vline(x=wybrany_profil, line_width=2, line_color="red", annotation_text=f"Profil {wybrany_profil}")
                fig_l.update_layout(title="KOLEINA LEWA [mm]", xaxis_title="Numer Profilu RSP", yaxis_title="Głębokość koleiny [mm]", height=420, hovermode="x unified")
                st.plotly_chart(fig_l, use_container_width=True)

            with sub_t2:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=df_plot["Profil"], y=df_plot["RSP_Rut_R"], mode="lines", name="RSP Koleina Prawa (Łata DSN)", line=dict(color="darkorange", width=2)))
                if has_lcms: fig_r.add_trace(go.Scatter(x=df_plot["Profil"], y=df_plot["LCMS_Rut_R"], mode="lines", name="LCMS Prawa (Dopasowana)", line=dict(color="forestgreen", width=2), connectgaps=False))
                fig_r.add_vline(x=wybrany_profil, line_width=2, line_color="red", annotation_text=f"Profil {wybrany_profil}")
                fig_r.update_layout(title="KOLEINA PRAWA [mm]", xaxis_title="Numer Profilu RSP", yaxis_title="Głębokość koleiny [mm]", height=420, hovermode="x unified")
                st.plotly_chart(fig_r, use_container_width=True)

            with sub_t3:
                fig_m = go.Figure()
                fig_m.add_trace(go.Scatter(x=df_plot["Profil"], y=df_plot["RSP_Rut_Max"], mode="lines", name="RSP Koleina Max (Łata DSN)", line=dict(color="crimson", width=2.5)))
                if has_lcms: fig_m.add_trace(go.Scatter(x=df_plot["Profil"], y=df_plot["LCMS_Rut_Max"], mode="lines", name="LCMS Koleina Max (Dopasowana)", line=dict(color="forestgreen", width=2.5), connectgaps=False))
                fig_m.add_vline(x=wybrany_profil, line_width=2, line_color="red", annotation_text=f"Profil {wybrany_profil}")
                fig_m.update_layout(title="KOLEINA MAKSYMALNA [mm]", xaxis_title="Numer Profilu RSP", yaxis_title="Głębokość koleiny [mm]", height=420, hovermode="x unified")
                st.plotly_chart(fig_m, use_container_width=True)

            with sub_t4:
                fig_all = go.Figure()
                fig_all.add_trace(go.Scatter(x=df_plot["Profil"], y=df_plot["RSP_Rut_L"], mode="lines", name="RSP Lewa", line=dict(color="royalblue", width=2)))
                fig_all.add_trace(go.Scatter(x=df_plot["Profil"], y=df_plot["RSP_Rut_R"], mode="lines", name="RSP Prawa", line=dict(color="darkorange", width=2)))
                if has_lcms:
                    fig_all.add_trace(go.Scatter(x=df_plot["Profil"], y=df_plot["LCMS_Rut_L"], mode="lines", name="LCMS Lewa", line=dict(color="forestgreen", width=1.5), connectgaps=False))
                    fig_all.add_trace(go.Scatter(x=df_plot["Profil"], y=df_plot["LCMS_Rut_R"], mode="lines", name="LCMS Prawa", line=dict(color="limegreen", width=1.5), connectgaps=False))
                fig_all.add_vline(x=wybrany_profil, line_width=2, line_color="red", annotation_text=f"Profil {wybrany_profil}")
                fig_all.update_layout(title="Zestawienie Zbiorcze: RSP vs LCMS", xaxis_title="Numer Profilu RSP", yaxis_title="Głębokość koleiny [mm]", height=450, hovermode="x unified")
                st.plotly_chart(fig_all, use_container_width=True)

            # MAPA GPS
            if df_rsp_gps is not None or df_xml_gps is not None:
                st.markdown("---")
                st.subheader("🗺️ MAPA GPS i Przesunięcie Terenowe")
                idx_rsp_gps = min(max(0, wybrany_profil - min_p), len(df_rsp_gps) - 1) if df_rsp_gps is not None else None
                row_rsp_gps = df_rsp_gps.iloc[idx_rsp_gps] if idx_rsp_gps is not None else None
                idx_lcms_gps = min(max(0, wybrany_profil_lcms - 1), len(df_xml_gps) - 1) if df_xml_gps is not None else None
                row_lcms_gps = df_xml_gps.iloc[idx_lcms_gps] if idx_lcms_gps is not None else None

                dist_m = None
                if row_rsp_gps is not None and row_lcms_gps is not None:
                    dist_m = haversine_distance_m(row_rsp_gps["LAT"], row_rsp_gps["LON"], row_lcms_gps["LAT"], row_lcms_gps["LON"])
                    dists_all = [haversine_distance_m(row_rsp_gps["LAT"], row_rsp_gps["LON"], la, lo) for la, lo in zip(df_xml_gps["LAT"].values, df_xml_gps["LON"].values)]
                    best_match_idx, best_match_dist = int(np.argmin(dists_all)), np.min(dists_all)
                    sugerowany_offset = (best_match_idx + 1) - wybrany_profil

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Odległość GPS (Obecny Offset)", f"{dist_m:.2f} m", delta=f"Przesunięcie: {całkowity_offset:+.1f} m")
                    mc2.metric("Sugerowany Offset Przestrzenny", f"{sugerowany_offset:+} m", delta_color="normal")
                    mc3.metric("Najmniejszy Dystans Terenowy", f"{best_match_dist:.2f} m", f"Punkt LCMS #{best_match_idx + 1}")

                fig_map = go.Figure()
                center_lat, center_lon = row_rsp_gps["LAT"] if row_rsp_gps is not None else (row_lcms_gps["LAT"] if row_lcms_gps is not None else 52.0), row_rsp_gps["LON"] if row_rsp_gps is not None else (row_lcms_gps["LON"] if row_lcms_gps is not None else 19.0)

                if df_rsp_gps is not None:
                    fig_map.add_trace(make_map_trace(lat=df_rsp_gps["LAT"], lon=df_rsp_gps["LON"], mode="markers+lines", marker=dict(size=5, color="royalblue"), line=dict(width=2, color="royalblue"), name=f"Trasa RSP GPS ({len(df_rsp_gps)} pkt)", text=[f"Profil RSP: {p_i}" for p_i in df_rsp_gps["Profil_idx"]]))
                    if row_rsp_gps is not None: fig_map.add_trace(make_map_trace(lat=[row_rsp_gps["LAT"]], lon=[row_rsp_gps["LON"]], mode="markers", marker=dict(size=14, color="red"), name=f"Profil RSP {wybrany_profil}", text=[f"Pozycja RSP nr {wybrany_profil}"]))
                if df_xml_gps is not None:
                    fig_map.add_trace(make_map_trace(lat=df_xml_gps["LAT"], lon=df_xml_gps["LON"], mode="markers+lines", marker=dict(size=4, color="forestgreen"), line=dict(width=1.5, color="forestgreen"), name=f"Trasa LCMS XML ({len(df_xml_gps)} pkt)", text=[f"Pikietaż LCMS: {st_m:.1f}m<br>Plik: {fn}" for st_m, fn in zip(df_xml_gps["Station_m"], df_xml_gps["Filename"])]))
                    if row_lcms_gps is not None: fig_map.add_trace(make_map_trace(lat=[row_lcms_gps["LAT"]], lon=[row_lcms_gps["LON"]], mode="markers", marker=dict(size=12, color="darkorange"), name=f"Profil LCMS {wybrany_profil_lcms}", text=[f"LCMS Profil {wybrany_profil_lcms}<br>Pikietaż: {row_lcms_gps['Station_m']}m"]))
                if row_rsp_gps is not None and row_lcms_gps is not None and dist_m is not None:
                    fig_map.add_trace(make_map_trace(lat=[row_rsp_gps["LAT"], row_lcms_gps["LAT"]], lon=[row_rsp_gps["LON"], row_lcms_gps["LON"]], mode="lines", line=dict(color="crimson", width=3), name=f"Przesunięcie terenowe: {dist_m:.2f} m", text=[f"Przesunięcie pozycji: {dist_m:.2f} m", f"Przesunięcie pozycji: {dist_m:.2f} m"]))

                map_config = dict(center=dict(lat=center_lat, lon=center_lon), zoom=15)
                try:
                    if hasattr(go, "Scattermap"): fig_map.update_layout(map_style="open-street-map", map=map_config, margin=dict(l=0, r=0, t=20, b=0), height=480)
                    else: fig_map.update_layout(mapbox_style="open-street-map", mapbox=map_config, margin=dict(l=0, r=0, t=20, b=0), height=480)
                except Exception: fig_map.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=480)
                st.plotly_chart(fig_map, use_container_width=True)

        # --- ZAKŁADKA 2: OCENA 50M DSN 2019 ---
        with tab_dsn50:
            st.header("📊 Ocena Stanu Koleiny dla Odcinków 50-metrowych (DSN 2019)")
            st.markdown(r"""Wzór normowy oceny odcinkowej $H_m$ = $E(h) + 2 D_h$
            **Klasy Stanu DSN:** 🟢 **A (0–10.9 mm)** | 🟡 **B (11–20.9 mm)** | 🟠 **C (21–30.9 mm)** | 🔴 **D (≥ 31 mm)**""")
            df_sec_dsn = compute_50m_dsn_sections_full(df_merged_ruts, step_m=1.0, section_m=50.0)
            sub_dsn_l, sub_dsn_p, sub_dsn_max = st.tabs(["👈 Lewa Koleina (L_KOL)", "👉 Prawa Koleina (P_KOL)", "🔝 Koleina Maksymalna (MAX_KOL)"])
            with sub_dsn_l: st.plotly_chart(make_50m_bar_chart(df_sec_dsn, "L_KOL", "Lewa Koleina (L_KOL)"), use_container_width=True)
            with sub_dsn_p: st.plotly_chart(make_50m_bar_chart(df_sec_dsn, "P_KOL", "Prawa Koleina (P_KOL)"), use_container_width=True)
            with sub_dsn_max: st.plotly_chart(make_50m_bar_chart(df_sec_dsn, "MAX_KOL", "Koleina Maksymalna (MAX_KOL)"), use_container_width=True)
            st.markdown("##### Tabela Zbiorcza Odcinkowej Oceny 50m (DSN 2019)")
            st.dataframe(df_sec_dsn, use_container_width=True)

        # --- ZAKŁADKA 3: ANALIZA PORÓWNAWCZA GDDKiA (BLAND-ALTMAN & RPD) ---
        with tab_ba_gddkia:
            st.header("📈 Analiza Porównawcza LCMS vs RSP (Metodologia GDDKiA)")
            if lcms_profiles is not None and len(lcms_profiles) > 0 and "LCMS_Rut_L" in df_merged_ruts.columns:
                rsp_l_v, rsp_r_v = df_merged_ruts["RSP_Rut_L"].values, df_merged_ruts["RSP_Rut_R"].values
                lcms_l_v, lcms_r_v = df_merged_ruts["LCMS_Rut_L"].values, df_merged_ruts["LCMS_Rut_R"].values

                col_ba1, col_ba2 = st.columns(2)
                with col_ba1:
                    fig_ba_r, m_diff_r, _, _, _ = build_bland_altman_fig(rsp_r_v, lcms_r_v, title="PRAWA KOLEINA (P_KOL)")
                    st.plotly_chart(fig_ba_r, use_container_width=True)
                with col_ba2:
                    fig_ba_l, m_diff_l, _, _, _ = build_bland_altman_fig(rsp_l_v, lcms_l_v, title="LEWA KOLEINA (L_KOL)")
                    st.plotly_chart(fig_ba_l, use_container_width=True)

                st.markdown("---")
                st.subheader("📊 Procentowa Różnica Względna (RPD)")
                col_rpd_p, col_rpd_l = st.columns(2)
                with col_rpd_p:
                    st.markdown("##### P_KOL (Prawa Koleina)")
                    st.table(build_rpd_table_df(rsp_r_v, lcms_r_v))
                with col_rpd_l:
                    st.markdown("##### L_KOL (Lewa Koleina)")
                    st.table(build_rpd_table_df(rsp_l_v, lcms_l_v))

                bias_status_p = "zawyża wyniki" if m_diff_r > 0 else "zaniża wyniki"
                bias_status_l = "zawyża wyniki" if m_diff_l > 0 else "zaniża wyniki"
                st.info(f"""
                **Wnioski Kluczowe:**
                1. **Błąd Systematyczny (Bias):**
                   * **P_KOL**: Średnia różnica wynosi **{m_diff_r:+.2f} mm** (LCMS {bias_status_p} względem RSP).
                   * **L_KOL**: Średnia różnica wynosi **{m_diff_l:+.2f} mm** (LCMS {bias_status_l} względem RSP).
                2. **Charakterystyka Rozrzutu:** Przy niskich koleinach metody wykazują zgodność. Ze wzrostem głębokości rozrzut rośnie ("stożek"), co jest typowe w ocenie GDDKiA.
                3. **Wnioski RPD:** Dominanta grupy o wysokim zróżnicowaniu (>30%) pozwala ocenić tendencję zawyżania/zaniżania bez błędów współczynnika zmienności (CV).
                """)
            else:
                st.warning("⚠️ Aby wygenerować wykresy Blanda-Altmana, wgraj pliki RSP oraz LCMS.")
    else:
        st.warning(f"Brak danych RSP dla profilu {wybrany_profil}.")
else:
    st.info("👈 Wgraj pliki z danymi RSP w panelu bocznym po lewej stronie.")
