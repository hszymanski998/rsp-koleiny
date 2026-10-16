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

st.sidebar.header("4. Geometria
