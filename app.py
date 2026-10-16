import io
import os
import re
import datetime
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
    """Zwraca klasę stanu koleiny wg DSN 2019."""
    if pd.isna(h_val):
        return "A"
    if h_val < 11.0:
        return "A"
    elif h_val < 21.0:
        return "B"
    elif h_val < 31.0:
        return "C"
    else:
        return "D"


def get_dsn_color(klasa):
    """Kolory dla klas DSN 2019."""
    if klasa == "A":
        return "#28a745"
    elif klasa == "B":
        return "#ffc107"
    elif klasa == "C":
        return "#fd7e14"
    else:
        return "#dc3545"


# ==========================================
# 2. PARSERY SUROWYCH PLIKÓW (.RSP, .GPX, .CSV, .XLSX, .2025)
# ==========================================
def parse_native_rsp_file(file_bytes):
    """Parsuje surowy plik Dynatest .rsp / .txt lub plik raportu XML .2025."""
    text = file_bytes.decode("utf-8", errors="ignore")
    
    # Obsługa formatu XML / .2025 GDDKiA w uploaderze RSP
    if "<KOLC" in text or "<ODCINEK>" in text:
        try:
            tree = ET.ElementTree(ET.fromstring(text))
            root = tree.getroot()
            profiles_data = []
            gps_data = []
            for idx, odc in enumerate(root.findall(".//ODCINEK")):
                p_idx = idx + 1
                try:
                    l_str = odc.findtext("L_KOL")
                    p_str = odc.findtext("P_KOL")
                    l_val = float(l_str.replace(",", ".")) if l_str and l_str.strip() else 0.0
                    p_val = float(p_str.replace(",", ".")) if p_str and p_str.strip() else 0.0
                    
                    # Tworzymy sztuczny profil 21 laserów tak, aby L1 i P1 odpowiadały odczytom L_KOL i P_KOL
                    row = [p_idx] + [0.0] * 21
                    row[1] = -l_val  # Lewa koleina
                    row[19] = -p_val # Prawa koleina
                    profiles_data.append(row)
                    
                    lat_s = odc.findtext("LAT")
                    lon_s = odc.findtext("LON")
                    km_p_s = odc.findtext("KM_P")
                    if lat_s and lon_s and float(lat_s) != 0:
                        gps_data.append({
                            "Km": float(km_p_s.replace(",", ".")) if km_p_s else p_idx,
                            "LAT": float(lat_s.replace(",", ".")),
                            "LON": float(lon_s.replace(",", ".")),
                            "Profil_idx": p_idx,
                        })
                except Exception:
                    continue
            if profiles_data:
                cols = ["Profil"] + [f"L{i+1}" for i in range(21)]
                df_prof = pd.DataFrame(profiles_data, columns=cols)
                df_gps = pd.DataFrame(gps_data) if gps_data else None
                return df_prof, df_gps, DEFAULT_RSP_X_LIST
        except Exception:
            pass

    lines = text.strip().split("\n")
    sensors = []
    profiles_data = []
    gps_data = []

    for line in lines:
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        code = parts[0]

        if code.isdigit() and 5201 <= int(code) <= 5221:
            try:
                x_mm = float(parts[-1])
                sensors.append((int(code), x_mm / 1000.0))
            except ValueError:
                pass

        elif code == "5401":
            try:
                readings = [float(x) for x in parts[3:24]]
                profiles_data.append([len(profiles_data) + 1] + readings)
            except (ValueError, IndexError):
                pass

        elif code == "5280":
            try:
                st_m = float(parts[1])
                lat = float(parts[5])
                lon = float(parts[6])
                ele = float(parts[7])
                gps_data.append({
                    "Km": st_m,
                    "LAT": lat,
                    "LON": lon,
                    "ELE": ele,
                    "Profil_idx": len(gps_data) + 1,
                })
            except (ValueError, IndexError):
                pass

    df_prof = None
    if profiles_data:
        cols = ["Profil"] + [f"L{i+1}" for i in range(21)]
        df_prof = pd.DataFrame(profiles_data, columns=cols)

    df_gps = pd.DataFrame(gps_data) if gps_data else None
    x_positions = (
        [s[1] for s in sorted(sensors, key=lambda x: x[0])]
        if len(sensors) == 21
        else None
    )

    return df_prof, df_gps, x_positions


def parse_gpx_cached(file_bytes):
    try:
        tree = ET.ElementTree(ET.fromstring(file_bytes))
        root = tree.getroot()
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        records = []
        for tag in ["rtept", "trkpt", "wpt"]:
            for pt in root.iter(tag):
                try:
                    lat = float(pt.attrib.get("lat"))
                    lon = float(pt.attrib.get("lon"))
                    ele_str = pt.findtext("ele")
                    ele = float(ele_str) if ele_str else 0.0
                    name = pt.findtext("name")
                    clean_km = (
                        float(name)
                        if name and name.replace(".", "", 1).replace("-", "").isdigit()
                        else len(records) + 1
                    )
                    records.append({
                        "LAT": lat,
                        "LON": lon,
                        "ELE": ele,
                        "Km": clean_km,
                        "Profil_idx": len(records) + 1,
                    })
                except (ValueError, TypeError):
                    continue
        return pd.DataFrame(records) if records else None
    except Exception:
        return None


def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return float(2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def get_file_bytes_and_name(uploaded_file, default_paths):
    if uploaded_file is not None:
        return uploaded_file.getvalue(), uploaded_file.name

    for path in default_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read(), os.path.basename(path)
    return None, None


@st.cache_data
def load_rsp_data(file_bytes, file_name):
    text = file_bytes.decode("utf-8", errors="ignore")
    if "<KOLC" in text or "<ODCINEK>" in text or file_name.lower().endswith((".2025", ".xml")):
        df_prof, _, _ = parse_native_rsp_file(file_bytes)
        if df_prof is not None:
            return df_prof

    if file_name.lower().endswith((".rsp", ".txt")):
        df_prof, _, _ = parse_native_rsp_file(file_bytes)
        if df_prof is not None:
            return df_prof

    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def parse_rsp_gps_cached(file_bytes, file_name):
    if file_name.lower().endswith(".gpx"):
        df_gpx = parse_gpx_cached(file_bytes)
        if df_gpx is not None:
            return df_gpx

    if file_name.lower().endswith((".rsp", ".txt", ".2025", ".xml")):
        _, df_gps, _ = parse_native_rsp_file(file_bytes)
        if df_gps is not None:
            return df_gps

    if file_name.lower().endswith((".csv", ".txt")):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), header=None, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), header=None, sep=";")
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), header=None)

    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(
            df_clean[col].astype(str).str.replace(",", "."), errors="coerce"
        )

    lat_col, lon_col = None, None
    for col in df_clean.columns:
        vals = df_clean[col].dropna()
        if len(vals) > 0:
            m = vals.mean()
            if 48.0 <= m <= 56.0 and lat_col is None:
                lat_col = col
            elif 13.0 <= m <= 25.0 and lon_col is None:
                lon_col = col

    if lat_col is None:
        lat_col = 5
    if lon_col is None:
        lon_col = 6

    res = pd.DataFrame({
        "LAT": df_clean[lat_col],
        "LON": df_clean[lon_col],
        "Profil_idx": np.arange(1, len(df_clean) + 1),
    })
    if 1 in df_clean.columns:
        res["Km"] = df_clean[1]

    return res.dropna(subset=["LAT", "LON"]).reset_index(drop=True)


@st.cache_data
def parse_lcms_raw_cached(file_bytes, file_name):
    """Pancerny parser LCMS z obsługą odwrócenia osi Z (koleiny w dół / do góry)."""
    text = file_bytes.decode("utf-8", errors="ignore")
    
    if file_name.lower().endswith((".2025", ".xml")) or "<KOLC" in text or "<ODCINEK>" in text:
        try:
            tree = ET.ElementTree(ET.fromstring(text))
            root = tree.getroot()
            profiles = {}
            prof_idx = 1
            for odc in root.findall(".//ODCINEK"):
                try:
                    l_kol_str = odc.findtext("L_KOL")
                    p_kol_str = odc.findtext("P_KOL")
                    rut_l = float(l_kol_str.replace(",", ".")) if l_kol_str and l_kol_str.strip() else 0.0
                    rut_r = float(p_kol_str.replace(",", ".")) if p_kol_str and p_kol_str.strip() else 0.0
                    
                    # Generujemy poprawny profil poprzeczny LCMS z uwzględnieniem odwróconych osi Z
                    x_arr = np.array([-1.2, -0.8, 0.0, 0.8, 1.2])
                    z_arr = np.array([0.0, -rut_l, 0.0, -rut_r, 0.0])
                    
                    profiles[prof_idx] = {
                        "x": x_arr,
                        "z": z_arr,
                        "rut_l": rut_l,
                        "rut_r": rut_r,
                    }
                    prof_idx += 1
                except Exception:
                    continue
            if profiles:
                return profiles
        except Exception:
            pass

    if file_name.lower().endswith((".xlsx", ".xls")):
        df_excel = pd.read_excel(io.BytesIO(file_bytes), header=None)
        raw_rows = [[v if pd.notna(v) else "" for v in r.values] for _, r in df_excel.iterrows()]
    else:
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        first_line = lines[0] if lines else ""
        sep = ";" if ";" in first_line else ("," if "," in first_line else "\t")
        raw_rows = [[p.strip() for p in line.split(sep)] for line in lines]

    data_row_indices = []
    for idx, row in enumerate(raw_rows):
        shift = 0
        while shift < len(row) and str(row[shift]).strip() == "":
            shift += 1
        if shift < len(row):
            val_a = str(row[shift]).replace(".0", "").strip()
            if val_a.isdigit() and len(val_a) > 0:
                num_count = sum(1 for check_idx in range(shift + 1, min(len(row), shift + 5)) if str(row[check_idx]).strip())
                if num_count > 0:
                    data_row_indices.append((idx, shift))

    profiles = {}
    prof_idx = 1
    for k in range(0, len(data_row_indices) - 1, 2):
        r_left_h, shift_l = data_row_indices[k]
        r_left_d = r_left_h + 1
        r_right_h, shift_r = data_row_indices[k + 1]
        r_right_d = r_right_h + 1

        if r_right_d >= len(raw_rows):
            break

        try:
            row_lh = raw_rows[r_left_h]
            row_ld = raw_rows[r_left_d]
            row_rh = raw_rows[r_right_h]
            row_rd = raw_rows[r_right_d]

            rut_l, rut_r = 0.0, 0.0

            def parse_rut(v):
                if isinstance(v, datetime.datetime):
                    return float(f"{v.day}.{v.month:02d}")
                v_str = str(v).replace(",", ".").strip()
                if v_str == "":
                    return 0.0
                v_lower = v_str.lower()
                mies = {"sty": "01", "lut": "02", "mar": "03", "kwi": "04", "maj": "05", "cze": "06",
                        "lip": "07", "sie": "08", "wrz": "09", "paź": "10", "paz": "10", "lis": "11", "gru": "12"}
                for m, num in mies.items():
                    if m in v_lower:
                        v_lower = v_lower.replace(m, num)
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", v_lower)
                return float(nums[0]) if nums else 0.0

            if len(row_lh) > shift_l + 1:
                rut_l = parse_rut(row_lh[shift_l + 1])
            if len(row_lh) > shift_l + 2:
                rut_r = parse_rut(row_lh[shift_l + 2])

            def parse_floats(arr):
                res = []
                for v in arr:
                    v_str = str(v).replace(",", ".").strip()
                    if v_str == "":
                        continue
                    try:
                        res.append(float(v_str))
                    except ValueError:
                        pass
                return res

            x_left = parse_floats(row_lh[shift_l + 3:])
            z_left = parse_floats(row_ld[shift_l + 3:])
            x_right = parse_floats(row_rh[shift_r + 3:])
            z_right = parse_floats(row_rd[shift_r + 3:])

            min_len_l = min(len(x_left), len(z_left))
            x_l_arr, z_l_arr = np.array(x_left[:min_len_l]), np.array(z_left[:min_len_l])

            min_len_r = min(len(x_right), len(z_right))
            x_r_arr, z_r_arr = np.array(x_right[:min_len_r]), np.array(z_right[:min_len_r])

            x_full = np.concatenate([x_l_arr, x_r_arr])
            if len(x_full) > 0 and np.max(np.abs(x_full)) > 20:
                x_full = x_full / 1000.0

            # KLUCZOWA ZMIANA: odwracamy oś Z LCMS (-z), żeby koleiny były w dół / poprawnie wyświetlane
            z_full = -np.concatenate([z_l_arr, z_r_arr])

            if len(x_full) > 0:
                sort_order = np.argsort(x_full)
                profiles[prof_idx] = {
                    "x": x_full[sort_order],
                    "z": z_full[sort_order],
                    "rut_l": rut_l,
                    "rut_r": rut_r,
                }
            else:
                profiles[prof_idx] = {
                    "x": np.array([]), "z": np.array([]), "rut_l": rut_l, "rut_r": rut_r,
                }
            prof_idx += 1
        except Exception:
            continue

    return profiles


@st.cache_data
def parse_lcms_xml(file_bytes):
    tree = ET.ElementTree(ET.fromstring(file_bytes))
    root = tree.getroot()
    records = []
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    for pic in root.findall("PicIndex"):
        station = pic.findtext("Station")
        lat = pic.findtext("LAT")
        lon = pic.findtext("LON")
        filename = pic.findtext("Filename")
        iddrogi = pic.findtext("IDDROGI")

        if lat and lon:
            records.append({
                "Station_m": float(station) if station else 0.0,
                "LAT": float(lat),
                "LON": float(lon),
                "Filename": filename if filename else "",
                "IDDROGI": iddrogi if iddrogi else "",
            })

    df_xml = pd.DataFrame(records)
    if not df_xml.empty:
        df_xml = df_xml.sort_values(by="Station_m").drop_duplicates(subset=["Station_m"]).reset_index(drop=True)
    return df_xml


def smooth_profile(z_vals, window_size=31):
    if len(z_vals) < window_size:
        return z_vals
    return pd.Series(z_vals).rolling(window=window_size, center=True, min_periods=1).mean().values


def get_upper_hull_pts(x_sec, y_sec):
    pts = sorted(list(zip(x_sec, y_sec)), key=lambda p: p[0])
    upper = []
    for p in pts:
        while len(upper) >= 2:
            o, a, b = upper[-2], upper[-1], p
            cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            if cross >= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    return upper


def get_upper_hull(x_sec, y_sec):
    upper = get_upper_hull_pts(x_sec, y_sec)
    x_h = [p[0] for p in upper]
    y_h = [p[1] for p in upper]
    return np.interp(x_sec, x_h, y_h)


@st.cache_data
def compute_all_rsp_ruts_cached(df_rsp, x_vals_tuple):
    col_profil = "Profil" if "Profil" in df_rsp.columns else df_rsp.columns[0]
    laser_cols = [c for c in df_rsp.columns if c != col_profil]

    prof_ids = df_rsp[col_profil].values
    data_mat = df_rsp[laser_cols].values.astype(float)
    ile_laserow = data_mat.shape[1]

    if ile_laserow == 0:
        return pd.DataFrame({
            "Profil": prof_ids,
            "RSP_Rut_L": [0.0] * len(prof_ids),
            "RSP_Rut_R": [0.0] * len(prof_ids),
            "RSP_Rut_Max": [0.0] * len(prof_ids),
        })

    # Jeśli dane pochodzą z XML (gdzie wstrzyknęliśmy L_KOL i P_KOL bezpośrednio w kolumny)
    if "L1" in df_rsp.columns and "L20" in df_rsp.columns and np.all(data_mat[:, 1:19] == 0):
        ruts_l = -data_mat[:, 1]
        ruts_r = -data_mat[:, 19]
        return pd.DataFrame({
            "Profil": prof_ids,
            "RSP_Rut_L": ruts_l,
            "RSP_Rut_R": ruts_r,
            "RSP_Rut_Max": np.maximum(ruts_l, ruts_r),
        })

    i_min_mid = int(ile_laserow * 0.3)
    i_max_mid = int(ile_laserow * 0.7)

    ruts_l, ruts_r = [], []
    y_mat = -data_mat
    x_arr = np.array(x_vals_tuple)

    for row in y_mat:
        idx_crown = i_min_mid + np.argmax(row[i_min_mid : i_max_mid + 1])
        x_left, y_left = x_arr[: idx_crown + 1], row[: idx_crown + 1]
        x_right, y_right = x_arr[idx_crown:], row[idx_crown:]

        lata_l = get_upper_hull(x_left, y_left)
        lata_r = get_upper_hull(x_right, y_right)

        ruts_l.append(float(np.max(lata_l - y_left)))
        ruts_r.append(float(np.max(lata_r - y_right)))

    return pd.DataFrame({
        "Profil": prof_ids,
        "RSP_Rut_L": ruts_l,
        "RSP_Rut_R": ruts_r,
        "RSP_Rut_Max": np.maximum(ruts_l, ruts_r),
    })


@st.cache_data
def compute_all_lcms_calc_ruts_cached(lcms_profiles):
    ruts_calc_l, ruts_calc_r = [], []
    prof_list = sorted(lcms_profiles.keys())

    for pid in prof_list:
        pdata = lcms_profiles[pid]
        x_pts, z_pts = pdata["x"], pdata["z"]
        if len(x_pts) < 2:
            ruts_calc_l.append(pdata.get("rut_l", 0.0))
            ruts_calc_r.append(pdata.get("rut_r", 0.0))
            continue

        z_smooth = smooth_profile(z_pts, window_size=31)
        mid_mask = (x_pts >= -0.5) & (x_pts <= 0.5)
        if np.any(mid_mask):
            idx_crown = np.where(mid_mask)[0][np.argmax(z_smooth[mid_mask])]
        else:
            idx_crown = np.argmin(np.abs(x_pts))

        x_l, z_l = x_pts[: idx_crown + 1], z_smooth[: idx_crown + 1]
        x_r, z_r = x_pts[idx_crown:], z_smooth[idx_crown:]

        lata_l = get_upper_hull(x_l, z_l)
        lata_r = get_upper_hull(x_r, z_r)

        ruts_calc_l.append(float(np.max(lata_l - z_l)) if len(z_l) > 0 else 0.0)
        ruts_calc_r.append(float(np.max(lata_r - z_r)) if len(z_r) > 0 else 0.0)

    return pd.DataFrame({
        "Profil": prof_list,
        "LCMS_Calc_Rut_L": ruts_calc_l,
        "LCMS_Calc_Rut_R": ruts_calc_r,
        "LCMS_Calc_Rut_Max": np.maximum(ruts_calc_l, ruts_calc_r),
    })


def auto_align_offset_clean(df_rsp_ruts, lcms_profiles, max_search=150):
    if df_rsp_ruts is None or lcms_profiles is None or len(lcms_profiles) == 0:
        return 0.0

    rsp_map = dict(zip(df_rsp_ruts["Profil"], df_rsp_ruts["RSP_Rut_Max"]))
    lcms_map = {
        p: max(data.get("rut_l", 0.0), data.get("rut_r", 0.0))
        for p, data in lcms_profiles.items()
    }

    min_rsp = int(df_rsp_ruts["Profil"].min())
    max_rsp = int(df_rsp_ruts["Profil"].max())
    common_profs = np.arange(min_rsp, max_rsp + 1)

    rsp_vec = np.array([rsp_map.get(p, np.nan) for p in common_profs])
    best_offset, max_corr = 0, -1e9

    for delta in range(-max_search, max_search + 1):
        lcms_vec = np.array([lcms_map.get(p + delta, np.nan) for p in common_profs])
        valid = ~np.isnan(rsp_vec) & ~np.isnan(lcms_vec)
        if np.sum(valid) > 10:
            r_sub = rsp_vec[valid] - np.mean(rsp_vec[valid])
            l_sub = lcms_vec[valid] - np.mean(lcms_vec[valid])
            denom = np.linalg.norm(r_sub) * np.linalg.norm(l_sub)
            if denom > 1e-9:
                corr = np.sum(r_sub * l_sub) / denom
                if corr > max_corr:
                    max_corr = corr
                    best_offset = delta
    return best_offset


def auto_align_section(df_rsp_ruts, lcms_profiles, p_start, p_end, max_search=50):
    sub_rsp = df_rsp_ruts[(df_rsp_ruts["Profil"] >= p_start) & (df_rsp_ruts["Profil"] <= p_end)]
    if len(sub_rsp) < 10:
        return 0.0
    return auto_align_offset_clean(sub_rsp, lcms_profiles, max_search=max_search)


def make_map_trace(lat, lon, mode="lines", marker=None, line=None, name="", text=None):
    kwargs = {"lat": lat, "lon": lon, "mode": mode, "name": name}
    if marker is not None:
        kwargs["marker"] = marker
    if line is not None:
        clean_line = line.copy()
        clean_line.pop("dash", None)
        kwargs["line"] = clean_line
    if text is not None:
        kwargs["text"] = text
        kwargs["hoverinfo"] = "text"

    if hasattr(go, "Scattermap"):
        return go.Scattermap(**kwargs)
    elif hasattr(go, "Scattermapbox"):
        return go.Scattermapbox(**kwargs)
    else:
        return go.Scattergeo(**kwargs)


def build_bland_altman_fig(rsp_v, lcms_v, title="PRAWA KOLEINA"):
    means = (rsp_v + lcms_v) / 2.0
    diffs = lcms_v - rsp_v
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=means, y=diffs, mode="markers", name="Pomiary",
        marker=dict(size=7, color="#ffffcc", line=dict(width=1, color="#333333")),
        hovertemplate="Średnia: %{x:.2f} mm<br>Różnica (LCMS-RSP): %{y:.2f} mm<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="lightgray")
    fig.add_hline(y=mean_diff, line_width=3, line_color="#0066cc", annotation_text=f"Średnia: {mean_diff:+.2f} mm", annotation_position="right")
    fig.add_hline(y=loa_upper, line_width=2, line_dash="dash", line_color="red", annotation_text=f"+1.96 SD: {loa_upper:+.2f} mm", annotation_position="right")
    fig.add_hline(y=loa_lower, line_width=2, line_dash="dash", line_color="red", annotation_text=f"-1.96 SD: {loa_lower:+.2f} mm", annotation_position="right")

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=18)),
        xaxis_title="ŚREDNIA (LCMS, RSP) [mm]",
        yaxis_title="RÓŻNICA (LCMS - RSP) [mm]",
        height=480,
        template="plotly_white",
    )
    return fig, mean_diff, sd_diff, loa_upper, loa_lower


def build_rpd_table_df(rsp_v, lcms_v):
    mean_v = (rsp_v + lcms_v) / 2.0
    diff_abs = np.abs(lcms_v - rsp_v)
    rpd = np.where(mean_v > 0, (diff_abs / mean_v) * 100.0, 0.0)

    m_low = rpd <= 30.0
    m_high = rpd > 30.0
    n_tot = len(rsp_v)

    def sub_stat(m_grp, label):
        n_grp = int(np.sum(m_grp))
        if n_grp == 0:
            return {"zakres RPD": label, "n": 0, "%": "0%", "RSP > LCMS": 0, "Wyniki równe": 0, "LCMS > RSP": 0, "%RSP > LCMS": "0%", "%Wyniki równe": "0%", "%LCMS > RSP": "0%"}
        r_sub = np.round(rsp_v[m_grp], 1)
        l_sub = np.round(lcms_v[m_grp], 1)
        rsp_gt = int(np.sum(r_sub > l_sub))
        eq = int(np.sum(r_sub == l_sub))
        lcms_gt = int(np.sum(l_sub > r_sub))
        return {
            "zakres RPD": label, "n": n_grp, "%": f"{(n_grp / n_tot)*100:.0f}%",
            "RSP > LCMS": rsp_gt, "Wyniki równe": eq, "LCMS > RSP": lcms_gt,
            "%RSP > LCMS": f"{(rsp_gt / n_grp)*100:.0f}%", "%Wyniki równe": f"{(eq / n_grp)*100:.0f}%", "%LCMS > RSP": f"{(lcms_gt / n_grp)*100:.0f}%",
        }

    return pd.DataFrame([sub_stat(m_low, "0-30%"), sub_stat(m_high, ">30%"), {"zakres RPD": "suma", "n": n_tot, "%": "100%", "RSP > LCMS": "-", "Wyniki równe": "-", "LCMS > RSP": "-", "%RSP > LCMS": "-", "%Wyniki równe": "-", "%LCMS > RSP": "-"}])


def compute_50m_dsn_sections_full(df_ruts, step_m=1.0, section_m=50.0):
    profs_per_sec = max(1, int(round(section_m / step_m)))
    results = []
    num_sections = int(np.ceil(len(df_ruts) / profs_per_sec))

    for i in range(num_sections):
        sub = df_ruts.iloc[i * profs_per_sec : (i + 1) * profs_per_sec]
        if len(sub) == 0:
            continue
        st_start, st_end = sub["Profil"].iloc[0], sub["Profil"].iloc[-1]
        rsp_l, rsp_r = sub["RSP_Rut_L"].values, sub["RSP_Rut_R"].values
        rsp_max = np.maximum(rsp_l, rsp_r)

        rec = {
            "Odcinek": f"{st_start} - {st_end} m",
            "RSP L_KOL E(h)": round(np.mean(rsp_l), 2), "RSP L_KOL Dh": round(np.std(rsp_l, ddof=0), 2),
            "RSP L_KOL Hm": int(np.round(np.mean(rsp_l) + 2 * np.std(rsp_l, ddof=0))),
            "RSP L_KOL Klasa": get_dsn_class(np.mean(rsp_l) + 2 * np.std(rsp_l, ddof=0)),
            "RSP P_KOL E(h)": round(np.mean(rsp_r), 2), "RSP P_KOL Dh": round(np.std(rsp_r, ddof=0), 2),
            "RSP P_KOL Hm": int(np.round(np.mean(rsp_r) + 2 * np.std(rsp_r, ddof=0))),
            "RSP P_KOL Klasa": get_dsn_class(np.mean(rsp_r) + 2 * np.std(rsp_r, ddof=0)),
            "RSP MAX_KOL E(h)": round(np.mean(rsp_max), 2), "RSP MAX_KOL Dh": round(np.std(rsp_max, ddof=0), 2),
            "RSP MAX_KOL Hm": int(np.round(np.mean(rsp_max) + 2 * np.std(rsp_max, ddof=0))),
            "RSP MAX_KOL Klasa": get_dsn_class(np.mean(rsp_max) + 2 * np.std(rsp_max, ddof=0)),
        }

        if "LCMS_Rut_L" in sub.columns:
            lcms_l, lcms_r = sub["LCMS_Rut_L"].values, sub["LCMS_Rut_R"].values
            lcms_max = np.maximum(lcms_l, lcms_r)
            rec.update({
                "LCMS L_KOL E(h)": round(np.mean(lcms_l), 2), "LCMS L_KOL Dh": round(np.std(lcms_l, ddof=0), 2),
                "LCMS L_KOL Hm": int(np.round(np.mean(lcms_l) + 2 * np.std(lcms_l, ddof=0))),
                "LCMS L_KOL Klasa": get_dsn_class(np.mean(lcms_l) + 2 * np.std(lcms_l, ddof=0)),
                "LCMS P_KOL E(h)": round(np.mean(lcms_r), 2), "LCMS P_KOL Dh": round(np.std(lcms_r, ddof=0), 2),
                "LCMS P_KOL Hm": int(np.round(np.mean(lcms_r) + 2 * np.std(lcms_r, ddof=0))),
                "LCMS P_KOL Klasa": get_dsn_class(np.mean(lcms_r) + 2 * np.std(lcms_r, ddof=0)),
                "LCMS MAX_KOL E(h)": round(np.mean(lcms_max), 2), "LCMS MAX_KOL Dh": round(np.std(lcms_max, ddof=0), 2),
                "LCMS MAX_KOL Hm": int(np.round(np.mean(lcms_max) + 2 * np.std(lcms_max, ddof=0))),
                "LCMS MAX_KOL Klasa": get_dsn_class(np.mean(lcms_max) + 2 * np.std(lcms_max, ddof=0)),
            })
        results.append(rec)
    return pd.DataFrame(results)


def make_50m_bar_chart(df_sec_dsn, param_prefix="P_KOL", title="Prawa Koleina (P_KOL)"):
    fig = go.Figure()
    col_rsp_hm, col_rsp_klasa = f"RSP {param_prefix} Hm", f"RSP {param_prefix} Klasa"
    fig.add_trace(go.Bar(
        x=df_sec_dsn["Odcinek"], y=df_sec_dsn[col_rsp_hm], name=f"RSP {title} (Hm)",
        marker_color=[get_dsn_color(k) for k in df_sec_dsn[col_rsp_klasa]],
        text=[f"{hm} mm ({k})" for hm, k in zip(df_sec_dsn[col_rsp_hm], df_sec_dsn[col_rsp_klasa])],
        textposition="auto",
    ))

    col_lcms_hm, col_lcms_klasa = f"LCMS {param_prefix} Hm", f"LCMS {param_prefix} Klasa"
    if col_lcms_hm in df_sec_dsn.columns:
        fig.add_trace(go.Bar(
            x=df_sec_dsn["Odcinek"], y=df_sec_dsn[col_lcms_hm], name=f"LCMS {title} (Hm)",
            marker_color=[get_dsn_color(k) for k in df_sec_dsn[col_lcms_klasa]], opacity=0.7,
            text=[f"{hm} mm ({k})" for hm, k in zip(df_sec_dsn[col_lcms_hm], df_sec_dsn[col_lcms_klasa])],
            textposition="auto",
        ))

    fig.add_hline(y=11, line_dash="dash", line_color="#ffc107", annotation_text="Próg B (11mm)")
    fig.add_hline(y=21, line_dash="dash", line_color="#fd7e14", annotation_text="Próg C (21mm)")
    fig.add_hline(y=31, line_dash="dash", line_color="#dc3545", annotation_text="Próg D (31mm)")
    fig.update_layout(title=f"Ocena Hm dla Odcinków 50m – {title}", xaxis_title="Odcinek Pikietażu", yaxis_title="Ocena Hm [mm]", barmode="group", height=480)
    return fig


# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="RSP vs LCMS – System Analizy Kolein DSN 2019", layout="wide")
st.title("⚡ System Analizy RSP + LCMS (DSN 2019 & GDDKiA)")

if "offset_pikitaz_input" not in st.session_state:
    st.session_state["offset_pikitaz_input"] = 0.0

st.sidebar.header("1. Dane RSP (Profilograf / XML / Surowe)")
uploaded_rsp = st.sidebar.file_uploader("Wgraj plik RSP / XML (.rsp / .2025 / .xml / .csv)", type=["rsp", "xlsx", "csv", "txt", "2025", "xml"], key="rsp_input")
uploaded_rsp_gps = st.sidebar.file_uploader("Wgraj plik GPS RSP (.gpx / .rsp / .csv / .xml)", type=["gpx", "rsp", "xlsx", "csv", "txt", "2025", "xml"], key="rsp_gps_input")

st.sidebar.header("2. Dane LCMS (Surowe CSV / Excel / XML)")
uploaded_lcms = st.sidebar.file_uploader("Wgraj plik LCMS (.csv / .xlsx / .2025 / .xml)", type=["csv", "xlsx", "2025", "xml"], key="lcms_input")
uploaded_xml = st.sidebar.file_uploader("Wgraj plik XML z LCMS (GPS)", type=["xml"], key="xml_input")

bytes_rsp, name_rsp = get_file_bytes_and_name(uploaded_rsp, ["data/rsp.rsp", "rsp.rsp", "data/rsp.2025"])
bytes_rsp_gps, name_rsp_gps = get_file_bytes_and_name(uploaded_rsp_gps, ["data/rsp_gps.gpx", "rsp_gps.gpx"])
bytes_lcms, name_lcms = get_file_bytes_and_name(uploaded_lcms, ["data/lcms.csv", "lcms.csv", "data/lcms.2025"])
bytes_xml, name_xml = get_file_bytes_and_name(uploaded_xml, ["data/lcms.xml", "lcms.xml"])

auto_x_from_rsp = DEFAULT_RSP_X_LIST

st.sidebar.header("3. Synchronizacja i Pikietaż")
def sync_offset_from_slider():
    st.session_state["offset_pikitaz_input"] = float(st.session_state["offset_slider_key"])
def sync_offset_from_input():
    st.session_state["offset_slider_key"] = int(round(st.session_state["offset_pikitaz_input"]))

if "offset_slider_key" not in st.session_state:
    st.session_state["offset_slider_key"] = int(round(st.session_state["offset_pikitaz_input"]))

st.sidebar.slider("🎚️ Suwak przesunięcia [profili]:", min_value=-200, max_value=200, value=int(round(st.session_state["offset_pikitaz_input"])), step=1, key="offset_slider_key", on_change=sync_offset_from_slider)
glowny_offset = st.sidebar.number_input("✍️ Wpisz dokładnie [m / profili]:", min_value=-500.0, max_value=500.0, value=float(st.session_state["offset_pikitaz_input"]), step=0.1, format="%.1f", key="offset_pikitaz_input", on_change=sync_offset_from_input)

podziel_na_sekcje = st.sidebar.checkbox("Dziel trasę na sekcje", value=False)
sekcje_offsets = {}

st.sidebar.header("4. Geometria Belki RSP")
typ_geometrii = st.sidebar.radio("Typ konfiguracji czujników:", ["Domyślna RSP (21 laserów)", "Stały odstęp"])
if typ_geometrii == "Domyślna RSP (21 laserów)":
    pozycja_x = DEFAULT_RSP_X_LIST
else:
    krok_laserow = st.sidebar.number_input("Odstęp [m]:", min_value=0.05, max_value=0.50, value=0.165, step=0.005)
    pozycja_x = [round((i - 10) * krok_laserow, 3) for i in range(21)]

zrodlo_lcms = st.sidebar.radio("Źródło kolein LCMS:", ["Z pliku LCMS (surowe B2/C2)", "Wyliczone z profilu (Wirtualna Łata)"])

st.sidebar.header("5. Ustawienia wyświetlania LCMS")
zaczep_prawa_krawedz = st.sidebar.checkbox("🔗 Zaczep prawą krawędź LCMS", value=True)
wyrownaj_spadek_lcms = st.sidebar.checkbox("Wyzeruj spadek poprzeczny LCMS", value=True)
tryb_zero_l11 = st.sidebar.checkbox("Ustaw środek jako Zero", value=True)
pokaz_late_rsp = st.sidebar.checkbox("Pokaż łatę RSP", value=True)
pokaz_late_lcms = st.sidebar.checkbox("Pokaż łatę LCMS", value=True)

df_rsp_gps = parse_rsp_gps_cached(bytes_rsp_gps, name_rsp_gps) if bytes_rsp_gps is not None else None
lcms_profiles = parse_lcms_raw_cached(bytes_lcms, name_lcms) if bytes_lcms is not None else None
df_xml_gps = parse_lcms_xml(bytes_xml) if bytes_xml is not None else None

if bytes_rsp is not None:
    df_rsp = load_rsp_data(bytes_rsp, name_rsp)
    col_profil = "Profil" if "Profil" in df_rsp.columns else df_rsp.columns[0]
    min_p, max_p = int(df_rsp[col_profil].min()), int(df_rsp[col_profil].max())

    def get_offset_for_profil(p_rsp):
        return glowny_offset

    wybrany_profil = st.sidebar.number_input(f"Numer profilu RSP [{min_p} - {max_p}]:", min_value=min_p, max_value=max_p, value=min_p, step=1)
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
                if zrodlo_lcms == "Z pliku LCMS (surowe B2/C2)":
                    rl = lcms_profiles[p_lcms]["rut_l"]
                    rr = lcms_profiles[p_lcms]["rut_r"]
                else:
                    rl = calc_dict_l.get(p_lcms, lcms_profiles[p_lcms]["rut_l"])
                    rr = calc_dict_r.get(p_lcms, lcms_profiles[p_lcms]["rut_r"])
                rmax = max(rl, rr)
            else:
                rl, rr, rmax = np.nan, np.nan, np.nan
            lcms_l_mapped.append(rl)
            lcms_r_mapped.append(rr)
            lcms_max_mapped.append(rmax)

        df_all_rsp_ruts["LCMS_Rut_L"] = lcms_l_mapped
        df_all_rsp_ruts["LCMS_Rut_R"] = lcms_r_mapped
        df_all_rsp_ruts["LCMS_Rut_Max"] = lcms_max_mapped
        df_merged_ruts = df_all_rsp_ruts.dropna(subset=["LCMS_Rut_L", "LCMS_Rut_R"]).copy()
    else:
        df_merged_ruts = df_all_rsp_ruts.copy()

    tab_main, tab_dsn50, tab_ba_gddkia = st.tabs([
        "📐 Przekrój Poprzeczny & Cała Trasa",
        "📊 Ocena 50m DSN 2019 (Klasy A-D)",
        "📈 Analiza Porównawcza GDDKiA",
    ])

    wiersz_rsp = df_rsp[df_rsp[col_profil] == wybrany_profil]
    if not wiersz_rsp.empty:
        kolumny_laserow = [c for c in df_rsp.columns if c != col_profil]
        odczyty_raw = wiersz_rsp.iloc[0][kolumny_laserow].values.astype(float)
        ile_laserow = len(odczyty_raw)

        if ile_laserow == 21 and np.all(odczyty_raw[1:19] == 0):
            # Tryb XML/2025 wgrany jako RSP
            koleina_lewa_mm = -odczyty_raw[1]
            koleina_prawa_mm = -odczyty_raw[19]
            max_koleina_rsp = max(koleina_lewa_mm, koleina_prawa_mm)
            x_vals = pozycja_x
            y_vals = np.zeros(21)
            y_vals[1] = -koleina_lewa_mm
            y_vals[19] = -koleina_prawa_mm
            lata_lewa_y = np.zeros(21)
            lata_prawa_y = np.zeros(21)
            idx_kol_l, idx_kol_p = 1, 19
        else:
            x_vals = pozycja_x
            y_vals = odczyty_raw
            i_min_mid, i_max_mid = int(ile_laserow * 0.3), int(ile_laserow * 0.7)
            idx_crown = i_min_mid + np.argmax(y_vals[i_min_mid : i_max_mid + 1])
            x_left, y_left = np.array(x_vals[: idx_crown + 1]), np.array(y_vals[: idx_crown + 1])
            x_right, y_right = np.array(x_vals[idx_crown:]), np.array(y_vals[idx_crown:])
            lata_lewa_y, lata_prawa_y = get_upper_hull(x_left, y_left), get_upper_hull(x_right, y_right)
            diff_l, diff_p = lata_lewa_y - y_left, lata_prawa_y - y_right
            idx_kol_l = np.argmax(diff_l)
            idx_kol_p_rel = np.argmax(diff_p)
            idx_kol_p = idx_crown + idx_kol_p_rel
            koleina_lewa_mm, koleina_prawa_mm = diff_l[idx_kol_l], diff_p[idx_kol_p_rel]
            max_koleina_rsp = max(koleina_lewa_mm, koleina_prawa_mm)

        kol_lcms_l, kol_lcms_p = None, None
        prof_lcms_data = lcms_profiles.get(wybrany_profil_lcms) if lcms_profiles else None
        if prof_lcms_data:
            kol_lcms_l, kol_lcms_p = prof_lcms_data.get("rut_l"), prof_lcms_data.get("rut_r")

        with tab_main:
            st.subheader(f"Profil RSP nr {wybrany_profil} (LCMS nr {wybrany_profil_lcms})")
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("RSP Koleina Lewa", f"{koleina_lewa_mm:.1f} mm (Klasa {get_dsn_class(koleina_lewa_mm)})")
            rc2.metric("RSP Koleina Prawa", f"{koleina_prawa_mm:.1f} mm (Klasa {get_dsn_class(koleina_prawa_mm)})")
            rc3.metric("RSP Koleina Maksymalna", f"{max_koleina_rsp:.1f} mm (Klasa {get_dsn_class(max_koleina_rsp)})")

            # WYKRES PRZEKROJU
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name="Profil RSP", line=dict(color="royalblue", width=3)))

            if prof_lcms_data is not None and len(prof_lcms_data.get("x", [])) > 0:
                x_lcms = prof_lcms_data["x"].copy()
                z_lcms = prof_lcms_data["z"].copy() # Jest już odwrócone w parserze!
                
                if zaczep_prawa_krawedz:
                    x_lcms = x_lcms + (np.max(x_vals) - np.max(x_lcms))

                fig.add_trace(go.Scatter(x=x_lcms, y=z_lcms, mode="lines+markers", name=f"Profil LCMS nr {wybrany_profil_lcms}", line=dict(color="forestgreen", width=2)))

            fig.update_layout(title="Przekrój Poprzeczny", xaxis_title="Pozycja [m]", yaxis_title="Wysokość [mm]", height=520)
            st.plotly_chart(fig, use_container_width=True)

            # WYKRESY CAŁEJ TRASY
            st.markdown("---")
            st.subheader("📈 Przebieg Kolein na Całej Trasie")
            fig_all = go.Figure()
            fig_all.add_trace(go.Scatter(x=df_all_rsp_ruts["Profil"], y=df_all_rsp_ruts["RSP_Rut_L"], name="RSP Lewa", line=dict(color="royalblue")))
            fig_all.add_trace(go.Scatter(x=df_all_rsp_ruts["Profil"], y=df_all_rsp_ruts["RSP_Rut_R"], name="RSP Prawa", line=dict(color="darkorange")))
            if "LCMS_Rut_L" in df_all_rsp_ruts.columns:
                fig_all.add_trace(go.Scatter(x=df_all_rsp_ruts["Profil"], y=df_all_rsp_ruts["LCMS_Rut_L"], name="LCMS Lewa", line=dict(color="forestgreen"), connectgaps=False))
                fig_all.add_trace(go.Scatter(x=df_all_rsp_ruts["Profil"], y=df_all_rsp_ruts["LCMS_Rut_R"], name="LCMS Prawa", line=dict(color="limegreen"), connectgaps=False))
            fig_all.update_layout(xaxis_title="Profil", yaxis_title="Koleina [mm]", height=420)
            st.plotly_chart(fig_all, use_container_width=True)

        with tab_dsn50:
            st.header("📊 Ocena Stanu Koleiny dla Odcinków 50-metrowych (DSN 2019)")
            df_sec_dsn = compute_50m_dsn_sections_full(df_merged_ruts, step_m=1.0, section_m=50.0)
            st.plotly_chart(make_50m_bar_chart(df_sec_dsn, param_prefix="P_KOL", title="Prawa Koleina"), use_container_width=True)
            st.dataframe(df_sec_dsn, use_container_width=True)

        with tab_ba_gddkia:
            st.header("📈 Analiza Porównawcza GDDKiA (Bland-Altman)")
            if lcms_profiles is not None and "LCMS_Rut_L" in df_merged_ruts.columns:
                fig_ba_r, _, _, _, _ = build_bland_altman_fig(df_merged_ruts["RSP_Rut_R"].values, df_merged_ruts["LCMS_Rut_R"].values, title="PRAWA KOLEINA")
                st.plotly_chart(fig_ba_r, use_container_width=True)
else:
    st.info("👈 Wgraj plik raportu .2025/XML lub RSP w panelu bocznym.")
