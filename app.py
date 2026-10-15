import io
import os
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
    -1.610,
    -1.444,
    -1.285,
    -1.145,
    -1.007,
    -0.850,
    -0.700,
    -0.550,
    -0.400,
    -0.200,
    0.000,
    0.200,
    0.400,
    0.550,
    0.700,
    0.850,
    1.003,
    1.140,
    1.280,
    1.440,
    1.618,
]


# ==========================================
# 1. PARSERY SUROWYCH PLIKÓW (.RSP, .GPX, .CSV, .XLSX)
# ==========================================
def parse_native_rsp_file(file_bytes):
    """Parsuje surowy plik Dynatest .rsp / .txt (linie 5201-5221, 5401, 5280)."""
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.strip().split("\n")

    sensors = []
    profiles_data = []
    gps_data = []

    for line in lines:
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        code = parts[0]

        # 5201-5221: Konfiguracja czujników X [mm]
        if code.isdigit() and 5201 <= int(code) <= 5221:
            try:
                x_mm = float(parts[-1])
                sensors.append((int(code), x_mm / 1000.0))
            except ValueError:
                pass

        # 5401: Odczyty profilu poprzecznego (21 laserów [mm])
        elif code == "5401":
            try:
                readings = [float(x) for x in parts[3:24]]
                profiles_data.append([len(profiles_data) + 1] + readings)
            except (ValueError, IndexError):
                pass

        # 5280: Pozycje GPS w pliku RSP
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
    """Parsuje pliki GPX śladu GPS z profilografu RSP."""
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
                        if name
                        and name.replace(".", "", 1).replace("-", "").isdigit()
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
    """Oblicza odległość w metrach na elipsoidzie WGS84."""
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
    """Pobiera plik z uploadu lub domyślny z folderu data/."""
    if uploaded_file is not None:
        return uploaded_file.getvalue(), uploaded_file.name

    for path in default_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read(), os.path.basename(path)
    return None, None


@st.cache_data
def load_rsp_data(file_bytes, file_name):
    """Wczytuje i keszuje dane RSP (zarówno Excel/CSV jak i surowy .rsp)."""
    if file_name.lower().endswith(".rsp") or file_name.lower().endswith(".txt"):
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
    """Parser GPS z plików GPX / RSP / CSV / Excel."""
    if file_name.lower().endswith(".gpx"):
        df_gpx = parse_gpx_cached(file_bytes)
        if df_gpx is not None:
            return df_gpx

    if file_name.lower().endswith(".rsp") or file_name.lower().endswith(".txt"):
        _, df_gps, _ = parse_native_rsp_file(file_bytes)
        if df_gps is not None:
            return df_gps

    if file_name.lower().endswith(".csv") or file_name.lower().endswith(".txt"):
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes), header=None, sep=None, engine="python"
            )
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
    """Parser surowych plików LCMS DSN (.csv oraz .xlsx)."""
    if file_name.lower().endswith(".csv"):
        try:
            df_lcms_raw = pd.read_csv(
                io.BytesIO(file_bytes), header=None, sep=";", engine="python"
            )
        except Exception:
            df_lcms_raw = pd.read_csv(
                io.BytesIO(file_bytes), header=None, sep=None, engine="python"
            )
    else:
        df_lcms_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)

    profiles = {}
    num_rows = len(df_lcms_raw)

    def is_valid_profile_id(val):
        s = str(val).strip().replace(".0", "")
        return s.isdigit() and len(s) > 0

    data_rows = []
    for idx in range(num_rows):
        val_a = df_lcms_raw.iloc[idx, 0]
        val_b = (
            str(df_lcms_raw.iloc[idx, 1])
            .replace(",", ".")
            .replace("-", "")
            .strip()
        )
        if is_valid_profile_id(val_a):
            try:
                float(val_b)
                data_rows.append(idx)
            except Exception:
                pass

    prof_idx = 1
    for k in range(0, len(data_rows) - 1, 2):
        r_left_h = data_rows[k]
        r_left_d = r_left_h + 1
        r_right_h = data_rows[k + 1]
        r_right_d = r_right_h + 1

        if r_right_d >= num_rows:
            break

        try:
            rut_l_str = str(df_lcms_raw.iloc[r_left_h, 1]).replace(",", ".").strip()
            rut_r_str = str(df_lcms_raw.iloc[r_left_h, 2]).replace(",", ".").strip()

            rut_l = (
                float(rut_l_str)
                if rut_l_str.replace(".", "", 1).replace("-", "").isdigit()
                else 0.0
            )
            rut_r = (
                float(rut_r_str)
                if rut_r_str.replace(".", "", 1).replace("-", "").isdigit()
                else 0.0
            )

            x_left = pd.to_numeric(
                df_lcms_raw.iloc[r_left_h, 3:]
                .astype(str)
                .str.replace(",", "."),
                errors="coerce",
            ).values
            z_left = pd.to_numeric(
                df_lcms_raw.iloc[r_left_d, 3:]
                .astype(str)
                .str.replace(",", "."),
                errors="coerce",
            ).values

            x_right = pd.to_numeric(
                df_lcms_raw.iloc[r_right_h, 3:]
                .astype(str)
                .str.replace(",", "."),
                errors="coerce",
            ).values
            z_right = pd.to_numeric(
                df_lcms_raw.iloc[r_right_d, 3:]
                .astype(str)
                .str.replace(",", "."),
                errors="coerce",
            ).values

            valid_l = ~np.isnan(x_left) & ~np.isnan(z_left)
            x_left, z_left = x_left[valid_l], z_left[valid_l]

            valid_r = ~np.isnan(x_right) & ~np.isnan(z_right)
            x_right, z_right = x_right[valid_r], z_right[valid_r]

            x_full = np.concatenate([x_left, x_right])
            if len(x_full) > 0 and np.max(np.abs(x_full)) > 20:
                x_full = x_full / 1000.0

            z_full = np.concatenate([z_left, z_right])

            sort_order = np.argsort(x_full)
            x_sorted = x_full[sort_order]
            z_sorted = z_full[sort_order]

            profiles[prof_idx] = {
                "x": x_sorted,
                "z": z_sorted,
                "rut_l": rut_l,
                "rut_r": rut_r,
            }
            prof_idx += 1
        except Exception:
            continue

    return profiles


@st.cache_data
def parse_lcms_xml(file_bytes):
    """Parser XML z LCMS ze ścisłym sortowaniem po pikietażu."""
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
        df_xml = (
            df_xml.sort_values(by="Station_m")
            .drop_duplicates(subset=["Station_m"])
            .reset_index(drop=True)
        )
    return df_xml


def smooth_profile(z_vals, window_size=31):
    """Filtruje makrostrukturę nawierzchni."""
    if len(z_vals) < window_size:
        return z_vals
    return (
        pd.Series(z_vals)
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .values
    )


def get_upper_hull_pts(x_sec, y_sec):
    """Górna otoczka wypukła (Wirtualna Łata)."""
    pts = sorted(list(zip(x_sec, y_sec)), key=lambda p: p[0])
    upper = []
    for p in pts:
        while len(upper) >= 2:
            o, a, b = upper[-2], upper[-1], p
            cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (
                b[0] - o[0]
            )
            if cross >= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    return upper


def get_upper_hull(x_sec, y_sec):
    """Interpolacja wartości prostej łaty DSN 2019."""
    upper = get_upper_hull_pts(x_sec, y_sec)
    x_h = [p[0] for p in upper]
    y_h = [p[1] for p in upper]
    return np.interp(x_sec, x_h, y_h)


@st.cache_data
def compute_all_rsp_ruts_cached(df_rsp, x_vals_tuple):
    """Przetwarza całą trasę RSP na potrzeby wykresu podłużnego."""
    col_profil = "Profil" if "Profil" in df_rsp.columns else df_rsp.columns[0]
    laser_cols = [c for c in df_rsp.columns if c != col_profil]

    prof_ids = df_rsp[col_profil].values
    data_mat = df_rsp[laser_cols].values.astype(float)

    ile_laserow = data_mat.shape[1]
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
    """Wylicza koleiny z Wirtualnej Łaty dla wszystkich profili LCMS."""
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
        if np.any(mid_mask):
            idx_crown = np.where(mid_mask)[0][np.argmax(z_smooth[mid_mask])]
        else:
            idx_crown = np.argmin(np.abs(x_pts))

        x_l, z_l = x_pts[: idx_crown + 1], z_smooth[: idx_crown + 1]
        x_r, z_r = x_pts[idx_crown:], z_smooth[idx_crown:]

        lata_l = get_upper_hull(x_l, z_l)
        lata_r = get_upper_hull(x_r, z_r)

        ruts_calc_l.append(
            float(np.max(lata_l - z_l)) if len(z_l) > 0 else 0.0
        )
        ruts_calc_r.append(
            float(np.max(lata_r - z_r)) if len(z_r) > 0 else 0.0
        )

    return pd.DataFrame({
        "Profil": prof_list,
        "LCMS_Calc_Rut_L": ruts_calc_l,
        "LCMS_Calc_Rut_R": ruts_calc_r,
        "LCMS_Calc_Rut_Max": np.maximum(ruts_calc_l, ruts_calc_r),
    })


def auto_align_offset_clean(rsp_s, lcms_s, max_search=100):
    """Automatycznie dopasowuje pikietaż na podstawie korelacji wykresu kolein."""
    r = np.array(rsp_s) - np.mean(rsp_s)
    l = np.array(lcms_s) - np.mean(lcms_s)

    best_offset = 0
    max_corr = -1e9

    for offset in range(-max_search, max_search + 1):
        if offset >= 0:
            r_sub = r[: len(r) - offset] if offset < len(r) else r[:0]
            l_sub = l[offset : offset + len(r_sub)]
        else:
            abs_off = abs(offset)
            r_sub = r[abs_off:]
            l_sub = l[: len(r_sub)]

        if len(r_sub) > 10 and len(r_sub) == len(l_sub):
            denom = np.linalg.norm(r_sub) * np.linalg.norm(l_sub)
            if denom > 1e-9:
                corr = np.sum(r_sub * l_sub) / denom
                if corr > max_corr:
                    max_corr = corr
                    best_offset = offset

    return best_offset


def make_map_trace(
    lat, lon, mode="lines", marker=None, line=None, name="", text=None
):
    """Bezpieczny kreator warstw mapy Plotly."""
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


# ==========================================
# 2. KONFIGURACJA STRONY STREAMLIT
# ==========================================
st.set_page_config(
    page_title="RSP vs LCMS – Szybka Analiza Kolein", layout="wide"
)
st.title("⚡ System Analizy RSP + LCMS (DSN 2019)")

if "offset_pikitaz_input" not in st.session_state:
    st.session_state["offset_pikitaz_input"] = 0.0

# ==========================================
# 3. PANEL BOCZNY (PRZEPISANE DANE I SYNCHRONIZACJA)
# ==========================================
st.sidebar.header("1. Dane RSP (Profilograf / Surowe)")
uploaded_rsp = st.sidebar.file_uploader(
    "Wgraj plik RSP (.rsp / .xlsx / .csv / .txt)",
    type=["rsp", "xlsx", "csv", "txt"],
    key="rsp_input",
)
uploaded_rsp_gps = st.sidebar.file_uploader(
    "Wgraj plik GPS RSP (.gpx / .rsp / .csv / .txt)",
    type=["gpx", "rsp", "xlsx", "csv", "txt"],
    key="rsp_gps_input",
)

st.sidebar.header("2. Dane LCMS (Surowe CSV / Excel)")
uploaded_lcms = st.sidebar.file_uploader(
    "Wgraj plik LCMS (.csv / .xlsx)", type=["csv", "xlsx"], key="lcms_input"
)
uploaded_xml = st.sidebar.file_uploader(
    "Wgraj plik XML z LCMS (GPS)", type=["xml"], key="xml_input"
)

bytes_rsp, name_rsp = get_file_bytes_and_name(
    uploaded_rsp,
    [
        "data/rsp.rsp",
        "data/rsp.xlsx",
        "data/rsp.csv",
        "rsp.rsp",
        "rsp.xlsx",
        "rsp.csv",
    ],
)
bytes_rsp_gps, name_rsp_gps = get_file_bytes_and_name(
    uploaded_rsp_gps,
    [
        "data/rsp_gps.gpx",
        "data/rsp_gps.csv",
        "data/rsp_gps.xlsx",
        "rsp_gps.gpx",
        "rsp_gps.csv",
    ],
)
bytes_lcms, name_lcms = get_file_bytes_and_name(
    uploaded_lcms, ["data/lcms.csv", "data/lcms.xlsx", "lcms.csv", "lcms.xlsx"]
)
bytes_xml, name_xml = get_file_bytes_and_name(
    uploaded_xml, ["data/lcms.xml", "lcms.xml"]
)

# Automatyczny odczyt czujników z surowego RSP
auto_x_from_rsp = None
if bytes_rsp is not None and (
    name_rsp.lower().endswith(".rsp") or name_rsp.lower().endswith(".txt")
):
    _, _, auto_x_from_rsp = parse_native_rsp_file(bytes_rsp)

st.sidebar.header("3. Synchronizacja i Pikietaż (Wpisz wartość)")
glowny_offset = st.sidebar.number_input(
    "Wpisz przesunięcie LCMS względem RSP [metry / profili]:",
    min_value=-500.0,
    max_value=500.0,
    value=float(st.session_state["offset_pikitaz_input"]),
    step=0.1,
    format="%.1f",
    key="offset_pikitaz_input",
)

podziel_na_sekcje = st.sidebar.checkbox(
    "Dziel trasę na sekcje (osobne przesunięcia dla odcinków)", value=False
)
sekcja_m = 500
sekcje_offsets = {}

if podziel_na_sekcje:
    sekcja_m = st.sidebar.select_slider(
        "Długość pojedynczej sekcji pikietażu [m]:",
        options=[250, 500, 1000, 2000],
        value=500,
    )
    st.sidebar.markdown("##### Ręczne offsety dla poszczególnych sekcji:")

st.sidebar.header("4. Geometria Belki RSP")
typ_geometrii = st.sidebar.radio(
    "Typ konfiguracji czujników RSP:",
    ["Wprowadzona w pliku / Domyślna RSP", "Stały odstęp (równomierny)"],
)

if typ_geometrii == "Wprowadzona w pliku / Domyślna RSP":
    default_x_val = (
        ", ".join(f"{x:.3f}" for x in auto_x_from_rsp)
        if auto_x_from_rsp
        else DEFAULT_RSP_X_STR
    )
    x_input_str = st.sidebar.text_area(
        "Pozycje X [m] dla 21 laserów:", value=default_x_val, height=90
    )
    try:
        pozycja_x = [
            float(val.strip().replace(",", "."))
            for val in x_input_str.split(",")
            if val.strip() != ""
        ]
    except Exception:
        pozycja_x = auto_x_from_rsp or DEFAULT_RSP_X_LIST
else:
    krok_laserow = st.sidebar.number_input(
        "Odstęp [m]:",
        min_value=0.05,
        max_value=0.50,
        value=0.165,
        step=0.005,
        format="%.3f",
    )
    pozycja_x = [round((i - 10) * krok_laserow, 3) for i in range(21)]

st.sidebar.header("5. Wybór Źródła Kolein LCMS")
zrodlo_lcms = st.sidebar.radio(
    "Wykresy podłużne i kafelki LCMS bazują na:",
    [
        "Z pliku LCMS (surowe B2/C2)",
        "Wyliczone z profilu (Wirtualna Łata)",
        "Oba źródła jednocześnie",
    ],
)

st.sidebar.header("6. Geometria i Dopasowanie Poprzeczne LCMS vs RSP")
zaczep_prawa_krawedz = st.sidebar.checkbox(
    "🔗 Zaczep prawą krawędź LCMS do krawędzi RSP", value=True
)
skaluj_szerokosc = st.sidebar.checkbox(
    "📐 Skaluj szerokość LCMS do szerokości belki RSP", value=False
)
manual_x_shift_lcms = st.sidebar.number_input(
    "Ręczne przesunięcie poprzeczne X LCMS [m]:",
    min_value=-2.0,
    max_value=2.0,
    value=0.0,
    step=0.01,
    format="%.2f",
)

test_stacjonarny = st.sidebar.checkbox(
    "Tryb testu stacjonarnego (Łata 0)", value=False
)
tryb_zero_l11 = st.sidebar.checkbox(
    "Ustaw L11 jako Zero (wysokość względna RSP)", value=True
)
pokaz_late_rsp = st.sidebar.checkbox("Pokaż łatę i koleiny RSP", value=True)
wyrownaj_spadek_lcms = st.sidebar.checkbox(
    "Wyzeruj spadek poprzeczny LCMS do wyświetlania", value=True
)
pokaz_late_lcms = st.sidebar.checkbox("Pokaż łatę i koleiny LCMS", value=True)

# Parsowanie plików GPS i LCMS
df_rsp_gps = None
if bytes_rsp_gps is not None:
    try:
        df_rsp_gps = parse_rsp_gps_cached(bytes_rsp_gps, name_rsp_gps)
        st.sidebar.success(f"GPS RSP: {len(df_rsp_gps)} pkt.")
    except Exception as e:
        st.sidebar.error(f"Błąd GPS RSP: {e}")

lcms_profiles = None
if bytes_lcms is not None:
    try:
        lcms_profiles = parse_lcms_raw_cached(bytes_lcms, name_lcms)
        st.sidebar.success(f"LCMS: {len(lcms_profiles)} profili.")
    except Exception as e:
        st.sidebar.error(f"Błąd LCMS: {e}")

df_xml_gps = None
if bytes_xml is not None:
    try:
        df_xml_gps = parse_lcms_xml(bytes_xml)
        st.sidebar.success(f"GPS LCMS: {len(df_xml_gps)} pkt.")
    except Exception as e:
        st.sidebar.error(f"Błąd XML: {e}")

# ==========================================
# 4. GŁÓWNA LOGIKA ANALIZY I WYKRESÓW
# ==========================================
if bytes_rsp is not None:
    df_rsp = load_rsp_data(bytes_rsp, name_rsp)

    col_profil = "Profil" if "Profil" in df_rsp.columns else df_rsp.columns[0]
    min_p = int(df_rsp[col_profil].min())
    max_p = int(df_rsp[col_profil].max())

    if podziel_na_sekcje:
        num_sekcji = int(np.ceil((max_p - min_p + 1) / sekcja_m))
        for s_i in range(num_sekcji):
            p_start_s = min_p + s_i * sekcja_m
            p_end_s = min(max_p, p_start_s + sekcja_m - 1)
            sekcje_offsets[s_i] = st.sidebar.number_input(
                f"Sekcja {s_i+1} ({p_start_s}-{p_end_s}m) [m]:",
                min_value=-100.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                format="%.1f",
            )

    st.sidebar.header("7. Wybór profilu")
    wybrany_profil = st.sidebar.number_input(
        f"Numer profilu RSP [{min_p} - {max_p}]:",
        min_value=min_p,
        max_value=max_p,
        value=min_p,
        step=1,
    )

    bazowy_offset = glowny_offset
    if podziel_na_sekcje:
        idx_s = int((wybrany_profil - min_p) // sekcja_m)
        dodatkowy_s_offset = sekcje_offsets.get(idx_s, 0.0)
    else:
        dodatkowy_s_offset = 0.0

    całkowity_offset = bazowy_offset + dodatkowy_s_offset
    wybrany_profil_lcms = int(round(wybrany_profil + całkowity_offset))

    # AUTOMATYCZNA SYNCHRONIZACJA BAZUJĄCA NA WYKRESIE BIEGU KOLEIN
    if lcms_profiles is not None:
        df_all_rsp_ruts = compute_all_rsp_ruts_cached(df_rsp, tuple(pozycja_x))
        list_lcms_p = sorted(lcms_profiles.keys())
        lcms_file_rut_max = np.maximum(
            [lcms_profiles[p]["rut_l"] for p in list_lcms_p],
            [lcms_profiles[p]["rut_r"] for p in list_lcms_p],
        )

        if st.sidebar.button("⚡ Automatyczna Synchronizacja Pikietażu"):
            sug_offset = auto_align_offset_clean(
                df_all_rsp_ruts["RSP_Rut_Max"].values, lcms_file_rut_max
            )
            st.session_state["offset_pikitaz_input"] = float(sug_offset)
            st.sidebar.success(
                f"✅ Optymalne przesunięcie pikietażu wynosi: {sug_offset:+} m"
            )
            st.rerun()

    wiersz_rsp = df_rsp[df_rsp[col_profil] == wybrany_profil]

    if not wiersz_rsp.empty:
        kolumny_laserow = [c for c in df_rsp.columns if c != col_profil]
        odczyty_raw = wiersz_rsp.iloc[0][kolumny_laserow].values.astype(float)
        ile_laserow = len(odczyty_raw)

        x_vals = (
            pozycja_x
            if len(pozycja_x) == ile_laserow
            else [
                round((i - (ile_laserow - 1) / 2) * 0.165, 3)
                for i in range(ile_laserow)
            ]
        )

        if tryb_zero_l11:
            idx_srodek = ile_laserow // 2
            ref_val = odczyty_raw[idx_srodek]
            y_vals = ref_val - odczyty_raw
            y_label = "Wysokość względna do L11 [mm]"
        else:
            y_vals = odczyty_raw
            y_label = "Odczyt surowy czujnika [mm]"

        # RSP Crown & Hull
        i_min_mid = int(ile_laserow * 0.3)
        i_max_mid = int(ile_laserow * 0.7)
        idx_crown = i_min_mid + np.argmax(y_vals[i_min_mid : i_max_mid + 1])

        x_left, y_left = (
            np.array(x_vals[: idx_crown + 1]),
            np.array(y_vals[: idx_crown + 1]),
        )
        x_right, y_right = (
            np.array(x_vals[idx_crown:]),
            np.array(y_vals[idx_crown:]),
        )

        lata_lewa_y = get_upper_hull(x_left, y_left)
        lata_prawa_y = get_upper_hull(x_right, y_right)

        diff_l = lata_lewa_y - y_left
        idx_kol_l = np.argmax(diff_l)
        koleina_lewa_mm = diff_l[idx_kol_l]

        diff_p = lata_prawa_y - y_right
        idx_kol_p_rel = np.argmax(diff_p)
        idx_kol_p = idx_crown + idx_kol_p_rel
        koleina_prawa_mm = diff_p[idx_kol_p_rel]

        max_koleina_rsp = max(koleina_lewa_mm, koleina_prawa_mm)

        if test_stacjonarny:
            st.subheader("🧪 Weryfikacja Testu Stacjonarnego (DSN 2019)")
            if max_koleina_rsp <= 0.5:
                st.success(
                    f"✅ **TEST POZYTYWNY**: Odchyłka wynosi"
                    f" **{max_koleina_rsp:.2f} mm** (≤ 0,5 mm)."
                )
            else:
                st.error(
                    f"❌ **TEST NEGATYWNY**: Odchyłka wynosi"
                    f" **{max_koleina_rsp:.2f} mm** i przekracza limit 0,5 mm dla"
                    " łaty!"
                )

        kol_lcms_l, kol_lcms_p = None, None
        prof_lcms_data = None
        if lcms_profiles and wybrany_profil_lcms in lcms_profiles:
            prof_lcms_data = lcms_profiles[wybrany_profil_lcms]
            kol_lcms_l = prof_lcms_data.get("rut_l", None)
            kol_lcms_p = prof_lcms_data.get("rut_r", None)

        st.subheader(
            f"Profil RSP nr {wybrany_profil} (LCMS nr {wybrany_profil_lcms},"
            f" przesunięcie: {całkowity_offset:+.1f} m)"
        )

        # KAFELKI INFORMACYJNE
        st.markdown("##### 🟦 Dane z Profilografu RSP")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric(
            "RSP Koleina Lewa",
            f"{koleina_lewa_mm:.1f} mm",
            f"Laser L{idx_kol_l+1} ({x_vals[idx_kol_l]:.2f}m)",
        )
        rc2.metric(
            "RSP Koleina Prawa",
            f"{koleina_prawa_mm:.1f} mm",
            f"Laser L{idx_kol_p+1} ({x_vals[idx_kol_p]:.2f}m)",
        )
        rc3.metric("RSP Koleina Maksymalna", f"{max_koleina_rsp:.1f} mm")

        # Obliczenia Wirtualnej Łaty dla LCMS
        calc_lcms_rut_l, calc_lcms_rut_r = None, None
        if prof_lcms_data is not None:
            x_lcms_orig = prof_lcms_data["x"].copy()
            z_lcms_raw = prof_lcms_data["z"]
            z_lcms_smooth = smooth_profile(z_lcms_raw, window_size=31)

            mid_mask_raw = (x_lcms_orig >= -0.5) & (x_lcms_orig <= 0.5)
            if np.any(mid_mask_raw):
                idx_c_lcms = np.where(mid_mask_raw)[0][
                    np.argmax(z_lcms_smooth[mid_mask_raw])
                ]
            else:
                idx_c_lcms = np.argmin(np.abs(x_lcms_orig))

            x_lcms_l_raw = x_lcms_orig[: idx_c_lcms + 1]
            z_lcms_l_raw = z_lcms_smooth[: idx_c_lcms + 1]
            x_lcms_r_raw = x_lcms_orig[idx_c_lcms:]
            z_lcms_r_raw = z_lcms_smooth[idx_c_lcms:]

            lata_raw_l = get_upper_hull(x_lcms_l_raw, z_lcms_l_raw)
            lata_raw_r = get_upper_hull(x_lcms_r_raw, z_lcms_r_raw)

            calc_lcms_rut_l = float(np.max(lata_raw_l - z_lcms_l_raw))
            calc_lcms_rut_r = float(np.max(lata_raw_r - z_lcms_r_raw))

        st.markdown("##### 🟩 Dane z Systemu LCMS")
        lc1, lc2, lc3 = st.columns(3)
        if kol_lcms_l is not None and kol_lcms_p is not None:
            # Reaktywny wybór wartości w zależności od opcji 5
            if zrodlo_lcms == "Z pliku LCMS (surowe B2/C2)":
                disp_l, disp_p = kol_lcms_l, kol_lcms_p
                label_sub = "(z pliku B2/C2)"
            elif zrodlo_lcms == "Wyliczone z profilu (Wirtualna Łata)":
                disp_l, disp_p = calc_lcms_rut_l, calc_lcms_rut_r
                label_sub = "(z Łaty DSN)"
            else:
                disp_l, disp_p = kol_lcms_l, kol_lcms_p
                label_sub = f"(Plik: {kol_lcms_l:.1f} / Łata: {calc_lcms_rut_l:.1f})"

            max_disp = max(disp_l, disp_p)
            lc1.metric(
                f"LCMS Koleina Lewa {label_sub}",
                f"{disp_l:.1f} mm",
                delta=f"{koleina_lewa_mm - disp_l:+.1f} mm vs RSP",
            )
            lc2.metric(
                f"LCMS Koleina Prawa {label_sub}",
                f"{disp_p:.1f} mm",
                delta=f"{koleina_prawa_mm - disp_p:+.1f} mm vs RSP",
            )
            lc3.metric(
                "LCMS Koleina Maksymalna",
                f"{max_disp:.1f} mm",
                delta=f"{max_koleina_rsp - max_disp:+.1f} mm vs RSP",
            )
        else:
            lc1.info("Brak profilu LCMS dla wybranego pikietażu.")

        # WYKRES PRZEKROJU POPRZECZNEGO
        fig = go.Figure()

        # Profil RSP (Zaokrąglona linia Spline)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name="Profil RSP (Spline)",
                line=dict(
                    shape="spline", smoothing=1.2, color="royalblue", width=3
                ),
                marker=dict(size=7, color="navy"),
                hovertemplate="Poz X: %{x} m<br>Wys Y: %{y:.1f} mm<extra></extra>",
            )
        )

        if pokaz_late_rsp:
            fig.add_trace(
                go.Scatter(
                    x=x_left,
                    y=lata_lewa_y,
                    mode="lines",
                    name="Łata RSP Lewa",
                    line=dict(color="crimson", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_right,
                    y=lata_prawa_y,
                    mode="lines",
                    name="Łata RSP Prawa",
                    line=dict(color="darkorange", width=2),
                )
            )

            x_kl, y_kl_prof, y_kl_lata = (
                x_vals[idx_kol_l],
                y_vals[idx_kol_l],
                lata_lewa_y[idx_kol_l],
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_kl, x_kl],
                    y=[y_kl_lata, y_kl_prof],
                    mode="lines+markers",
                    name=f"Głębokość RSP Lewa ({koleina_lewa_mm:.1f} mm)",
                    line=dict(color="crimson", width=3),
                    marker=dict(size=6, symbol="triangle-down"),
                )
            )
            fig.add_annotation(
                x=x_kl,
                y=(y_kl_lata + y_kl_prof) / 2,
                text=f" <b>RSP: {koleina_lewa_mm:.1f} mm</b>",
                showarrow=False,
                font=dict(color="crimson", size=12),
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="crimson",
                borderwidth=1,
                xanchor="right",
            )

            x_kp, y_kp_prof, y_kp_lata = (
                x_vals[idx_kol_p],
                y_vals[idx_kol_p],
                lata_prawa_y[idx_kol_p_rel],
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_kp, x_kp],
                    y=[y_kp_lata, y_kp_prof],
                    mode="lines+markers",
                    name=f"Głębokość RSP Prawa ({koleina_prawa_mm:.1f} mm)",
                    line=dict(color="darkorange", width=3),
                    marker=dict(size=6, symbol="triangle-down"),
                )
            )
            fig.add_annotation(
                x=x_kp,
                y=(y_kp_prof + y_kp_lata) / 2,
                text=f" <b>RSP: {koleina_prawa_mm:.1f} mm</b>",
                showarrow=False,
                font=dict(color="darkorange", size=12),
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="darkorange",
                borderwidth=1,
                xanchor="right",
            )

        # Obróbka i Łata LCMS z precyzyjną geometrią X
        if prof_lcms_data is not None:
            x_lcms_orig = prof_lcms_data["x"].copy()
            z_lcms_raw = prof_lcms_data["z"]
            z_lcms_smooth = smooth_profile(z_lcms_raw, window_size=31)

            rsp_x_max = np.max(x_vals)
            rsp_x_min = np.min(x_vals)

            x_lcms = x_lcms_orig.copy()

            # Precyzyjne dopasowanie geometrii krawędzi
            if skaluj_szerokosc and len(x_lcms) > 1:
                lcms_min = np.min(x_lcms)
                lcms_max = np.max(x_lcms)
                x_lcms = rsp_x_min + (x_lcms - lcms_min) * (
                    (rsp_x_max - rsp_x_min) / (lcms_max - lcms_min)
                )
            elif zaczep_prawa_krawedz and len(x_lcms) > 0:
                shift_right = rsp_x_max - np.max(x_lcms)
                x_lcms = x_lcms + shift_right

            x_lcms = x_lcms + manual_x_shift_lcms

            mid_mask_raw = (x_lcms >= -0.5) & (x_lcms <= 0.5)
            if np.any(mid_mask_raw):
                idx_crown_lcms = np.where(mid_mask_raw)[0][
                    np.argmax(z_lcms_smooth[mid_mask_raw])
                ]
            else:
                idx_crown_lcms = np.argmin(np.abs(x_lcms))

            x_lcms_l = x_lcms[: idx_crown_lcms + 1]
            z_lcms_l_raw = z_lcms_smooth[: idx_crown_lcms + 1]
            x_lcms_r = x_lcms[idx_crown_lcms:]
            z_lcms_r_raw = z_lcms_smooth[idx_crown_lcms:]

            lata_raw_l = get_upper_hull(x_lcms_l, z_lcms_l_raw)
            lata_raw_r = get_upper_hull(x_lcms_r, z_lcms_r_raw)

            if wyrownaj_spadek_lcms and len(x_lcms) > 10:
                p_trend = np.polyfit(x_lcms, z_lcms_smooth, 1)
                trend_line = p_trend[0] * x_lcms
                z_lcms_leveled = z_lcms_smooth - trend_line
                lata_plot_l = lata_raw_l - trend_line[: idx_crown_lcms + 1]
                lata_plot_r = lata_raw_r - trend_line[idx_crown_lcms:]
            else:
                z_lcms_leveled = z_lcms_smooth
                lata_plot_l = lata_raw_l
                lata_plot_r = lata_raw_r

            idx_mid_lcms = np.argmin(np.abs(x_lcms))
            z_ref_lcms = z_lcms_leveled[idx_mid_lcms]

            z_lcms_plot = z_lcms_leveled - z_ref_lcms
            lata_plot_l_zero = lata_plot_l - z_ref_lcms
            lata_plot_r_zero = lata_plot_r - z_ref_lcms

            fig.add_trace(
                go.Scatter(
                    x=x_lcms,
                    y=z_lcms_plot,
                    mode="lines",
                    name=f"Profil LCMS nr {wybrany_profil_lcms}",
                    line=dict(color="forestgreen", width=2),
                )
            )

            if pokaz_late_lcms:
                fig.add_trace(
                    go.Scatter(
                        x=x_lcms_l,
                        y=lata_plot_l_zero,
                        mode="lines",
                        name="Łata LCMS Lewa",
                        line=dict(color="limegreen", width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_lcms_r,
                        y=lata_plot_r_zero,
                        mode="lines",
                        name="Łata LCMS Prawa",
                        line=dict(color="lime", width=2),
                    )
                )

                # Wartość do wyświetlenia w etykietach LCMS w zależności od opcji 5
                val_l_show = (
                    kol_lcms_l
                    if zrodlo_lcms == "Z pliku LCMS (surowe B2/C2)"
                    and kol_lcms_l is not None
                    else calc_lcms_rut_l
                )
                val_r_show = (
                    kol_lcms_p
                    if zrodlo_lcms == "Z pliku LCMS (surowe B2/C2)"
                    and kol_lcms_p is not None
                    else calc_lcms_rut_r
                )

                # Etykieta LCMS LEWA
                diff_lcms_l = lata_plot_l_zero - (
                    z_lcms_plot[: idx_crown_lcms + 1]
                )
                idx_k_l_c = np.argmax(diff_lcms_l)
                x_kl_c = x_lcms_l[idx_k_l_c]
                y_kl_c_prof = z_lcms_plot[idx_k_l_c]
                y_kl_c_lata = lata_plot_l_zero[idx_k_l_c]

                fig.add_trace(
                    go.Scatter(
                        x=[x_kl_c, x_kl_c],
                        y=[y_kl_c_lata, y_kl_c_prof],
                        mode="lines+markers",
                        name=f"Głębokość LCMS Lewa ({val_l_show:.1f} mm)",
                        line=dict(color="forestgreen", width=2.5),
                        marker=dict(size=5, symbol="triangle-down"),
                    )
                )
                fig.add_annotation(
                    x=x_kl_c,
                    y=(y_kl_c_lata + y_kl_c_prof) / 2,
                    text=f"<b>LCMS: {val_l_show:.1f} mm</b> ",
                    showarrow=False,
                    font=dict(color="forestgreen", size=12),
                    bgcolor="rgba(235, 255, 235, 0.95)",
                    bordercolor="forestgreen",
                    borderwidth=1,
                    xanchor="left",
                )

                # Etykieta LCMS PRAWA
                diff_lcms_r = lata_plot_r_zero - (z_lcms_plot[idx_crown_lcms:])
                idx_k_r_c_rel = np.argmax(diff_lcms_r)
                idx_k_r_c = idx_crown_lcms + idx_k_r_c_rel
                x_kp_c = x_lcms[idx_k_r_c]
                y_kp_c_prof = z_lcms_plot[idx_k_r_c]
                y_kp_c_lata = lata_plot_r_zero[idx_k_r_c_rel]

                fig.add_trace(
                    go.Scatter(
                        x=[x_kp_c, x_kp_c],
                        y=[y_kp_c_lata, y_kp_c_prof],
                        mode="lines+markers",
                        name=f"Głębokość LCMS Prawa ({val_r_show:.1f} mm)",
                        line=dict(color="forestgreen", width=2.5),
                        marker=dict(size=5, symbol="triangle-down"),
                    )
                )
                fig.add_annotation(
                    x=x_kp_c,
                    y=(y_kp_c_lata + y_kp_c_prof) / 2,
                    text=f"<b>LCMS: {val_r_show:.1f} mm</b> ",
                    showarrow=False,
                    font=dict(color="forestgreen", size=12),
                    bgcolor="rgba(235, 255, 235, 0.95)",
                    bordercolor="forestgreen",
                    borderwidth=1,
                    xanchor="left",
                )

        fig.add_vline(
            x=0,
            line_width=1,
            line_dash="dot",
            line_color="gray",
            annotation_text="Oś L11 (0.0 m)",
        )

        fig.update_layout(
            title=f"Przekrój Poprzeczny – Profil RSP nr {wybrany_profil}",
            xaxis_title="Pozycja czujnika na belce [m]",
            yaxis_title=y_label,
            height=580,
            hovermode="x unified",
            xaxis=dict(range=[-1.85, 1.85], dtick=0.2),
            yaxis=dict(autorange=True if tryb_zero_l11 else "reversed"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # WYKRESY PODŁUŻNE NA CAŁEJ TRASIE
        st.markdown("---")
        st.subheader("📈 Przebieg Kolein na Całej Trasie (RSP vs LCMS)")

        df_all_rsp_ruts = compute_all_rsp_ruts_cached(df_rsp, tuple(pozycja_x))

        has_lcms = lcms_profiles is not None and len(lcms_profiles) > 0
        if has_lcms:
            list_lcms_p = sorted(lcms_profiles.keys())
            df_lcms_calc_all = compute_all_lcms_calc_ruts_cached(lcms_profiles)

            lcms_file_rut_l = [lcms_profiles[p]["rut_l"] for p in list_lcms_p]
            lcms_file_rut_r = [lcms_profiles[p]["rut_r"] for p in list_lcms_p]
            lcms_file_rut_max = np.maximum(lcms_file_rut_l, lcms_file_rut_r)

            lcms_calc_rut_l = df_lcms_calc_all["LCMS_Calc_Rut_L"].values
            lcms_calc_rut_r = df_lcms_calc_all["LCMS_Calc_Rut_R"].values
            lcms_calc_rut_max = df_lcms_calc_all["LCMS_Calc_Rut_Max"].values

            # Przesunięcia sekcyjne pikietażu
            lcms_p_shifted = []
            for p in list_lcms_p:
                if podziel_na_sekcje:
                    idx_s_p = int((p - min_p) // sekcja_m)
                    off_s = bazowy_offset + sekcje_offsets.get(idx_s_p, 0.0)
                else:
                    off_s = bazowy_offset
                lcms_p_shifted.append(p - off_s)

        tab_l, tab_r, tab_max, tab_all = st.tabs([
            "👈 Koleina Lewa",
            "👉 Koleina Prawa",
            "🔝 Koleina Maksymalna",
            "📊 Zbiorcze",
        ])

        with tab_l:
            fig_l = go.Figure()
            fig_l.add_trace(
                go.Scatter(
                    x=df_all_rsp_ruts["Profil"],
                    y=df_all_rsp_ruts["RSP_Rut_L"],
                    mode="lines",
                    name="RSP Koleina Lewa (Łata DSN)",
                    line=dict(color="royalblue", width=2),
                )
            )
            if has_lcms:
                if zrodlo_lcms in [
                    "Z pliku LCMS (surowe B2/C2)",
                    "Oba źródła jednocześnie",
                ]:
                    fig_l.add_trace(
                        go.Scatter(
                            x=lcms_p_shifted,
                            y=lcms_file_rut_l,
                            mode="lines",
                            name="LCMS Lewa (Z pliku B2)",
                            line=dict(color="forestgreen", width=2),
                        )
                    )
                if zrodlo_lcms in [
                    "Wyliczone z profilu (Wirtualna Łata)",
                    "Oba źródła jednocześnie",
                ]:
                    fig_l.add_trace(
                        go.Scatter(
                            x=lcms_p_shifted,
                            y=lcms_calc_rut_l,
                            mode="lines",
                            name="LCMS Lewa (Wirtualna Łata)",
                            line=dict(
                                color="limegreen", width=1.5, dash="dash"
                            ),
                        )
                    )

            fig_l.add_vline(
                x=wybrany_profil,
                line_width=2,
                line_color="red",
                annotation_text=f"Profil {wybrany_profil}",
            )
            fig_l.update_layout(
                title="KOLEINA LEWA [mm]",
                xaxis_title="Numer Profilu RSP",
                yaxis_title="Głębokość koleiny [mm]",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig_l, use_container_width=True)

        with tab_r:
            fig_r = go.Figure()
            fig_r.add_trace(
                go.Scatter(
                    x=df_all_rsp_ruts["Profil"],
                    y=df_all_rsp_ruts["RSP_Rut_R"],
                    mode="lines",
                    name="RSP Koleina Prawa (Łata DSN)",
                    line=dict(color="darkorange", width=2),
                )
            )
            if has_lcms:
                if zrodlo_lcms in [
                    "Z pliku LCMS (surowe B2/C2)",
                    "Oba źródła jednocześnie",
                ]:
                    fig_r.add_trace(
                        go.Scatter(
                            x=lcms_p_shifted,
                            y=lcms_file_rut_r,
                            mode="lines",
                            name="LCMS Prawa (Z pliku C2)",
                            line=dict(color="forestgreen", width=2),
                        )
                    )
                if zrodlo_lcms in [
                    "Wyliczone z profilu (Wirtualna Łata)",
                    "Oba źródła jednocześnie",
                ]:
                    fig_r.add_trace(
                        go.Scatter(
                            x=lcms_p_shifted,
                            y=lcms_calc_rut_r,
                            mode="lines",
                            name="LCMS Prawa (Wirtualna Łata)",
                            line=dict(
                                color="limegreen", width=1.5, dash="dash"
                            ),
                        )
                    )

            fig_r.add_vline(
                x=wybrany_profil,
                line_width=2,
                line_color="red",
                annotation_text=f"Profil {wybrany_profil}",
            )
            fig_r.update_layout(
                title="KOLEINA PRAWA [mm]",
                xaxis_title="Numer Profilu RSP",
                yaxis_title="Głębokość koleiny [mm]",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with tab_max:
            fig_m = go.Figure()
            fig_m.add_trace(
                go.Scatter(
                    x=df_all_rsp_ruts["Profil"],
                    y=df_all_rsp_ruts["RSP_Rut_Max"],
                    mode="lines",
                    name="RSP Koleina Max (Łata DSN)",
                    line=dict(color="crimson", width=2.5),
                )
            )
            if has_lcms:
                if zrodlo_lcms in [
                    "Z pliku LCMS (surowe B2/C2)",
                    "Oba źródła jednocześnie",
                ]:
                    fig_m.add_trace(
                        go.Scatter(
                            x=lcms_p_shifted,
                            y=lcms_file_rut_max,
                            mode="lines",
                            name="LCMS Koleina Max (Z pliku B2/C2)",
                            line=dict(color="forestgreen", width=2.5),
                        )
                    )
                if zrodlo_lcms in [
                    "Wyliczone z profilu (Wirtualna Łata)",
                    "Oba źródła jednocześnie",
                ]:
                    fig_m.add_trace(
                        go.Scatter(
                            x=lcms_p_shifted,
                            y=lcms_calc_rut_max,
                            mode="lines",
                            name="LCMS Koleina Max (Wirtualna Łata)",
                            line=dict(
                                color="limegreen", width=1.5, dash="dash"
                            ),
                        )
                    )

            fig_m.add_vline(
                x=wybrany_profil,
                line_width=2,
                line_color="red",
                annotation_text=f"Profil {wybrany_profil}",
            )
            fig_m.update_layout(
                title="KOLEINA MAKSYMALNA [mm]",
                xaxis_title="Numer Profilu RSP",
                yaxis_title="Głębokość koleiny [mm]",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig_m, use_container_width=True)

        with tab_all:
            fig_all = go.Figure()
            fig_all.add_trace(
                go.Scatter(
                    x=df_all_rsp_ruts["Profil"],
                    y=df_all_rsp_ruts["RSP_Rut_L"],
                    mode="lines",
                    name="RSP Lewa",
                    line=dict(color="royalblue", width=2),
                )
            )
            fig_all.add_trace(
                go.Scatter(
                    x=df_all_rsp_ruts["Profil"],
                    y=df_all_rsp_ruts["RSP_Rut_R"],
                    mode="lines",
                    name="RSP Prawa",
                    line=dict(color="darkorange", width=2),
                )
            )

            if has_lcms:
                if zrodlo_lcms in [
                    "Z pliku LCMS (surowe B2/C2)",
                    "Oba źródła jednocześnie",
                ]:
                    fig_all.add_trace(
                        go.Scatter(
                            x=lcms_p_shifted,
                            y=lcms_file_rut_l,
                            mode="lines",
                            name="LCMS Lewa (Plik B2)",
                            line=dict(color="forestgreen", width=1.5),
                        )
                    )
                    fig_all.add_trace(
                        go.Scatter(
                            x=lcms_p_shifted,
                            y=lcms_file_rut_r,
                            mode="lines",
                            name="LCMS Prawa (Plik C2)",
                            line=dict(color="limegreen", width=1.5),
                        )
                    )
                if zrodlo_lcms in [
                    "Wyliczone z profilu (Wirtualna Łata)",
                    "Oba źródła jednocześnie",
                ]:
                    fig_all.add_trace(
                        go.Scatter(
                            x=lcms_p_shifted,
                            y=lcms_calc_rut_l,
                            mode="lines",
                            name="LCMS Lewa (Łata)",
                            line=dict(
                                color="forestgreen", width=1.0, dash="dash"
                            ),
                        )
                    )
                    fig_all.add_trace(
                        go.Scatter(
                            x=lcms_p_shifted,
                            y=lcms_calc_rut_r,
                            mode="lines",
                            name="LCMS Prawa (Łata)",
                            line=dict(
                                color="limegreen", width=1.0, dash="dash"
                            ),
                        )
                    )

            fig_all.add_vline(
                x=wybrany_profil,
                line_width=2,
                line_color="red",
                annotation_text=f"Profil {wybrany_profil}",
            )
            fig_all.update_layout(
                title="Zestawienie Zbiorcze: RSP vs LCMS",
                xaxis_title="Numer Profilu RSP",
                yaxis_title="Głębokość koleiny [mm]",
                height=450,
                hovermode="x unified",
            )
            st.plotly_chart(fig_all, use_container_width=True)

        # MAPA GPS
        if df_rsp_gps is not None or df_xml_gps is not None:
            st.markdown("---")
            st.subheader("🗺️ MAPA GPS i Przesunięcie Terenowe")

            idx_rsp_gps = (
                min(max(0, wybrany_profil - min_p), len(df_rsp_gps) - 1)
                if df_rsp_gps is not None
                else None
            )
            row_rsp_gps = (
                df_rsp_gps.iloc[idx_rsp_gps] if idx_rsp_gps is not None else None
            )

            idx_lcms_gps = (
                min(max(0, wybrany_profil_lcms - 1), len(df_xml_gps) - 1)
                if df_xml_gps is not None
                else None
            )
            row_lcms_gps = (
                df_xml_gps.iloc[idx_lcms_gps]
                if idx_lcms_gps is not None
                else None
            )

            dist_m = None
            if row_rsp_gps is not None and row_lcms_gps is not None:
                dist_m = haversine_distance_m(
                    row_rsp_gps["LAT"],
                    row_rsp_gps["LON"],
                    row_lcms_gps["LAT"],
                    row_lcms_gps["LON"],
                )

                lats_lcms = df_xml_gps["LAT"].values
                lons_lcms = df_xml_gps["LON"].values
                dists_all = [
                    haversine_distance_m(
                        row_rsp_gps["LAT"], row_rsp_gps["LON"], la, lo
                    )
                    for la, lo in zip(lats_lcms, lons_lcms)
                ]
                best_match_idx = int(np.argmin(dists_all))
                best_match_dist = dists_all[best_match_idx]
                sugerowany_offset = (best_match_idx + 1) - wybrany_profil

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric(
                    "Odległość GPS (Obecny Offset)",
                    f"{dist_m:.2f} m",
                    delta=f"Przesunięcie: {całkowity_offset:+.1f} m",
                )
                mc2.metric(
                    "Sugerowany Offset Przestrzenny",
                    f"{sugerowany_offset:+} m",
                    delta_color="normal",
                )
                mc3.metric(
                    "Najmniejszy Dystans Terenowy",
                    f"{best_match_dist:.2f} m",
                    f"Punkt LCMS #{best_match_idx + 1}",
                )

            fig_map = go.Figure()
            center_lat, center_lon = 52.0, 19.0

            if row_rsp_gps is not None:
                center_lat, center_lon = (
                    row_rsp_gps["LAT"],
                    row_rsp_gps["LON"],
                )
            elif row_lcms_gps is not None:
                center_lat, center_lon = (
                    row_lcms_gps["LAT"],
                    row_lcms_gps["LON"],
                )

            if df_rsp_gps is not None:
                fig_map.add_trace(
                    make_map_trace(
                        lat=df_rsp_gps["LAT"],
                        lon=df_rsp_gps["LON"],
                        mode="markers+lines",
                        marker=dict(size=5, color="royalblue"),
                        line=dict(width=2, color="royalblue"),
                        name=f"Trasa RSP GPS ({len(df_rsp_gps)} pkt)",
                        text=[
                            f"Profil RSP: {p_i}"
                            for p_i in df_rsp_gps["Profil_idx"]
                        ],
                    )
                )
                if row_rsp_gps is not None:
                    fig_map.add_trace(
                        make_map_trace(
                            lat=[row_rsp_gps["LAT"]],
                            lon=[row_rsp_gps["LON"]],
                            mode="markers",
                            marker=dict(size=14, color="red"),
                            name=f"Profil RSP {wybrany_profil}",
                            text=[f"Pozycja RSP nr {wybrany_profil}"],
                        )
                    )

            if df_xml_gps is not None:
                fig_map.add_trace(
                    make_map_trace(
                        lat=df_xml_gps["LAT"],
                        lon=df_xml_gps["LON"],
                        mode="markers+lines",
                        marker=dict(size=4, color="forestgreen"),
                        line=dict(width=1.5, color="forestgreen"),
                        name=f"Trasa LCMS XML ({len(df_xml_gps)} pkt)",
                        text=[
                            f"Pikietaż LCMS: {st_m:.1f}m<br>Plik: {fn}"
                            for st_m, fn in zip(
                                df_xml_gps["Station_m"], df_xml_gps["Filename"]
                            )
                        ],
                    )
                )
                if row_lcms_gps is not None:
                    fig_map.add_trace(
                        make_map_trace(
                            lat=[row_lcms_gps["LAT"]],
                            lon=[row_lcms_gps["LON"]],
                            mode="markers",
                            marker=dict(size=12, color="darkorange"),
                            name=f"Profil LCMS {wybrany_profil_lcms}",
                            text=[
                                f"LCMS Profil {wybrany_profil_lcms}<br>Pikietaż:"
                                f" {row_lcms_gps['Station_m']}m"
                            ],
                        )
                    )

            if (
                row_rsp_gps is not None
                and row_lcms_gps is not None
                and dist_m is not None
            ):
                fig_map.add_trace(
                    make_map_trace(
                        lat=[row_rsp_gps["LAT"], row_lcms_gps["LAT"]],
                        lon=[row_rsp_gps["LON"], row_lcms_gps["LON"]],
                        mode="lines",
                        line=dict(color="crimson", width=3),
                        name=f"Przesunięcie terenowe: {dist_m:.2f} m",
                        text=[
                            f"Przesunięcie pozycji: {dist_m:.2f} m",
                            f"Przesunięcie pozycji: {dist_m:.2f} m",
                        ],
                    )
                )

            map_config = dict(
                center=dict(lat=center_lat, lon=center_lon), zoom=15
            )

            try:
                if hasattr(go, "Scattermap"):
                    fig_map.update_layout(
                        map_style="open-street-map",
                        map=map_config,
                        margin=dict(l=0, r=0, t=20, b=0),
                        height=480,
                    )
                else:
                    fig_map.update_layout(
                        mapbox_style="open-street-map",
                        mapbox=map_config,
                        margin=dict(l=0, r=0, t=20, b=0),
                        height=480,
                    )
            except Exception:
                fig_map.update_layout(
                    margin=dict(l=0, r=0, t=20, b=0), height=480
                )

            st.plotly_chart(fig_map, use_container_width=True)

        st.subheader("Odczyty Czujników RSP [mm]:")
        df_tbl = pd.DataFrame(
            [y_vals],
            columns=[
                f"L{i+1} ({x_vals[i]:.2f}m)" for i in range(ile_laserow)
            ],
        )
        st.dataframe(df_tbl, use_container_width=True)

    else:
        st.warning(f"Brak danych RSP dla profilu {wybrany_profil}.")
else:
    st.info("👈 Wgraj pliki z danymi RSP w panelu bocznym po lewej stronie.")
