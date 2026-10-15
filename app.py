import io
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 0. KONFIGURACJA STRONY I STAŁE GEOMETRII
# ==========================================
st.set_page_config(
    page_title="RSP vs LCMS – System Analizy Kolein DSN 2019",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
# 1. ZABEZPIECZENIE TYPÓW DLA PYARROW
# ==========================================
def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Konwertuje kolumny 'object' na ciągi tekstowe, chroniąc PyArrow przed awariami."""
    if df is None or df.empty:
        return df
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].astype(str)
    return df_clean


# ==========================================
# 2. METODY DSN 2019 I GEOMETRIA PROFILU
# ==========================================
def get_dsn_class(h_val):
    if h_val is None or (isinstance(h_val, float) and np.isnan(h_val)):
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
    mapping = {"A": "#28a745", "B": "#ffc107", "C": "#fd7e14", "D": "#dc3545"}
    return mapping.get(klasa, "#6c757d")


def safe_float(val):
    if val is None or not str(val).strip():
        return None
    try:
        return float(str(val).replace(",", ".").strip())
    except ValueError:
        return None


def smooth_profile(z_vals, window_size=31):
    if len(z_vals) < window_size:
        return z_vals
    return (
        pd.Series(z_vals)
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .values
    )


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
    if len(x_sec) < 2:
        return y_sec
    upper = get_upper_hull_pts(x_sec, y_sec)
    x_h = [p[0] for p in upper]
    y_h = [p[1] for p in upper]
    return np.interp(x_sec, x_h, y_h)


def auto_align_transverse_x(x_rsp, y_rsp, x_lcms, z_lcms):
    """Wylicza optymalne przesunięcie X, aby idealnie nałożyć profil LCMS na RSP."""
    x_grid = np.linspace(-1.5, 1.5, 300)
    y_rsp_interp = np.interp(x_grid, x_rsp, y_rsp, left=np.nan, right=np.nan)

    best_shift = 0.0
    min_err = 1e9

    for shift in np.arange(-1.0, 1.05, 0.01):
        z_lcms_interp = np.interp(
            x_grid, x_lcms + shift, z_lcms, left=np.nan, right=np.nan
        )
        valid = ~np.isnan(y_rsp_interp) & ~np.isnan(z_lcms_interp)
        if np.sum(valid) > 50:
            diff = (y_rsp_interp[valid] - np.mean(y_rsp_interp[valid])) - (
                z_lcms_interp[valid] - np.mean(z_lcms_interp[valid])
            )
            err = np.mean(diff**2)
            if err < min_err:
                min_err = err
                best_shift = shift
    return float(np.round(best_shift, 2))


def get_total_offset_for_profil(
    p, base_offset, split_sections, sec_size, sec_offsets, min_p
):
    """Dynamiczne pobieranie offsetu podłużnego dla dowolnego profilu na trasie."""
    if not split_sections:
        return base_offset
    sec_idx = int((p - min_p) // sec_size)
    return base_offset + sec_offsets.get(sec_idx, 0.0)


# ==========================================
# 3. PARSERY PLIKÓW RSP, LCMS, XML, GPX
# ==========================================
def parse_kolc_xml(file_bytes):
    try:
        tree = ET.ElementTree(ET.fromstring(file_bytes))
        root = tree.getroot()
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        records = []
        odcinki = root.findall("ODCINEK")
        if not odcinki:
            odcinki = [e for e in root.iter() if e.tag == "ODCINEK"]

        for idx, odc in enumerate(odcinki):
            km_p = odc.findtext("KM_P")
            km_k = odc.findtext("KM_K")
            l_kol = odc.findtext("L_KOL")
            p_kol = odc.findtext("P_KOL")
            pel_kol = odc.findtext("PEL_KOL")
            lat = odc.findtext("LAT")
            lon = odc.findtext("LON")

            st_p = safe_float(km_p)
            st_m = st_p * 1000.0 if st_p is not None else float(idx + 1)

            records.append({
                "Profil": idx + 1,
                "Km_P": st_p,
                "Km_K": safe_float(km_k),
                "Station_m": st_m,
                "L_KOL": safe_float(l_kol) or 0.0,
                "P_KOL": safe_float(p_kol) or 0.0,
                "PEL_KOL": safe_float(pel_kol) or 0.0,
                "LAT": safe_float(lat),
                "LON": safe_float(lon),
            })
        df = pd.DataFrame(records)
        return df if not df.empty else None
    except Exception:
        return None


def parse_native_rsp_file(file_bytes):
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.strip().split("\n")
    sensors, profiles_data, gps_data = [], [], []

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
                st_m, lat, lon, ele = (
                    float(parts[1]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                )
                gps_data.append({
                    "Km": st_m,
                    "LAT": lat,
                    "LON": lon,
                    "ELE": ele,
                    "Profil_idx": len(gps_data) + 1,
                })
            except (ValueError, IndexError):
                pass

    df_prof = (
        pd.DataFrame(
            profiles_data,
            columns=["Profil"] + [f"L{i+1}" for i in range(21)],
        )
        if profiles_data
        else None
    )
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
                    lat, lon = float(pt.attrib.get("lat")), float(
                        pt.attrib.get("lon")
                    )
                    ele_str = pt.findtext("ele")
                    ele = float(ele_str) if ele_str else 0.0
                    name = pt.findtext("name")
                    clean_km = (
                        float(name)
                        if name
                        and name.replace(".", "", 1)
                        .replace("-", "")
                        .isdigit()
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


@st.cache_data
def load_rsp_data(file_bytes, file_name):
    ext = file_name.lower()
    df_kolc = parse_kolc_xml(file_bytes)
    if df_kolc is not None:
        l_cols_exist = [
            c for c in df_kolc.columns if c.startswith("L") and c[1:].isdigit()
        ]
        if len(l_cols_exist) == 0:
            for i in range(1, 22):
                df_kolc[f"L{i}"] = 0.0
            if "L_KOL" in df_kolc.columns and "P_KOL" in df_kolc.columns:
                df_kolc["L6"] = df_kolc["L_KOL"]
                df_kolc["L16"] = df_kolc["P_KOL"]
        return df_kolc

    if ext.endswith(".rsp") or ext.endswith(".txt"):
        df_prof, _, _ = parse_native_rsp_file(file_bytes)
        if df_prof is not None:
            return df_prof

    if ext.endswith(".csv") or ext.endswith(".txt"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            pass

    if ext.endswith(".xlsx") or ext.endswith(".xls"):
        try:
            df = pd.read_excel(io.BytesIO(file_bytes))
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            pass

    raise ValueError(f"Nie rozpoznano formatu pliku RSP ({file_name}).")


@st.cache_data
def parse_rsp_gps_cached(file_bytes, file_name):
    ext = file_name.lower()
    df_kolc = parse_kolc_xml(file_bytes)
    if df_kolc is not None and "LAT" in df_kolc.columns:
        df_gps = df_kolc.dropna(subset=["LAT", "LON"]).copy()
        if not df_gps.empty:
            df_gps["Profil_idx"] = df_gps["Profil"]
            return df_gps

    if ext.endswith(".gpx"):
        df_gpx = parse_gpx_cached(file_bytes)
        if df_gpx is not None:
            return df_gpx

    if ext.endswith(".rsp") or ext.endswith(".txt"):
        _, df_gps, _ = parse_native_rsp_file(file_bytes)
        if df_gps is not None:
            return df_gps

    return None


@st.cache_data
def parse_single_lcms_raw(file_bytes, file_name):
    df_kolc = parse_kolc_xml(file_bytes)
    if df_kolc is not None:
        records = []
        for _, r in df_kolc.iterrows():
            st_m = r.get("Station_m", None)
            records.append({
                "Station_m": st_m,
                "x": np.array([-1.5, 1.5]),
                "z": np.array([0.0, 0.0]),
                "rut_l": float(r["L_KOL"]),
                "rut_r": float(r["P_KOL"]),
            })
        return records

    if file_name.lower().endswith(".xlsx") or file_name.lower().endswith(".xls"):
        try:
            df_excel = pd.read_excel(io.BytesIO(file_bytes), header=None)
            raw_rows = [
                [
                    str(v).replace(",", ".").strip() if pd.notna(v) else ""
                    for v in r.values
                ]
                for _, r in df_excel.iterrows()
            ]
        except Exception:
            return []
    else:
        text = file_bytes.decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        first_line = lines[0] if lines else ""
        sep = ";" if ";" in first_line else ("," if "," in first_line else "\t")
        raw_rows = [
            [p.strip().replace(",", ".") for p in line.split(sep)]
            for line in lines
        ]

    data_row_indices = []
    for idx, row in enumerate(raw_rows):
        if len(row) > 1:
            val_a = row[0].replace(".0", "").strip()
            if val_a.isdigit() and len(val_a) > 0:
                try:
                    float(row[1])
                    data_row_indices.append(idx)
                except ValueError:
                    pass

    records = []
    for k in range(0, len(data_row_indices) - 1, 2):
        r_left_h = data_row_indices[k]
        r_left_d = r_left_h + 1
        r_right_h = data_row_indices[k + 1]
        r_right_d = r_right_h + 1

        if r_right_d >= len(raw_rows):
            break

        try:
            row_lh, row_ld = raw_rows[r_left_h], raw_rows[r_left_d]
            row_rh, row_rd = raw_rows[r_right_h], raw_rows[r_right_d]

            st_val = safe_float(row_lh[0])
            rut_l = (
                float(row_lh[1])
                if len(row_lh) > 1 and row_lh[1] != ""
                else 0.0
            )
            rut_r = (
                float(row_lh[2])
                if len(row_lh) > 2 and row_lh[2] != ""
                else 0.0
            )

            x_left = [float(v) for v in row_lh[3:] if v != ""]
            z_left = [float(v) for v in row_ld[3:] if v != ""]
            x_right = [float(v) for v in row_rh[3:] if v != ""]
            z_right = [float(v) for v in row_rd[3:] if v != ""]

            min_len_l = min(len(x_left), len(z_left))
            x_l_arr, z_l_arr = (
                np.array(x_left[:min_len_l]),
                np.array(z_left[:min_len_l]),
            )
            min_len_r = min(len(x_right), len(z_right))
            x_r_arr, z_r_arr = (
                np.array(x_right[:min_len_r]),
                np.array(z_right[:min_len_r]),
            )

            x_full = np.concatenate([x_l_arr, x_r_arr])
            if len(x_full) > 0 and np.max(np.abs(x_full)) > 20:
                x_full = x_full / 1000.0

            z_full = np.concatenate([z_l_arr, z_r_arr])
            sort_order = np.argsort(x_full)

            records.append({
                "Station_m": st_val,
                "x": x_full[sort_order],
                "z": z_full[sort_order],
                "rut_l": rut_l,
                "rut_r": rut_r,
            })
        except Exception:
            continue

    return records


def load_multiple_lcms_files(uploaded_files_list):
    all_records = []
    for uf in uploaded_files_list:
        content = uf.getvalue()
        fname = uf.name
        recs = parse_single_lcms_raw(content, fname)
        all_records.extend(recs)

    if not all_records:
        return {}

    profiles = {}
    for idx, r in enumerate(all_records):
        st_m = r["Station_m"]
        pid = int(round(st_m)) if st_m is not None else idx + 1

        # Zapobieganie nadpisywaniu przy nakładających się indeksach w osobnych plikach
        while pid in profiles and r["Station_m"] is None:
            pid += 1

        profiles[pid] = {
            "x": r["x"],
            "z": r["z"],
            "rut_l": r["rut_l"],
            "rut_r": r["rut_r"],
            "Station_m": st_m,
        }

    return profiles


def parse_lcms_xml_multiple(uploaded_xml_list):
    dfs = []
    for uf in uploaded_xml_list:
        df_k = parse_kolc_xml(uf.getvalue())
        if df_k is not None and "LAT" in df_k.columns:
            res = df_k.dropna(subset=["LAT", "LON"]).copy()
            if not res.empty:
                res["Filename"] = uf.name
                dfs.append(res)
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        return merged.sort_values(by="Station_m").reset_index(drop=True)
    return None


# ==========================================
# 4. PRZELICZENIA I ANALIZY ZBIORCZE
# ==========================================
@st.cache_data
def compute_all_rsp_ruts_cached(df_rsp, x_vals_tuple):
    col_profil = "Profil" if "Profil" in df_rsp.columns else df_rsp.columns[0]
    l_cols = [
        c for c in df_rsp.columns if c.startswith("L") and c[1:].isdigit()
    ]
    laser_cols = (
        sorted(l_cols, key=lambda c: int(c[1:]))
        if l_cols
        else [c for c in df_rsp.columns if c != col_profil]
    )

    prof_ids = df_rsp[col_profil].values
    data_mat = df_rsp[laser_cols].values.astype(float)
    ile_laserow = data_mat.shape[1]
    i_min_mid, i_max_mid = int(ile_laserow * 0.3), int(ile_laserow * 0.7)

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


def auto_align_offset_clean(rsp_s, lcms_s, max_search=100):
    r = np.array(rsp_s) - np.mean(rsp_s)
    l = np.array(lcms_s) - np.mean(lcms_s)
    best_offset, max_corr = 0, -1e9

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


def build_bland_altman_fig(rsp_v, lcms_v, title="PRAWA KOLEINA"):
    means = (rsp_v + lcms_v) / 2.0
    diffs = lcms_v - rsp_v
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=means,
            y=diffs,
            mode="markers",
            name="Pomiary",
            marker=dict(
                size=7, color="#ffffcc", line=dict(width=1, color="#333333")
            ),
            hovertemplate=(
                "Średnia: %{x:.2f} mm<br>Różnica (LCMS-RSP): %{y:.2f}"
                " mm<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="lightgray")
    fig.add_hline(
        y=mean_diff,
        line_width=3,
        line_color="#0066cc",
        annotation_text=f"Średnia: {mean_diff:+.2f} mm",
        annotation_position="right",
        annotation_font=dict(color="#0066cc", size=13),
    )
    fig.add_hline(
        y=loa_upper,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"+1.96 SD: {loa_upper:+.2f} mm",
        annotation_position="right",
        annotation_font=dict(color="red", size=12),
    )
    fig.add_hline(
        y=loa_lower,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"-1.96 SD: {loa_lower:+.2f} mm",
        annotation_position="right",
        annotation_font=dict(color="red", size=12),
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=18)),
        xaxis_title="ŚREDNIA (LCMS, RSP) [mm]",
        yaxis_title="RÓŻNICA (LCMS - RSP) [mm]",
        height=480,
        template="plotly_white",
        hovermode="closest",
    )
    return fig, mean_diff, sd_diff, loa_upper, loa_lower


def build_rpd_table_df(rsp_v, lcms_v):
    mean_v = (rsp_v + lcms_v) / 2.0
    diff_abs = np.abs(lcms_v - rsp_v)
    rpd = np.where(mean_v > 0, (diff_abs / mean_v) * 100.0, 0.0)
    m_low, m_high, n_tot = rpd <= 30.0, rpd > 30.0, len(rsp_v)

    def sub_stat(m_grp, label):
        n_grp = int(np.sum(m_grp))
        if n_grp == 0:
            return {
                "zakres RPD": label,
                "n": "0",
                "%": "0%",
                "RSP > LCMS": "0",
                "Wyniki równe": "0",
                "LCMS > RSP": "0",
                "%RSP > LCMS": "0%",
                "%Wyniki równe": "0%",
                "%LCMS > RSP": "0%",
            }
        r_sub, l_sub = np.round(rsp_v[m_grp], 1), np.round(lcms_v[m_grp], 1)
        rsp_gt, eq, lcms_gt = (
            int(np.sum(r_sub > l_sub)),
            int(np.sum(r_sub == l_sub)),
            int(np.sum(l_sub > r_sub)),
        )
        return {
            "zakres RPD": label,
            "n": str(n_grp),
            "%": f"{(n_grp / n_tot)*100:.0f}%",
            "RSP > LCMS": str(rsp_gt),
            "Wyniki równe": str(eq),
            "LCMS > RSP": str(lcms_gt),
            "%RSP > LCMS": f"{(rsp_gt / n_grp)*100:.0f}%",
            "%Wyniki równe": f"{(eq / n_grp)*100:.0f}%",
            "%LCMS > RSP": f"{(lcms_gt / n_grp)*100:.0f}%",
        }

    return pd.DataFrame([
        sub_stat(m_low, "0-30%"),
        sub_stat(m_high, ">30%"),
        {
            "zakres RPD": "suma",
            "n": str(n_tot),
            "%": "100%",
            "RSP > LCMS": "-",
            "Wyniki równe": "-",
            "LCMS > RSP": "-",
            "%RSP > LCMS": "-",
            "%Wyniki równe": "-",
            "%LCMS > RSP": "-",
        },
    ])


def compute_50m_dsn_sections_full(df_ruts, step_m=1.0, section_m=50.0):
    profs_per_sec = max(1, int(round(section_m / step_m)))
    results = []
    num_sections = int(np.ceil(len(df_ruts) / float(profs_per_sec)))

    for i in range(num_sections):
        sub = df_ruts.iloc[i * profs_per_sec : (i + 1) * profs_per_sec]
        if len(sub) == 0:
            continue
        st_start, st_end = sub["Profil"].iloc[0], sub["Profil"].iloc[-1]

        rsp_l, rsp_r = sub["RSP_Rut_L"].values, sub["RSP_Rut_R"].values
        rsp_max = np.maximum(rsp_l, rsp_r)

        e_l, d_l = np.mean(rsp_l), np.std(rsp_l, ddof=0)
        e_r, d_r = np.mean(rsp_r), np.std(rsp_r, ddof=0)
        e_m, d_m = np.mean(rsp_max), np.std(rsp_max, ddof=0)

        hm_l, hm_r, hm_m = (
            int(np.round(e_l + 2 * d_l)),
            int(np.round(e_r + 2 * d_r)),
            int(np.round(e_m + 2 * d_m)),
        )

        rec = {
            "Odcinek": f"{st_start} - {st_end} m",
            "RSP L_KOL E(h)": round(e_l, 2),
            "RSP L_KOL Dh": round(d_l, 2),
            "RSP L_KOL Hm": hm_l,
            "RSP L_KOL Klasa": get_dsn_class(hm_l),
            "RSP P_KOL E(h)": round(e_r, 2),
            "RSP P_KOL Dh": round(d_r, 2),
            "RSP P_KOL Hm": hm_r,
            "RSP P_KOL Klasa": get_dsn_class(hm_r),
            "RSP MAX_KOL E(h)": round(e_m, 2),
            "RSP MAX_KOL Dh": round(d_m, 2),
            "RSP MAX_KOL Hm": hm_m,
            "RSP MAX_KOL Klasa": get_dsn_class(hm_m),
        }

        if (
            "LCMS_Rut_L" in sub.columns
            and not sub["LCMS_Rut_L"].dropna().empty
        ):
            sub_lcms = sub.dropna(subset=["LCMS_Rut_L", "LCMS_Rut_R"])
            if len(sub_lcms) > 0:
                lcms_l, lcms_r = (
                    sub_lcms["LCMS_Rut_L"].values,
                    sub_lcms["LCMS_Rut_R"].values,
                )
                lcms_max = np.maximum(lcms_l, lcms_r)
                e_ll, d_ll = np.mean(lcms_l), np.std(lcms_l, ddof=0)
                e_rr, d_rr = np.mean(lcms_r), np.std(lcms_r, ddof=0)
                e_mm, d_mm = np.mean(lcms_max), np.std(lcms_max, ddof=0)
                hm_ll, hm_rr, hm_mm = (
                    int(np.round(e_ll + 2 * d_ll)),
                    int(np.round(e_rr + 2 * d_rr)),
                    int(np.round(e_mm + 2 * d_mm)),
                )

                rec.update({
                    "LCMS L_KOL E(h)": round(e_ll, 2),
                    "LCMS L_KOL Dh": round(d_ll, 2),
                    "LCMS L_KOL Hm": hm_ll,
                    "LCMS L_KOL Klasa": get_dsn_class(hm_ll),
                    "LCMS P_KOL E(h)": round(e_rr, 2),
                    "LCMS P_KOL Dh": round(d_rr, 2),
                    "LCMS P_KOL Hm": hm_rr,
                    "LCMS P_KOL Klasa": get_dsn_class(hm_rr),
                    "LCMS MAX_KOL E(h)": round(e_mm, 2),
                    "LCMS MAX_KOL Dh": round(d_mm, 2),
                    "LCMS MAX_KOL Hm": hm_mm,
                    "LCMS MAX_KOL Klasa": get_dsn_class(hm_mm),
                })

        results.append(rec)

    return pd.DataFrame(results)


def make_50m_bar_chart(
    df_sec_dsn, param_prefix="P_KOL", title="Prawa Koleina (P_KOL)"
):
    fig = go.Figure()
    col_rsp_hm, col_rsp_klasa = (
        f"RSP {param_prefix} Hm",
        f"RSP {param_prefix} Klasa",
    )
    colors_rsp = [get_dsn_color(k) for k in df_sec_dsn[col_rsp_klasa]]

    fig.add_trace(
        go.Bar(
            x=df_sec_dsn["Odcinek"],
            y=df_sec_dsn[col_rsp_hm],
            name=f"RSP {title} (Hm)",
            marker_color=colors_rsp,
            text=[
                f"{hm} mm ({k})"
                for hm, k in zip(
                    df_sec_dsn[col_rsp_hm], df_sec_dsn[col_rsp_klasa]
                )
            ],
            textposition="auto",
        )
    )

    col_lcms_hm, col_lcms_klasa = (
        f"LCMS {param_prefix} Hm",
        f"LCMS {param_prefix} Klasa",
    )
    if col_lcms_hm in df_sec_dsn.columns:
        colors_lcms = [get_dsn_color(k) for k in df_sec_dsn[col_lcms_klasa]]
        fig.add_trace(
            go.Bar(
                x=df_sec_dsn["Odcinek"],
                y=df_sec_dsn[col_lcms_hm],
                name=f"LCMS {title} (Hm)",
                marker_color=colors_lcms,
                opacity=0.7,
                text=[
                    f"{hm} mm ({k})"
                    for hm, k in zip(
                        df_sec_dsn[col_lcms_hm], df_sec_dsn[col_lcms_klasa]
                    )
                ],
                textposition="auto",
            )
        )

    fig.add_hline(
        y=11,
        line_dash="dash",
        line_color="#ffc107",
        annotation_text="Próg B (11mm)",
    )
    fig.add_hline(
        y=21,
        line_dash="dash",
        line_color="#fd7e14",
        annotation_text="Próg C (21mm)",
    )
    fig.add_hline(
        y=31,
        line_dash="dash",
        line_color="#dc3545",
        annotation_text="Próg D (31mm)",
    )

    fig.update_layout(
        title=f"Ocena Hm dla Odcinków 50m – {title}",
        xaxis_title="Odcinek Pikietażu",
        yaxis_title="Ocena Hm [mm]",
        barmode="group",
        height=480,
    )
    return fig


# ==========================================
# 5. INICJALIZACJA SESSION STATE & PANEL
# ==========================================
if "glowny_offset_val" not in st.session_state:
    st.session_state["glowny_offset_val"] = 0.0
if "manual_x_shift_val" not in st.session_state:
    st.session_state["manual_x_shift_val"] = 0.0

st.title("⚡ System Analizy RSP + LCMS (DSN 2019 & GDDKiA)")

st.sidebar.header("1. Dane RSP (Profilograf / Surowe)")
uploaded_rsp = st.sidebar.file_uploader(
    "Wgraj plik RSP (.rsp / .xlsx / .csv / .txt / .xml / .2025)",
    type=None,
    key="rsp_input",
)
uploaded_rsp_gps = st.sidebar.file_uploader(
    "Wgraj plik GPS RSP (.gpx / .rsp / .csv / .txt / .xml / .2025)",
    type=None,
    key="rsp_gps_input",
)

st.sidebar.header("2. Dane LCMS (Wieloplikowo)")
uploaded_lcms_list = st.sidebar.file_uploader(
    "Wgraj pliki LCMS (.csv / .xlsx / .xml / .2025)",
    type=None,
    accept_multiple_files=True,
    key="lcms_input_list",
)
uploaded_xml_list = st.sidebar.file_uploader(
    "Wgraj pliki XML GPS z LCMS",
    type=None,
    accept_multiple_files=True,
    key="xml_input_list",
)

bytes_rsp, name_rsp = None, None
if uploaded_rsp is not None:
    bytes_rsp, name_rsp = uploaded_rsp.getvalue(), uploaded_rsp.name
elif os.path.exists("data/rsp.rsp"):
    with open("data/rsp.rsp", "rb") as f:
        bytes_rsp, name_rsp = f.read(), "rsp.rsp"

bytes_rsp_gps, name_rsp_gps = None, None
if uploaded_rsp_gps is not None:
    bytes_rsp_gps, name_rsp_gps = (
        uploaded_rsp_gps.getvalue(),
        uploaded_rsp_gps.name,
    )

auto_x_from_rsp = None
if bytes_rsp is not None and (
    name_rsp.lower().endswith(".rsp") or name_rsp.lower().endswith(".txt")
):
    _, _, auto_x_from_rsp = parse_native_rsp_file(bytes_rsp)

st.sidebar.header("3. Synchronizacja Podłużna (Pikietaż)")
glowny_offset = st.sidebar.number_input(
    "✍️ Bazowe przesunięcie LCMS względem RSP [m]:",
    min_value=-500.0,
    max_value=500.0,
    value=st.session_state["glowny_offset_val"],
    step=1.0,
    format="%.1f",
    key="offset_input_widget",
)
st.session_state["glowny_offset_val"] = glowny_offset

podziel_na_sekcje = st.sidebar.checkbox(
    "Dziel trasę na sekcje (osobne przesunięcia)", value=False
)
sekcja_m, sekcje_offsets = 500, {}
if podziel_na_sekcje:
    sekcja_m = st.sidebar.select_slider(
        "Długość sekcji [m]:", options=[250, 500, 1000, 2000], value=500
    )

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
    ["Z pliku LCMS (surowe B2/C2)", "Wyliczone z profilu (Wirtualna Łata)"],
)

st.sidebar.header("6. Dopasowanie Poprzeczne X (Lewo/Prawo)")
tryb_dopasowania_x = st.sidebar.radio(
    "Tryb bazowy dla X:",
    [
        "Swobodne przesunięcie (suwak)",
        "Zaczep do prawej krawędzi RSP",
        "Wyśrodkuj osie (Zero w środku)",
    ],
    index=0,
)

manual_x_shift_lcms = st.sidebar.slider(
    "↔️ Przesunięcie poprzeczne X LCMS [m]:",
    min_value=-2.00,
    max_value=2.00,
    value=st.session_state["manual_x_shift_val"],
    step=0.01,
    format="%.2f",
    key="manual_x_slider_widget",
)
st.session_state["manual_x_shift_val"] = manual_x_shift_lcms

skaluj_szerokosc = st.sidebar.checkbox(
    "📐 Skaluj szerokość LCMS do RSP", value=False
)
tryb_zero_l11 = st.sidebar.checkbox("Ustaw L11 jako Zero", value=True)
pokaz_late_rsp = st.sidebar.checkbox("Pokaż łatę i koleiny RSP", value=True)
wyrownaj_spadek_lcms = st.sidebar.checkbox(
    "Wyzeruj spadek poprzeczny LCMS", value=True
)
pokaz_late_lcms = st.sidebar.checkbox("Pokaż łatę i koleiny LCMS", value=True)

# Wczytywanie danych
df_rsp_gps = (
    parse_rsp_gps_cached(bytes_rsp_gps, name_rsp_gps)
    if bytes_rsp_gps
    else None
)
if df_rsp_gps is not None:
    st.sidebar.success(f"GPS RSP: {len(df_rsp_gps)} pkt.")

lcms_profiles = (
    load_multiple_lcms_files(uploaded_lcms_list)
    if uploaded_lcms_list
    else None
)
if lcms_profiles:
    st.sidebar.success(
        f"LCMS: {len(lcms_profiles)} profili z {len(uploaded_lcms_list)}"
        " plików!"
    )

df_xml_gps = (
    parse_lcms_xml_multiple(uploaded_xml_list)
    if uploaded_xml_list
    else None
)
if df_xml_gps is not None:
    st.sidebar.success(f"GPS LCMS: Łącznie {len(df_xml_gps)} pkt z XML!")


# ==========================================
# 6. GŁÓWNA LOGIKA I ZAKŁADKI (TABS)
# ==========================================
if bytes_rsp is not None:
    df_rsp = load_rsp_data(bytes_rsp, name_rsp)
    col_profil = "Profil" if "Profil" in df_rsp.columns else df_rsp.columns[0]
    min_p, max_p = int(df_rsp[col_profil].min()), int(df_rsp[col_profil].max())

    if podziel_na_sekcje:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📍 Przesunięcia Sekcyjne")
        num_sekcji = int(np.ceil((max_p - min_p + 1) / float(sekcja_m)))
        for s_i in range(num_sekcji):
            p_start_s = min_p + s_i * sekcja_m
            p_end_s = min(max_p, p_start_s + sekcja_m - 1)
            sekcje_offsets[s_i] = st.sidebar.number_input(
                f"Sekcja {s_i+1} ({p_start_s}-{p_end_s}m) [m]:",
                min_value=-200.0,
                max_value=200.0,
                value=0.0,
                step=0.5,
                format="%.1f",
                key=f"sec_off_input_{s_i}",
            )

    st.sidebar.header("7. Wybór profilu")
    wybrany_profil = st.sidebar.number_input(
        f"Numer profilu RSP [{min_p} - {max_p}]:",
        min_value=min_p,
        max_value=max_p,
        value=min_p,
        step=1,
    )

    bazowy_offset = st.session_state["glowny_offset_val"]
    całkowity_offset = get_total_offset_for_profil(
        wybrany_profil,
        bazowy_offset,
        podziel_na_sekcje,
        sekcja_m,
        sekcje_offsets,
        min_p,
    )
    wybrany_profil_lcms = int(round(wybrany_profil + całkowity_offset))

    if lcms_profiles:
        df_all_rsp_ruts = compute_all_rsp_ruts_cached(df_rsp, tuple(pozycja_x))
        list_lcms_p = sorted(lcms_profiles.keys())
        lcms_file_rut_max = np.maximum(
            [lcms_profiles[p]["rut_l"] for p in list_lcms_p],
            [lcms_profiles[p]["rut_r"] for p in list_lcms_p],
        )

        if st.sidebar.button("⚡ Automatyczna Synchronizacja Pikietażu"):
            sug_offset = float(
                auto_align_offset_clean(
                    df_all_rsp_ruts["RSP_Rut_Max"].values, lcms_file_rut_max
                )
            )
            st.session_state["glowny_offset_val"] = sug_offset
            st.sidebar.info(f"💡 Ustawiono sugerowany offset: {sug_offset:+} m")
            st.rerun()

    tab_main, tab_map, tab_dsn50, tab_ba_gddkia, tab_export = st.tabs([
        "📐 Przekrój Poprzeczny & Trasa",
        "🗺️ Mapa GPS Trasy",
        "📊 Ocena 50m DSN 2019 (Klasy A-D)",
        "📈 Analiza GDDKiA (Bland-Altman & RPD)",
        "📥 Eksport Raportu (Excel)",
    ])

    wiersz_rsp = df_rsp[df_rsp[col_profil] == wybrany_profil]

    if not wiersz_rsp.empty:
        l_cols = [
            c for c in df_rsp.columns if c.startswith("L") and c[1:].isdigit()
        ]
        kolumny_laserow = (
            sorted(l_cols, key=lambda c: int(c[1:]))
            if l_cols
            else [c for c in df_rsp.columns if c != col_profil]
        )

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
            ref_val = odczyty_raw[ile_laserow // 2]
            y_vals = ref_val - odczyty_raw
            y_label = "Wysokość względna do L11 [mm]"
        else:
            y_vals = odczyty_raw
            y_label = "Odczyt surowy czujnika [mm]"

        i_min_mid, i_max_mid = int(ile_laserow * 0.3), int(ile_laserow * 0.7)
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

        diff_l, diff_p = lata_lewa_y - y_left, lata_prawa_y - y_right
        idx_kol_l = np.argmax(diff_l)
        idx_kol_p_rel = np.argmax(diff_p)
        idx_kol_p = idx_crown + idx_kol_p_rel

        koleina_lewa_mm, koleina_prawa_mm = (
            diff_l[idx_kol_l],
            diff_p[idx_kol_p_rel],
        )
        max_koleina_rsp = max(koleina_lewa_mm, koleina_prawa_mm)

        kol_lcms_l, kol_lcms_p, prof_lcms_data = None, None, None
        if lcms_profiles and wybrany_profil_lcms in lcms_profiles:
            prof_lcms_data = lcms_profiles[wybrany_profil_lcms]
            kol_lcms_l, kol_lcms_p = prof_lcms_data.get(
                "rut_l", None
            ), prof_lcms_data.get("rut_r", None)

        calc_lcms_rut_l, calc_lcms_rut_r = 0.0, 0.0

        if prof_lcms_data is not None:
            x_lcms_orig = prof_lcms_data["x"].copy()
            z_lcms_raw = prof_lcms_data["z"]
            z_lcms_smooth = smooth_profile(z_lcms_raw, window_size=31)

            rsp_x_max, rsp_x_min = np.max(x_vals), np.min(x_vals)
            x_lcms = x_lcms_orig.copy()

            if (
                tryb_dopasowania_x == "Zaczep do prawej krawędzi RSP"
                and len(x_lcms) > 0
            ):
                x_lcms = x_lcms + (rsp_x_max - np.max(x_lcms))
            elif (
                tryb_dopasowania_x == "Wyśrodkuj osie (Zero w środku)"
                and len(x_lcms) > 0
            ):
                lcms_mid = (np.min(x_lcms) + np.max(x_lcms)) / 2.0
                x_lcms = x_lcms - lcms_mid

            if skaluj_szerokosc and len(x_lcms) > 1:
                lcms_min, lcms_max = np.min(x_lcms), np.max(x_lcms)
                x_lcms = rsp_x_min + (x_lcms - lcms_min) * (
                    (rsp_x_max - rsp_x_min) / (lcms_max - lcms_min)
                )

            x_lcms = x_lcms + manual_x_shift_lcms

            if wyrownaj_spadek_lcms and len(x_lcms) > 10:
                p_trend = np.polyfit(x_lcms, z_lcms_smooth, 1)
                z_lcms_leveled = z_lcms_smooth - p_trend[0] * x_lcms
            else:
                z_lcms_leveled = z_lcms_smooth

            idx_mid_lcms = np.argmin(np.abs(x_lcms))
            z_ref_lcms = z_lcms_leveled[idx_mid_lcms]
            z_lcms_plot = z_lcms_leveled - z_ref_lcms

            # Obliczenie wirtualnej łaty dla LCMS
            mid_mask_lcms = (x_lcms >= -0.5) & (x_lcms <= 0.5)
            if np.any(mid_mask_lcms):
                idx_c_lcms = np.where(mid_mask_lcms)[0][
                    np.argmax(z_lcms_plot[mid_mask_lcms])
                ]
            else:
                idx_c_lcms = np.argmin(np.abs(x_lcms))

            x_lcms_l, z_lcms_l = (
                x_lcms[: idx_c_lcms + 1],
                z_lcms_plot[: idx_c_lcms + 1],
            )
            x_lcms_r, z_lcms_r = (
                x_lcms[idx_c_lcms:],
                z_lcms_plot[idx_c_lcms:],
            )

            lata_lcms_l = get_upper_hull(x_lcms_l, z_lcms_l)
            lata_lcms_r = get_upper_hull(x_lcms_r, z_lcms_r)

            diff_lcms_l = lata_lcms_l - z_lcms_l
            diff_lcms_r = lata_lcms_r - z_lcms_r

            idx_max_lcms_l = (
                np.argmax(diff_lcms_l) if len(diff_lcms_l) > 0 else 0
            )
            idx_max_lcms_r = (
                np.argmax(diff_lcms_r) if len(diff_lcms_r) > 0 else 0
            )

            calc_lcms_rut_l = (
                float(diff_lcms_l[idx_max_lcms_l])
                if len(diff_lcms_l) > 0
                else 0.0
            )
            calc_lcms_rut_r = (
                float(diff_lcms_r[idx_max_lcms_r])
                if len(diff_lcms_r) > 0
                else 0.0
            )

            if st.sidebar.button("⚡ Dopasuj poprzecznie (X) LCMS do RSP"):
                best_x_shift = auto_align_transverse_x(
                    np.array(x_vals), y_vals, x_lcms_orig, z_lcms_smooth
                )
                st.session_state["manual_x_shift_val"] = best_x_shift
                st.sidebar.info(
                    f"💡 Ustawiono przesunięcie X: {best_x_shift:+.2f} m"
                )
                st.rerun()

        # TAB 1: PRZEKRÓJ POPRZECZNY I TRASA
        with tab_main:
            st.subheader(
                f"Profil RSP nr {wybrany_profil} (Dopasowany LCMS nr"
                f" {wybrany_profil_lcms}, Offset Podłużny: {całkowity_offset:+.1f}"
                f" m, Poprzeczny X: {manual_x_shift_lcms:+.2f} m)"
            )

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric(
                "RSP Koleina Lewa",
                f"{koleina_lewa_mm:.1f} mm",
                f"Klasa {get_dsn_class(koleina_lewa_mm)}",
            )
            rc2.metric(
                "RSP Koleina Prawa",
                f"{koleina_prawa_mm:.1f} mm",
                f"Klasa {get_dsn_class(koleina_prawa_mm)}",
            )
            rc3.metric(
                "RSP Koleina Max",
                f"{max_koleina_rsp:.1f} mm",
                f"Klasa {get_dsn_class(max_koleina_rsp)}",
            )

            lc1, lc2, lc3 = st.columns(3)
            if kol_lcms_l is not None and kol_lcms_p is not None:
                disp_l = (
                    kol_lcms_l
                    if zrodlo_lcms == "Z pliku LCMS (surowe B2/C2)"
                    else calc_lcms_rut_l
                )
                disp_p = (
                    kol_lcms_p
                    if zrodlo_lcms == "Z pliku LCMS (surowe B2/C2)"
                    else calc_lcms_rut_r
                )
                lc1.metric(
                    "LCMS Koleina Lewa",
                    f"{disp_l:.1f} mm",
                    delta=f"{koleina_lewa_mm - disp_l:+.1f} mm vs RSP",
                )
                lc2.metric(
                    "LCMS Koleina Prawa",
                    f"{disp_p:.1f} mm",
                    delta=f"{koleina_prawa_mm - disp_p:+.1f} mm vs RSP",
                )
                lc3.metric(
                    "LCMS Koleina Max",
                    f"{max(disp_l, disp_p):.1f} mm",
                    delta=f"{max_koleina_rsp - max(disp_l, disp_p):+.1f} mm vs RSP",
                )
            else:
                lc1.info(
                    "Brak dopasowanego profilu LCMS dla obecnego pikietażu."
                )

            # WYKRES POPRZECZNY Z PRZESUNIĘCIEM X I OBU WIRTUALNYMI ŁATAMI
            fig = go.Figure()

            # Profil RSP
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name="Profil RSP (Spline)",
                    line=dict(
                        shape="spline",
                        smoothing=1.2,
                        color="royalblue",
                        width=3,
                    ),
                    marker=dict(size=7, color="navy"),
                )
            )

            # Wirtualna łata RSP
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
                        name=f"RSP Lewa ({koleina_lewa_mm:.1f} mm)",
                        line=dict(color="crimson", width=3),
                        marker=dict(size=6, symbol="triangle-down"),
                    )
                )
                fig.add_annotation(
                    x=x_kl,
                    y=(y_kl_lata + y_kl_prof) / 2,
                    text=(
                        f"<b>RSP Lewa: {koleina_lewa_mm:.1f} mm</b><br>Klasa"
                        f" {get_dsn_class(koleina_lewa_mm)}"
                    ),
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="crimson",
                    font=dict(color="crimson", size=11),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="crimson",
                    borderwidth=1,
                    ax=-60,
                    ay=-30,
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
                        name=f"RSP Prawa ({koleina_prawa_mm:.1f} mm)",
                        line=dict(color="darkorange", width=3),
                        marker=dict(size=6, symbol="triangle-down"),
                    )
                )
                fig.add_annotation(
                    x=x_kp,
                    y=(y_kp_prof + y_kp_lata) / 2,
                    text=(
                        f"<b>RSP Prawa: {koleina_prawa_mm:.1f} mm</b><br>Klasa"
                        f" {get_dsn_class(koleina_prawa_mm)}"
                    ),
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="darkorange",
                    font=dict(color="darkorange", size=11),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="darkorange",
                    borderwidth=1,
                    ax=60,
                    ay=-30,
                )

            # Profil LCMS oraz Wirtualna Łata LCMS
            if prof_lcms_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=x_lcms,
                        y=z_lcms_plot,
                        mode="lines",
                        name=f"Profil LCMS nr {wybrany_profil_lcms}",
                        line=dict(color="seagreen", width=2),
                    )
                )

                if pokaz_late_lcms and len(x_lcms_l) > 1 and len(x_lcms_r) > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=x_lcms_l,
                            y=lata_lcms_l,
                            mode="lines",
                            name="Łata LCMS Lewa",
                            line=dict(
                                color="mediumseagreen", width=2, dash="dash"
                            ),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_lcms_r,
                            y=lata_lcms_r,
                            mode="lines",
                            name="Łata LCMS Prawa",
                            line=dict(color="darkgreen", width=2, dash="dash"),
                        )
                    )

                    x_lcms_kl = x_lcms_l[idx_max_lcms_l]
                    z_lcms_kl_prof = z_lcms_l[idx_max_lcms_l]
                    z_lcms_kl_lata = lata_lcms_l[idx_max_lcms_l]
                    fig.add_trace(
                        go.Scatter(
                            x=[x_lcms_kl, x_lcms_kl],
                            y=[z_lcms_kl_lata, z_lcms_kl_prof],
                            mode="lines+markers",
                            name=f"LCMS Łata Lewa ({calc_lcms_rut_l:.1f} mm)",
                            line=dict(color="mediumseagreen", width=2),
                            marker=dict(size=5, symbol="triangle-down"),
                        )
                    )
                    fig.add_annotation(
                        x=x_lcms_kl,
                        y=(z_lcms_kl_lata + z_lcms_kl_prof) / 2,
                        text=(
                            f"<b>Łata LCMS Lewa: {calc_lcms_rut_l:.1f}"
                            f" mm</b><br>Klasa {get_dsn_class(calc_lcms_rut_l)}"
                        ),
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="seagreen",
                        font=dict(color="seagreen", size=10),
                        bgcolor="rgba(235, 255, 235, 0.95)",
                        bordercolor="seagreen",
                        borderwidth=1,
                        ax=-70,
                        ay=35,
                    )

                    x_lcms_kp = x_lcms_r[idx_max_lcms_r]
                    z_lcms_kp_prof = z_lcms_r[idx_max_lcms_r]
                    z_lcms_kp_lata = lata_lcms_r[idx_max_lcms_r]
                    fig.add_trace(
                        go.Scatter(
                            x=[x_lcms_kp, x_lcms_kp],
                            y=[z_lcms_kp_lata, z_lcms_kp_prof],
                            mode="lines+markers",
                            name=f"LCMS Łata Prawa ({calc_lcms_rut_r:.1f} mm)",
                            line=dict(color="darkgreen", width=2),
                            marker=dict(size=5, symbol="triangle-down"),
                        )
                    )
                    fig.add_annotation(
                        x=x_lcms_kp,
                        y=(z_lcms_kp_lata + z_lcms_kp_prof) / 2,
                        text=(
                            f"<b>Łata LCMS Prawa: {calc_lcms_rut_r:.1f}"
                            f" mm</b><br>Klasa {get_dsn_class(calc_lcms_rut_r)}"
                        ),
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="darkgreen",
                        font=dict(color="darkgreen", size=10),
                        bgcolor="rgba(235, 255, 235, 0.95)",
                        bordercolor="darkgreen",
                        borderwidth=1,
                        ax=70,
                        ay=35,
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
                height=550,
                xaxis=dict(range=[-2.2, 2.2], dtick=0.2),
                yaxis=dict(autorange=True if tryb_zero_l11 else "reversed"),
            )
            st.plotly_chart(fig, width="stretch", key="fig_cross_section")

            # WYKRES PODŁUŻNY Z DYNAMICZNYM SEKCJA-PO-SEKCJI DOPASOWANIEM
            st.markdown("---")
            st.subheader("📈 PRZEBIEG KOLEIN NA CAŁEJ TRASIE")
            df_all_rsp_ruts = compute_all_rsp_ruts_cached(
                df_rsp, tuple(pozycja_x)
            )

            if lcms_profiles:
                lcms_aligned_max = []
                for p_rsp in df_all_rsp_ruts["Profil"]:
                    off_p = get_total_offset_for_profil(
                        p_rsp,
                        bazowy_offset,
                        podziel_na_sekcje,
                        sekcja_m,
                        sekcje_offsets,
                        min_p,
                    )
                    target_lcms_p = int(round(p_rsp + off_p))
                    if target_lcms_p in lcms_profiles:
                        p_data = lcms_profiles[target_lcms_p]
                        lcms_aligned_max.append(
                            max(p_data["rut_l"], p_data["rut_r"])
                        )
                    else:
                        lcms_aligned_max.append(np.nan)

                fig_m = go.Figure()
                fig_m.add_trace(
                    go.Scatter(
                        x=df_all_rsp_ruts["Profil"],
                        y=df_all_rsp_ruts["RSP_Rut_Max"],
                        mode="lines",
                        name="RSP Koleina Max",
                        line=dict(color="crimson", width=2),
                    )
                )
                fig_m.add_trace(
                    go.Scatter(
                        x=df_all_rsp_ruts["Profil"],
                        y=lcms_aligned_max,
                        mode="lines",
                        name="LCMS Max (Dopasowane)",
                        line=dict(color="forestgreen", width=2),
                    )
                )
                fig_m.add_vline(
                    x=wybrany_profil,
                    line_width=2,
                    line_color="red",
                    annotation_text=f"Profil RSP {wybrany_profil}",
                )
                fig_m.update_layout(
                    title=(
                        "Maksymalna Koleina na Całej Trasie [mm] (Dopasowanie"
                        " Sekcyjne)"
                    ),
                    xaxis_title="Numer Profilu RSP",
                    yaxis_title="Koleina [mm]",
                    height=400,
                )
                st.plotly_chart(
                    fig_m, width="stretch", key="fig_longitudinal_max"
                )

        # TAB 2: MAPA GPS
        with tab_map:
            st.header("🗺️ Mapa Trasy i Klas Stanu DSN 2019")
            if df_rsp_gps is not None and not df_rsp_gps.empty:
                df_map_data = pd.merge(
                    df_rsp_gps,
                    df_all_rsp_ruts,
                    left_on="Profil_idx",
                    right_on="Profil",
                    how="inner",
                )
                df_map_data["Klasa"] = df_map_data["RSP_Rut_Max"].apply(
                    get_dsn_class
                )

                fig_map = px.scatter_mapbox(
                    df_map_data,
                    lat="LAT",
                    lon="LON",
                    color="Klasa",
                    size="RSP_Rut_Max",
                    hover_name="Profil",
                    hover_data={
                        "RSP_Rut_Max": ":.1f mm",
                        "RSP_Rut_L": ":.1f mm",
                        "RSP_Rut_R": ":.1f mm",
                    },
                    color_discrete_map={
                        "A": "#28a745",
                        "B": "#ffc107",
                        "C": "#fd7e14",
                        "D": "#dc3545",
                    },
                    zoom=13,
                    height=600,
                )
                fig_map.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig_map, width="stretch", key="gps_map_plotly")
            else:
                st.info(
                    "💡 Wgraj plik GPS (.gpx / .rsp z rekordu 5280 / XML), aby"
                    " wyświetlić interaktywną mapę."
                )

        # TAB 3: OCENA 50M DSN 2019 DLA CAŁEJ TRASY
        with tab_dsn50:
            st.header("📊 Ocena Odcinków 50m wg Normy DSN 2019")
            df_merged_ruts = df_all_rsp_ruts.copy()
            if lcms_profiles:
                lcms_l_vals, lcms_r_vals = [], []
                for p_rsp in df_merged_ruts["Profil"]:
                    off_p = get_total_offset_for_profil(
                        p_rsp,
                        bazowy_offset,
                        podziel_na_sekcje,
                        sekcja_m,
                        sekcje_offsets,
                        min_p,
                    )
                    target_lcms_p = int(round(p_rsp + off_p))
                    if target_lcms_p in lcms_profiles:
                        lcms_l_vals.append(
                            lcms_profiles[target_lcms_p]["rut_l"]
                        )
                        lcms_r_vals.append(
                            lcms_profiles[target_lcms_p]["rut_r"]
                        )
                    else:
                        lcms_l_vals.append(np.nan)
                        lcms_r_vals.append(np.nan)

                df_merged_ruts["LCMS_Rut_L"] = lcms_l_vals
                df_merged_ruts["LCMS_Rut_R"] = lcms_r_vals

            df_sec_dsn = compute_50m_dsn_sections_full(
                df_merged_ruts, step_m=1.0, section_m=50.0
            )

            st.plotly_chart(
                make_50m_bar_chart(
                    df_sec_dsn, "MAX_KOL", "Koleina Maksymalna (MAX_KOL)"
                ),
                width="stretch",
                key="fig_dsn_50m_bar",
            )
            st.dataframe(sanitize_df(df_sec_dsn), width="stretch")

        # TAB 4: BLAND-ALTMAN & RPD GDDKiA DLA CAŁEJ TRASY
        with tab_ba_gddkia:
            st.header("📈 Analiza Statystyczna Blanda-Altmana i RPD")
            if lcms_profiles and "LCMS_Rut_L" in df_merged_ruts.columns:
                df_valid_ba = df_merged_ruts.dropna(
                    subset=["LCMS_Rut_L", "LCMS_Rut_R"]
                )
                if not df_valid_ba.empty:
                    col_ba1, col_ba2 = st.columns(2)

                    with col_ba1:
                        rsp_r_v, lcms_r_v = (
                            df_valid_ba["RSP_Rut_R"].values,
                            df_valid_ba["LCMS_Rut_R"].values,
                        )
                        fig_ba_r, mean_r, sd_r, loa_u_r, loa_l_r = (
                            build_bland_altman_fig(
                                rsp_r_v, lcms_r_v, title="PRAWA KOLEINA (P_KOL)"
                            )
                        )
                        st.plotly_chart(
                            fig_ba_r, width="stretch", key="ba_prawa_chart"
                        )
                        st.table(
                            sanitize_df(build_rpd_table_df(rsp_r_v, lcms_r_v))
                        )

                    with col_ba2:
                        rsp_l_v, lcms_l_v = (
                            df_valid_ba["RSP_Rut_L"].values,
                            df_valid_ba["LCMS_Rut_L"].values,
                        )
                        fig_ba_l, mean_l, sd_l, loa_u_l, loa_l_l = (
                            build_bland_altman_fig(
                                rsp_l_v, lcms_l_v, title="LEWA KOLEINA (L_KOL)"
                            )
                        )
                        st.plotly_chart(
                            fig_ba_l, width="stretch", key="ba_lewa_chart"
                        )
                        st.table(
                            sanitize_df(build_rpd_table_df(rsp_l_v, lcms_l_v))
                        )
                else:
                    st.warning(
                        "⚠️ Brak dopasowanych profili LCMS dla wskazanych"
                        " przesunięć."
                    )
            else:
                st.warning("⚠️ Brak powiązanych danych LCMS do analizy.")

        # TAB 5: EKSPORT RAPORTU
        with tab_export:
            st.header("📥 Wygeneruj Raport Zbiorczy w Formacie Excel")
            st.write(
                "Pobierz gotowy skoroszyt zawierający profile surowe, przeliczenia"
                " DSN 2019 dla odcinków 50m oraz statystykę Blanda-Altmana."
            )

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                sanitize_df(df_merged_ruts).to_excel(
                    writer, sheet_name="Surowe Koleiny", index=False
                )
                if "df_sec_dsn" in locals():
                    sanitize_df(df_sec_dsn).to_excel(
                        writer, sheet_name="Ocena DSN 50m", index=False
                    )

            st.download_button(
                label="💾 Pobierz Raport Excel (.xlsx)",
                data=output.getvalue(),
                file_name=f"Raport_RSP_LCMS_{wybrany_profil}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

else:
    st.info("👈 Wgraj plik RSP w panelu po lewej stronie, aby rozpocząć pracę.")
