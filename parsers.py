import io
import xml.etree.ElementTree as ET
from calculations import safe_float
import numpy as np
import pandas as pd
import streamlit as st


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
            profiles_data, columns=["Profil"] + [f"L{i+1}" for i in range(21)]
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
            x_l_arr, z_l_arr = np.array(x_left[:min_len_l]), np.array(
                z_left[:min_len_l]
            )
            min_len_r = min(len(x_right), len(z_right))
            x_r_arr, z_r_arr = np.array(x_right[:min_len_r]), np.array(
                z_right[:min_len_r]
            )

            x_full = np.concatenate([x_l_arr, x_r_arr])
            if len(x_full) > 0 and np.max(np.abs(x_full)) > 20:
                x_full = x_full / 1000.0

            z_full = np.concatenate([z_l_arr, z_r_arr])
            sort_order = np.argsort(x_full)

            records.append({
                "Station_m": None,
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

    has_station = any(r["Station_m"] is not None for r in all_records)
    profiles = {}

    if has_station:
        all_records = sorted(
            all_records,
            key=lambda r: (
                r["Station_m"] if r["Station_m"] is not None else 0.0
            ),
        )
        min_st = min(
            r["Station_m"] for r in all_records if r["Station_m"] is not None
        )
        for r in all_records:
            st_m = r["Station_m"]
            pid = (
                int(round(st_m - min_st)) + 1
                if st_m is not None
                else len(profiles) + 1
            )
            profiles[pid] = {
                "x": r["x"],
                "z": r["z"],
                "rut_l": r["rut_l"],
                "rut_r": r["rut_r"],
                "Station_m": st_m,
            }
    else:
        for idx, r in enumerate(all_records):
            pid = idx + 1
            profiles[pid] = {
                "x": r["x"],
                "z": r["z"],
                "rut_l": r["rut_l"],
                "rut_r": r["rut_r"],
                "Station_m": None,
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
