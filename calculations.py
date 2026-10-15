import numpy as np
import pandas as pd
import streamlit as st


def get_dsn_class(h_val):
    if h_val < 11.0:
        return "A"
    elif h_val < 21.0:
        return "B"
    elif h_val < 31.0:
        return "C"
    else:
        return "D"


def get_dsn_color(klasa):
    if klasa == "A":
        return "#28a745"
    elif klasa == "B":
        return "#ffc107"
    elif klasa == "C":
        return "#fd7e14"
    else:
        return "#dc3545"


def safe_float(val):
    if val is None or not str(val).strip():
        return None
    try:
        return float(str(val).replace(",", ".").strip())
    except ValueError:
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
    upper = get_upper_hull_pts(x_sec, y_sec)
    x_h = [p[0] for p in upper]
    y_h = [p[1] for p in upper]
    return np.interp(x_sec, x_h, y_h)


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


@st.cache_data
def compute_all_lcms_calc_ruts_cached(lcms_profiles):
    ruts_calc_l, ruts_calc_r = [], []
    prof_list = sorted(lcms_profiles.keys())

    for pid in prof_list:
        pdata = lcms_profiles[pid]
        x_pts, z_pts = pdata["x"], pdata["z"]
        if len(x_pts) < 10:
            ruts_calc_l.append(pdata.get("rut_l", 0.0))
            ruts_calc_r.append(pdata.get("rut_r", 0.0))
            continue

        z_smooth = smooth_profile(z_pts, window_size=31)
        mid_mask = (x_pts >= -0.5) & (x_pts <= 0.5)
        idx_crown = (
            np.where(mid_mask)[0][np.argmax(z_smooth[mid_mask])]
            if np.any(mid_mask)
            else np.argmin(np.abs(x_pts))
        )

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
                "n": 0,
                "%": "0%",
                "RSP > LCMS": 0,
                "Wyniki równe": 0,
                "LCMS > RSP": 0,
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
            "n": n_grp,
            "%": f"{(n_grp / n_tot)*100:.0f}%",
            "RSP > LCMS": rsp_gt,
            "Wyniki równe": eq,
            "LCMS > RSP": lcms_gt,
            "%RSP > LCMS": f"{(rsp_gt / n_grp)*100:.0f}%",
            "%Wyniki równe": f"{(eq / n_grp)*100:.0f}%",
            "%LCMS > RSP": f"{(lcms_gt / n_grp)*100:.0f}%",
        }

    return pd.DataFrame([
        sub_stat(m_low, "0-30%"),
        sub_stat(m_high, ">30%"),
        {
            "zakres RPD": "suma",
            "n": n_tot,
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
    num_sections = int(np.ceil(len(df_ruts) / profs_per_sec))

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

        if "LCMS_Rut_L" in sub.columns:
            lcms_l, lcms_r = sub["LCMS_Rut_L"].values, sub["LCMS_Rut_R"].values
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
