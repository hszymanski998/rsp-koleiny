import io
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ==========================================
# 1. FUNKCJE PARSUJĄCE Z CACHEM (IO & GPS)
# ==========================================
@st.cache_data
def load_rsp_data(file_bytes, file_name):
    """Wczytuje i keszuje dane RSP z wykorzystaniem io.BytesIO."""
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def parse_rsp_gps_cached(file_bytes, file_name):
    """Szybki, keszowany parser pliku GPS z profilografu RSP (.xlsx / .csv / .txt)."""
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

    # Automatyczne wykrywanie kolumn LAT (~48-56 N) i LON (~13-25 E)
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
        lat_col = 5  # Domyślnie kolumna F w Excelu RSP
    if lon_col is None:
        lon_col = 6  # Domyślnie kolumna G w Excelu RSP

    res = pd.DataFrame({
        "LAT": df_clean[lat_col],
        "LON": df_clean[lon_col],
        "Profil_idx": np.arange(1, len(df_clean) + 1),
    })

    if 1 in df_clean.columns:
        res["Km"] = df_clean[1]

    return res.dropna(subset=["LAT", "LON"]).reset_index(drop=True)


@st.cache_data
def parse_lcms_excel_cached(file_bytes, file_name):
    """Szybki, keszowany parser pliku LCMS z czyszczeniem danych."""
    if file_name.lower().endswith(".csv"):
        df_lcms_raw = pd.read_csv(
            io.BytesIO(file_bytes), header=None, sep=None, engine="python"
        )
    else:
        df_lcms_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)

    profiles = {}
    num_rows = len(df_lcms_raw)

    data_rows = []
    for idx in range(num_rows):
        val_a = str(df_lcms_raw.iloc[idx, 0]).strip()
        val_b = (
            str(df_lcms_raw.iloc[idx, 1])
            .replace(",", ".")
            .replace("-", "")
            .strip()
        )
        if val_a.isdigit():
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
            rut_l_str = str(df_lcms_raw.iloc[r_left_h, 1]).replace(",", ".")
            rut_r_str = str(df_lcms_raw.iloc[r_left_h, 2]).replace(",", ".")
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
                x_full = x_full / 1000.0  # mm na metry

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
    """Szybki parser pliku XML z LCMS do DataFrame Pandas."""
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

    return pd.DataFrame(records)


def smooth_profile(z_vals, window_size=31):
    """Filtruje makrostrukturę kruszywa i szum lasera filtrem średniej kroczącej."""
    if len(z_vals) < window_size:
        return z_vals
    return (
        pd.Series(z_vals)
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .values
    )


def get_upper_hull_pts(x_sec, y_sec):
    """Wyznacza punkty kontaktowe górnej otoczki wypukłej (Wirtualna Łata)."""
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
    """Oblicza wartości wirtualnej łaty DSN 2019 na całej długości sekcji."""
    upper = get_upper_hull_pts(x_sec, y_sec)
    x_h = [p[0] for p in upper]
    y_h = [p[1] for p in upper]
    return np.interp(x_sec, x_h, y_h)


def make_map_trace(lat, lon, mode, marker, line=None, name="", text=None):
    """Kreator warstwy mapy kompatybilny z każdą wersją Plotly."""
    kwargs = {"lat": lat, "lon": lon, "mode": mode, "marker": marker, "name": name}
    if line is not None:
        kwargs["line"] = line
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

# ==========================================
# 3. PANEL BOCZNY
# ==========================================
st.sidebar.header("1. Dane RSP (Profilograf)")
uploaded_rsp = st.sidebar.file_uploader(
    "Wgraj plik profilu RSP (.xlsx / .csv)", type=["xlsx", "csv"], key="rsp_input"
)
uploaded_rsp_gps = st.sidebar.file_uploader(
    "Wgraj plik GPS RSP (.xlsx / .csv / .txt)",
    type=["xlsx", "csv", "txt"],
    key="rsp_gps_input",
)

st.sidebar.header("2. Dane LCMS (Opcjonalnie)")
uploaded_lcms = st.sidebar.file_uploader(
    "Wgraj plik profilu LCMS (.xlsx / .csv)",
    type=["xlsx", "csv"],
    key="lcms_input",
)
uploaded_xml = st.sidebar.file_uploader(
    "Wgraj plik XML z LCMS (GPS)", type=["xml"], key="xml_input"
)

st.sidebar.header("3. Synchronizacja i Pikietaż (Drift)")
offset_pikitaz = st.sidebar.slider(
    "Przesunięcie pikietażu LCMS względem RSP [liczba profili]:",
    min_value=-200,
    max_value=200,
    value=0,
    step=1,
)

st.sidebar.header("4. Geometria belki RSP")
typ_geometrii = st.sidebar.radio(
    "Rozstaw czujników RSP:",
    ["Zmienny / Niestandardowy", "Stały odstęp (równomierny)"],
)

if typ_geometrii == "Stały odstęp (równomierny)":
    krok_laserow = st.sidebar.number_input(
        "Odstęp między laserami [m]:",
        min_value=0.05,
        max_value=0.50,
        value=0.165,
        step=0.005,
        format="%.3f",
    )
    pozycja_x = [round((i - 10) * krok_laserow, 3) for i in range(21)]
else:
    domyslne_x_str = (
        "-1.65, -1.48, -1.32, -1.16, -1.00, -0.84, -0.68, -0.52, -0.36, -0.18, "
        "0.0, 0.18, 0.36, 0.52, 0.68, 0.84, 1.00, 1.16, 1.32, 1.48, 1.65"
    )
    x_input = st.sidebar.text_area(
        "Wpisz pozycje X [m] dla L1..L21 (rozdzielone przecinkami):",
        value=domyslne_x_str,
        height=80,
    )
    try:
        pozycja_x = [
            float(val.strip().replace(",", "."))
            for val in x_input.split(",")
            if val.strip() != ""
        ]
    except Exception:
        pozycja_x = [round((i - 10) * 0.165, 3) for i in range(21)]

st.sidebar.header("5. Opcje Wykresu")
test_stacjonarny = st.sidebar.checkbox(
    "Tryb testu stacjonarnego (Łata 0)", value=False
)
tryb_zero_l11 = st.sidebar.checkbox(
    "Ustaw L11 jako Zero (wysokość względna RSP)", value=True
)
pokaz_late_rsp = st.sidebar.checkbox("Pokaż łatę i koleiny RSP", value=True)
wyrownaj_spadek_lcms = st.sidebar.checkbox(
    "Wyzeruj spadek poprzeczny LCMS (Wyprostuj profil)", value=True
)
pokaz_late_lcms = st.sidebar.checkbox("Pokaż łatę i koleiny LCMS", value=True)

# Parsowanie opcjonalnych plików GPS
df_rsp_gps = None
if uploaded_rsp_gps is not None:
    try:
        df_rsp_gps = parse_rsp_gps_cached(
            uploaded_rsp_gps.getvalue(), uploaded_rsp_gps.name
        )
        st.sidebar.success(f"Wczytano GPS RSP: {len(df_rsp_gps)} punktów.")
    except Exception as e:
        st.sidebar.error(f"Błąd GPS RSP: {e}")

lcms_profiles = None
if uploaded_lcms is not None:
    try:
        with st.spinner("Przetwarzanie pliku LCMS..."):
            lcms_profiles = parse_lcms_excel_cached(
                uploaded_lcms.getvalue(), uploaded_lcms.name
            )
        st.sidebar.success(f"Wczytano LCMS: {len(lcms_profiles)} profili.")
    except Exception as e:
        st.sidebar.error(f"Błąd odczytu LCMS: {e}")

df_xml_gps = None
if uploaded_xml is not None:
    try:
        df_xml_gps = parse_lcms_xml(uploaded_xml.getvalue())
        st.sidebar.success(f"Wczytano GPS LCMS: {len(df_xml_gps)} punktów.")
    except Exception as e:
        st.sidebar.error(f"Błąd XML: {e}")

# ==========================================
# 4. GŁÓWNA LOGIKA ANALIZY I WYKRESÓW
# ==========================================
if uploaded_rsp is not None:
    df_rsp = load_rsp_data(uploaded_rsp.getvalue(), uploaded_rsp.name)

    col_profil = "Profil" if "Profil" in df_rsp.columns else df_rsp.columns[0]
    min_p = int(df_rsp[col_profil].min())
    max_p = int(df_rsp[col_profil].max())

    st.sidebar.header("6. Wybór profilu")
    wybrany_profil = st.sidebar.number_input(
        f"Wpisz numer profilu [{min_p} - {max_p}]:",
        min_value=min_p,
        max_value=max_p,
        value=min_p,
        step=1,
    )

    wybrany_profil_lcms = wybrany_profil + offset_pikitaz
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

        max_koleina = max(koleina_lewa_mm, koleina_prawa_mm)

        if test_stacjonarny:
            st.subheader("🧪 Weryfikacja Testu Stacjonarnego (DSN 2019)")
            if max_koleina <= 0.5:
                st.success(
                    f"✅ **TEST POZYTYWNY**: Odchyłka wynosi **{max_koleina:.2f} mm**"
                    " (≤ 0,5 mm)."
                )
            else:
                st.error(
                    f"❌ **TEST NEGATYWNY**: Odchyłka wynosi **{max_koleina:.2f} mm** i"
                    " przekracza limit 0,5 mm dla łaty!"
                )

        kol_lcms_l, kol_lcms_p = None, None
        prof_lcms_data = None
        if lcms_profiles and wybrany_profil_lcms in lcms_profiles:
            prof_lcms_data = lcms_profiles[wybrany_profil_lcms]
            kol_lcms_l = prof_lcms_data.get("rut_l", None)
            kol_lcms_p = prof_lcms_data.get("rut_p", None)

        st.subheader(
            f"Profil RSP nr {wybrany_profil} (LCMS z offsetem:"
            f" {wybrany_profil_lcms})"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "RSP Koleina Lewa",
            f"{koleina_lewa_mm:.1f} mm",
            f"Laser L{idx_kol_l+1} ({x_vals[idx_kol_l]}m)",
        )
        c2.metric(
            "RSP Koleina Prawa",
            f"{koleina_prawa_mm:.1f} mm",
            f"Laser L{idx_kol_p+1} ({x_vals[idx_kol_p]}m)",
        )

        if kol_lcms_l is not None and kol_lcms_p is not None:
            c3.metric(
                "LCMS Koleina Lewa",
                f"{kol_lcms_l:.1f} mm",
                delta=f"{koleina_lewa_mm - kol_lcms_l:+.1f} mm vs RSP",
            )
            c4.metric(
                "LCMS Koleina Prawa",
                f"{kol_lcms_p:.1f} mm",
                delta=f"{koleina_prawa_mm - kol_lcms_p:+.1f} mm vs RSP",
            )
        else:
            c3.metric("RSP Maksymalna", f"{max_koleina:.1f} mm")

        fig = go.Figure()

        # RSP Plot
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name="Profil RSP (21 laserów)",
                line=dict(color="royalblue", width=3),
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
                    line=dict(color="crimson", width=2, dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_right,
                    y=lata_prawa_y,
                    mode="lines",
                    name="Łata RSP Prawa",
                    line=dict(color="darkorange", width=2, dash="dash"),
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
                text=f" <b>{koleina_lewa_mm:.1f} mm</b>",
                showarrow=False,
                font=dict(color="crimson", size=13),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="crimson",
                borderwidth=1,
                xanchor="right" if x_kl < 0 else "left",
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
                text=f" <b>{koleina_prawa_mm:.1f} mm</b>",
                showarrow=False,
                font=dict(color="darkorange", size=13),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="darkorange",
                borderwidth=1,
                xanchor="left" if x_kp > 0 else "right",
            )

        # LCMS Plot
        if prof_lcms_data is not None:
            x_lcms_raw = prof_lcms_data["x"]
            z_lcms_raw = prof_lcms_data["z"]

            z_lcms_smooth = smooth_profile(z_lcms_raw, window_size=31)

            if wyrownaj_spadek_lcms and len(x_lcms_raw) > 10:
                p_trend = np.polyfit(x_lcms_raw, z_lcms_smooth, 1)
                z_lcms_leveled = z_lcms_smooth - (p_trend[0] * x_lcms_raw)
            else:
                z_lcms_leveled = z_lcms_smooth

            idx_mid_lcms = np.argmin(np.abs(x_lcms_raw))
            z_lcms_plot = z_lcms_leveled - z_lcms_leveled[idx_mid_lcms]

            fig.add_trace(
                go.Scatter(
                    x=x_lcms_raw,
                    y=z_lcms_plot,
                    mode="lines",
                    name=f"Profil LCMS nr {wybrany_profil_lcms} (Wygładzony)",
                    line=dict(color="forestgreen", width=2.5),
                )
            )

            mid_mask_lcms = (x_lcms_raw >= -0.5) & (x_lcms_raw <= 0.5)
            if np.any(mid_mask_lcms):
                idx_crown_lcms = np.where(mid_mask_lcms)[0][
                    np.argmax(z_lcms_plot[mid_mask_lcms])
                ]
            else:
                idx_crown_lcms = idx_mid_lcms

            x_lcms_l = x_lcms_raw[: idx_crown_lcms + 1]
            z_lcms_l = z_lcms_plot[: idx_crown_lcms + 1]
            x_lcms_r = x_lcms_raw[idx_crown_lcms:]
            z_lcms_r = z_lcms_plot[idx_crown_lcms:]

            lata_lcms_l_y = get_upper_hull(x_lcms_l, z_lcms_l)
            lata_lcms_r_y = get_upper_hull(x_lcms_r, z_lcms_r)

            diff_lcms_l = lata_lcms_l_y - z_lcms_l
            idx_kol_lcms_l = np.argmax(diff_lcms_l)
            calc_lcms_rut_l = diff_lcms_l[idx_kol_lcms_l]

            diff_lcms_r = lata_lcms_r_y - z_lcms_r
            idx_kol_lcms_r_rel = np.argmax(diff_lcms_r)
            idx_kol_lcms_r = idx_crown_lcms + idx_kol_lcms_r_rel
            calc_lcms_rut_r = diff_lcms_r[idx_kol_lcms_r_rel]

            if pokaz_late_lcms:
                fig.add_trace(
                    go.Scatter(
                        x=x_lcms_l,
                        y=lata_lcms_l_y,
                        mode="lines",
                        name="Łata LCMS Lewa",
                        line=dict(color="limegreen", width=2, dash="dash"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_lcms_r,
                        y=lata_lcms_r_y,
                        mode="lines",
                        name="Łata LCMS Prawa",
                        line=dict(color="lime", width=2, dash="dash"),
                    )
                )

                x_kl_c = x_lcms_l[idx_kol_lcms_l]
                y_kl_c_prof = z_lcms_l[idx_kol_lcms_l]
                y_kl_c_lata = lata_lcms_l_y[idx_kol_lcms_l]

                fig.add_trace(
                    go.Scatter(
                        x=[x_kl_c, x_kl_c],
                        y=[y_kl_c_lata, y_kl_c_prof],
                        mode="lines+markers",
                        name=f"Głębokość LCMS Lewa ({calc_lcms_rut_l:.1f} mm)",
                        line=dict(color="forestgreen", width=2.5),
                        marker=dict(size=5, symbol="triangle-down"),
                    )
                )
                fig.add_annotation(
                    x=x_kl_c,
                    y=(y_kl_c_lata + y_kl_c_prof) / 2,
                    text=f" <b>LCMS: {calc_lcms_rut_l:.1f} mm</b>",
                    showarrow=False,
                    font=dict(color="forestgreen", size=12),
                    bgcolor="rgba(235, 255, 235, 0.9)",
                    bordercolor="forestgreen",
                    borderwidth=1,
                    xanchor="right" if x_kl_c < 0 else "left",
                )

                x_kp_c = x_lcms_raw[idx_kol_lcms_r]
                y_kp_c_prof = z_lcms_plot[idx_kol_lcms_r]
                y_kp_c_lata = lata_lcms_r_y[idx_kol_lcms_r_rel]

                fig.add_trace(
                    go.Scatter(
                        x=[x_kp_c, x_kp_c],
                        y=[y_kp_c_lata, y_kp_c_prof],
                        mode="lines+markers",
                        name=f"Głębokość LCMS Prawa ({calc_lcms_rut_r:.1f} mm)",
                        line=dict(color="forestgreen", width=2.5),
                        marker=dict(size=5, symbol="triangle-down"),
                    )
                )
                fig.add_annotation(
                    x=x_kp_c,
                    y=(y_kp_c_lata + y_kp_c_prof) / 2,
                    text=f" <b>LCMS: {calc_lcms_rut_r:.1f} mm</b>",
                    showarrow=False,
                    font=dict(color="forestgreen", size=12),
                    bgcolor="rgba(235, 255, 235, 0.9)",
                    bordercolor="forestgreen",
                    borderwidth=1,
                    xanchor="left" if x_kp_c > 0 else "right",
                )

        fig.add_vline(
            x=0,
            line_width=1,
            line_dash="dot",
            line_color="gray",
            annotation_text="Oś L11 (0.0 m)",
        )

        fig.update_layout(
            title=(
                f"Przekrój Poprzeczny Nawierzchni – Profil RSP nr {wybrany_profil}"
            ),
            xaxis_title="Szerokość / Pozycja czujnika na belce [m]",
            yaxis_title=y_label,
            height=580,
            hovermode="x unified",
            xaxis=dict(range=[-1.75, 1.75], dtick=0.3),
            yaxis=dict(autorange=True if tryb_zero_l11 else "reversed"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Longitudinal Chart
        if lcms_profiles:
            st.markdown("---")
            st.subheader("📈 Przebieg Kolein na Całej Trasie (Wykres Podłużny)")

            list_lcms_p = sorted(lcms_profiles.keys())
            lcms_rut_l = [lcms_profiles[p]["rut_l"] for p in list_lcms_p]
            lcms_rut_r = [lcms_profiles[p]["rut_r"] for p in list_lcms_p]
            lcms_p_shifted = [p - offset_pikitaz for p in list_lcms_p]

            fig_long = go.Figure()
            fig_long.add_trace(
                go.Scatter(
                    x=lcms_p_shifted,
                    y=lcms_rut_l,
                    mode="lines",
                    name="LCMS Koleina Lewa",
                    line=dict(color="forestgreen", width=1.5),
                )
            )
            fig_long.add_trace(
                go.Scatter(
                    x=lcms_p_shifted,
                    y=lcms_rut_r,
                    mode="lines",
                    name="LCMS Koleina Prawa",
                    line=dict(color="lightgreen", width=1.5, dash="dot"),
                )
            )

            fig_long.add_vline(
                x=wybrany_profil,
                line_width=2,
                line_color="red",
                annotation_text=f"Profil RSP {wybrany_profil}",
            )

            fig_long.update_layout(
                title="Porównanie Podłużne Głębokości Kolein [mm] na Długości Trasy",
                xaxis_title="Numer Profilu RSP (Pikietaż)",
                yaxis_title="Głębokość koleiny [mm]",
                height=400,
                hovermode="x unified",
            )
            st.plotly_chart(fig_long, use_container_width=True)

        # Map GPS (Dwa Ślady: RSP + LCMS)
        if df_rsp_gps is not None or df_xml_gps is not None:
            st.markdown("---")
            st.subheader("🗺️ Pozycja Profilu na Mapie GPS")

            fig_map = go.Figure()
            center_lat, center_lon = 52.0, 19.0

            # 1. Ślad GPS z RSP
            if df_rsp_gps is not None:
                fig_map.add_trace(
                    make_map_trace(
                        lat=df_rsp_gps["LAT"],
                        lon=df_rsp_gps["LON"],
                        mode="markers+lines",
                        marker=dict(size=5, color="royalblue"),
                        line=dict(width=2, color="royalblue"),
                        name="Trasa RSP GPS",
                        text=[
                            f"Profil RSP: {p_i}"
                            for p_i in df_rsp_gps["Profil_idx"]
                        ],
                    )
                )
                idx_rsp_gps = min(
                    max(0, wybrany_profil - min_p), len(df_rsp_gps) - 1
                )
                row_rsp_gps = df_rsp_gps.iloc[idx_rsp_gps]
                center_lat, center_lon = (
                    row_rsp_gps["LAT"],
                    row_rsp_gps["LON"],
                )

                fig_map.add_trace(
                    make_map_trace(
                        lat=[row_rsp_gps["LAT"]],
                        lon=[row_rsp_gps["LON"]],
                        mode="markers",
                        marker=dict(size=14, color="red"),
                        name=f"Profil RSP {wybrany_profil}",
                        text=[f"Pozycja Profilu RSP nr {wybrany_profil}"],
                    )
                )

            # 2. Ślad GPS z LCMS (XML)
            if df_xml_gps is not None:
                fig_map.add_trace(
                    make_map_trace(
                        lat=df_xml_gps["LAT"],
                        lon=df_xml_gps["LON"],
                        mode="markers+lines",
                        marker=dict(size=5, color="forestgreen"),
                        line=dict(width=2, color="forestgreen"),
                        name="Trasa LCMS XML",
                        text=[
                            f"Pikietaż LCMS: {st_m:.1f}m<br>Plik: {fn}"
                            for st_m, fn in zip(
                                df_xml_gps["Station_m"], df_xml_gps["Filename"]
                            )
                        ],
                    )
                )
                idx_lcms_gps = min(
                    max(0, wybrany_profil_lcms - 1), len(df_xml_gps) - 1
                )
                row_lcms_gps = df_xml_gps.iloc[idx_lcms_gps]

                if df_rsp_gps is None:
                    center_lat, center_lon = (
                        row_lcms_gps["LAT"],
                        row_lcms_gps["LON"],
                    )

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

            map_config = dict(
                center=dict(lat=center_lat, lon=center_lon), zoom=14
            )

            try:
                if hasattr(go, "Scattermap"):
                    fig_map.update_layout(
                        map_style="open-street-map",
                        map=map_config,
                        margin=dict(l=0, r=0, t=20, b=0),
                        height=450,
                    )
                else:
                    fig_map.update_layout(
                        mapbox_style="open-street-map",
                        mapbox=map_config,
                        margin=dict(l=0, r=0, t=20, b=0),
                        height=450,
                    )
            except Exception:
                fig_map.update_layout(
                    margin=dict(l=0, r=0, t=20, b=0), height=450
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
