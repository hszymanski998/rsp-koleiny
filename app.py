import os
import calculations as calc
import charts
import numpy as np
import pandas as pd
import parsers
import plotly.graph_objects as go
import streamlit as st

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

st.set_page_config(
    page_title="RSP vs LCMS – System Analizy Kolein DSN 2019", layout="wide"
)
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

st.sidebar.header("2. Dane LCMS (Możesz wgrać WIELE plików naraz)")
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
    _, _, auto_x_from_rsp = parsers.parse_native_rsp_file(bytes_rsp)

st.sidebar.header("3. Synchronizacja i Pikietaż")
glowny_offset = st.sidebar.number_input(
    "✍️ Przesunięcie pikietażu LCMS względem RSP [m]:",
    min_value=-500.0,
    max_value=500.0,
    value=0.0,
    step=1.0,
    format="%.1f",
)

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
    [
        "Z pliku LCMS (surowe B2/C2)",
        "Wyliczone z profilu (Wirtualna Łata)",
        "Oba źródła jednocześnie",
    ],
)

st.sidebar.header("6. Geometria i Dopasowanie Poprzeczne")
zaczep_prawa_krawedz = st.sidebar.checkbox(
    "🔗 Zaczep prawą krawędź LCMS do RSP", value=True
)
skaluj_szerokosc = st.sidebar.checkbox(
    "📐 Skaluj szerokość LCMS do RSP", value=False
)
manual_x_shift_lcms = st.sidebar.number_input(
    "Ręczne przesunięcie X LCMS [m]:",
    min_value=-2.0,
    max_value=2.0,
    value=0.0,
    step=0.01,
    format="%.2f",
)

test_stacjonarny = st.sidebar.checkbox(
    "Tryb testu stacjonarnego (Łata 0)", value=False
)
tryb_zero_l11 = st.sidebar.checkbox("Ustaw L11 jako Zero", value=True)
pokaz_late_rsp = st.sidebar.checkbox("Pokaż łatę i koleiny RSP", value=True)
wyrownaj_spadek_lcms = st.sidebar.checkbox(
    "Wyzeruj spadek poprzeczny LCMS", value=True
)
pokaz_late_lcms = st.sidebar.checkbox("Pokaż łatę i koleiny LCMS", value=True)

df_rsp_gps = (
    parsers.parse_rsp_gps_cached(bytes_rsp_gps, name_rsp_gps)
    if bytes_rsp_gps
    else None
)
if df_rsp_gps is not None:
    st.sidebar.success(f"GPS RSP: {len(df_rsp_gps)} pkt.")

lcms_profiles = (
    parsers.load_multiple_lcms_files(uploaded_lcms_list)
    if uploaded_lcms_list
    else None
)
if lcms_profiles:
    st.sidebar.success(
        f"LCMS: Łącznie {len(lcms_profiles)} profili z"
        f" {len(uploaded_lcms_list)} plików!"
    )

df_xml_gps = (
    parsers.parse_lcms_xml_multiple(uploaded_xml_list)
    if uploaded_xml_list
    else None
)
if df_xml_gps is not None:
    st.sidebar.success(f"GPS LCMS: Łącznie {len(df_xml_gps)} pkt z XML!")

if bytes_rsp is not None:
    df_rsp = parsers.load_rsp_data(bytes_rsp, name_rsp)
    col_profil = "Profil" if "Profil" in df_rsp.columns else df_rsp.columns[0]
    min_p, max_p = int(df_rsp[col_profil].min()), int(df_rsp[col_profil].max())

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
    dodatkowy_s_offset = (
        sekcje_offsets.get(int((wybrany_profil - min_p) // sekcja_m), 0.0)
        if podziel_na_sekcje
        else 0.0
    )
    całkowity_offset = bazowy_offset + dodatkowy_s_offset
    wybrany_profil_lcms = int(round(wybrany_profil + całkowity_offset))

    if lcms_profiles:
        df_all_rsp_ruts = calc.compute_all_rsp_ruts_cached(
            df_rsp, tuple(pozycja_x)
        )
        list_lcms_p = sorted(lcms_profiles.keys())
        lcms_file_rut_max = np.maximum(
            [lcms_profiles[p]["rut_l"] for p in list_lcms_p],
            [lcms_profiles[p]["rut_r"] for p in list_lcms_p],
        )

        if st.sidebar.button("⚡ Automatyczna Synchronizacja Pikietażu"):
            sug_offset = calc.auto_align_offset_clean(
                df_all_rsp_ruts["RSP_Rut_Max"].values, lcms_file_rut_max
            )
            st.sidebar.info(f"💡 Sugerowany offset: {sug_offset:+} m")

    tab_main, tab_dsn50, tab_ba_gddkia = st.tabs([
        "📐 Przekrój Poprzeczny & Cała Trasa",
        "📊 Ocena 50m DSN 2019 (Klasy A-D)",
        "📈 Analiza Porównawcza GDDKiA (Bland-Altman & RPD)",
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

        lata_lewa_y = calc.get_upper_hull(x_left, y_left)
        lata_prawa_y = calc.get_upper_hull(x_right, y_right)

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

        calc_lcms_rut_l, calc_lcms_rut_r = None, None
        if prof_lcms_data is not None:
            x_lcms_orig, z_lcms_raw = (
                prof_lcms_data["x"].copy(),
                prof_lcms_data["z"],
            )
            z_lcms_smooth = calc.smooth_profile(z_lcms_raw, window_size=31)
            mid_mask_raw = (x_lcms_orig >= -0.5) & (x_lcms_orig <= 0.5)
            idx_c_lcms = (
                np.where(mid_mask_raw)[0][
                    np.argmax(z_lcms_smooth[mid_mask_raw])
                ]
                if np.any(mid_mask_raw)
                else np.argmin(np.abs(x_lcms_orig))
            )

            x_lcms_l_raw, z_lcms_l_raw = (
                x_lcms_orig[: idx_c_lcms + 1],
                z_lcms_smooth[: idx_c_lcms + 1],
            )
            x_lcms_r_raw, z_lcms_r_raw = (
                x_lcms_orig[idx_c_lcms:],
                z_lcms_smooth[idx_c_lcms:],
            )

            calc_lcms_rut_l = float(
                np.max(
                    calc.get_upper_hull(x_lcms_l_raw, z_lcms_l_raw)
                    - z_lcms_l_raw
                )
            )
            calc_lcms_rut_r = float(
                np.max(
                    calc.get_upper_hull(x_lcms_r_raw, z_lcms_r_raw)
                    - z_lcms_r_raw
                )
            )

        with tab_main:
            st.subheader(
                f"Profil RSP nr {wybrany_profil} (LCMS nr"
                f" {wybrany_profil_lcms}, przesunięcie:"
                f" {całkowity_offset:+.1f} m)"
            )
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric(
                "RSP Koleina Lewa",
                f"{koleina_lewa_mm:.1f} mm (Klasa"
                f" {calc.get_dsn_class(koleina_lewa_mm)})",
            )
            rc2.metric(
                "RSP Koleina Prawa",
                f"{koleina_prawa_mm:.1f} mm (Klasa"
                f" {calc.get_dsn_class(koleina_prawa_mm)})",
            )
            rc3.metric(
                "RSP Koleina Max",
                f"{max_koleina_rsp:.1f} mm (Klasa"
                f" {calc.get_dsn_class(max_koleina_rsp)})",
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
                    f"{disp_l:.1f} mm (Klasa {calc.get_dsn_class(disp_l)})",
                    delta=f"{koleina_lewa_mm - disp_l:+.1f} mm vs RSP",
                )
                lc2.metric(
                    "LCMS Koleina Prawa",
                    f"{disp_p:.1f} mm (Klasa {calc.get_dsn_class(disp_p)})",
                    delta=f"{koleina_prawa_mm - disp_p:+.1f} mm vs RSP",
                )
                lc3.metric(
                    "LCMS Koleina Max",
                    f"{max(disp_l, disp_p):.1f} mm",
                    delta=f"{max_koleina_rsp - max(disp_l, disp_p):+.1f} mm vs RSP",
                )
            else:
                lc1.info("Brak profilu LCMS dla wybranego pikietażu.")

            # WYKRES PRZEKROJU POPRZECZNEGO Z WSZYSTKIMI SZCZEGÓŁAMI
            fig = go.Figure()

            # 1. Krzywa RSP (Spline - zaokrąglona)
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
                    hovertemplate=(
                        "Poz X: %{x} m<br>Wys Y: %{y:.1f} mm<extra></extra>"
                    ),
                )
            )

            # 2. Łata RSP i linie głębokości kolein RSP
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

            # 3. Profil LCMS, wyrównanie spadku, łata LCMS i linie głębokości
            if prof_lcms_data is not None:
                x_lcms_orig = prof_lcms_data["x"].copy()
                z_lcms_raw = prof_lcms_data["z"]
                z_lcms_smooth = calc.smooth_profile(z_lcms_raw, window_size=31)

                rsp_x_max, rsp_x_min = np.max(x_vals), np.min(x_vals)
                x_lcms = x_lcms_orig.copy()

                if skaluj_szerokosc and len(x_lcms) > 1:
                    lcms_min, lcms_max = np.min(x_lcms), np.max(x_lcms)
                    x_lcms = rsp_x_min + (x_lcms - lcms_min) * (
                        (rsp_x_max - rsp_x_min) / (lcms_max - lcms_min)
                    )
                elif zaczep_prawa_krawedz and len(x_lcms) > 0:
                    x_lcms = x_lcms + (rsp_x_max - np.max(x_lcms))

                x_lcms = x_lcms + manual_x_shift_lcms

                mid_mask_raw = (x_lcms >= -0.5) & (x_lcms <= 0.5)
                idx_crown_lcms = (
                    np.where(mid_mask_raw)[0][
                        np.argmax(z_lcms_smooth[mid_mask_raw])
                    ]
                    if np.any(mid_mask_raw)
                    else np.argmin(np.abs(x_lcms))
                )

                x_lcms_l, z_lcms_l_raw = (
                    x_lcms[: idx_crown_lcms + 1],
                    z_lcms_smooth[: idx_crown_lcms + 1],
                )
                x_lcms_r, z_lcms_r_raw = (
                    x_lcms[idx_crown_lcms:],
                    z_lcms_smooth[idx_crown_lcms:],
                )

                lata_raw_l = calc.get_upper_hull(x_lcms_l, z_lcms_l_raw)
                lata_raw_r = calc.get_upper_hull(x_lcms_r, z_lcms_r_raw)

                if wyrownaj_spadek_lcms and len(x_lcms) > 10:
                    p_trend = np.polyfit(x_lcms, z_lcms_smooth, 1)
                    trend_line = p_trend[0] * x_lcms
                    z_lcms_leveled = z_lcms_smooth - trend_line
                    lata_plot_l = lata_raw_l - trend_line[: idx_crown_lcms + 1]
                    lata_plot_r = lata_raw_r - trend_line[idx_crown_lcms:]
                else:
                    z_lcms_leveled = z_lcms_smooth
                    lata_plot_l, lata_plot_r = lata_raw_l, lata_raw_r

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
                            name=(
                                f"Głębokość LCMS Lewa ({val_l_show:.1f} mm)"
                            ),
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

                    diff_lcms_r = lata_plot_r_zero - (
                        z_lcms_plot[idx_crown_lcms:]
                    )
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
                            name=(
                                f"Głębokość LCMS Prawa ({val_r_show:.1f} mm)"
                            ),
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

            # 4. Linia centralna osi L11
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
            st.plotly_chart(
                fig, use_container_width=True, key="fig_cross_section"
            )

            # WYKRES PODŁUŻNY CAŁEJ TRASY
            st.markdown("---")
            st.subheader(
                "📈 Przebieg Kolein na Całej Trasie (RSP vs Scalone Pliki LCMS)"
            )

            df_all_rsp_ruts = calc.compute_all_rsp_ruts_cached(
                df_rsp, tuple(pozycja_x)
            )

            if lcms_profiles:
                list_lcms_p = sorted(lcms_profiles.keys())
                lcms_file_rut_l = [
                    lcms_profiles[p]["rut_l"] for p in list_lcms_p
                ]
                lcms_file_rut_r = [
                    lcms_profiles[p]["rut_r"] for p in list_lcms_p
                ]
                lcms_p_shifted = [
                    p
                    - (
                        bazowy_offset
                        + (
                            sekcje_offsets.get(
                                int((p - min_p) // sekcja_m), 0.0
                            )
                            if podziel_na_sekcje
                            else 0.0
                        )
                    )
                    for p in list_lcms_p
                ]

                fig_m = go.Figure()
                fig_m.add_trace(
                    go.Scatter(
                        x=df_all_rsp_ruts["Profil"],
                        y=df_all_rsp_ruts["RSP_Rut_Max"],
                        mode="lines",
                        name="RSP Koleina Max",
                        line=dict(color="crimson", width=2.5),
                    )
                )
                fig_m.add_trace(
                    go.Scatter(
                        x=lcms_p_shifted,
                        y=np.maximum(lcms_file_rut_l, lcms_file_rut_r),
                        mode="lines",
                        name="LCMS Scalone Pliki Max",
                        line=dict(color="forestgreen", width=2.5),
                    )
                )
                fig_m.add_vline(
                    x=wybrany_profil,
                    line_width=2,
                    line_color="red",
                    annotation_text=f"Profil {wybrany_profil}",
                )
                fig_m.update_layout(
                    title="KOLEINA MAKSYMALNA NA CAŁEJ TRASIE [mm]",
                    xaxis_title="Numer Profilu RSP",
                    yaxis_title="Koleina [mm]",
                    height=420,
                )
                st.plotly_chart(
                    fig_m, use_container_width=True, key="fig_longitudinal_max"
                )

        with tab_dsn50:
            st.header(
                "📊 Ocena Stanu Koleiny dla Odcinków 50-metrowych (DSN 2019)"
            )
            df_merged_ruts = df_all_rsp_ruts.copy()
            if lcms_profiles:
                list_lcms_p = sorted(lcms_profiles.keys())
                df_lcms_sub = pd.DataFrame({
                    "Profil": [int(p - całkowity_offset) for p in list_lcms_p],
                    "LCMS_Rut_L": [
                        lcms_profiles[p]["rut_l"] for p in list_lcms_p
                    ],
                    "LCMS_Rut_R": [
                        lcms_profiles[p]["rut_r"] for p in list_lcms_p
                    ],
                })
                df_merged_ruts = pd.merge(
                    df_merged_ruts, df_lcms_sub, on="Profil", how="inner"
                )

            df_sec_dsn = calc.compute_50m_dsn_sections_full(
                df_merged_ruts, step_m=1.0, section_m=50.0
            )
            st.plotly_chart(
                charts.make_50m_bar_chart(
                    df_sec_dsn, "MAX_KOL", "Koleina Maksymalna (MAX_KOL)"
                ),
                use_container_width=True,
                key="fig_dsn_50m_bar",
            )
            st.dataframe(df_sec_dsn, use_container_width=True)

        with tab_ba_gddkia:
            st.header("📈 Analiza Porównawcza LCMS vs RSP (Metodologia GDDKiA)")
            if lcms_profiles and "LCMS_Rut_L" in df_merged_ruts.columns:
                rsp_r_v, lcms_r_v = (
                    df_merged_ruts["RSP_Rut_R"].values,
                    df_merged_ruts["LCMS_Rut_R"].values,
                )
                fig_ba_r, _, _, _, _ = charts.build_bland_altman_fig(
                    rsp_r_v, lcms_r_v, title="PRAWA KOLEINA (P_KOL)"
                )
                st.plotly_chart(
                    fig_ba_r, use_container_width=True, key="fig_bland_altman"
                )
                st.table(calc.build_rpd_table_df(rsp_r_v, lcms_r_v))
            else:
                st.warning(
                    "⚠️ Wgraj plik RSP oraz pliki LCMS, aby zobaczyć analizę"
                    " Blanda-Altmana."
                )

else:
    st.info("👈 Wgraj pliki w panelu bocznym po lewej stronie.")
