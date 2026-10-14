import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def get_upper_hull(x_sec, y_sec):
    """Oblicza wirtualną łatę (górną otoczkę wypukłą) dla dowolnego (również zmiennego) rozstawu punktów X."""
    pts = list(zip(x_sec, y_sec))
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

    x_h = [p[0] for p in upper]
    y_h = [p[1] for p in upper]
    return np.interp(x_sec, x_h, y_h)


st.set_page_config(
    page_title="Przekrój Poprzeczny RSP (DSN 2019)", layout="wide"
)
st.title("📐 Przekrój Poprzeczny RSP z Wirtualną Łatą")

# 1. Menu boczne
st.sidebar.header("1. Wczytaj dane")
uploaded_file = st.sidebar.file_uploader(
    "Wgraj plik (.xlsx lub .csv)", type=["xlsx", "csv"]
)

st.sidebar.header("2. Geometria belki RSP")
typ_geometrii = st.sidebar.radio(
    "Rozstaw czujników:",
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
    domyslne_x_str = "-1.65, -1.48, -1.32, -1.16, -1.00, -0.84, -0.68, -0.52, -0.36, -0.18, 0.0, 0.18, 0.36, 0.52, 0.68, 0.84, 1.00, 1.16, 1.32, 1.48, 1.65"
    x_input = st.sidebar.text_area(
        "Wpisz pozycje X [m] dla L1..L21 (rozdzielone przecinkami):",
        value=domyslne_x_str,
        height=100,
        help="Wprowadź 21 wartości pozycji poprzecznych od lewej do prawej.",
    )
    try:
        pozycja_x = [
            float(val.strip().replace(",", "."))
            for val in x_input.split(",")
            if val.strip() != ""
        ]
    except Exception:
        st.sidebar.error("Błąd w formacie liczb! Używam domyślnych pozycji.")
        pozycja_x = [round((i - 10) * 0.165, 3) for i in range(21)]

st.sidebar.header("3. Opcje analizy i DSN 2019")
test_stacjonarny = st.sidebar.checkbox(
    "Tryb testu stacjonarnego (Łata 0)",
    value=False,
    help="Tylko dla pomiarów kontrolnych na łacie stacjonarnej (limit <= 0.5 mm).",
)

tryb_zero_l11 = st.sidebar.checkbox(
    "Ustaw L11 jako Zero (wysokość względna)",
    value=True,
)

pokaz_wysokosc = st.sidebar.checkbox(
    "Wizualizuj głębokość kolein (pionowe linie od łaty)",
    value=True,
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    col_profil = "Profil" if "Profil" in df.columns else df.columns[0]
    min_p = int(df[col_profil].min())
    max_p = int(df[col_profil].max())

    st.sidebar.header("4. Wybór profilu")
    wybrany_profil = st.sidebar.number_input(
        f"Wpisz numer profilu [{min_p} - {max_p}]:",
        min_value=min_p,
        max_value=max_p,
        value=min_p,
        step=1,
    )

    wiersz = df[df[col_profil] == wybrany_profil]

    if not wiersz.empty:
        kolumny_laserow = [c for c in df.columns if c != col_profil]
        odczyty_raw = wiersz.iloc[0][kolumny_laserow].values.astype(float)

        ile_laserow = len(odczyty_raw)

        if len(pozycja_x) != ile_laserow:
            st.warning(
                f"Liczba pozycji X ({len(pozycja_x)}) nie odpowiada liczbie czujników w pliku ({ile_laserow}). Zastosowano automatyczny podział."
            )
            x_vals = [
                round((i - (ile_laserow - 1) / 2) * 0.165, 3)
                for i in range(ile_laserow)
            ]
        else:
            x_vals = pozycja_x

        # Wyznaczenie wysokości Y
        if tryb_zero_l11:
            idx_srodek = ile_laserow // 2
            ref_val = odczyty_raw[idx_srodek]
            y_vals = ref_val - odczyty_raw
            y_label = "Wysokość względna do L11 [mm]"
        else:
            y_vals = odczyty_raw
            y_label = "Odczyt surowy czujnika [mm]"

        # Szukanie najwyższego punktu korony jezdni w strefie środkowej
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

        # Odległość pionowa (głębokość koleiny od łaty)
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
                    f"✅ **TEST POZYTYWNY**: Maksymalna odchyłka wynosi **{max_koleina:.2f} mm** (≤ 0,5 mm)."
                )
            else:
                st.error(
                    f"❌ **TEST NEGATYWNY**: Odchyłka wynosi **{max_koleina:.2f} mm** i przekracza limit 0,5 mm!"
                )

        # Podsumowanie liczbowe
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Koleina Lewa (od łaty)",
            f"{koleina_lewa_mm:.1f} mm",
            delta=f"Laser L{idx_kol_l+1} ({x_vals[idx_kol_l]}m)",
            delta_color="inverse",
        )
        c2.metric(
            "Koleina Prawa (od łaty)",
            f"{koleina_prawa_mm:.1f} mm",
            delta=f"Laser L{idx_kol_p+1} ({x_vals[idx_kol_p]}m)",
            delta_color="inverse",
        )
        c3.metric("Maksymalna Koleina", f"{max_koleina:.1f} mm")

        # Wykres
        fig = go.Figure()

        # Profil nawierzchni
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name="Profil nawierzchni",
                line=dict(color="royalblue", width=3),
                marker=dict(size=7, color="navy"),
                hovertemplate="Poz X: %{x} m<br>Wys Y: %{y:.1f} mm<extra></extra>",
            )
        )

        # Łata lewa
        fig.add_trace(
            go.Scatter(
                x=x_left,
                y=lata_lewa_y,
                mode="lines",
                name="Wirtualna Łata Lewa",
                line=dict(color="crimson", width=2, dash="dash"),
            )
        )

        # Łata prawa
        fig.add_trace(
            go.Scatter(
                x=x_right,
                y=lata_prawa_y,
                mode="lines",
                name="Wirtualna Łata Prawa",
                line=dict(color="darkorange", width=2, dash="dash"),
            )
        )

        # Rysowanie pionowych odcinków wysokości kolein (od łaty do dołka)
        if pokaz_wysokosc:
            # Pionowa linia lewej koleiny
            x_kl = x_vals[idx_kol_l]
            y_kl_prof = y_vals[idx_kol_l]
            y_kl_lata = lata_lewa_y[idx_kol_l]

            fig.add_trace(
                go.Scatter(
                    x=[x_kl, x_kl],
                    y=[y_kl_lata, y_kl_prof],
                    mode="lines+markers",
                    name=f"Głębokość Lewa ({koleina_lewa_mm:.1f} mm)",
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
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="crimson",
                borderwidth=1,
                xanchor="right" if x_kl < 0 else "left",
            )

            # Pionowa linia prawej koleiny
            x_kp = x_vals[idx_kol_p]
            y_kp_prof = y_vals[idx_kol_p]
            y_kp_lata = lata_prawa_y[idx_kol_p_rel]

            fig.add_trace(
                go.Scatter(
                    x=[x_kp, x_kp],
                    y=[y_kp_lata, y_kp_prof],
                    mode="lines+markers",
                    name=f"Głębokość Prawa ({koleina_prawa_mm:.1f} mm)",
                    line=dict(color="darkorange", width=3),
                    marker=dict(size=6, symbol="triangle-down"),
                )
            )

            fig.add_annotation(
                x=x_kp,
                y=(y_kp_lata + y_kp_prof) / 2,
                text=f" <b>{koleina_prawa_mm:.1f} mm</b>",
                showarrow=False,
                font=dict(color="darkorange", size=13),
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="darkorange",
                borderwidth=1,
                xanchor="left" if x_kp > 0 else "right",
            )

        # Oś zero L11
        fig.add_vline(
            x=0,
            line_width=1,
            line_dash="dot",
            line_color="gray",
            annotation_text="Oś L11 (0.0 m)",
        )

        autorange_y = True if tryb_zero_l11 else "reversed"

        fig.update_layout(
            title=f"Przekrój poprzeczny RSP – Profil nr {wybrany_profil}",
            xaxis_title="Szerokość / Pozycja czujnika na belce [m]",
            yaxis_title=y_label,
            height=600,
            hovermode="x unified",
            xaxis=dict(range=[-1.75, 1.75], dtick=0.3),
            yaxis=dict(autorange=autorange_y),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tabela odczytów
        st.subheader("Odczyty czujników z rzeczywistymi pozycjami [m]:")
        df_tbl = pd.DataFrame(
            [y_vals],
            columns=[
                f"L{i+1} ({x_vals[i]:.2f}m)" for i in range(ile_laserow)
            ],
        )
        st.dataframe(df_tbl, use_container_width=True)

    else:
        st.warning(f"Brak danych dla profilu {wybrany_profil}.")
else:
    st.info("👈 Wgraj plik z danymi w panelu bocznym po lewej stronie.")
