import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Przekrój Poprzeczny RSP (21 laserów)", layout="wide"
)
st.title("📐 Przekrój Poprzeczny RSP (-1.65m do +1.65m)")

# 1. Menu boczne
st.sidebar.header("1. Wczytaj dane")
uploaded_file = st.sidebar.file_uploader(
    "Wgraj plik (.xlsx lub .csv)", type=["xlsx", "csv"]
)

st.sidebar.header("2. Geometria belki RSP")
krok_laserow = st.sidebar.number_input(
    "Odstęp między laserami [m]:",
    min_value=0.05,
    max_value=0.50,
    value=0.165,
    step=0.005,
    format="%.3f",
)

st.sidebar.header("3. Opcje widoku")
odwroc_y = st.sidebar.checkbox(
    "Odwróć oś Y (koleiny w dół)",
    value=True,
    help="Zaznaczone: większe wartości odczytu rysują dołek w dół.",
)
odwroc_x = st.sidebar.checkbox(
    "Zamień lewą z prawą (odwróć oś X)",
    value=False,
    help="Zaznaczone: zamienia kierunek czujników L1-L21.",
)

# Wyliczenie pozycji poprzecznej dla 21 laserów
pozycja_x = [round((i - 10) * krok_laserow, 3) for i in range(21)]

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
        pomiary = wiersz.iloc[0][kolumny_laserow].values

        ile_laserow = len(pomiary)
        if ile_laserow == 21:
            x_vals = pozycja_x
        else:
            srodek = (ile_laserow - 1) / 2
            x_vals = [
                round((i - srodek) * krok_laserow, 3)
                for i in range(ile_laserow)
            ]

        # Wykres przekroju poprzecznego
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=pomiary,
                mode="lines+markers",
                name=f"Profil {wybrany_profil}",
                line=dict(color="royalblue", width=3),
                marker=dict(size=7, color="firebrick"),
                hovertemplate="Pozycja: %{x} m<br>Głębokość: %{y} mm<extra></extra>",
            )
        )

        # Ustawienia kierunku osi dla Plotly (True = standardowo, "reversed" = odwrócona)
        y_mode = "reversed" if odwroc_y else True
        x_mode = "reversed" if odwroc_x else True

        # Pionowa linia osi L11 (0.0 m)
        fig.add_vline(
            x=0,
            line_width=1,
            line_dash="dot",
            line_color="gray",
            annotation_text="Oś L11 (0.0 m)",
        )

        fig.update_layout(
            title=f"Przekrój poprzeczny nawierzchni – Profil nr {wybrany_profil}",
            xaxis_title="Szerokość pasa / Szerokość belki RSP [m]",
            yaxis_title="Odczyt / Głębokość [mm]",
            height=550,
            hovermode="x unified",
            xaxis=dict(autorange=x_mode, dtick=0.3),
            yaxis=dict(autorange=y_mode),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tabela pod wykresem
        st.subheader("Odczyty czujników z pozycją na belce:")
        df_tbl = pd.DataFrame(
            [pomiary],
            columns=[
                f"L{i+1} ({x_vals[i]}m)" for i in range(len(pomiary))
            ],
        )
        st.dataframe(df_tbl, use_container_width=True)

    else:
        st.warning(f"Brak danych dla profilu {wybrany_profil}.")
else:
    st.info("👈 Wgraj plik z danymi w panelu bocznym po lewej stronie.")