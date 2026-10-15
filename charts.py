from calculations import get_dsn_color
import numpy as np
import plotly.graph_objects as go


def make_map_trace(
    lat, lon, mode="lines", marker=None, line=None, name="", text=None
):
    kwargs = {"lat": lat, "lon": lon, "mode": mode, "name": name}
    if marker is not None:
        kwargs["marker"] = marker
    if line is not None:
        clean_line = line.copy()
        clean_line.pop("dash", None)
        kwargs["line"] = clean_line
    if text is not None:
        kwargs["text"], kwargs["hoverinfo"] = text, "text"

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
    fig.add_trace(
        go.Scatter(
            x=means,
            y=diffs,
            mode="markers",
            name="Pomiary",
            marker=dict(
                size=7,
                color="#ffffcc",
                line=dict(width=1, color="#333333"),
            ),
            hovertemplate=(
                "Średnia: %{x:.2f} mm<br>Różnica (LCMS-RSP): %{y:.2f}"
                " mm<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=0, line_width=1, line_dash="solid", line_color="lightgray"
    )
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
