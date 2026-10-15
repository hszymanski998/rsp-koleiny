import numpy as np
import pandas as pd


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
