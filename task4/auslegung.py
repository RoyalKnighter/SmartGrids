"""
Auslegung von Ortsnetztrafos und Leitungen nach den Spitzenwerten einer Zeitreihe (Task 4, Teil 3).

    tab = spitzen(sim, ["LV Trafo R0-R1", "LV Line C1-C2"])        # Spitzenwerte und Auslastung
    trafo_kandidaten(net, s_max_mva=0.42, max_auslastung=0.7)       # passende pandapower-Standardtrafos
    leitung_kandidaten(net, i_max_a=190, art="ol", max_auslastung=0.7)
"""

import numpy as np
import pandas as pd
import pandapower as pp

from diagrams import rating


def spitzen(s, elemente):
    """
    Spitzenwerte je Element über alle konvergierten Zeitpunkte:
    P max, S max mit Zeitpunkt, I max mit Zeitpunkt, Bemessungswert und höchste Auslastung.
    """
    r = s.results
    c = r.converged
    P, S, I = (r.quantity(s.net, k)[c] for k in ("P", "S", "I"))
    rows = {}
    for e in elemente:
        row = {}
        if e in P:
            row["P max [MW]"] = P[e].abs().max()
        if e in S:
            row["S max [MVA]"] = S[e].max()
            row["Zeitpunkt S max"] = f"{S[e].idxmax():%a %d.%m. %H:%M}"
        if e in I:
            row["I max [A]"] = I[e].max()
            row["Zeitpunkt I max"] = f"{I[e].idxmax():%a %d.%m. %H:%M}"
        s_r, i_r = rating(s.net, e, "S"), rating(s.net, e, "I")
        if s_r is not None:
            row["S_r [MVA]"] = s_r
            row["I_r [A]"] = i_r
            row["Auslastung max [%]"] = 100 * (row["I max [A]"] / i_r if e in I else row["S max [MVA]"] / s_r)
        idx = s.net.line.index[s.net.line["name"] == e]
        if len(idx):
            row["Typ"] = s.net.line.at[idx[0], "std_type"]
        idx = s.net.trafo.index[s.net.trafo["name"] == e]
        if len(idx):
            row["Typ"] = s.net.trafo.at[idx[0], "std_type"] or "CIGRE (ohne Standardtyp)"
        rows[e] = row
    return pd.DataFrame(rows).T


def trafo_kandidaten(net, s_max_mva, max_auslastung=0.7, vn_hv_kv=20.0, vn_lv_kv=0.4):
    """
    pandapower-Standardtrafos mit passender Spannung, sortiert nach Bemessungsleistung.
    geeignet: Spitzenlast ≤ max_auslastung · S_r (Reserve für Last- und Stromspitzen).
    """
    st = pp.available_std_types(net, "trafo")
    st = st[(st.vn_hv_kv == vn_hv_kv) & (st.vn_lv_kv == vn_lv_kv)].sort_values("sn_mva")
    out = st[["sn_mva", "vk_percent", "vkr_percent", "pfe_kw", "i0_percent", "vector_group", "shift_degree",
              "tap_side", "tap_min", "tap_max", "tap_step_percent"]].copy()
    out["Auslastung bei Spitze [%]"] = 100 * s_max_mva / out.sn_mva
    out["geeignet"] = out["Auslastung bei Spitze [%]"] <= 100 * max_auslastung
    return out


def leitung_kandidaten(net, i_max_a, art=None, max_auslastung=0.7, voltage_rating="LV"):
    """
    pandapower-Standardleitungen der Spannungsebene (LV/MV/HV), optional nur Kabel (art="cs") oder
    Freileitungen (art="ol"), sortiert nach Bemessungsstrom.
    geeignet: Spitzenstrom ≤ max_auslastung · I_r.
    """
    st = pp.available_std_types(net, "line")
    st = st[st.voltage_rating == voltage_rating]
    if art:
        st = st[st.type == art]
    st = st.sort_values("max_i_ka")
    out = st[["type", "q_mm2", "r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka"]].copy()
    out["I_r [A]"] = out.max_i_ka * 1000
    out["Auslastung bei Spitze [%]"] = 100 * i_max_a / out["I_r [A]"]
    out["geeignet"] = out["Auslastung bei Spitze [%]"] <= 100 * max_auslastung
    return out.drop(columns="max_i_ka")


def kleinster_geeigneter(kandidaten):
    """Name des kleinsten geeigneten Typs oder None."""
    g = kandidaten[kandidaten["geeignet"]]
    return g.index[0] if len(g) else None
