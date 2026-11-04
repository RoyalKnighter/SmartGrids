"""
CIGRE-Niederspannungsnetz (create_cigre_network_lv) als Ortsnetz für Task 4, Teil 2.

Das CIGRE-LV-Netz [13] besteht aus drei Teilnetzen, die je über einen eigenen
Ortsnetztransformator 20/0,4 kV an der 20-kV-Sammelschiene "Bus 0" hängen:
    R  Wohngebiet (residential), Kabel,       Trafo R0-R1 0,50 MVA
    I  Industrie (industrial),   Kabel,       Trafo I0-I1 0,15 MVA
    C  Gewerbe (commercial),     Freileitung, Trafo C0-C1 0,30 MVA
Die Teilnetze sind über die Leistungsschalter S1–S3 (Bus–Bus) mit Bus 0 verbunden. Beim Anschluss an
das MV-Netz wird Bus 0 mit dem gewählten MV-Bus vereinigt; das externe Netz des LV-Netzes entfällt.
Alle Namen erhalten das Präfix "LV" (z. B. "LV Bus R1", "LV S1"), damit sie eindeutig bleiben.

Verwendung:
    lv = lv_net()                                   # LV-Netz mit Koordinaten für den Netzplan
    s = ref.copy("w4_lv_bus5", overwrite=True)
    attach(s, "Bus 5")                              # anhängen und Profile zuordnen
"""

import json
import re

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

from elements import Profile

PREFIX = "LV"

# Profilzuordnung der LV-Lasten (Namen nach dem Anhängen mit Präfix)
PROFILE_RULES = [
    ("load", r"^LV Load R", Profile.HAUSHALT),       # Wohngebiet → BDEW H25
    ("load", r"^LV Load I", Profile.GEWERBE),        # Industrie → BDEW G25 (kein eigenes Industrieprofil)
    ("load", r"^LV Load C", Profile.GEWERBE),        # Gewerbe → BDEW G25
]

# Daten der LV-Leitungstypen nach der vom Nutzer angegebenen Quelle (Tabelle der CIGRE-LV-Leitungstypen):
# Querschnitt, R und L des Phasenleiters, R und L des Neutralleiters, Bemessungsstrom.
# Quelle: Smart Energy Management for Smart Grid, MDPI Books,
# https://mdpi-res.com/bookfiles/book/7980/Smart_Energy_Management_for_Smart_Grid.pdf
LINE_TYPE_DATA = pd.DataFrame(
    [("UG1", 240, 0.163, 0.433, 0.490, 1.50, 430), ("UG2", 150, 0.266, 0.481, 0.733, 1.81, 325),
     ("UG3", 120, 0.326, 0.503, 0.860, 2.01, 290), ("UG4", 25, 1.54, 0.656, 2.33, 4.63, 120),
     ("UG5", 35, 1.11, 0.621, 1.93, 4.03, 145), ("UG6", 70, 0.569, 0.554, 1.29, 2.75, 215),
     ("OH1", 50, 0.387, 0.939, 0.689, 1.50, 172), ("OH2", 35, 0.524, 0.977, 0.838, 1.56, 145),
     ("OH3", 16, 1.15, 1.05, 1.84, 1.69, 93), ("I1", 150, 0.266, 0.481, 0.733, 1.81, 325)],
    columns=["Typ", "Querschnitt [mm²]", "R Phase [Ω/km]", "L Phase [mH/km]", "R Neutral [Ω/km]",
             "L Neutral [mH/km]", "I_r [A]"]).set_index("Typ")


def apply_line_limits(net):
    """Thermische Grenzströme (max_i_ka) der LV-Leitungen aus LINE_TYPE_DATA setzen (ersetzt den Platzhalter 1 kA)."""
    for idx, t in net.line["std_type"].items():
        if t in LINE_TYPE_DATA.index:
            net.line.at[idx, "max_i_ka"] = LINE_TYPE_DATA.at[t, "I_r [A]"] / 1000
    return net


# Lage des LV-Netzes im gemeinsamen Netzplan (rechts neben dem CIGRE-MV-Netz)
X_OFFSET = 18.0
Y_TOP = 15.0
X_SCALE = 1.7          # LV-Netz in x-Richtung strecken, damit Busse und Beschriftungen Platz haben


def lv_net(x_offset=X_OFFSET, y_top=Y_TOP, x_scale=X_SCALE):
    """
    CIGRE-LV-Netz mit umgerechneten Koordinaten für den gemeinsamen Netzplan:
    gespiegelt (Abgänge nach unten), rechts neben das MV-Netz verschoben; die 20-kV-Busse R0, I0, C0
    werden über ihre Teilnetze gelegt, damit die Ortsnetztrafos sichtbar sind.
    """
    net = pn.create_cigre_network_lv()
    xy = {}
    for idx, g in net.bus["geo"].items():
        x, y = json.loads(g)["coordinates"][:2] if isinstance(g, str) else g["coordinates"][:2]
        xy[idx] = [x, y]
    # 20-kV-Busse der Teilnetze über den ersten LV-Bus des Teilnetzes legen
    for _, t in net.trafo.iterrows():
        xy[t.hv_bus][0] = xy[t.lv_bus][0]
    for idx, (x, y) in xy.items():
        net.bus.at[idx, "geo"] = json.dumps({"type": "Point",
                                             "coordinates": [x * x_scale + x_offset, y_top - (y - 1.0)]})
    return apply_line_limits(net)


def lv_net_alone():
    """
    Unverändertes CIGRE-LV-Netz (eigenes externes Netz an Bus 0, 20 kV, 1,0 pu); nur die Koordinaten
    sind für den Netzplan aufgelöst (gleiche Darstellung wie im angeschlossenen Fall).
    """
    return lv_net(x_offset=0.0, y_top=12.0)


def attach(s, at_bus, prefix=PREFIX):
    """LV-Netz an `at_bus` der Simulation `s` anhängen und den LV-Lasten Profile zuordnen."""
    s.attachNetwork(lv_net(), at_bus, prefix=prefix)
    s.assignProfilesByName(rules=PROFILE_RULES)
    return s


def bus_groups(net):
    """Busnamen getrennt nach Ebene: 110 kV, 20 kV (MV), 20 kV im LV-Netz (R0/I0/C0), 0,4 kV (LV)."""
    names = net.bus["name"]
    v = net.bus["vn_kv"]
    lv_part = names.str.startswith(f"{PREFIX} ")
    return {"HS": list(names[v > 100]),
            "MS": list(names[(v > 1) & (v <= 100) & ~lv_part]),
            "MS_LV": list(names[(v > 1) & lv_part]),
            "NS": list(names[v <= 1])}


def element_groups(net):
    """Leitungen und Trafos getrennt nach MV- und LV-Teil."""
    lines, trafos = net.line["name"], net.trafo["name"]
    if not (net.bus["vn_kv"] > 100).any():          # LV-Netz allein: alle Leitungen und Trafos gehören zum Ortsnetz
        return {"line_MS": [], "line_NS": list(lines), "trafo_HS": [], "trafo_NS": list(trafos)}
    return {"line_MS": list(lines[~lines.str.startswith(f"{PREFIX} ")]),
            "line_NS": list(lines[lines.str.startswith(f"{PREFIX} ")]),
            "trafo_HS": list(trafos[~trafos.str.startswith(f"{PREFIX} ")]),
            "trafo_NS": list(trafos[trafos.str.startswith(f"{PREFIX} ")])}


def _extrem(df, cols, fn):
    sub = df[cols]
    if sub.empty:
        return np.nan, "–"
    st = sub.stack()
    if st.empty:
        return np.nan, "–"
    key = st.idxmin() if fn == "min" else st.idxmax()
    return st.loc[key], key[1] if isinstance(key, tuple) else key


def kennwerte(vm, va, ll, tl, net, converged=None):
    """
    Höchste und niedrigste Werte getrennt nach MV und LV. vm, va, ll, tl sind Tabellen
    (Zeilen = Zeitpunkte oder ein einzelner Zeitpunkt, Spalten = Elementnamen).
    Rückgabe: (Werte, Orte) als zwei Series.
    """
    b, e = bus_groups(net), element_groups(net)
    werte, orte = {}, {}

    def add(label, df, cols, fn):
        v, where = _extrem(df, cols, fn)
        werte[label], orte[label] = v, where

    add("U min MS [pu]", vm, b["MS"], "min")
    add("U max MS [pu]", vm, b["MS"], "max")
    add("U min NS [pu]", vm, b["NS"], "min")
    add("U max NS [pu]", vm, b["NS"], "max")
    add("Winkel min MS [°]", va, b["MS"], "min")
    add("Winkel max MS [°]", va, b["MS"], "max")
    add("Winkel min NS [°]", va, b["NS"], "min")
    add("Winkel max NS [°]", va, b["NS"], "max")
    add("Leitung min MS [%]", ll, e["line_MS"], "min")
    add("Leitung max MS [%]", ll, e["line_MS"], "max")
    add("Leitung min NS [%]", ll, e["line_NS"], "min")
    add("Leitung max NS [%]", ll, e["line_NS"], "max")
    add("Trafo min HS/MS [%]", tl, e["trafo_HS"], "min")
    add("Trafo max HS/MS [%]", tl, e["trafo_HS"], "max")
    add("Ortsnetztrafo min [%]", tl, e["trafo_NS"], "min")
    add("Ortsnetztrafo max [%]", tl, e["trafo_NS"], "max")
    v_all, where_all = _extrem(vm, b["MS"] + b["NS"], "min")
    werte["U min gesamt [pu]"], orte["U min gesamt [pu]"] = v_all, where_all
    return pd.Series(werte), pd.Series(orte)


def kennwerte_sim(s):
    """Kennwerte einer Zeitreihe (nur konvergierte Zeitpunkte) und Anzahl konvergierter Lastflüsse."""
    r = s.results
    c = r.converged
    w, o = kennwerte(r.vm_pu[c], r.va_degree[c], r.line_loading[c], r.trafo_loading[c], s.net)
    g = bus_groups(s.net)
    vm = r.vm_pu[c]
    alle = g["MS"] + g["NS"]
    w["U max gesamt [pu]"] = vm[alle].max().max()
    o["U max gesamt [pu]"] = vm[alle].max().idxmax()
    for lvl, cols in (("MS", g["MS"]), ("NS", g["NS"])):
        if cols:
            w[f"Stunden U < 0,90 {lvl}"] = int((vm[cols].min(axis=1) < 0.90).sum())
            w[f"Stunden U < 0,95 {lvl}"] = int((vm[cols].min(axis=1) < 0.95).sum())
            w[f"Stunden U > 1,05 {lvl}"] = int((vm[cols].max(axis=1) > 1.05).sum())
            w[f"Stunden U > 1,10 {lvl}"] = int((vm[cols].max(axis=1) > 1.10).sum())
    ext = r.ext_grid_p_mw[c].iloc[:, 0]
    w["Verluste [MWh]"] = float((r.line_pl_mw[c].sum(axis=1) + r.trafo_pl_mw[c].sum(axis=1)).sum()) \
        if len(r.line_pl_mw.columns) else np.nan
    w["Stunden Rückspeisung 110 kV"] = int((ext < 0).sum())
    w["max. Rückspeisung 110 kV [MW]"] = max(0.0, -ext.min())
    e = element_groups(s.net)
    for label, cols, df in (("Leitung MS", e["line_MS"], r.line_loading), ("Leitung NS", e["line_NS"], r.line_loading),
                            ("Trafo HS/MS", e["trafo_HS"], r.trafo_loading), ("Ortsnetztrafo", e["trafo_NS"], r.trafo_loading)):
        if cols:
            w[f"Stunden {label} > 100 %"] = int((df[c][cols].max(axis=1) > 100).sum())
    w["konvergiert"] = int(c.sum())
    w["Zeitpunkte"] = len(c)
    return w, o


def kennwerte_static(net):
    """Kennwerte eines statischen Lastflusses (nach pp.runpp)."""
    row = lambda tab, col: pd.DataFrame([net[f"res_{tab}"][col].to_numpy()], columns=net[tab]["name"].to_numpy())
    return kennwerte(row("bus", "vm_pu"), row("bus", "va_degree"), row("line", "loading_percent"),
                     row("trafo", "loading_percent"), net)


# ---------------------------------------------------------------------------
# Haushalte mit PV bzw. PV und Speicher (Szenario 4.2)
# ---------------------------------------------------------------------------

def prosumer(s, anteil_pv, anteil_speicher, muster=r"^LV Load R"):
    """
    Haushaltslasten des Ortsnetzes aufteilen: Von jeder Haushaltslast (Name passt auf `muster`)
    behält der Anteil 1 − anteil_pv − anteil_speicher das Profil H25, der Anteil anteil_pv wird als
    eigene Last mit P25 (Haushalt mit PV, Netzbezug) und anteil_speicher als eigene Last mit S25
    (Haushalt mit PV und Speicher, Netzbezug) angelegt.
    Skalierung wie bisher über die Jahreshöchstleistung: Jede Teillast erhält den entsprechenden
    Anteil der ursprünglichen Höchstleistung p_mw (die Winter-Abendspitze ändert sich durch PV kaum).
    Rückgabe: Tabelle der Aufteilung.
    """
    from elements import Load, Profile
    if anteil_pv < 0 or anteil_speicher < 0 or anteil_pv + anteil_speicher > 1:
        raise ValueError("Anteile müssen zwischen 0 und 1 liegen und zusammen höchstens 1 ergeben.")
    rows = []
    loads = s.net.load[s.net.load["name"].str.contains(muster, regex=True) & (s.net.load["profile"] == "HAUSHALT")]
    for idx, r in loads.iterrows():
        p, bus = float(r.p_mw), s.net.bus.at[r.bus, "name"]
        cos = float(r.cos_phi) if not pd.isna(r.cos_phi) else None
        rest = 1 - anteil_pv - anteil_speicher
        s.changeParameter(r["name"], "p_mw", p * rest)
        if anteil_pv > 0:
            s.createElement(Load(f"{r['name']} PV", bus=bus, profile=Profile.HAUSHALT_PV, p_mw=p * anteil_pv, cos_phi=cos))
        if anteil_speicher > 0:
            s.createElement(Load(f"{r['name']} PV+Speicher", bus=bus, profile=Profile.HAUSHALT_PV_SPEICHER,
                                 p_mw=p * anteil_speicher, cos_phi=cos))
        rows.append({"Last": r["name"], "Bus": bus, "p_mw vorher": p, "H25": p * rest,
                     "P25": p * anteil_pv, "S25": p * anteil_speicher})
    return pd.DataFrame(rows).set_index("Last")
