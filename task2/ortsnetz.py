"""
Ortsnetz für Aufgabe 4: Erweiterung des bestehenden Netzes um den Niederspannungsteil
des Kerber-Dorfnetzes.

Quelle der Netzstruktur:
    G. Kerber, "Aufnahmefähigkeit von Niederspannungsverteilnetzen für die Einspeisung
    aus Photovoltaikkleinanlagen", Dissertation, Technische Universität München, 2011.
    Umsetzung in pandapower: pandapower.networks.create_kerber_dorfnetz()

Aufbau
------
- Unverändert aus dem bestehenden Netz: externes Netz, Bus 1 (20 kV), Transformator
  0,4 MVA 20/0,4 kV mit Stufenschalter, Bus 2 (NS-Sammelschiene), Leitung zu Bus 3 mit Last.
- Neu: 6 Abgänge des Kerber-Dorfnetzes an Bus 2 (57 Hausanschlüsse). Kerbers eigene
  10-kV-Seite und sein Transformator werden nicht übernommen; seine NS-Sammelschiene
  ("main_busbar") wird durch Bus 2 ersetzt.
- Neu (optional): Mittelspannungsleitung zwischen einem Umspannwerk (externes Netz) und
  Bus 1 sowie die Windenergieanlage an Bus 1.
- Jede Leitung erhält am Anfang und am Ende einen Leistungsschalter (Aufgabe 1).
"""

import re
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top

import base_net as bn

warnings.simplefilter(action="ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------

# Kabeldaten der Kerber-Netze (wie in pandapower hinterlegt). Die Widerstände entsprechen
# den Höchstwerten nach DIN EN 60228 für Aluminiumleiter (50 mm²: 0,641 Ω/km,
# 150 mm²: 0,206 Ω/km bei 20 °C).
CABLE_MAIN = {"name": "NAYY 4x150", "r_ohm_per_km": 0.2067, "x_ohm_per_km": 0.080425,
              "c_nf_per_km": 830.0, "max_i_ka": 0.275}
CABLE_BRANCH = {"name": "NAYY 4x50", "r_ohm_per_km": 0.6417, "x_ohm_per_km": 0.084823,
                "c_nf_per_km": 670.0, "max_i_ka": 0.141}

# Mittelspannungsleitung zum Umspannwerk (nur mit Windenergieanlage)
MV_CABLE_STD_TYPE = "NA2XS2Y 1x185 RM/25 12/20 kV"


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _add_breakers(net, line, from_bus, to_bus):
    """Leistungsschalter am Anfang und Ende einer Leitung (Normalzustand: geschlossen)."""
    in_ka = net.line.at[line, "max_i_ka"]
    name = net.line.at[line, "name"]
    s1 = pp.create_switch(net, bus=from_bus, element=line, et="l", type="CB", closed=True,
                          name=f"CB {name} Start", in_ka=in_ka)
    s2 = pp.create_switch(net, bus=to_bus, element=line, et="l", type="CB", closed=True,
                          name=f"CB {name} Ende", in_ka=in_ka)
    return s1, s2


def _ensure_breakers(net):
    """Ergänzt fehlende Schalter an bestehenden Leitungen (falls build_base_net keine anlegt)."""
    for line in net.line.index:
        existing = net.switch[(net.switch.et == "l") & (net.switch.element == line)]
        if existing.empty:
            _add_breakers(net, line, net.line.at[line, "from_bus"], net.line.at[line, "to_bus"])


# ---------------------------------------------------------------------------
# Netzaufbau
# ---------------------------------------------------------------------------

def build_ortsnetz(with_mv_line=True, mv_line_km=5.0):
    """
    Erzeugt das Ortsnetz.

    Parameter
    ---------
    with_mv_line : True -> externes Netz an einem neuen Bus "Bus 0 UW" (Umspannwerk, 20 kV),
                   verbunden über eine MS-Kabelleitung mit Bus 1. Die Windenergieanlage wird
                   an Bus 1 angeschlossen und beeinflusst so die Spannung an der Ortsnetzstation.
    mv_line_km   : Länge der MS-Leitung [km]

    Rückgabe
    --------
    net, idx, households
        idx        : Dictionary der wichtigsten Elemente (wie build_base_net, ergänzt um
                     "bus0", "mv_line", "wind", "eval_bus")
        households : DataFrame mit einer Zeile je Hausanschluss
                     (load, bus, feeder, position, distance_m)
    """
    net, idx = bn.build_base_net()
    _ensure_breakers(net)

    # Bestehende Last an Bus 3 als Haushalt des Abgangs 0 ("Bestand")
    household_rows = [{"load": idx["load"], "bus": idx["bus3"], "feeder": 0, "position": 1,
                       "name": net.load.at[idx["load"], "name"]}]

    # --- Kerber-Dorfnetz einlesen und NS-Teil übertragen -------------------
    kerber = pn.create_kerber_dorfnetz()
    k_main = kerber.trafo.at[kerber.trafo.index[0], "lv_bus"]
    bus_map = {k_main: idx["bus2"]}

    for kb in kerber.bus.index:
        if kb == k_main or kerber.bus.at[kb, "vn_kv"] != 0.4:
            continue
        bus_map[kb] = pp.create_bus(net, vn_kv=0.4, name=f"D {kerber.bus.at[kb, 'name']}",
                                    type=kerber.bus.at[kb, "type"])

    for kl in kerber.line.index:
        k_name = kerber.line.at[kl, "name"]
        cable = CABLE_BRANCH if k_name.startswith("branchout") else CABLE_MAIN
        fb, tb = bus_map[kerber.line.at[kl, "from_bus"]], bus_map[kerber.line.at[kl, "to_bus"]]
        line = pp.create_line_from_parameters(
            net, from_bus=fb, to_bus=tb, length_km=kerber.line.at[kl, "length_km"],
            r_ohm_per_km=cable["r_ohm_per_km"], x_ohm_per_km=cable["x_ohm_per_km"],
            c_nf_per_km=cable["c_nf_per_km"], max_i_ka=cable["max_i_ka"],
            name=f"D {k_name}", type="cs")
        net.line.at[line, "std_type"] = cable["name"]
        _add_breakers(net, line, fb, tb)

    for kload in kerber.load.index:
        kb = kerber.load.at[kload, "bus"]
        k_name = kerber.bus.at[kb, "name"]                       # z. B. loadbus_3_7
        feeder, position = map(int, re.findall(r"_(\d+)_(\d+)$", k_name)[0])
        load = pp.create_load(net, bus=bus_map[kb], p_mw=0.0, q_mvar=0.0, name=f"HH {feeder}-{position}")
        household_rows.append({"load": load, "bus": bus_map[kb], "feeder": feeder,
                               "position": position, "name": f"HH {feeder}-{position}"})

    # --- optionale MS-Leitung und Windenergieanlage -------------------------
    if with_mv_line:
        bus0 = pp.create_bus(net, vn_kv=20.0, name="Bus 0 UW", type="b")
        net.ext_grid.at[idx["ext"], "bus"] = bus0                # externes Netz ins Umspannwerk
        mv_line = pp.create_line(net, from_bus=bus0, to_bus=idx["bus1"], length_km=mv_line_km,
                                 std_type=MV_CABLE_STD_TYPE, name="MS-Leitung UW - ONS")
        _add_breakers(net, mv_line, bus0, idx["bus1"])
        wind = pp.create_sgen(net, bus=idx["bus1"], p_mw=0.0, q_mvar=0.0, name="WEA Vestas V90-2.0 MW",
                              type="WP")
        idx.update({"bus0": bus0, "mv_line": mv_line, "wind": wind})

    # --- Entfernungen und Auswertungsbus ------------------------------------
    households = pd.DataFrame(household_rows).set_index("load")
    graph = top.create_nxgraph(net, include_trafos=False)
    dist = nx.single_source_dijkstra_path_length(graph, idx["bus2"], weight="weight")
    households["distance_m"] = [dist[b] * 1000 for b in households.bus]

    idx["eval_bus"] = int(households.sort_values("distance_m").bus.iloc[-1])
    return net, idx, households


def assign_pv(households, share=0.5):
    """
    Legt fest, welche Haushalte eine PV-Anlage haben. In jedem Abgang wird der Anteil
    `share` der Haushalte gleichmäßig über die Länge des Abgangs verteilt ausgewählt
    (deterministisch, damit die Ergebnisse reproduzierbar sind).
    Rückgabe: boolesche Series (Index = load)
    """
    has_pv = pd.Series(False, index=households.index)
    for _, group in households.groupby("feeder"):
        ordered = group.sort_values("distance_m").index
        n = int(round(share * len(ordered)))
        if n > 0:
            picks = np.unique(np.round(np.linspace(0, len(ordered) - 1, n)).astype(int))
            has_pv[ordered[picks]] = True
    return has_pv


def add_pv_systems(net, households, has_pv, kwp):
    """Legt für alle Haushalte mit PV eine PV-Anlage (sgen) am Hausanschluss an."""
    rows = []
    for load in households.index[has_pv]:
        bus = households.at[load, "bus"]
        sgen = pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, type="PV",
                              name=f"PV {households.at[load, 'name'][3:]}")
        rows.append({"sgen": sgen, "load": load, "bus": bus, "feeder": households.at[load, "feeder"],
                     "kwp": kwp})
    return pd.DataFrame(rows).set_index("sgen")


def describe_network(net, idx, households, pv=None):
    """Übersichtstabelle der Abgänge."""
    rows = []
    for feeder, g in households.groupby("feeder"):
        rows.append({
            "Abgang": "Bestand (Bus 3)" if feeder == 0 else feeder,
            "Hausanschlüsse": len(g),
            "PV-Anlagen": 0 if pv is None else int((pv.feeder == feeder).sum()),
            "max. Entfernung [m]": round(g.distance_m.max()),
        })
    return pd.DataFrame(rows).set_index("Abgang")
