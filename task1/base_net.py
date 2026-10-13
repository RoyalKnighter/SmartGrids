"""
Hilfsfunktionen für die Untersuchung "Netzparameter und Busspannungen".

Enthält:
- build_base_net():          Ausgangsnetz (Baseline) aus dem pandapower-Minimalbeispiel
- build_110kV_net():         Erweiterung mit 110-kV-Ebene und zusätzlichem 110/20-kV-Trafo
- build_110kV_net_simple():  110 kV ohne Topologieänderung (nur Parameter angepasst)
- build_2_load_net():        Baseline mit zweiter Last an Bus 3
- vary_parameter():          Variiert einen Parameter eines Bauteils, rechnet für jeden Wert
                             einen Lastfluss auf einer frischen Netzkopie und gibt die
                             gewünschten Ergebnisse als Tabelle zurück
"""

import re
import warnings

import pandas as pd
import pandapower as pp
from pandapower.create import (
    create_empty_network,
    create_bus,
    create_ext_grid,
    create_load,
    create_line,
    create_transformer,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Netze
# ---------------------------------------------------------------------------

def build_base_net():
    """Ausgangsnetz: ext_grid -> Bus 1 (20 kV) -> Trafo -> Bus 2 (0,4 kV) -> Leitung -> Bus 3 -> Last."""
    net = create_empty_network()

    # create buses
    bus1 = create_bus(net, vn_kv=20., name="Bus 1")
    bus2 = create_bus(net, vn_kv=0.4, name="Bus 2")
    bus3 = create_bus(net, vn_kv=0.4, name="Bus 3")

    # create bus elements
    ext = create_ext_grid(net, bus=bus1, vm_pu=1.02, name="Grid Connection")
    load = create_load(net, bus=bus3, p_mw=0.05, q_mvar=0.005, name="Load")

    # create branch elements
    trafo = create_transformer(net, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV", name="Trafo")
    line = create_line(net, from_bus=bus2, to_bus=bus3, length_km=0.1, std_type="NAYY 4x50 SE", name="Line")

    idx = {
        "bus1": bus1, "bus2": bus2, "bus3": bus3,
        "ext": ext, "load": load, "trafo": trafo, "line": line,
    }
    return net, idx


def build_110kV_net():
    """Erweiterung: zusätzliche 110-kV-Ebene mit 110/20-kV-Umspanntrafo vor dem Ortsnetztrafo."""
    net = create_empty_network()

    # create buses
    bus1 = create_bus(net, vn_kv=110., name="Bus 1")
    bus2 = create_bus(net, vn_kv=20., name="Bus 2")
    bus3 = create_bus(net, vn_kv=0.4, name="Bus 3")
    bus4 = create_bus(net, vn_kv=0.4, name="Bus 4")

    # create bus elements
    ext = create_ext_grid(net, bus=bus1, vm_pu=1.02, name="Grid Connection")
    load = create_load(net, bus=bus4, p_mw=0.05, q_mvar=0.005, name="Load")

    # create branch elements
    trafo1 = create_transformer(net, hv_bus=bus1, lv_bus=bus2, std_type="25 MVA 110/20 kV", name="Trafo 1")
    trafo2 = create_transformer(net, hv_bus=bus2, lv_bus=bus3, std_type="0.4 MVA 20/0.4 kV", name="Trafo 2")
    line = create_line(net, from_bus=bus3, to_bus=bus4, length_km=0.1, std_type="NAYY 4x50 SE", name="Line")

    idx = {
        "bus1": bus1, "bus2": bus2, "bus3": bus3, "bus4": bus4,
        "ext": ext, "load": load, "trafo1": trafo1, "trafo2": trafo2, "line": line,
    }
    return net, idx


def build_110kV_net_simple():
    """110 kV ohne Topologieänderung: gleiche Busse und Elemente wie die Baseline,
    nur Nennspannung von Bus 1 und Oberspannung des Trafos auf 110 kV gesetzt."""
    net, idx = build_base_net()
    net.bus.at[idx["bus1"], "vn_kv"] = 110.
    net.trafo.at[idx["trafo"], "vn_hv_kv"] = 110.
    net.trafo.at[idx["trafo"], "std_type"] = None          # Standardtyp passt nicht mehr
    net.trafo.at[idx["trafo"], "name"] = "Trafo 110/0.4 kV"
    return net, idx


def build_2_load_net():
    """Baseline mit zweiter Last an Bus 3."""
    net = create_empty_network()

    # create buses
    bus1 = create_bus(net, vn_kv=20., name="Bus 1")
    bus2 = create_bus(net, vn_kv=0.4, name="Bus 2")
    bus3 = create_bus(net, vn_kv=0.4, name="Bus 3")

    # create bus elements
    ext = create_ext_grid(net, bus=bus1, vm_pu=1.02, name="Grid Connection")
    load1 = create_load(net, bus=bus3, p_mw=0.05, q_mvar=0.005, name="Load 1")
    load2 = create_load(net, bus=bus3, p_mw=0.02, q_mvar=0.005, name="Load 2")

    # create branch elements
    trafo = create_transformer(net, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV", name="Trafo")
    line = create_line(net, from_bus=bus2, to_bus=bus3, length_km=0.1, std_type="NAYY 4x50 SE", name="Line")

    idx = {
        "bus1": bus1, "bus2": bus2, "bus3": bus3,
        "ext": ext, "load1": load1, "load2": load2, "trafo": trafo, "line": line,
    }
    return net, idx


# ---------------------------------------------------------------------------
# Parametervariation
# ---------------------------------------------------------------------------

# Zuordnung: Schlüssel aus idx (ohne Ziffern am Ende) -> Elementtabelle in net
_KEY_TO_TABLE = {
    "bus": "bus",
    "ext": "ext_grid",
    "load": "load",
    "trafo": "trafo",
    "line": "line",
}


def _table_for_key(key):
    """'trafo' -> 'trafo', 'load2' -> 'load', 'bus3' -> 'bus', 'ext' -> 'ext_grid'."""
    base = re.sub(r"\d+$", "", key)
    if base not in _KEY_TO_TABLE:
        raise KeyError(f"Unbekanntes Bauteil '{key}'. Erlaubt: {list(_KEY_TO_TABLE)} (ggf. mit Nummer).")
    return _KEY_TO_TABLE[base]


def _read_result(net, idx, source):
    """Liest einen Wert anhand von 'tabelle.spalte.bauteil', z. B. 'res_bus.vm_pu.bus3'."""
    try:
        table, column, key = source.split(".")
    except ValueError:
        raise ValueError(f"'{source}' hat nicht das Format 'tabelle.spalte.bauteil', z. B. 'res_bus.vm_pu.bus3'.")
    return getattr(net, table).at[idx[key], column]


def vary_parameter(component, parameter, values, outputs,
                   build_fn=build_base_net, include_baseline=True):
    """
    Variiert einen Parameter eines Bauteils und gibt die Ergebnisse als Tabelle zurück.

    Für jeden Wert wird ein frisches Netz mit build_fn() erzeugt, sodass jeder
    Rechenpunkt vom selben Ausgangsnetz ausgeht. Nicht konvergierte Fälle werden
    markiert und ihre Ergebniswerte bleiben leer (NaN).

    Parameter
    ---------
    component : str
        Schlüssel des Bauteils aus idx, z. B. "trafo", "line", "load", "load2".
    parameter : str
        Spalte in der Elementtabelle, die geändert wird, z. B. "tap_pos".
    values : list
        Neue Werte für den Parameter, z. B. [-2, -1, 0, 1, 2].
    outputs : list of [str, str]
        2D-Liste mit [Quelle, Spaltenname]. Die Quelle hat das Format
        "ergebnistabelle.spalte.bauteil", z. B.
            [["res_bus.vm_pu.bus2",            "vm Bus 2 [pu]"],
             ["res_line.loading_percent.line", "Auslastung Leitung [%]"]]
    build_fn : callable
        Funktion, die (net, idx) zurückgibt. Standard: build_base_net.
    include_baseline : bool
        Wenn True, wird der Ausgangswert des Parameters automatisch mitgerechnet
        und in der Spalte "Baseline" markiert.

    Rückgabe
    --------
    pandas.DataFrame mit einer Zeile pro Wert.
    """
    table = _table_for_key(component)

    # Ausgangswert bestimmen und ggf. zu den Werten hinzufügen
    net0, idx0 = build_fn()
    base_value = getattr(net0, table).at[idx0[component], parameter]
    values = list(values)
    if include_baseline and base_value not in values:
        values.append(base_value)
    values = sorted(values)

    rows = []
    for value in values:
        net, idx = build_fn()                                   # frische Netzkopie
        getattr(net, table).at[idx[component], parameter] = value

        row = {f"{component}.{parameter}": value}
        if include_baseline:
            row["Baseline"] = "x" if value == base_value else ""

        try:
            pp.runpp(net)
            row["konvergiert"] = True
            for source, name in outputs:
                row[name] = _read_result(net, idx, source)
        except pp.LoadflowNotConverged:
            row["konvergiert"] = False
            for _, name in outputs:
                row[name] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)