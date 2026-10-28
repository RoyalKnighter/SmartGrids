"""
Schalterstellungen S1–S3 und Auswertungen für Task 4.

Die Trennschalter S1 (Line 14-8), S2 (Line 6-7) und S3 (Line 11-4) sind im CIGRE-MV-Netz im
Normalbetrieb offen. Der Arbeitsauftrag verlangt alle Kombinationen, also 2³ = 8 Schaltzustände.

Schlüssel eines Schaltzustands: drei Ziffern in der Reihenfolge S1, S2, S3, 1 = geschlossen.
    "000" = alle offen (Normalbetrieb, Baseline), "100" = nur S1 geschlossen, "111" = alle geschlossen

Statische Lastflüsse:
    res = statisch_alle(lambda: pn.create_cigre_network_mv(with_der=False))
    vergleich(res, "vm_pu")                      # Busse × Schaltzustände

Zeitreihen mit der Simulationssoftware:
    sims = zeitreihe_alle(basis_sim, prefix="w4_baseline")
    kennzahlen(sims)
"""

import copy
import itertools

import numpy as np
import pandas as pd
import pandapower as pp
import matplotlib.pyplot as plt

SWITCHES = ["S1", "S2", "S3"]
SWITCH_LINE = {"S1": "Line 14-8", "S2": "Line 6-7", "S3": "Line 11-4"}
KEYS = ["".join(bits) for bits in itertools.product("01", repeat=3)]      # 000, 001, …, 111


def zustand(key):
    """{"S1": bool, "S2": bool, "S3": bool} aus einem Schlüssel wie "101"."""
    if len(key) != 3 or set(key) - {"0", "1"}:
        raise ValueError(f"Schlüssel '{key}' ungültig, erwartet z. B. '000' oder '101' (S1 S2 S3, 1 = geschlossen).")
    return {s: c == "1" for s, c in zip(SWITCHES, key)}


def label(key, kurz=False):
    """Lesbare Bezeichnung, z. B. 'S1 zu · S2 offen · S3 offen' bzw. kurz 'S1' / 'alle offen'."""
    z = zustand(key)
    if kurz:
        zu = [s for s in SWITCHES if z[s]]
        return "alle offen" if not zu else "+".join(zu) + " zu"
    return " · ".join(f"{s} {'zu' if z[s] else 'offen'}" for s in SWITCHES)


def schalten(net, key):
    """Schaltzustand in einem pandapower-Netz setzen (das Netz wird verändert)."""
    for s, closed in zustand(key).items():
        idx = net.switch.index[net.switch["name"] == s]
        if not len(idx):
            raise KeyError(f"Schalter '{s}' nicht im Netz.")
        net.switch.loc[idx, "closed"] = closed
    return net


# ---------------------------------------------------------------------------
# Statische Lastflüsse
# ---------------------------------------------------------------------------

def lastfluss(net):
    """
    AC-Lastfluss (Newton-Raphson) mit Konvergenzprüfung.
    Rückgabe: dict mit converged, iterations und Ergebnistabellen (Index = Elementname) oder,
    falls nicht konvergiert, nur converged=False und die Fehlermeldung.
    """
    try:
        pp.runpp(net, algorithm="nr")
    except pp.LoadflowNotConverged as e:
        return {"converged": False, "fehler": str(e), "net": net}
    if not net.converged:
        return {"converged": False, "fehler": "net.converged = False", "net": net}
    bn = net.bus["name"]
    bus = pd.DataFrame({"U_n [kV]": net.bus.vn_kv.to_numpy(),
                        "vm_pu": net.res_bus.vm_pu.to_numpy(),
                        "U [kV]": (net.res_bus.vm_pu * net.bus.vn_kv).to_numpy(),
                        "va_degree": net.res_bus.va_degree.to_numpy(),
                        "P_einspeisung [MW]": -net.res_bus.p_mw.to_numpy()}, index=bn.to_numpy())
    line = pd.DataFrame({"von": bn.loc[net.line.from_bus].to_numpy(), "nach": bn.loc[net.line.to_bus].to_numpy(),
                         "p_from_mw": net.res_line.p_from_mw.to_numpy(), "q_from_mvar": net.res_line.q_from_mvar.to_numpy(),
                         "p_to_mw": net.res_line.p_to_mw.to_numpy(),
                         "i_ka": net.res_line.i_ka.to_numpy(), "max_i_ka": net.line.max_i_ka.to_numpy(),
                         "loading_percent": net.res_line.loading_percent.to_numpy(),
                         "pl_mw": net.res_line.pl_mw.to_numpy()}, index=net.line["name"].to_numpy())
    trafo = pd.DataFrame({"p_hv_mw": net.res_trafo.p_hv_mw.to_numpy(), "q_hv_mvar": net.res_trafo.q_hv_mvar.to_numpy(),
                          "p_lv_mw": net.res_trafo.p_lv_mw.to_numpy(),
                          "loading_percent": net.res_trafo.loading_percent.to_numpy(),
                          "pl_mw": net.res_trafo.pl_mw.to_numpy()}, index=net.trafo["name"].to_numpy())
    p_load = net.res_load.p_mw.sum() if len(net.load) else 0.0
    p_sgen = net.res_sgen.p_mw.sum() if "res_sgen" in net and len(net.sgen) else 0.0
    bilanz = pd.Series({"Last P [MW]": p_load, "Erzeugung P [MW]": p_sgen,
                        "Bezug ext. Netz P [MW]": net.res_ext_grid.p_mw.sum(),
                        "Bezug ext. Netz Q [Mvar]": net.res_ext_grid.q_mvar.sum(),
                        "Verluste Leitungen [MW]": net.res_line.pl_mw.sum(),
                        "Verluste Trafos [MW]": net.res_trafo.pl_mw.sum()})
    bilanz["Verluste gesamt [MW]"] = bilanz["Verluste Leitungen [MW]"] + bilanz["Verluste Trafos [MW]"]
    bilanz["Verluste [% der Last]"] = 100 * bilanz["Verluste gesamt [MW]"] / p_load if p_load else np.nan
    return {"converged": True, "iterations": int(net._ppc["iterations"]), "bus": bus, "line": line,
            "trafo": trafo, "bilanz": bilanz, "net": net}


def statisch_alle(netz_fabrik, keys=KEYS):
    """
    Statischer Lastfluss für alle Schaltzustände. netz_fabrik() muss jedes Mal ein neues,
    unverändertes Netz liefern (gleiche Randbedingungen für jeden Vergleich).
    Rückgabe: {key: Ergebnis von lastfluss()}
    """
    out = {}
    for key in keys:
        net = schalten(netz_fabrik(), key)
        out[key] = lastfluss(net)
    return out


def konvergenz(ergebnisse):
    """Übersicht: Schaltzustand, konvergiert, Iterationen."""
    rows = []
    for key, r in ergebnisse.items():
        rows.append({"Schlüssel": key, "Schaltzustand": label(key), "konvergiert": "ja" if r["converged"] else "nein",
                     "Iterationen": r.get("iterations", np.nan), "Hinweis": r.get("fehler", "")})
    return pd.DataFrame(rows).set_index("Schlüssel")


_QUELLE = {"vm_pu": "bus", "va_degree": "bus", "U [kV]": "bus", "P_einspeisung [MW]": "bus",
           "p_from_mw": "line", "q_from_mvar": "line", "i_ka": "line", "loading_percent": "line", "pl_mw": "line",
           "p_hv_mw": "trafo", "trafo_loading": "trafo", "trafo_pl_mw": "trafo"}


def vergleich(ergebnisse, groesse, kurz=True):
    """
    Eine Ergebnisgröße aller Schaltzustände nebeneinander (Zeilen = Elemente, Spalten = Zustände).
    groesse: "vm_pu", "va_degree", "p_from_mw", "i_ka", "loading_percent" (Leitungen), "pl_mw",
             "p_hv_mw", "trafo_loading", "trafo_pl_mw"
    Nicht konvergierte Zustände erscheinen als NaN.
    """
    tab = _QUELLE[groesse]
    col = {"trafo_loading": "loading_percent", "trafo_pl_mw": "pl_mw"}.get(groesse, groesse)
    cols = {}
    for key, r in ergebnisse.items():
        name = label(key, kurz=True) if kurz else key
        cols[name] = r[tab][col] if r["converged"] else np.nan
    df = pd.DataFrame(cols)
    return df


def bilanzen(ergebnisse):
    """Leistungsbilanz und Kennzahlen je Schaltzustand."""
    rows = {}
    for key, r in ergebnisse.items():
        if not r["converged"]:
            rows[label(key, kurz=True)] = pd.Series(dtype=float)
            continue
        b = r["bilanz"].copy()
        bus20 = r["bus"][r["bus"]["U_n [kV]"] < 100]
        b["U min [pu]"] = bus20.vm_pu.min()
        b["Bus U min"] = bus20.vm_pu.idxmin()
        b["Leitung max [%]"] = r["line"].loading_percent.max()
        b["Leitung mit max"] = r["line"].loading_percent.idxmax()
        b["Trafo max [%]"] = r["trafo"].loading_percent.max()
        rows[label(key, kurz=True)] = b
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Zeitreihen
# ---------------------------------------------------------------------------

def zeitreihe_alle(basis, prefix, keys=KEYS):
    """
    Für jeden Schaltzustand eine Kopie der Simulation `basis` anlegen, die Schalter setzen und
    rechnen. Alle Kopien starten vom selben Stand (Baseline = Stand von `basis`).
    Mit sim.ONLY_RUN_IF_CHANGED = True wird nur neu gerechnet, wenn sich etwas geändert hat.
    Rückgabe: {key: Simulation}
    """
    sims = {}
    for key in keys:
        s = basis.copy(f"{prefix}_{key}", overwrite=True)
        for sw, closed in zustand(key).items():
            if bool(s.net.switch.loc[s.net.switch["name"] == sw, "closed"].iloc[0]) != closed:
                s.changeParameter(sw, "closed", closed)
        s.runSimulation()
        sims[key] = s
    return sims


def kennzahlen(sims, v_min=0.9, v_max=1.1):
    """Kennzahlen aller Zeitreihen nebeneinander (Zeilen = Schaltzustände)."""
    rows = {}
    for key, s in sims.items():
        r = s.results
        conv = r.converged
        vm = r.vm_pu[conv]
        slack = set(s.net.bus.loc[s.net.ext_grid.bus, "name"])
        vm20 = vm[[c for c in vm.columns if c not in slack]]
        ll, tl = r.line_loading[conv], r.trafo_loading[conv]
        loss = (r.line_pl_mw.sum(axis=1) + r.trafo_pl_mw.sum(axis=1))[conv] if len(r.line_pl_mw.columns) else None
        st_min = vm20.stack()
        (t_min, b_min) = st_min.idxmin()
        st_l = ll.stack()
        (t_l, n_l) = st_l.idxmax()
        rows[label(key, kurz=True)] = {
            "konvergiert": f"{int(conv.sum())}/{len(conv)}",
            "U min [pu]": st_min.min(), "Bus U min": b_min, "Zeitpunkt U min": f"{pd.Timestamp(t_min):%a %d.%m. %H:%M}",
            "U max 20 kV [pu]": vm20.max().max(),
            f"Stunden U < {v_min}": int((vm.min(axis=1) < v_min).sum()),
            f"Stunden U > {v_max}": int((vm.max(axis=1) > v_max).sum()),
            "Leitung max [%]": st_l.max(), "Leitung": n_l, "Zeitpunkt Leitung max": f"{pd.Timestamp(t_l):%a %d.%m. %H:%M}",
            "Trafo max [%]": tl.max().max(), "Stunden Trafo > 100 %": int((tl.max(axis=1) > 100).sum()),
            "Verluste [MWh]": loss.sum() if loss is not None else np.nan,
            "Verluste [% der Last]": 100 * loss.sum() / r.load_p_mw[conv].sum().sum() if loss is not None else np.nan,
        }
    return pd.DataFrame(rows).T


def je_element(sims, tabelle, stat="min"):
    """Eine Ergebnistabelle aller Zeitreihen zusammengefasst je Element (Zeilen) und Schaltzustand (Spalten)."""
    out = {}
    for key, s in sims.items():
        df = getattr(s.results, tabelle)[s.results.converged]
        out[label(key, kurz=True)] = df.agg(stat)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Diagramme
# ---------------------------------------------------------------------------

def heatmap(df, title, cbar_label, cmap="viridis", vmin=None, vmax=None, fmt="{:.3f}", grenze=None, ax=None):
    """
    Tabelle als Heatmap mit Zahlen in den Feldern (Zeilen = Elemente, Spalten = Schaltzustände).
    grenze: Werte darüber werden fett und mit schwarzem Rahmen markiert (z. B. 100 % Auslastung).
    """
    data = df.astype(float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.25 * len(data.columns) + 3, 0.38 * len(data) + 1.6))
    else:
        fig = ax.figure
    im = ax.imshow(data.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=9)
    norm = im.norm
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iat[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8)
                continue
            rgba = im.cmap(norm(v))
            light = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.55
            over = grenze is not None and v > grenze
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="black" if light else "white", fontweight="bold" if over else "normal")
            if over:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig
