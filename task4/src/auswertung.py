"""
Gemeinsame Konstanten und Auswertungsfunktionen für Task 4.

Spannungsband (Teil 3):
    V_MIN, V_MAX            Arbeitsgrenzen 0,90 … 1,10 pu (DIN EN 50160)
    V_PLAN_MIN, V_PLAN_MAX  planerisches Spannungsband 0,95 … 1,05 pu
    PLAN_MARKER             Markierungen des Planungsbands für Spannungsdiagramme

Netz:
    baseline_net()          unverändertes CIGRE-MV-Netz ohne dezentrale Erzeuger (Referenz, Teil 1)
    MS_BUSSE                die 20-kV-Busse Bus 1 … Bus 14 (ohne Bus 0 = 110 kV)

Auswertung:
    band_stunden(s)         Stunden außerhalb von Planungsband und Arbeitsgrenzen, getrennt nach MS und NS
    lv_regler(net)          die drei rONT-Regler aus Teil 3b (ohne Backup-Plan)
    ausfall_status(row)     Kurztext für eine Zeile von regelung.ausfall_analyse
    strangpfad(net, a, b)   Busse und Entfernung entlang eines Strangs
    spannungsprofil(...)    Spannungen entlang eines Strangs für ausgewählte Zeitpunkte
"""

import pandas as pd
import matplotlib.pyplot as plt
import pandapower.networks as pn

V_MIN, V_MAX = 0.90, 1.10
V_PLAN_MIN, V_PLAN_MAX = 0.95, 1.05
PLAN_MARKER = [("orange", V_PLAN_MIN, "Planungsband 0,95 pu", "--"),
               ("orange", V_PLAN_MAX, "Planungsband 1,05 pu", "--")]
MS_BUSSE = [f"Bus {i}" for i in range(1, 15)]
TAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def baseline_net():
    """Unverändertes CIGRE-MV-Netz ohne dezentrale Erzeuger (Referenz für alle Vergleiche)."""
    return pn.create_cigre_network_mv(with_der=False)


def band_stunden(s):
    """Stunden außerhalb des Planungsbands und der Arbeitsgrenzen sowie U min/max, getrennt nach MS und NS."""
    import lvnetz
    g = lvnetz.bus_groups(s.net)
    vm = s.results.vm_pu[s.results.converged.astype(bool)]
    out = {}
    for lvl, cols in (("MS", g["MS"]), ("NS", g["NS"])):
        if not cols:
            continue
        lo, hi = vm[cols].min(axis=1), vm[cols].max(axis=1)
        out[f"Stunden < {V_PLAN_MIN} {lvl}"] = int((lo < V_PLAN_MIN).sum())
        out[f"Stunden > {V_PLAN_MAX} {lvl}"] = int((hi > V_PLAN_MAX).sum())
        out[f"Stunden < {V_MIN} {lvl}"] = int((lo < V_MIN).sum())
        out[f"Stunden > {V_MAX} {lvl}"] = int((hi > V_MAX).sum())
        out[f"U min {lvl} [pu]"] = lo.min()
        out[f"U max {lvl} [pu]"] = hi.max()
    return pd.Series(out)


def lv_regler(net, u_min=V_PLAN_MIN, u_max=V_PLAN_MAX):
    """Die drei rONT-Regler aus Teil 3b (je Teilnetz R, I, C), ohne Backup-Plan."""
    import regelung as rg
    return [r for r in rg.standard_regler(net, u_min, u_max) if isinstance(r, rg.TapRegler)]


def ausfall_status(row):
    """Kurztext für eine Zeile der (n-1)-Prüfung (regelung.ausfall_analyse)."""
    if row.get("Bemerkung"):
        return "✘ " + row["Bemerkung"]
    k = int(row["nicht versorgte Busse"])
    if k > 0:
        return f"{k} Bus unversorgt" if k == 1 else f"{k} Busse unversorgt"
    return "✔" if row["zulässig"] else "✘ Grenze verletzt"


def strangpfad(net, start, ende):
    """Busse auf dem Weg von start nach ende und die Entfernung vom Start in km (über die Leitungslängen)."""
    import networkx as nx
    import pandapower.topology as top
    graph = top.create_nxgraph(net, include_trafos=False)
    idx = dict(zip(net.bus["name"], net.bus.index))
    pfad = nx.shortest_path(graph, idx[start], idx[ende])
    km = [0.0]
    for a, b in zip(pfad[:-1], pfad[1:]):
        km.append(km[-1] + min(d["weight"] for d in graph.get_edge_data(a, b).values()))
    return [net.bus.at[b, "name"] for b in pfad], km


def spannungsprofil(sims, start, ende, zeitpunkte, titel=None):
    """
    Spannung entlang des Strangs start → ende für mehrere Simulationen (dict Name → Simulation) zu den
    angegebenen Zeitpunkten (dict Titel → Zeitstempel); ein Diagramm je Zeitpunkt.
    """
    erste = next(iter(sims.values()))
    busse, km = strangpfad(erste.net, start, ende)
    fig, axes = plt.subplots(1, len(zeitpunkte), figsize=(7 * len(zeitpunkte), 4.5), sharey=True, squeeze=False)
    for ax, (t_titel, t) in zip(axes[0], zeitpunkte.items()):
        for (name, s), stil in zip(sims.items(), ("o-", "s--", "^:", "d-.")):
            ax.plot(km, s.results.vm_pu.loc[t, busse], stil, label=name)
        for v, farbe in ((V_MIN, "red"), (V_PLAN_MIN, "orange"), (V_PLAN_MAX, "orange"), (V_MAX, "red")):
            ax.axhline(v, color=farbe, ls="--", lw=1)
        ax.set_xticks(km, [b.replace("LV Bus ", "") for b in busse])
        oben = ax.secondary_xaxis("top")
        oben.set_xticks(km, [f"{x * 1000:.0f} m" for x in km], fontsize=7)
        ax.set_title(f"{t_titel}\n{TAGE[t.weekday()]} {t:%d.%m. %H:%M} Uhr", fontsize=10)
        ax.set_xlabel("Bus auf dem Strang (oben: Entfernung von der Sammelschiene)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Spannung [pu]")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.suptitle(titel or f"Spannungsfall entlang des Strangs ({start} → {ende})")
    plt.tight_layout()
    plt.show()


# Netz am Ende von Teil 4 (gewählte Kombination aus Teil 4a + Haushalte mit PV und Speicher, Variante „heute“)
PV_HAUSHALTE = {"heute": (0.083, 0.067), "Zukunft": (0.10, 0.40)}   # (Anteil nur PV, Anteil PV + Speicher)
KOMBI = "w4_kombi_W1_P12"                                             # Teil 4a: Wind an Bus 1, PV an Bus 12


def mit_pv_haushalten(basis, name, variante="heute"):
    """Kopie der Simulation `basis` mit Haushalten mit PV und Speicher im Ortsnetz (lvnetz.prosumer)."""
    import lvnetz
    s = basis.copy(name, overwrite=True)
    tab = lvnetz.prosumer(s, *PV_HAUSHALTE[variante])
    s.save(results=False)
    return s, tab


def netz_ende_teil4(name="w4_t5_basis"):
    """Netz am Ende von Teil 4: Kombination aus Teil 4a mit PV-Haushalten (heute). Rückgabe: (Simulation, Tabelle)."""
    import simulation as sim
    return mit_pv_haushalten(sim.loadSimulation(KOMBI), name, "heute")
