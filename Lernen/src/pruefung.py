"""
Szenarien für das Netz am Ende von Teil 4 (Wind + PV + Ortsnetz mit PV-Haushalten, Variante „heute“).

    basis()                       Ausgangsnetz (Kopie des Netzes aus Teil 4, eigener Name, nichts überschrieben)
    szenario(name, ...)           Variante rechnen: Profiländerungen, Monate, Netzänderungen, Regelung
    kennzahlen(sims)              Vergleichstabelle (Spannungen mit Ort, Auslastungen, Rückspeisung, Regelung)
    differenz(tab, bezug)         Änderung gegenüber einer Spalte der Kennzahltabelle
    verlauf(sims, ...)            Zeitverlauf: Erzeugung/Last, Austausch, Spannungen, Auslastungen, rONT-Stufe
    profil_vorschau(s, ...)       Leistung einzelner Elemente vor und nach den Profiländerungen
    leistung(s, muster, tabelle)  Summe der Wirkleistung aller Elemente, deren Name auf das Muster passt
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import simulation as sim
import auswertung as aw
import regelung as rg
import lvnetz
import plots
from auswertung import V_MIN, V_MAX, V_PLAN_MIN, V_PLAN_MAX

PREFIX = "w4_p_"
BASIS = PREFIX + "basis"
ONT = {"R": "LV Trafo R0-R1", "I": "LV Trafo I0-I1", "C": "LV Trafo C0-C1"}
MS_TRAFOS = ["Trafo 0-1", "Trafo 0-12"]
PV_ALLE = r"^PV-Freifläche$|^LV PV"          # PV-Freiflächenanlage und alle PV-Anlagen der Haushalte
PV_HAUSHALTE = r"^LV PV"
WIND = r"^Windpark$"
HAUSHALTE = r"^LV Load R"                   # Haushaltslasten im Wohngebiet
ALLE_LASTEN = r"Load"


def basis(variante="heute"):
    """Netz am Ende von Teil 4 (Kombination aus Teil 4a + PV-Haushalte) unter eigenem Namen."""
    name = BASIS if variante == "heute" else f"{BASIS}_{variante.lower()}"
    s, _ = aw.mit_pv_haushalten(sim.loadSimulation(aw.KOMBI), name, variante)
    return s


def szenario(name, basis_sim, aenderungen=None, monate=None, netz=None, regler=True):
    """
    Variante rechnen und zurückgeben.
    aenderungen : Liste von Profiländerungen (profilaenderung.immer/einmal/periodisch)
    monate      : Monat oder Liste von Monaten (Standard: alle 12 typischen Wochen)
    netz        : Funktion f(s), die das Netz ändert (z. B. Schalter schließen, Leitung außer Betrieb)
    regler      : True = rONT-Regelung aus Teil 3b (drei Ortsnetztrafos); False = keine Regelung
    """
    s = basis_sim.copy(PREFIX + name, overwrite=True)
    if monate is not None:
        s.changeSetting("months", monate)
    if aenderungen:
        s.changeSetting("profile_edits", aenderungen)
    if netz is not None:
        netz(s)
    s.runSimulation(controllers=aw.lv_regler(s.net) if regler else None, verbose=False)
    return s


def leistung(s, muster, tabelle="sgen"):
    """Summe der Wirkleistung [MW] je Stunde aller Elemente der Tabelle (sgen, load, storage), deren Name passt."""
    df = getattr(s.results, f"{tabelle}_p_mw")
    spalten = [c for c in df.columns if re.search(muster, str(c))]
    return df[spalten].sum(axis=1)


def _ort(serie, extrem):
    i = serie.idxmin() if extrem == "min" else serie.idxmax()
    return f"{serie[i]:.3f} ({str(i).replace('LV Bus ', '').replace('LV Line ', '').replace('LV Trafo ', '')})"


def _stufen(s):
    try:
        return int(rg.schaltvorgaenge(s).filter(like="Stufen").sum())
    except Exception:
        return 0


def kennzahlen(sims):
    """Vergleichstabelle; Zeilen = Kennzahlen, Spalten = Varianten (dict Name → Simulation)."""
    out = {}
    for k, s in sims.items():
        r = s.results
        ok = r.converged.astype(bool)
        g = lvnetz.bus_groups(s.net)
        vm = r.vm_pu[ok]
        ms, ns = vm[g["MS"]], vm[g["NS"]]
        ll = r.line_loading[ok]
        lv_lines = [c for c in ll.columns if str(c).startswith("LV ")]
        ms_lines = [c for c in ll.columns if c not in lv_lines]
        tl = r.trafo_loading[ok]
        ext = r.ext_grid_p_mw[ok].sum(axis=1)
        p_r = r.trafo_p_mw[ok][ONT["R"]]
        z = {"Stunden gerechnet": len(r.info), "davon nicht konvergiert": int((~ok).sum())}
        if ok.any():
            z |= {"U min MS [pu]": _ort(ms.min(), "min"), "U max MS [pu]": _ort(ms.max(), "max"),
                  "U min NS [pu]": _ort(ns.min(), "min"), "U max NS [pu]": _ort(ns.max(), "max"),
                  "Stunden außerhalb 0,95 … 1,05 (MS)": int(((ms.min(axis=1) < V_PLAN_MIN) | (ms.max(axis=1) > V_PLAN_MAX)).sum()),
                  "Stunden außerhalb 0,95 … 1,05 (NS)": int(((ns.min(axis=1) < V_PLAN_MIN) | (ns.max(axis=1) > V_PLAN_MAX)).sum()),
                  "Stunden außerhalb 0,90 … 1,10 (alle)": int(((vm.min(axis=1) < V_MIN) | (vm.max(axis=1) > V_MAX)).sum()),
                  "Leitung max MS [%]": _ort(ll[ms_lines].max(), "max").replace(".000", ""),
                  "Leitung max NS [%]": _ort(ll[lv_lines].max(), "max"),
                  "Trafo 0-1 max [%]": f"{tl['Trafo 0-1'].max():.1f}", "Trafo 0-12 max [%]": f"{tl['Trafo 0-12'].max():.1f}",
                  "Ortsnetztrafo R max [%]": f"{tl[ONT['R']].max():.1f}",
                  "Stunden mit Überlastung (> 100 %)": int(((ll.max(axis=1) > 100) | (tl.max(axis=1) > 100)).sum()),
                  "Rückspeisung 110 kV [h]": int((ext < -0.001).sum()), "max. Rückspeisung 110 kV [MW]": f"{max(0, -ext.min()):.2f}",
                  "max. Bezug 110 kV [MW]": f"{ext.max():.2f}",
                  "Rückspeisung Wohngebiet [h]": int((p_r < -0.001).sum()),
                  "max. Rückspeisung Wohngebiet [kW]": f"{1000 * max(0, -p_r.min()):.0f}",
                  "rONT-Stufenschritte": _stufen(s),
                  "Verluste [MWh]": f"{(r.line_pl_mw[ok].sum(axis=1) + r.trafo_pl_mw[ok].sum(axis=1)).sum():.2f}"}
        out[k] = z
    return pd.DataFrame(out)


def zahl(tab, zeile, spalte):
    """Zahlenwert aus der Kennzahltabelle (Text wie '0,951 (Bus 11)' → 0.951)."""
    return float(str(tab.at[zeile, spalte]).split()[0])


def differenz(tab, bezug):
    """Änderung jeder numerischen Kennzahl gegenüber der Spalte `bezug` (Orte werden weggelassen)."""
    num = tab.apply(lambda col: col.map(lambda v: float(str(v).split()[0]) if pd.notna(v) else np.nan))
    return num.sub(num[bezug], axis=0).drop(columns=bezug)


def verlauf(sims, titel="", tage=None, monat=None):
    """
    Zeitverlauf mehrerer Varianten: Erzeugung und Last, Austausch mit dem 110-kV-Netz, niedrigste und höchste
    Spannung (MS und NS), Auslastung von Trafo 0-1 und Ortsnetztrafo R, Stufe des rONT im Wohngebiet.
    tage: Liste von Datumsangaben ("2025-07-16") zur Einschränkung; monat: nur diese typische Woche.
    """
    erste = next(iter(sims.values())).results
    idx = erste.info.index
    if monat is not None:
        idx = idx[erste.info["month"] == monat]
    if tage is not None:
        tage = [pd.Timestamp(t).normalize() for t in tage]
        idx = idx[pd.DatetimeIndex(idx).normalize().isin(tage)]
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.2, 1.4, 1.2, 0.8]})
    stile = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1))]
    farben = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (name, s), st, fb in zip(sims.items(), stile * 3, farben * 3):
        r = s.results
        i = idx.intersection(r.info.index)
        ok = r.converged.astype(bool).reindex(i).fillna(False).to_numpy()
        g = lvnetz.bus_groups(s.net)
        def m(x):                                       # nicht konvergierte Stunden als Lücke
            x = x.reindex(i).astype(float).copy()
            x[~ok] = np.nan
            return x
        wind = leistung(s, WIND).reindex(i)
        pv = leistung(s, PV_ALLE).reindex(i)
        last = r.load_p_mw.sum(axis=1).reindex(i)
        axes[0].plot(i, wind, color="tab:blue", ls=st, lw=1.2, label=f"Wind – {name}")
        axes[0].plot(i, pv, color="goldenrod", ls=st, lw=1.2, label=f"PV gesamt – {name}")
        axes[0].plot(i, last, color="black", ls=st, lw=1.0, label=f"Last gesamt – {name}")
        axes[1].plot(i, m(r.ext_grid_p_mw.sum(axis=1)), color=fb, ls=st, label=name)
        axes[2].plot(i, m(r.vm_pu[g["MS"]].min(axis=1)), color=fb, ls=st, lw=1.2, label=f"U min MS – {name}")
        axes[2].plot(i, m(r.vm_pu[g["NS"]].min(axis=1)), color=fb, ls=st, lw=0.8, alpha=0.6, label=f"U min NS – {name}")
        axes[2].plot(i, m(r.vm_pu[g["NS"]].max(axis=1)), color=fb, ls=st, lw=0.8, alpha=0.6)
        axes[3].plot(i, m(r.trafo_loading["Trafo 0-1"]), color=fb, ls=st, lw=1.2, label=f"Trafo 0-1 – {name}")
        axes[3].plot(i, m(r.trafo_loading[ONT["R"]]), color=fb, ls=st, lw=0.8, alpha=0.6, label=f"Ortsnetztrafo R – {name}")
        if hasattr(r, "tap_pos") and ONT["R"] in r.tap_pos:
            axes[4].step(i, r.tap_pos[ONT["R"]].reindex(i), where="post", color=fb, ls=st, label=name)
    axes[0].set_ylabel("Leistung [MW]")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylabel("Austausch 110 kV [MW]\n(< 0 Rückspeisung)")
    for v, c in ((V_MIN, "red"), (V_PLAN_MIN, "orange"), (V_PLAN_MAX, "orange"), (V_MAX, "red")):
        axes[2].axhline(v, color=c, ls="--", lw=0.8)
    axes[2].set_ylabel("Spannung [pu]\n(dünn: NS min/max)")
    axes[3].axhline(100, color="red", ls="--", lw=0.8)
    axes[3].set_ylabel("Auslastung [%]")
    axes[4].set_ylabel("Stufe rONT R\n(+ senkt)")
    axes[4].set_yticks(range(-2, 3))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    if tage is not None:                                # einzelne Tage: Stundenachse statt Wochenachse
        import matplotlib.dates as mdates
        axes[4].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
        axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.\n%H:%M"))
        axes[4].set_xlim(idx[0], idx[-1] + pd.Timedelta(hours=1))
    else:
        plots.week_axis(axes[4], idx)
    fig.suptitle(titel, y=1.0)
    plt.tight_layout()
    plt.show()


def profil_vorschau(s, muster, aenderungen, monat="JULY", tabelle="sgen", titel=None):
    """
    Leistung der Elemente (Summe über alle, deren Name auf `muster` passt) vor und nach den Profiländerungen,
    für die typische Woche `monat`. Es wird kein Lastfluss gerechnet.
    """
    import timeseries as tsm
    from profiles import Month
    import profilaenderung as pa
    pt = tsm.build_profile_table([Month[monat]])
    _, p, _ = s._element_matrices(tabelle, pt)
    namen = s.net[tabelle]["name"].to_numpy()
    spalten = [j for j, n in enumerate(namen) if re.search(muster, str(n))]
    vorher = p[:, spalten].sum(axis=1)
    nachher = pa.anwenden(p, namen, pt, aenderungen)[:, spalten].sum(axis=1)
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(pt.index, vorher, color="gray", lw=1.5, label="Originalprofil")
    ax.plot(pt.index, nachher, color="tab:red", lw=1.2, ls="--", label="bearbeitete Kopie")
    ax.set_ylabel("Leistung [MW]")
    ax.set_title(titel or f"Profiländerung für „{muster}“ ({len(spalten)} Elemente)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    plots.week_axis(ax, pt.index)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------------------------------
# Spannung entlang eines Strangs (vom 110-kV-Netz bis zum Strangende)
# ---------------------------------------------------------------------------------------------------------------

STRAENGE = {"Abgang 1": "Bus 11", "Abgang 2": "Bus 14", "Wohngebiet": "LV Bus R15", "Wohngebiet (R18)": "LV Bus R18",
            "Gewerbe": "LV Bus C12", "Industrie": "LV Bus I2"}
TYP_FARBE = {"HS/MS-Trafo": "#d9d9d9", "MS-Leitung": "#c6dbef", "Ortsnetztrafo": "#fdd0a2", "NS-Leitung": "#c7e9c0"}


def _kurz(name):
    return str(name).replace("LV Bus ", "")


def kette(s, ende, start="Bus 0"):
    """
    Busse auf dem Weg von `start` (110 kV) nach `ende` im aktuellen Schaltzustand der Simulation, mit dem Element
    zum Vorgänger, seinem Typ (HS/MS-Trafo, MS-Leitung, Ortsnetztrafo, NS-Leitung) und der Länge.
    """
    import networkx as nx
    import pandapower.topology as top
    net = getattr(s, "net", s)                                   # Simulation oder pandapower-Netz
    g = top.create_nxgraph(net, include_trafos=True, respect_switches=True)
    idx = dict(zip(net.bus["name"], net.bus.index))
    pfad = nx.shortest_path(g, idx[start], idx[ende])
    zeilen = [{"Bus": net.bus.at[pfad[0], "name"], "Element": "", "Typ": "", "Länge [m]": 0.0}]
    for a, b in zip(pfad[:-1], pfad[1:]):
        kanten = g.get_edge_data(a, b)
        (tab, i), d = next(iter(kanten.items()))
        niedrig = min(net.bus.at[a, "vn_kv"], net.bus.at[b, "vn_kv"])
        if tab.startswith("trafo"):
            typ = "HS/MS-Trafo" if niedrig > 1 else "Ortsnetztrafo"
            name, laenge = net.trafo.at[i, "name"], 0.0
        elif tab == "switch":                                    # Bus-Bus-Schalter: keine Impedanz
            typ = "MS-Leitung" if niedrig > 1 else "NS-Leitung"
            name, laenge = f"Schalter {net.switch.at[i, 'name']}", 0.0
        else:
            typ = "MS-Leitung" if niedrig > 1 else "NS-Leitung"
            name, laenge = net.line.at[i, "name"], 1000 * float(net.line.at[i, "length_km"])
        zeilen.append({"Bus": net.bus.at[b, "name"], "Element": name, "Typ": typ, "Länge [m]": laenge})
    return pd.DataFrame(zeilen)


def fall_reihe(s, ende):
    """Spannungsfall [pu] je Stunde auf den Leitungen des Strangs: von der Sammelschiene hinter dem letzten
    Trafo bis zum Strangende (> 0 fällt, < 0 steigt). Die Trafostufe ist darin nicht enthalten."""
    k = kette(s, ende)
    trafos = k.index[k["Typ"].str.contains("trafo", case=False)]
    start = k.at[trafos[-1], "Bus"] if len(trafos) else k.at[0, "Bus"]
    v = s.results.vm_pu[s.results.converged.astype(bool)]
    return v[start] - v[ende]


def zeitpunkt(s, ende, wahl="min"):
    """
    Stunde für das Profil: 'min'/'max' = niedrigste/höchste Spannung am Strangende; 'fall'/'steig' = größter
    Spannungsfall bzw. größte Spannungsanhebung auf den Leitungen des Strangs; sonst ein Zeitstempel.
    """
    if wahl in ("fall", "steig"):
        f = fall_reihe(s, ende)
        return f.idxmax() if wahl == "fall" else f.idxmin()
    if wahl in ("min", "max"):
        v = s.results.vm_pu[ende][s.results.converged.astype(bool)]
        return v.idxmin() if wahl == "min" else v.idxmax()
    return pd.Timestamp(wahl)


def profil_werte(s, ende, wahl="min"):
    """Spannung je Bus entlang des Strangs zur gewählten Stunde und Spannungsfall je Abschnitt (Prozentpunkte)."""
    k = kette(s, ende)
    t = zeitpunkt(s, ende, wahl)
    k["U [pu]"] = s.results.vm_pu.loc[t, k["Bus"]].to_numpy()
    k["ΔU zum Vorgänger [%-Pkt.]"] = (-k["U [pu]"].diff() * 100).fillna(0.0)    # > 0: Spannung fällt
    return k, t


def gemeinsame_stunde(sims, ende, wahl):
    """
    'fall'/'steig'/'min'/'max': dieselbe Stunde für alle Varianten, nämlich die Stunde, in der irgendeine Variante die niedrigste
    (höchste) Spannung am Strangende hat; so werden alle Varianten zum selben Zeitpunkt verglichen.
    'min_je'/'max_je': für jede Variante ihre eigene kritische Stunde. Sonst unverändert (Zeitstempel oder dict).
    """
    if isinstance(wahl, str) and wahl in ("min", "max", "fall", "steig"):
        kand = {n: zeitpunkt(s, ende, wahl) for n, s in sims.items()}
        if wahl in ("min", "max"):
            werte = {n: sims[n].results.vm_pu.at[t, ende] for n, t in kand.items()}
        else:
            werte = {n: fall_reihe(sims[n], ende).at[t] for n, t in kand.items()}
        bester = min(werte, key=werte.get) if wahl in ("min", "steig") else max(werte, key=werte.get)
        return kand[bester]
    if isinstance(wahl, str) and wahl.endswith("_je"):
        return {n: wahl[:-3] for n in sims}
    return wahl


def profil(sims, strang="Wohngebiet", wahl="fall", titel=None):
    """
    Spannungsprofil vom 110-kV-Netz bis zum Strangende für mehrere Varianten (dict Name → Simulation).
    wahl: gemeinsame Stunde für alle Varianten, bestimmt von der Variante mit dem extremsten Wert:
          'fall' = größter Spannungsfall auf den Leitungen des Strangs (Standard), 'steig' = größte Anhebung
          (Rückspeisung), 'min'/'max' = niedrigste/höchste Spannung am Strangende; mit Endung '_je' (z. B.
          'fall_je') je Variante die eigene Stunde; außerdem ein Zeitstempel oder ein dict Name → Wahl.
    Oben: Spannung an jedem Bus, Hintergrund nach Abschnitt (HS/MS-Trafo, MS-Leitung, Ortsnetztrafo, NS-Leitung).
    Unten: Spannungsfall je Abschnittstyp (> 0 Spannung fällt, < 0 Spannung steigt; beim Ortsnetztrafo enthält er
    die Stufe des rONT). Rückgabe: Tabelle der Spannungen und Spannungsfälle.
    """
    ende = STRAENGE.get(strang, strang)
    wahl = gemeinsame_stunde(sims, ende, wahl)
    werte = {}
    for name, s in sims.items():
        w = wahl.get(name, "min") if isinstance(wahl, dict) else wahl
        werte[name] = profil_werte(s, ende, w)
    gleich = len({tuple(k["Bus"]) for k, _ in werte.values()}) == 1
    zeilen = 1 if gleich else len(werte)
    fig = plt.figure(figsize=(15, 4.6 * zeilen + 3.6))
    gs = fig.add_gridspec(zeilen + 1, 1, height_ratios=[4.6] * zeilen + [3.0])
    stile = ["o-", "s--", "^:", "d-.", "v-", "P--"]
    farben = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    achsen = []
    for z in range(zeilen):
        ax = fig.add_subplot(gs[z])
        achsen.append(ax)
        eintraege = list(werte.items()) if gleich else [list(werte.items())[z]]
        k0 = eintraege[0][1][0]
        x = np.arange(len(k0))
        for j in range(1, len(k0)):                                  # Abschnitte als Hintergrund
            ax.axvspan(j - 1, j, color=TYP_FARBE[k0.at[j, "Typ"]], alpha=0.6, lw=0)
        for (name, (k, t)), st, fb in zip(eintraege, stile * 3, farben * 3):
            i = list(werte).index(name)
            ax.plot(x, k["U [pu]"], stile[i % 6], color=farben[i % 10], lw=1.6, ms=5,
                    label=f"{name} – {t:%d.%m. %H:%M}")
        for v, c in ((V_MIN, "red"), (V_PLAN_MIN, "orange"), (V_PLAN_MAX, "orange"), (V_MAX, "red")):
            ax.axhline(v, color=c, ls="--", lw=0.9)
        ax.set_xticks(x, [_kurz(b) for b in k0["Bus"]], fontsize=8, rotation=0)
        oben = ax.secondary_xaxis("top")
        oben.set_xticks(x[1:] - 0.5, ["Trafo" if "trafo" in t.lower() else f"{l:.0f} m"
                                      for l, t in zip(k0["Länge [m]"][1:], k0["Typ"][1:])], fontsize=7)
        ax.set_xlim(-0.3, len(k0) - 0.7)
        ax.set_ylabel("Spannung [pu]")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    from matplotlib.patches import Patch
    achsen[-1].legend(handles=achsen[-1].get_legend_handles_labels()[0] +
                      [Patch(color=c, alpha=0.6, label=t) for t, c in TYP_FARBE.items()],
                      fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    achsen[0].set_title(titel or f"Spannung entlang des Strangs: {strang} (oben: Länge des Abschnitts)", pad=24)
    # Spannungsfall je Abschnittstyp
    axb = fig.add_subplot(gs[zeilen])
    typen = list(TYP_FARBE)
    abschnitt = pd.DataFrame({name: k.groupby("Typ")["ΔU zum Vorgänger [%-Pkt.]"].sum().reindex(typen).fillna(0.0)
                              for name, (k, _) in werte.items()})
    breite = 0.8 / len(werte)
    for i, name in enumerate(werte):
        pos = np.arange(len(typen)) + (i - (len(werte) - 1) / 2) * breite
        balken = axb.bar(pos, abschnitt[name], breite, color=farben[i % 10], label=name)
        axb.bar_label(balken, fmt="%.2f", fontsize=7, padding=2)
    axb.axhline(0, color="black", lw=0.8)
    axb.set_xticks(range(len(typen)), typen)
    axb.set_ylabel("Spannungsfall [%-Punkte]\n(> 0 fällt, < 0 steigt)")
    axb.grid(alpha=0.3, axis="y")
    axb.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.show()
    tab = pd.concat({name: k.set_index(k["Bus"].map(_kurz))[["Element", "U [pu]", "ΔU zum Vorgänger [%-Pkt.]"]]
                     for name, (k, _) in werte.items()}, axis=1)
    return tab


def abschnitte(sims, strang="Wohngebiet", wahl="fall"):
    """Kurztabelle: Stunde, Spannung an Sammelschiene und Strangende, Spannungsfall je Abschnittstyp."""
    ende = STRAENGE.get(strang, strang)
    wahl = gemeinsame_stunde(sims, ende, wahl)
    out = {}
    for name, s in sims.items():
        w = wahl.get(name, "min") if isinstance(wahl, dict) else wahl
        k, t = profil_werte(s, ende, w)
        z = {"Stunde": f"{t:%d.%m. %H:%M}", "U 110 kV [pu]": k["U [pu]"].iloc[0], "U Strangende [pu]": k["U [pu]"].iloc[-1]}
        for typ in TYP_FARBE:
            if (k["Typ"] == typ).any():
                z[f"ΔU {typ} [%-Pkt.]"] = k.loc[k["Typ"] == typ, "ΔU zum Vorgänger [%-Pkt.]"].sum()
        z["ΔU gesamt [%-Pkt.]"] = (k["U [pu]"].iloc[0] - k["U [pu]"].iloc[-1]) * 100
        out[name] = z
    return pd.DataFrame(out).T


def spannung_zeit_ort(s, strang="Wohngebiet", monat=None, tage=None, titel=None, vmin=0.92, vmax=1.08):
    """
    Spannung aller Busse eines Strangs über die Zeit als Farbbild (x: Zeit, y: Bus vom 110-kV-Netz bis zum
    Strangende). Zeigt auf einen Blick, wann und wo die Spannung am tiefsten bzw. höchsten ist.
    """
    ende = STRAENGE.get(strang, strang)
    k = kette(s, ende)
    r = s.results
    idx = r.info.index
    if monat is not None:
        idx = idx[r.info["month"] == monat]
    if tage is not None:
        idx = idx[pd.DatetimeIndex(idx).normalize().isin([pd.Timestamp(t) for t in tage])]
    v = r.vm_pu.loc[idx, k["Bus"]].T
    fig, ax = plt.subplots(figsize=(15, 0.32 * len(k) + 2))
    im = ax.pcolormesh(np.arange(len(idx) + 1), np.arange(len(k) + 1), v.to_numpy(), cmap="RdYlBu_r",
                       vmin=vmin, vmax=vmax, shading="flat")
    ax.set_yticks(np.arange(len(k)) + 0.5, [_kurz(b) for b in k["Bus"]], fontsize=8)
    ax.invert_yaxis()
    tick = np.arange(0, len(idx), 6 if len(idx) <= 48 else 24)
    ax.set_xticks(tick, [f"{aw.TAGE[pd.Timestamp(idx[i]).weekday()]} " + pd.Timestamp(idx[i]).strftime("%d.%m.\n%H:%M")
                         for i in tick], fontsize=7)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Spannung [pu]")
    for v_ in (V_PLAN_MIN, V_PLAN_MAX):
        cb.ax.axhline(v_, color="black", lw=1)
    ax.set_title(titel or f"Spannung entlang des Strangs {strang} über die Zeit ({s.name})")
    plt.tight_layout()
    plt.show()
