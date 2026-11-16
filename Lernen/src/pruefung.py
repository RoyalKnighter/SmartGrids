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
