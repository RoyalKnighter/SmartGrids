"""
Live-Lastfluss für das Netz am Ende von Teil 4 (statische Werte einer Stunde oder Bemessungswerte).

    app()                  interaktive Oberfläche (ipywidgets) im Notebook
    LiveNetz(net)          Netz mit Lastfluss, rONT-Regelung und Auswertung (auch ohne Oberfläche nutzbar)
    startpunkte()          Startzustände: Bemessungswerte und typische Stunden aus den Referenzläufen

Jede Last, jeder Erzeuger und jeder Speicher bekommt einen Regler für Wirk- und Blindleistung; an jedem Bus
können eigene Lasten oder Erzeuger ergänzt werden. Nach jeder Änderung wird ein AC-Lastfluss (pp.runpp)
gerechnet; gezeigt werden das Spannungsprofil eines Strangs (aktuell und Ausgangszustand), die Spannung an
allen Bussen und die Auslastung der Betriebsmittel.

Vorzeichen (Verbraucherzählpfeilsystem wie in pandapower): Last p > 0 = Bezug, q > 0 = induktiv;
Erzeuger (sgen) p > 0 = Einspeisung, q > 0 = Blindleistungsabgabe (übererregt); Speicher p > 0 = Laden.
"""

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp

import pruefung as pr
from auswertung import V_MIN, V_MAX, V_PLAN_MIN, V_PLAN_MAX

ONT = {"R": "LV Trafo R0-R1", "I": "LV Trafo I0-I1", "C": "LV Trafo C0-C1"}
SCHALTER = ["S1", "S2", "S3"]


# ---------------------------------------------------------------------------------------------- Startzustände

def _netz_stunde(s, zeit):
    """Netz mit den Werten einer Stunde einer gerechneten Simulation (Lasten, Erzeuger, Speicher, Stufen)."""
    r = s.results
    net = copy.deepcopy(s.net)
    for table in ("load", "sgen", "storage"):
        p = getattr(r, f"{table}_p_mw", None)
        if p is None or not len(net[table]) or not len(p.columns):
            continue
        net[table]["p_mw"] = net[table]["name"].map(p.loc[zeit]).fillna(0.0).to_numpy()
        q = getattr(r, f"{table}_q_mvar", None)
        if q is not None and len(q.columns):
            net[table]["q_mvar"] = net[table]["name"].map(q.loc[zeit]).fillna(0.0).to_numpy()
    if hasattr(r, "tap_pos") and len(r.tap_pos.columns):
        tap = net.trafo["name"].map(r.tap_pos.loc[zeit])
        net.trafo.loc[tap.notna(), "tap_pos"] = tap[tap.notna()].astype(int)
    return net


def startpunkte(basis=None):
    """
    Startzustände (Name → pandapower-Netz): Bemessungswerte des Netzes und typische Stunden aus den
    Referenzläufen der Prüfungsszenarien (werden bei Bedarf gerechnet, eine Woche je Monat).
    """
    b = basis or pr.basis()
    netze = {}
    nenn = copy.deepcopy(b.net)
    if len(nenn.storage):
        nenn.storage["p_mw"] = 0.0
    netze["Bemessungswerte (alle Lasten und Erzeuger mit Nennleistung)"] = nenn
    ref = {m: pr.szenario(f"ref_{m.lower()}", b, monate=[m]) for m in ("JANUARY", "JUNE", "JULY")}
    rj, rs, rn = ref["JANUARY"].results, ref["JULY"].results, ref["JUNE"].results
    t = rj.load_p_mw.sum(axis=1).idxmax()
    netze[f"Winter: Höchstlast ({t:%d.%m. %H:%M})"] = _netz_stunde(ref["JANUARY"], t)
    t = rj.ext_grid_p_mw.sum(axis=1).idxmin()
    netze[f"Winter: Starkwind, größte Rückspeisung ({t:%d.%m. %H:%M})"] = _netz_stunde(ref["JANUARY"], t)
    t = pr.leistung(ref["JULY"], pr.PV_ALLE).idxmax()
    netze[f"Sommer: Mittag, größte PV-Leistung ({t:%d.%m. %H:%M})"] = _netz_stunde(ref["JULY"], t)
    t = rn.load_p_mw.sum(axis=1).idxmin()
    netze[f"Sommer: Nacht, Schwachlast ({t:%d.%m. %H:%M})"] = _netz_stunde(ref["JUNE"], t)
    return netze


# ---------------------------------------------------------------------------------------------- Netz

def teilnetz(net, bus):
    """MS, Wohngebiet (R), Industrie (I) oder Gewerbe (C) für einen Bus."""
    name = str(net.bus.at[bus, "name"])
    if name.startswith("LV Bus R"):
        return "Wohngebiet (R)"
    if name.startswith("LV Bus I"):
        return "Industrie (I)"
    if name.startswith("LV Bus C"):
        return "Gewerbe (C)"
    return "MS-Netz"


class LiveNetz:
    """Statisches Netz mit Lastfluss, einfacher rONT-Regelung und Auswertung."""

    def __init__(self, net):
        self.net0 = copy.deepcopy(net)
        self.net = copy.deepcopy(net)
        self.ok = False
        self.ref = None                      # Ergebnis des Ausgangszustands (Spannungen)

    def reset(self):
        self.net = copy.deepcopy(self.net0)

    def rechne(self, ront=False, u_min=V_PLAN_MIN, u_max=V_PLAN_MAX):
        """AC-Lastfluss; mit ront=True stufen die drei Ortsnetztrafos wie in Teil 3b (±2 Stufen)."""
        try:
            pp.runpp(self.net, numba=False)
            self.ok = True
        except Exception:
            self.ok = False
            return False
        if ront:
            for _ in range(6):
                geaendert = False
                for kurz, name in ONT.items():
                    i = self.net.trafo.index[self.net.trafo["name"] == name]
                    if not len(i):
                        continue
                    i = i[0]
                    busse = self.net.bus.index[self.net.bus["name"].str.startswith(f"LV Bus {kurz}")
                                               & (self.net.bus["name"] != f"LV Bus {kurz}0")]
                    v = self.net.res_bus.loc[busse, "vm_pu"]
                    tap = int(self.net.trafo.at[i, "tap_pos"])
                    schritt = 0.025
                    if v.min() < u_min and v.max() + schritt <= u_max and tap > -2:
                        self.net.trafo.at[i, "tap_pos"] = tap - 1          # Stufe − hebt an
                        geaendert = True
                    elif v.max() > u_max and v.min() - schritt >= u_min and tap < 2:
                        self.net.trafo.at[i, "tap_pos"] = tap + 1          # Stufe + senkt ab
                        geaendert = True
                if not geaendert:
                    break
                try:
                    pp.runpp(self.net, numba=False)
                except Exception:
                    self.ok = False
                    return False
        return True

    def element_hinzufuegen(self, bus_name, art, p_mw, q_mvar=0.0, name=None):
        """Eigene Last ('Last') oder eigenen Erzeuger ('Erzeuger') an einem Bus ergänzen; Rückgabe (Tabelle, Index)."""
        bus = int(self.net.bus.index[self.net.bus["name"] == bus_name][0])
        nr = len(self.net.load) + len(self.net.sgen) + 1
        if art == "Last":
            name = name or f"Eigene Last {nr} ({bus_name})"
            return "load", pp.create_load(self.net, bus, p_mw=p_mw, q_mvar=q_mvar, name=name)
        name = name or f"Eigener Erzeuger {nr} ({bus_name})"
        return "sgen", pp.create_sgen(self.net, bus, p_mw=p_mw, q_mvar=q_mvar, name=name)

    # ------------------------------------------------------------------ Auswertung
    def zusammenfassung(self):
        n = self.net
        if not self.ok:
            return {"Lastfluss": "nicht konvergiert"}
        vm = n.res_bus["vm_pu"].dropna()
        ns = vm[n.bus.loc[vm.index, "vn_kv"] < 1]
        ms = vm[(n.bus.loc[vm.index, "vn_kv"] >= 1) & (n.bus.loc[vm.index, "vn_kv"] < 100)]
        ll = n.res_line["loading_percent"].dropna()
        tl = n.res_trafo["loading_percent"].dropna()
        name = lambda tab, i: str(n[tab].at[i, "name"])
        bname = lambda i: str(n.bus.at[i, "name"]).replace("LV Bus ", "")
        verl = n.res_line["pl_mw"].sum() + n.res_trafo["pl_mw"].sum()
        z = {"Lastfluss": "konvergiert",
             "Austausch 110 kV": f"{n.res_ext_grid['p_mw'].sum():+.2f} MW / {n.res_ext_grid['q_mvar'].sum():+.2f} Mvar (+ = Bezug)",
             "U min MS": f"{ms.min():.3f} pu ({bname(ms.idxmin())})", "U max MS": f"{ms.max():.3f} pu ({bname(ms.idxmax())})",
             "U min NS": f"{ns.min():.3f} pu ({bname(ns.idxmin())})", "U max NS": f"{ns.max():.3f} pu ({bname(ns.idxmax())})",
             "Leitung max": f"{ll.max():.1f} % ({name('line', ll.idxmax())})",
             "Trafo max": f"{tl.max():.1f} % ({name('trafo', tl.idxmax())})",
             "Verluste": f"{1000 * verl:.0f} kW",
             "rONT-Stufen (R / I / C)": " / ".join(
                 str(int(n.trafo.loc[n.trafo['name'] == t, 'tap_pos'].iloc[0])) for t in ONT.values()
                 if (n.trafo['name'] == t).any())}
        aus_band = int(((vm < V_PLAN_MIN) | (vm > V_PLAN_MAX)).sum())
        z["Busse außerhalb 0,95 … 1,05 pu"] = aus_band
        z["Busse außerhalb 0,90 … 1,10 pu"] = int(((vm < V_MIN) | (vm > V_MAX)).sum())
        z["Betriebsmittel > 100 %"] = int((ll > 100).sum() + (tl > 100).sum())
        return z

    def abbildung(self, strang="Wohngebiet", ref=None):
        """Spannungsprofil des Strangs (aktuell und Ausgangszustand), Spannung aller Busse, Auslastungen."""
        n = self.net
        fig = plt.figure(figsize=(15, 11))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1], width_ratios=[3, 1.2])
        ax = fig.add_subplot(gs[0, :])
        if not self.ok:
            ax.text(0.5, 0.5, "Lastfluss nicht konvergiert", ha="center", va="center", fontsize=16, color="red",
                    transform=ax.transAxes)
            plt.show()
            return
        ende = pr.STRAENGE.get(strang, strang)
        k = pr.kette(n, ende)
        x = np.arange(len(k))
        for j in range(1, len(k)):
            ax.axvspan(j - 1, j, color=pr.TYP_FARBE[k.at[j, "Typ"]], alpha=0.6, lw=0)
        u = n.res_bus.loc[[int(n.bus.index[n.bus["name"] == b][0]) for b in k["Bus"]], "vm_pu"].to_numpy()
        if ref is not None:
            u0 = ref.reindex(k["Bus"]).to_numpy()
            ax.plot(x, u0, "s--", color="gray", lw=1.2, ms=4, label="Ausgangszustand")
        ax.plot(x, u, "o-", color="tab:blue", lw=2, ms=6, label="aktuell")
        for xi, ui in zip(x, u):
            ax.annotate(f"{ui:.3f}", (xi, ui), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
        for v, c in ((V_MIN, "red"), (V_PLAN_MIN, "orange"), (V_PLAN_MAX, "orange"), (V_MAX, "red")):
            ax.axhline(v, color=c, ls="--", lw=0.9)
        ax.set_xticks(x, [pr._kurz(b) for b in k["Bus"]], fontsize=8)
        oben = ax.secondary_xaxis("top")
        oben.set_xticks(x[1:] - 0.5, ["Trafo" if "trafo" in t.lower() else f"{l:.0f} m"
                                      for l, t in zip(k["Länge [m]"][1:], k["Typ"][1:])], fontsize=7)
        ax.set_xlim(-0.3, len(k) - 0.7)
        ax.set_ylabel("Spannung [pu]")
        ax.set_title(f"Spannungsprofil {strang}", pad=22)
        from matplotlib.patches import Patch
        ax.legend(handles=ax.get_legend_handles_labels()[0] + [Patch(color=c, alpha=0.6, label=t) for t, c in pr.TYP_FARBE.items()],
                  fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(alpha=0.3, axis="y")
        # Spannung aller Busse
        axv = fig.add_subplot(gs[1, :])
        b = n.bus[(n.bus["vn_kv"] < 100)].copy()
        b["U"] = n.res_bus.loc[b.index, "vm_pu"]
        b["Netz"] = [teilnetz(n, i) for i in b.index]
        import re as _re
        def _ordnung(name):                                     # Bus 1 … 14, dann R, I, C, je nach Nummer
            m = _re.match(r"^(LV Bus )?([A-Z]*)\s*(\d+)$", str(name).replace("Bus ", "Bus  ").replace("LV Bus  ", "LV Bus "))
            gruppe = {"": 0, "R": 1, "I": 2, "C": 3}
            if str(name).startswith("Bus "):
                return (0, int(str(name).split()[1]))
            buchst = str(name).replace("LV Bus ", "")[0]
            return (gruppe.get(buchst, 4), int(_re.sub(r"\D", "", str(name)) or 0))
        b = b.loc[sorted(b.index, key=lambda i: _ordnung(b.at[i, "name"]))]
        farbe = np.where((b["U"] < V_MIN) | (b["U"] > V_MAX), "red",
                         np.where((b["U"] < V_PLAN_MIN) | (b["U"] > V_PLAN_MAX), "orange", "tab:blue"))
        axv.bar(range(len(b)), b["U"] - 1.0, bottom=1.0, color=farbe, width=0.8)
        if ref is not None:
            axv.scatter(range(len(b)), ref.reindex(b["name"]).to_numpy(), marker="_", color="black", s=60,
                        label="Ausgangszustand", zorder=3)
            axv.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        for v, c in ((V_MIN, "red"), (V_PLAN_MIN, "orange"), (V_PLAN_MAX, "orange"), (V_MAX, "red")):
            axv.axhline(v, color=c, ls="--", lw=0.9)
        axv.set_xticks(range(len(b)), [str(v).replace("LV Bus ", "") for v in b["name"]], fontsize=6, rotation=90)
        axv.set_ylim(min(0.88, b["U"].min() - 0.01), max(1.12, b["U"].max() + 0.01))
        axv.set_ylabel("Spannung [pu]")
        axv.set_title("Spannung an allen Bussen (orange: außerhalb 0,95 … 1,05, rot: außerhalb 0,90 … 1,10)")
        axv.grid(alpha=0.3, axis="y")
        # Auslastung Leitungen (die 15 höchsten) und Trafos
        axl = fig.add_subplot(gs[2, 0])
        ll = n.res_line["loading_percent"].dropna().sort_values(ascending=False).head(15)
        axl.bar(range(len(ll)), ll, color=np.where(ll > 100, "red", "tab:blue"))
        axl.set_xticks(range(len(ll)), [str(n.line.at[i, "name"]).replace("LV Line ", "") for i in ll.index],
                       fontsize=7, rotation=60)
        axl.axhline(100, color="red", ls="--", lw=0.9)
        axl.set_ylabel("Auslastung [%]")
        axl.set_title("Leitungen (die 15 höchsten)")
        axl.grid(alpha=0.3, axis="y")
        axt = fig.add_subplot(gs[2, 1])
        tl = n.res_trafo["loading_percent"]
        axt.bar(range(len(tl)), tl, color=np.where(tl > 100, "red", "tab:orange"))
        axt.set_xticks(range(len(tl)), [str(v).replace("LV Trafo ", "ONT ") for v in n.trafo.loc[tl.index, "name"]],
                       fontsize=7, rotation=60)
        axt.axhline(100, color="red", ls="--", lw=0.9)
        axt.set_title("Transformatoren")
        axt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------------------------- Oberfläche

def _bereich(p0, nenn, art):
    """Reglerbereich für P [MW]: großzügig um den Start- und Nennwert."""
    m = max(abs(p0), abs(nenn), 0.01)
    if art == "storage":
        return -1.5 * m, 1.5 * m
    return 0.0, (2.0 if art == "load" else 1.5) * m


def app(netze=None, strang="Wohngebiet"):
    """Interaktive Oberfläche (ipywidgets). netze: dict Name → Netz (Standard: startpunkte())."""
    import ipywidgets as w
    from IPython.display import display, clear_output

    netze = netze or startpunkte()
    stand = {"live": None, "regler": {}, "ref": None}
    ausgabe = w.Output()
    info = w.HTML()

    start = w.Dropdown(options=list(netze), description="Start:", layout=w.Layout(width="620px"),
                       style={"description_width": "60px"})
    laden = w.Button(description="Laden / zurücksetzen", button_style="warning", icon="refresh")
    strang_w = w.Dropdown(options=list(pr.STRAENGE), value=strang, description="Strang:",
                          style={"description_width": "60px"})
    ext = w.FloatSlider(value=1.03, min=0.95, max=1.08, step=0.005, description="U 110 kV [pu]",
                        continuous_update=False, readout_format=".3f", style={"description_width": "110px"})
    schalter = {s: w.Checkbox(value=False, description=f"{s} geschlossen", indent=False) for s in SCHALTER}
    ront_auto = w.Checkbox(value=True, description="rONT automatisch regeln (0,95 … 1,05 pu)", indent=False)
    taps = {k: w.IntSlider(value=0, min=-2, max=2, description=f"Stufe {k}", continuous_update=False,
                           style={"description_width": "60px"}, layout=w.Layout(width="260px")) for k in ONT}
    gruppen_box = w.Accordion()

    neu_bus = w.Dropdown(description="Bus:", style={"description_width": "40px"})
    neu_art = w.ToggleButtons(options=["Last", "Erzeuger"], value="Erzeuger")
    neu_p = w.FloatText(value=1.0, description="P [MW]", style={"description_width": "50px"}, layout=w.Layout(width="170px"))
    neu_q = w.FloatText(value=0.0, description="Q [Mvar]", style={"description_width": "60px"}, layout=w.Layout(width="180px"))
    neu_btn = w.Button(description="Element hinzufügen", button_style="success", icon="plus")

    beschaeftigt = {"an": False}

    def aktualisieren(*_):
        if beschaeftigt["an"] or stand["live"] is None:
            return
        live = stand["live"]
        n = live.net
        n.ext_grid["vm_pu"] = ext.value
        for s_, cb in schalter.items():
            n.switch.loc[n.switch["name"] == s_, "closed"] = cb.value
        for (tab, i), (sp, sq) in stand["regler"].items():
            n[tab].at[i, "p_mw"] = sp.value
            n[tab].at[i, "q_mvar"] = sq.value
        if not ront_auto.value:
            for k, sl in taps.items():
                n.trafo.loc[n.trafo["name"] == ONT[k], "tap_pos"] = sl.value
        live.rechne(ront=ront_auto.value)
        if ront_auto.value:                                     # Regler zeigen die gewählte Stufe
            beschaeftigt["an"] = True
            for k, sl in taps.items():
                sl.value = int(n.trafo.loc[n.trafo["name"] == ONT[k], "tap_pos"].iloc[0])
            beschaeftigt["an"] = False
        z = live.zusammenfassung()
        farbe = "green" if live.ok and z.get("Busse außerhalb 0,90 … 1,10 pu", 1) == 0 and z.get("Betriebsmittel > 100 %", 1) == 0 else "red"
        info.value = (f"<div style='border-left:4px solid {farbe};padding-left:8px'>"
                      + "<br>".join(f"<b>{k}:</b> {v}" for k, v in z.items()) + "</div>")
        with ausgabe:
            clear_output(wait=True)
            live.abbildung(strang_w.value, ref=stand["ref"])

    def zeile(tab, i, n):
        name = str(n[tab].at[i, "name"])
        bus = str(n.bus.at[n[tab].at[i, "bus"], "name"]).replace("LV Bus ", "")
        p0, q0 = float(n[tab].at[i, "p_mw"]), float(n[tab].at[i, "q_mvar"])
        if tab == "storage":
            nenn = float(n[tab].at[i, "max_p_mw"]) if "max_p_mw" in n[tab] else p0
        else:
            nenn = stand["nenn"].get((tab, name), p0)                # Nennleistung aus den Bemessungswerten
        lo, hi = _bereich(p0, nenn, tab)
        schritt = (hi - lo) / 400
        sp = w.FloatSlider(value=p0, min=lo, max=hi, step=schritt, description="P [MW]", continuous_update=False,
                           readout_format=".3f", style={"description_width": "55px"}, layout=w.Layout(width="420px"))
        qb = max(abs(hi), abs(lo)) / 2
        sq = w.FloatSlider(value=float(np.clip(q0, -qb, qb)), min=-qb, max=qb, step=qb / 200, description="Q [Mvar]",
                           continuous_update=False, readout_format=".3f", style={"description_width": "65px"},
                           layout=w.Layout(width="420px"))
        sp.observe(aktualisieren, "value")
        sq.observe(aktualisieren, "value")
        stand["regler"][(tab, i)] = (sp, sq)
        return w.HBox([w.Label(f"{name} @ {bus}", layout=w.Layout(width="260px")), sp, sq])

    def gruppen_bauen():
        n = stand["live"].net
        stand["regler"] = {}
        gruppen = {}
        for tab, art in (("sgen", "Erzeuger"), ("load", "Lasten"), ("storage", "Speicher")):
            for i in n[tab].index:
                name = str(n[tab].at[i, "name"])
                if name.startswith(("Eigene", "Eigener")):
                    g = "Eigene Elemente"
                elif tab == "storage":
                    g = "Heimspeicher (Wohngebiet)"
                else:
                    g = f"{art} – {teilnetz(n, n[tab].at[i, 'bus'])}"
                gruppen.setdefault(g, []).append(zeile(tab, i, n))
        reihenfolge = sorted(gruppen, key=lambda g: (g != "Eigene Elemente", "MS" not in g, g))
        gruppen_box.children = [w.VBox(gruppen[g]) for g in reihenfolge]
        for j, g in enumerate(reihenfolge):
            gruppen_box.set_title(j, f"{g} ({len(gruppen[g])})")

    def laden_klick(*_):
        beschaeftigt["an"] = True
        net = netze[start.value]
        live = LiveNetz(net)
        stand["live"] = live
        nenn_netz = netze[list(netze)[0]]
        stand["nenn"] = {(tab, nm): float(pv) for tab in ("load", "sgen")
                         for nm, pv in zip(nenn_netz[tab]["name"], nenn_netz[tab]["p_mw"])}
        ext.value = float(net.ext_grid["vm_pu"].iloc[0])
        for s_, cb in schalter.items():
            m = net.switch["name"] == s_
            cb.value = bool(net.switch.loc[m, "closed"].iloc[0]) if m.any() else False
        for k, sl in taps.items():
            m = net.trafo["name"] == ONT[k]
            sl.value = int(net.trafo.loc[m, "tap_pos"].iloc[0]) if m.any() else 0
        neu_bus.options = [b for b in net.bus["name"] if net.bus.loc[net.bus["name"] == b, "vn_kv"].iloc[0] < 100]
        gruppen_bauen()
        live.rechne(ront=ront_auto.value)
        stand["ref"] = pd.Series(live.net.res_bus["vm_pu"].to_numpy(), index=live.net.bus["name"]) if live.ok else None
        beschaeftigt["an"] = False
        aktualisieren()

    def neu_klick(*_):
        if stand["live"] is None:
            return
        stand["live"].element_hinzufuegen(neu_bus.value, neu_art.value, neu_p.value, neu_q.value)
        beschaeftigt["an"] = True
        gruppen_bauen()
        gruppen_box.selected_index = 0
        beschaeftigt["an"] = False
        aktualisieren()

    laden.on_click(laden_klick)
    neu_btn.on_click(neu_klick)
    for wdg in [ext, strang_w, ront_auto, *schalter.values(), *taps.values()]:
        wdg.observe(aktualisieren, "value")

    kopf = w.VBox([
        w.HBox([start, laden]),
        w.HBox([strang_w, ext]),
        w.HBox([*schalter.values()]),
        w.HBox([ront_auto, *taps.values()]),
        w.HTML("<b>Element ergänzen</b> (Last: P > 0 Bezug; Erzeuger: P > 0 Einspeisung, Q > 0 übererregt)"),
        w.HBox([neu_bus, neu_art, neu_p, neu_q, neu_btn]),
        w.HTML("<b>Lasten, Erzeuger und Speicher</b> (Regler: Wirk- und Blindleistung; Speicher P > 0 = Laden)"),
        gruppen_box,
    ])
    display(w.VBox([kopf, info, ausgabe]))
    laden_klick()
