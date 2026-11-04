"""
Regler für die Zeitreihensimulation (Task 4, Teil 3).

Ein Regler wird an runSimulation(controllers=[...]) übergeben. Nach jedem Lastfluss ruft die
Simulation step(net) jedes Reglers auf. Verstellt ein Regler etwas (Rückgabe True), wird dieselbe
Stunde neu gerechnet, bis kein Regler mehr eingreift. Der Zustand wird in die nächste Stunde
übernommen. In der Realität entspricht das einem Regler mit einer Verzögerungszeit im Bereich
von Sekunden bis Minuten: Er greift innerhalb der Stunde ein, nicht erst in der nächsten.

TapRegler         regelbarer Ortsnetztransformator (LV)
SchaltPlan        Backup-Plan für das MV-Netz bei Unterspannung über die Trennschalter S1–S3 (Notfallmaßnahme)
Wiederversorgung  Backup-Plan für das MV-Netz nach dem Ausfall eines Betriebsmittels ((n-1)-Fall)

ausfall_analyse / wiederversorgungsplan: statische (n-1)-Prüfung mit allen Schaltoptionen

Stufenschalter in pandapower (tap_side = "hv"): tap_pos = −1 erhöht die US-Spannung um eine Stufe
(2,5 %), tap_pos = +1 senkt sie.
"""

import copy
import math

import numpy as np
import pandas as pd
import pandapower as pp


class TapRegler:
    """
    Regelbarer Ortsnetztransformator mit Fernmessung im eigenen Teilnetz.

    Messgröße : niedrigste und höchste Spannung der überwachten NS-Busse (busse), z. B. aller Busse
                des Teilnetzes; entspricht Sensoren an den Strangenden.
    Sollband  : u_min … u_max (Planungsband, Standard 0,95 … 1,05 pu)
    Stellbereich: tap_min … tap_max des Trafos (Standardtypen: −2 … +2, je 2,5 %)

    Regelgesetz (je Prüfung):
      - U_min < u_min: so viele Stufen höher, wie die Abweichung in Stufen beträgt
        (bis 1 Stufe: 1, bis 2 Stufen: 2, …), aber nur so viele, dass U_max + Stufen · Δu ≤ u_max bleibt.
      - U_max > u_max: entsprechend tiefer, solange U_min − Stufen · Δu ≥ u_min bleibt.
      - Beide Grenzen im Band und Stufe ≠ 0: eine Stufe Richtung Neutralstellung, wenn danach
        beide Grenzen mit der Hysterese weiter im Band liegen (verhindert Hin- und Herschalten).
      Die Wirkung einer Stufe wird für diese Prüfungen mit dem Faktor `sicherheit` (Standard 1,2)
      vorsichtig geschätzt, weil eine Stufe von 2,5 % die Spannung am Strangende etwas stärker ändert.
    """

    def __init__(self, trafo, busse, u_min=0.95, u_max=1.05, hysterese=0.005, sicherheit=1.2, name=None):
        self.trafo = trafo
        self.busse = list(busse)
        self.u_min, self.u_max, self.hysterese = u_min, u_max, hysterese
        self.sicherheit = sicherheit        # Vorhersage der Spannungsänderung je Stufe mit Zuschlag
        self.name = name or f"TapRegler {trafo}"

    def config(self):
        return {"typ": "TapRegler", "trafo": self.trafo, "busse": sorted(self.busse), "u_min": self.u_min,
                "u_max": self.u_max, "hysterese": self.hysterese, "sicherheit": self.sicherheit}

    def reset(self, net):
        idx = net.trafo.index[net.trafo["name"] == self.trafo]
        if not len(idx):
            raise KeyError(f"Trafo '{self.trafo}' nicht im Netz.")
        self.ti = idx[0]
        t = net.trafo.loc[self.ti]
        if pd.isna(t.tap_min) or pd.isna(t.tap_max) or pd.isna(t.tap_step_percent):
            raise ValueError(f"Trafo '{self.trafo}' hat keinen Stufenschalter (tap_min/tap_max/tap_step_percent).")
        self.tap_min, self.tap_max = int(t.tap_min), int(t.tap_max)
        self.du = float(t.tap_step_percent) / 100
        self.vorzeichen = -1 if t.tap_side == "hv" else 1          # Stufen, um die US-Spannung zu erhöhen
        name_to_idx = dict(zip(net.bus["name"], net.bus.index))
        self.bi = [name_to_idx[b] for b in self.busse]
        net.trafo.at[self.ti, "tap_pos"] = int(t.tap_neutral) if not pd.isna(t.tap_neutral) else 0

    def _set(self, net, hoeher):
        """hoeher > 0: Spannung um hoeher Stufen anheben, < 0: absenken. Rückgabe: tatsächlich verstellt."""
        tap = int(net.trafo.at[self.ti, "tap_pos"])
        neu = int(np.clip(tap + self.vorzeichen * hoeher, self.tap_min, self.tap_max))
        if neu == tap:
            return False
        net.trafo.at[self.ti, "tap_pos"] = neu
        return True

    def step(self, net):
        vm = net.res_bus.vm_pu.loc[self.bi]
        lo, hi = vm.min(), vm.max()
        du_v = self.du * self.sicherheit                              # vorsichtig geschätzte Wirkung je Stufe
        if lo < self.u_min:
            k = math.ceil((self.u_min - lo) / self.du - 1e-9)
            while k > 0 and hi + k * du_v > self.u_max:              # keine Überspannung erzeugen
                k -= 1
            return self._set(net, k) if k > 0 else False
        if hi > self.u_max:
            k = math.ceil((hi - self.u_max) / self.du - 1e-9)
            while k > 0 and lo - k * du_v < self.u_min:
                k -= 1
            return self._set(net, -k) if k > 0 else False
        tap = int(net.trafo.at[self.ti, "tap_pos"])
        if tap == 0:
            return False
        richtung = -np.sign(self.vorzeichen * tap)                   # Richtung zur Neutralstellung (in Stufen „höher“)
        if richtung < 0 and lo - du_v >= self.u_min + self.hysterese:
            return self._set(net, -1)
        if richtung > 0 and hi + du_v <= self.u_max - self.hysterese:
            return self._set(net, +1)
        return False


class SchaltPlan:
    """
    Backup-Plan für das MV-Netz: Trennschalter stufenweise schließen, wenn die niedrigste
    MV-Spannung eine Schwelle unterschreitet, und wieder öffnen, wenn genug Reserve besteht.

    stufen: Liste von (Schalter, Auslöseschwelle, Rücknahmeschwelle), z. B.
        [(["S2", "S3"], 0.94, 0.95),      # Stufe 1: Maschen in Abgang 1 schließen
         (["S1"],       0.92, 0.96)]      # Stufe 2: Abgänge über S1 kuppeln (Notfall)
    Messgröße: niedrigste Spannung der überwachten MV-Busse (busse).
    Je Prüfung wird höchstens eine Stufe zu- oder abgeschaltet; Rücknahme in umgekehrter Reihenfolge.
    """

    def __init__(self, busse, stufen, name="SchaltPlan MV", wiederversorgung=None):
        self.busse = list(busse)
        self.stufen = [(list(sw), float(an), float(aus)) for sw, an, aus in stufen]
        self.name = name
        self.wiederversorgung = wiederversorgung    # Schalter, die die Wiederversorgung geschlossen hat, bleiben zu

    def config(self):
        return {"typ": "SchaltPlan", "busse": sorted(self.busse), "stufen": self.stufen}

    def reset(self, net):
        name_to_idx = dict(zip(net.bus["name"], net.bus.index))
        self.bi = [name_to_idx[b] for b in self.busse]
        self.si = [[int(net.switch.index[net.switch["name"] == s][0]) for s in sw] for sw, _, _ in self.stufen]
        for idx in self.si:                                          # Ausgangszustand: alle Stufen offen
            net.switch.loc[idx, "closed"] = False
        self.stufe = 0

    def step(self, net):
        lo = net.res_bus.vm_pu.loc[self.bi].min()                    # nicht versorgte Busse (NaN) zählen nicht
        if self.stufe < len(self.stufen) and lo < self.stufen[self.stufe][1]:
            idx = self.si[self.stufe]
            neu = not net.switch.loc[idx, "closed"].all()
            net.switch.loc[idx, "closed"] = True
            self.stufe += 1
            return bool(neu)
        if self.stufe > 0 and lo >= self.stufen[self.stufe - 1][2]:
            gesperrt = self.wiederversorgung.geschlossen if self.wiederversorgung else set()
            idx = [i for i in self.si[self.stufe - 1] if i not in gesperrt]
            geaendert = bool(len(idx)) and bool(net.switch.loc[idx, "closed"].any())
            net.switch.loc[idx, "closed"] = False
            self.stufe -= 1
            return geaendert
        return False


class Wiederversorgung:
    """
    Wiederversorgung nach dem Ausfall eines Betriebsmittels (Schwab, Kap. 11.3.1): Das gestörte
    Betriebsmittel ist herausgetrennt (in_service = False); danach wird die im Plan vorgesehene
    Trennstelle geschlossen, damit alle ungestörten Stationen weiter versorgt werden.

    plan: {Name des Betriebsmittels: [Trennschalter]}, z. B. {"Line 2-3": ["S1"], "Line 4-5": ["S2"]}
    Die geschlossenen Schalter bleiben zu, solange das Betriebsmittel ausgefallen ist; ein
    SchaltPlan mit wiederversorgung=... öffnet sie nicht.
    """

    def __init__(self, plan, name="Wiederversorgung MV"):
        self.plan = {k: list(v) for k, v in plan.items()}
        self.name = name
        self.geschlossen = set()

    def config(self):
        return {"typ": "Wiederversorgung", "plan": self.plan}

    def reset(self, net):
        self.elemente = {}
        for element, schalter in self.plan.items():
            for table in ("line", "trafo"):
                idx = net[table].index[net[table]["name"] == element]
                if len(idx):
                    sw = [int(net.switch.index[net.switch["name"] == s][0]) for s in schalter]
                    self.elemente[element] = (table, int(idx[0]), sw)
        self.geschlossen = set()

    def step(self, net):
        geaendert = False
        for element, (table, idx, sw) in self.elemente.items():
            if not net[table].at[idx, "in_service"]:
                offen = [i for i in sw if not net.switch.at[i, "closed"]]
                if offen:
                    net.switch.loc[offen, "closed"] = True
                    geaendert = True
                self.geschlossen.update(sw)
        return geaendert


def ausfall_analyse(net, ausfaelle, optionen, mv_busse, u_min=0.90, grenze=100.0):
    """
    (n-1)-Prüfung: Jedes Betriebsmittel aus `ausfaelle` wird einzeln außer Betrieb genommen und für jede
    Schaltoption aus `optionen` (Tupel von Trennschaltern, () = keine Maßnahme) ein Lastfluss gerechnet.

    Rückgabe: DataFrame mit einer Zeile je (Ausfall, Maßnahme):
      nicht versorgte Busse, niedrigste MS- und NS-Spannung, höchste Auslastung der MS-Leitungen und der
      110/20-kV-Trafos und ob die Maßnahme zulässig ist (alle Busse versorgt, U ≥ u_min, Auslastung ≤ grenze).
    """
    mv_idx = net.bus.index[net.bus["name"].isin(mv_busse)]
    mv_lines = net.line.index[net.line.from_bus.isin(mv_idx) & net.line.to_bus.isin(mv_idx)]
    hs_trafos = net.trafo.index[net.trafo.lv_bus.isin(mv_idx)]
    rows = []
    for element in ausfaelle:
        table = "line" if element in set(net.line["name"]) else "trafo"
        for option in optionen:
            n = copy.deepcopy(net)
            n[table].loc[n[table]["name"] == element, "in_service"] = False
            for s in option:
                n.switch.loc[n.switch["name"] == s, "closed"] = True
            row = {"Ausfall": element, "Maßnahme": " + ".join(f"{s} schließen" for s in option) or "keine"}
            try:
                pp.runpp(n)
            except pp.LoadflowNotConverged:
                row.update({"nicht versorgte Busse": np.nan, "zulässig": False, "Bemerkung": "Lastfluss konvergiert nicht"})
                rows.append(row)
                continue
            vm = n.res_bus.vm_pu
            versorgt = n.bus.index[vm.notna()]
            ll = n.res_line.loading_percent.loc[mv_lines]
            ll = ll[n.line.loc[mv_lines, "in_service"]]
            row.update({
                "nicht versorgte Busse": int(vm.isna().sum()),
                "U min MS [pu]": vm.loc[mv_idx].min(),
                "U min NS [pu]": vm[n.bus.vn_kv < 1].min(),
                "Leitung max MS [%]": ll.max(),
                "am stärksten belastet": n.line.at[ll.idxmax(), "name"],
                "Trafo max HS/MS [%]": n.res_trafo.loading_percent.loc[hs_trafos].max(),
            })
            row["zulässig"] = bool(row["nicht versorgte Busse"] == 0 and row["U min MS [pu]"] >= u_min
                                   and row["Leitung max MS [%]"] <= grenze and row["Trafo max HS/MS [%]"] <= grenze)
            row["Bemerkung"] = ""
            rows.append(row)
    return pd.DataFrame(rows)


def wiederversorgungsplan(analyse):
    """Je Ausfall die erste zulässige Maßnahme (Reihenfolge der Optionen = Vorrang). Rückgabe: DataFrame."""
    plan = []
    for element, df in analyse.groupby("Ausfall", sort=False):
        ohne = df.iloc[0]
        ok = df[df["zulässig"]]
        wahl = ok.iloc[0] if len(ok) else None
        plan.append({"Ausfall": element,
                     "ohne Maßnahme nicht versorgt": ohne["nicht versorgte Busse"],
                     "Maßnahme": wahl["Maßnahme"] if wahl is not None else "keine zulässige Maßnahme",
                     "U min MS [pu]": wahl["U min MS [pu]"] if wahl is not None else np.nan,
                     "U min NS [pu]": wahl["U min NS [pu]"] if wahl is not None else np.nan,
                     "Leitung max MS [%]": wahl["Leitung max MS [%]"] if wahl is not None else np.nan,
                     "am stärksten belastet": wahl["am stärksten belastet"] if wahl is not None else "",
                     "Trafo max HS/MS [%]": wahl["Trafo max HS/MS [%]"] if wahl is not None else np.nan})
    return pd.DataFrame(plan).set_index("Ausfall")


def regle_statisch(net, controllers, max_iter=10):
    """
    Regler auf einen einzelnen Lastfluss anwenden (z. B. Lastspitzen-Test).
    Rückgabe: Liste der Schritte (Iteration, Regler, Stufenstellungen, geschlossene Schalter, U min).
    """
    for c in controllers:
        c.reset(net)
    pp.runpp(net)
    protokoll = [_zustand(net, 0, "Start")]
    for it in range(1, max_iter + 1):
        changed = [c.name for c in controllers if c.step(net)]
        if not changed:
            break
        pp.runpp(net, init="auto")
        protokoll.append(_zustand(net, it, ", ".join(changed)))
    return pd.DataFrame(protokoll).set_index("Iteration")


def _zustand(net, it, wer):
    return {"Iteration": it, "Eingriff": wer,
            "Stufen": ", ".join(f"{n}: {int(t)}" for n, t in zip(net.trafo["name"], net.trafo["tap_pos"])
                                 if not pd.isna(t) and str(n).startswith("LV")),
            "geschlossene Trennschalter": ", ".join(s for s, c in zip(net.switch["name"], net.switch["closed"])
                                                   if c and s in ("S1", "S2", "S3")) or "keine",
            "U min MS/NS [pu]": net.res_bus.vm_pu[net.bus.vn_kv < 100].min(),
            "U max MS/NS [pu]": net.res_bus.vm_pu[net.bus.vn_kv < 100].max(),
            "U min NS [pu]": net.res_bus.vm_pu[net.bus.vn_kv < 1].min(),
            "U max NS [pu]": net.res_bus.vm_pu[net.bus.vn_kv < 1].max()}


def schaltvorgaenge(s):
    """
    Anzahl Stufen- und Schalterbewegungen je Trafo bzw. Schalter (Summe der Änderungen von Stunde zu Stunde).
    Ausgangszustand vor der ersten Stunde: Neutralstellung bzw. Trennschalter offen (Normalschaltzustand);
    ein Schalter, der schon in der ersten Stunde geschlossen ist, zählt also als eine Schaltung.
    """
    r = s.results
    tap = r.tap_pos.dropna(axis=1, how="all")
    out = {f"Stufenschritte {c}": int(tap[c].diff().abs().sum()) for c in tap.columns}
    sw = r.switch_closed[[c for c in ("S1", "S2", "S3") if c in r.switch_closed]].astype(int)
    out.update({f"Schaltungen {c}": int(sw[c].diff().fillna(sw[c].iloc[0]).abs().sum()) for c in sw.columns})
    return pd.Series(out)


def standard_regler(net, u_min=0.95, u_max=1.05):
    """
    Regler aus Teil 3b für ein Netz mit angeschlossenem CIGRE-LV-Ortsnetz:
    drei rONT-Regler (je Teilnetz R, I, C) und der Backup-Plan für das MV-Netz.
    """
    import lvnetz
    g = lvnetz.bus_groups(net)
    regler = []
    for x in "RIC":
        trafo = f"LV Trafo {x}0-{x}1"
        if trafo in set(net.trafo["name"]):
            regler.append(TapRegler(trafo, [b for b in g["NS"] if b.startswith(f"LV Bus {x}")], u_min=u_min, u_max=u_max))
    regler.append(SchaltPlan(g["MS"], [(["S2", "S3"], 0.94, 0.95), (["S1"], 0.92, 0.96)]))
    return regler
