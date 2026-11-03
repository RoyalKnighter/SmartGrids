"""
Regler für die Zeitreihensimulation (Task 4, Teil 3).

Ein Regler wird an runSimulation(controllers=[...]) übergeben. Nach jedem Lastfluss ruft die
Simulation step(net) jedes Reglers auf. Verstellt ein Regler etwas (Rückgabe True), wird dieselbe
Stunde neu gerechnet, bis kein Regler mehr eingreift. Der Zustand wird in die nächste Stunde
übernommen. In der Realität entspricht das einem Regler mit einer Verzögerungszeit im Bereich
von Sekunden bis Minuten: Er greift innerhalb der Stunde ein, nicht erst in der nächsten.

TapRegler       regelbarer Ortsnetztransformator (LV)
SchaltPlan      Backup-Plan für das MV-Netz über die Trennschalter S1–S3 (Notfallmaßnahme)

Stufenschalter in pandapower (tap_side = "hv"): tap_pos = −1 erhöht die US-Spannung um eine Stufe
(2,5 %), tap_pos = +1 senkt sie.
"""

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

    def __init__(self, busse, stufen, name="SchaltPlan MV"):
        self.busse = list(busse)
        self.stufen = [(list(sw), float(an), float(aus)) for sw, an, aus in stufen]
        self.name = name

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
        lo = net.res_bus.vm_pu.loc[self.bi].min()
        if self.stufe < len(self.stufen) and lo < self.stufen[self.stufe][1]:
            net.switch.loc[self.si[self.stufe], "closed"] = True
            self.stufe += 1
            return True
        if self.stufe > 0 and lo >= self.stufen[self.stufe - 1][2]:
            net.switch.loc[self.si[self.stufe - 1], "closed"] = False
            self.stufe -= 1
            return True
        return False


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
    """Anzahl Stufen- und Schalterbewegungen je Trafo bzw. Schalter (Summe der Änderungen von Stunde zu Stunde)."""
    r = s.results
    tap = r.tap_pos.dropna(axis=1, how="all")
    out = {f"Stufenschritte {c}": int(tap[c].diff().abs().sum()) for c in tap.columns}
    sw = r.switch_closed[[c for c in ("S1", "S2", "S3") if c in r.switch_closed]].astype(int)
    out.update({f"Schaltungen {c}": int(sw[c].diff().abs().sum()) for c in sw.columns})
    return pd.Series(out)
