"""
Bewertung von Anschlussvarianten (Task 4, Teil 2 und Teil 4) nach einheitlichen Kriterien.

Ausschluss: In mindestens einer Stunde
    - Spannung unter 0,90 pu oder über 1,10 pu (MS- oder NS-Bus) oder
    - Leitung bzw. Trafo über 100 % Auslastung.
Kriterien, die bei allen Varianten verletzt sind, unterscheiden die Varianten nicht und werden für
die Auswahl ausgeklammert (bleiben aber als Befund in der Tabelle).

Auswahl unter den verbleibenden Varianten: Rangsumme über
    niedrigste höchste Spannung, höchste niedrigste Spannung,
    niedrigste höchste Auslastung von MS-Leitungen, NS-Leitungen, Trafos 110/20 kV und Ortsnetztrafos.
Bei gleicher Rangsumme entscheiden die geringeren Netzverluste.
"""

import numpy as np
import pandas as pd

import lvnetz

KRITERIEN = {
    "U < 0,90 pu (MS)": "Stunden U < 0,90 MS",
    "U < 0,90 pu (NS)": "Stunden U < 0,90 NS",
    "U > 1,10 pu (MS)": "Stunden U > 1,10 MS",
    "U > 1,10 pu (NS)": "Stunden U > 1,10 NS",
    "Leitung MS > 100 %": "Stunden Leitung MS > 100 %",
    "Leitung NS > 100 %": "Stunden Leitung NS > 100 %",
    "Trafo HS/MS > 100 %": "Stunden Trafo HS/MS > 100 %",
    "Ortsnetztrafo > 100 %": "Stunden Ortsnetztrafo > 100 %",
}

# Kennwert → True: kleiner ist besser, False: größer ist besser
ZIELE = {"U max gesamt [pu]": True, "U min gesamt [pu]": False,
         "Leitung max MS [%]": True, "Leitung max NS [%]": True,
         "Trafo max HS/MS [%]": True, "Ortsnetztrafo max [%]": True}


def kennwerte(sims):
    """Kennwerte und Orte aller Varianten: (werte, orte) als DataFrames, Zeilen = Varianten."""
    werte, orte = {}, {}
    for k, s in sims.items():
        werte[k], orte[k] = lvnetz.kennwerte_sim(s)
    return pd.DataFrame(werte).T, pd.DataFrame(orte).T


def ausschluss(werte):
    """
    Ausschlusstabelle: Stunden je Kriterium, Begründung und Ergebnis.
    Rückgabe: (Tabelle, Liste der verbleibenden Varianten, immer verletzte Kriterien)
    """
    stunden = pd.DataFrame({k: werte.get(c, pd.Series(0, index=werte.index)).fillna(0).astype(int)
                            for k, c in KRITERIEN.items()})
    verletzt = stunden > 0
    immer = [k for k in verletzt if verletzt[k].all()] if len(verletzt) > 1 else []
    wirksam = [k for k in verletzt if k not in immer]
    tab = stunden.copy()
    tab["ausgeschlossen wegen"] = [", ".join(k for k in wirksam if verletzt.at[v, k]) or "–" for v in tab.index]
    tab["Ergebnis"] = np.where(verletzt[wirksam].any(axis=1), "ausgeschlossen", "bleibt")
    return tab, list(tab.index[tab["Ergebnis"] == "bleibt"]), immer


# Genauigkeit für den Vergleich: kleinere Unterschiede gelten als gleich (Ränge werden geteilt)
GENAUIGKEIT = {"[pu]": 3, "[%]": 1}


def rangfolge(werte, kandidaten):
    """
    Rangtabelle der Kandidaten (1 = bester Wert je Kriterium) und gewählte Variante (kleinste Rangsumme).
    Die Kennwerte werden vor dem Vergleich gerundet (Spannung auf 0,001 pu, Auslastung auf 0,1 %), damit
    rechnerisch gleiche Varianten gleiche Ränge erhalten. Bei gleicher Rangsumme entscheiden die
    geringeren Netzverluste.
    """
    if not kandidaten:
        return pd.DataFrame(), None
    v = werte.loc[kandidaten, list(ZIELE)].astype(float)
    for k in v:
        dez = next((d for einheit, d in GENAUIGKEIT.items() if einheit in k), 3)
        v[k] = v[k].round(dez)
    raenge = pd.DataFrame({f"Rang {k.split(' [')[0]}": v[k].rank(ascending=asc, method="min").astype(int)
                           for k, asc in ZIELE.items()})
    raenge["Rangsumme"] = raenge.sum(axis=1)
    tab = pd.concat([v, raenge], axis=1)
    if "Verluste [MWh]" in werte:
        tab["Verluste [MWh]"] = werte.loc[kandidaten, "Verluste [MWh]"].astype(float).round(1)
        tab = tab.sort_values(["Rangsumme", "Verluste [MWh]"])
    else:
        tab = tab.sort_values("Rangsumme")
    return tab, tab.index[0]
