"""
Profile kopieren und bearbeiten (Szenarien für einzelne Elemente).

Eine Profiländerung wirkt auf die Leistung der ausgewählten Elemente (Lasten und Erzeuger), nachdem sie
aus ihrem Profil berechnet wurde. Das Profil selbst (z. B. PV) bleibt für alle anderen Elemente unverändert;
die ausgewählten Elemente bekommen also eine bearbeitete Kopie ihres Profils.

Die Änderungen werden als Einstellung an die Simulation gehängt und gehören damit zum Fingerabdruck
(gespeicherte Ergebnisse werden nur wiederverwendet, wenn die Änderungen gleich sind):

    import profilaenderung as pa
    s.changeSetting("profile_edits", [
        pa.immer("^Windpark$", faktor=0.3),                                  # Dunkelflaute: 30 % Wind
        pa.einmal("PV", faktor=0.2, datum="2025-07-23", von=11, bis=14),     # Wolken am 23.07. 11–14 Uhr
        pa.periodisch("^LV Load R", faktor=1.3, alle_h=24, start_h=17, dauer_h=5),  # jeden Abend +30 %
    ])

Auswahl der Elemente: regulärer Ausdruck auf den Namen (Lasten und Erzeuger, `re.search`).
Art der Änderung (genau eine angeben):
    faktor  : Leistung mit dem Faktor multiplizieren (0,3 = 30 % der normalen Leistung)
    wert_mw : Leistung fest auf diesen Wert setzen [MW] (je Element)
    plus_mw : diesen Wert zur Leistung addieren [MW] (je Element)
Die Blindleistung folgt der Wirkleistung über den cos φ des Elements. Speicher rechnen ihren Fahrplan
mit der geänderten PV- und Lastleistung.
Zeitraum:
    immer(...)       : alle Stunden (optional nur bestimmte Monate)
    einmal(...)      : ein Tag (datum oder monat + wochentag) von Stunde `von` bis `bis`,
                       oder ohne Tag: die ganze typische Woche des Monats
    periodisch(...)  : alle `alle_h` Stunden für `dauer_h` Stunden, beginnend bei Stunde `start_h` der Woche
                       (Montag 0 Uhr = 0); alle_h=24 heißt täglich, alle_h=168 wöchentlich
Mehrere Änderungen werden nacheinander angewendet.
"""

import re

import numpy as np
import pandas as pd

ARTEN = ("faktor", "wert_mw", "plus_mw")
WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def _aenderung(elemente, zeit, faktor, wert_mw, plus_mw, name):
    gesetzt = {k: v for k, v in (("faktor", faktor), ("wert_mw", wert_mw), ("plus_mw", plus_mw)) if v is not None}
    if len(gesetzt) != 1:
        raise ValueError("Genau eine Art angeben: faktor, wert_mw oder plus_mw.")
    re.compile(elemente)
    (art, wert), = gesetzt.items()
    return {"name": name, "elemente": elemente, "art": art, "wert": float(wert), "zeit": zeit}


def _monate(monate):
    if monate is None:
        return None
    from profiles import Month
    monate = [monate] if isinstance(monate, (str, Month)) else list(monate)
    return [m.name if isinstance(m, Month) else Month[str(m).upper()].name for m in monate]


def immer(elemente, faktor=None, wert_mw=None, plus_mw=None, monate=None, name=None):
    """Änderung in allen Stunden (optional nur in den angegebenen Monaten)."""
    return _aenderung(elemente, {"typ": "immer", "monate": _monate(monate)}, faktor, wert_mw, plus_mw, name)


def einmal(elemente, faktor=None, wert_mw=None, plus_mw=None, datum=None, monat=None, wochentag=None,
           von=0, bis=24, name=None):
    """
    Einmalige Änderung an einem Tag von Stunde `von` bis `bis` (bis ausschließlich).
    Tag: datum="2025-07-23" oder monat="JULY" und wochentag=2 (0 = Montag). Nur monat: die ganze Woche.
    Die Daten der typischen Wochen zeigt `typische_wochen()`.
    """
    if datum is None and monat is None:
        raise ValueError("datum oder monat angeben.")
    if not 0 <= von < bis <= 24:
        raise ValueError("Es muss 0 ≤ von < bis ≤ 24 gelten.")
    zeit = {"typ": "einmal", "datum": str(pd.Timestamp(datum).date()) if datum is not None else None,
            "monat": _monate(monat)[0] if monat is not None else None,
            "wochentag": None if wochentag is None else int(wochentag), "von": int(von), "bis": int(bis)}
    return _aenderung(elemente, zeit, faktor, wert_mw, plus_mw, name)


def periodisch(elemente, faktor=None, wert_mw=None, plus_mw=None, alle_h=24, start_h=0, dauer_h=1,
               monate=None, name=None):
    """
    Wiederkehrende Änderung: alle `alle_h` Stunden für `dauer_h` Stunden, erstmals ab Stunde `start_h`
    der Woche (Montag 0 Uhr = 0). Beispiele: täglich 17–22 Uhr → alle_h=24, start_h=17, dauer_h=5;
    jeden zweiten Tag 12–14 Uhr → alle_h=48, start_h=12, dauer_h=2; samstags ganztägig → alle_h=168,
    start_h=120, dauer_h=24.
    """
    if alle_h <= 0 or dauer_h <= 0 or dauer_h > alle_h:
        raise ValueError("Es muss 0 < dauer_h ≤ alle_h gelten.")
    zeit = {"typ": "periodisch", "alle_h": int(alle_h), "start_h": int(start_h), "dauer_h": int(dauer_h),
            "monate": _monate(monate)}
    return _aenderung(elemente, zeit, faktor, wert_mw, plus_mw, name)


def maske(zeit, profile_table):
    """Bool-Array je Zeitschritt: in welchen Stunden gilt die Änderung."""
    idx = pd.DatetimeIndex(profile_table.index)
    monat = profile_table["month"].to_numpy()
    how = profile_table["hour_of_week"].to_numpy()
    m = np.ones(len(idx), dtype=bool)
    if zeit.get("monate"):
        m &= np.isin(monat, zeit["monate"])
    typ = zeit["typ"]
    if typ == "immer":
        return m
    if typ == "einmal":
        if zeit.get("datum"):
            m &= np.asarray(idx.normalize() == pd.Timestamp(zeit["datum"]))
        else:
            m &= monat == zeit["monat"]
            if zeit.get("wochentag") is not None:
                m &= np.asarray(idx.weekday == zeit["wochentag"])
        return m & (np.asarray(idx.hour) >= zeit["von"]) & (np.asarray(idx.hour) < zeit["bis"])
    if typ == "periodisch":
        rel = how - zeit["start_h"]
        return m & (rel >= 0) & (rel % zeit["alle_h"] < zeit["dauer_h"])
    raise ValueError(f"Unbekannter Zeittyp '{typ}'.")


def anwenden(p, namen, profile_table, aenderungen):
    """
    Profiländerungen auf die Leistungsmatrix p (T × n, MW) der Elemente `namen` anwenden.
    Rückgabe: geänderte Kopie von p.
    """
    if not aenderungen:
        return p
    p = p.copy()
    namen = list(namen)
    for a in aenderungen:
        spalten = [j for j, n in enumerate(namen) if re.search(a["elemente"], str(n))]
        if not spalten:
            continue
        zeilen = maske(a["zeit"], profile_table)
        if not zeilen.any():
            continue
        block = p[np.ix_(zeilen, spalten)]
        if a["art"] == "faktor":
            block = block * a["wert"]
        elif a["art"] == "wert_mw":
            block = np.full_like(block, a["wert"])
        else:
            block = block + a["wert"]
        p[np.ix_(zeilen, spalten)] = block
    return p


def betroffene_elemente(s, aenderungen):
    """Welche Lasten und Erzeuger von welcher Änderung betroffen sind (zur Kontrolle)."""
    zeilen = []
    for i, a in enumerate(aenderungen):
        for table in ("load", "sgen"):
            for n in s.net[table]["name"]:
                if re.search(a["elemente"], str(n)):
                    zeilen.append({"Änderung": a.get("name") or f"Nr. {i + 1}", "Tabelle": table, "Element": n})
    return pd.DataFrame(zeilen)


def beschreibung(a):
    """Kurzer Text zu einer Änderung."""
    art = {"faktor": f"× {a['wert']:g}", "wert_mw": f"= {a['wert']:g} MW", "plus_mw": f"+ {a['wert']:g} MW"}[a["art"]]
    z = a["zeit"]
    if z["typ"] == "immer":
        wann = "immer" + (f" ({', '.join(z['monate'])})" if z.get("monate") else "")
    elif z["typ"] == "einmal":
        tag = z["datum"] or f"{z['monat']}" + (f", {WOCHENTAGE[z['wochentag']]}" if z.get("wochentag") is not None else "")
        wann = f"einmal: {tag}, {z['von']}–{z['bis']} Uhr"
    else:
        wann = (f"alle {z['alle_h']} h für {z['dauer_h']} h ab Wochenstunde {z['start_h']}"
                + (f" ({', '.join(z['monate'])})" if z.get("monate") else ""))
    return f"{a.get('name') or ''} [{a['elemente']}] {art}, {wann}".strip()


def typische_wochen():
    """Daten (Montag bis Sonntag) der 12 typischen Wochen, damit man einen Tag für einmal() wählen kann."""
    import timeseries as tsm
    from profiles import Month
    pt = tsm.build_profile_table(list(Month))
    zeilen = {}
    for m in Month:
        tage = pd.DatetimeIndex(pt.index[pt["month"] == m.name]).normalize().unique()
        zeilen[m.label] = {WOCHENTAGE[t.weekday()]: f"{t:%Y-%m-%d}" for t in tage}
    return pd.DataFrame(zeilen).T[WOCHENTAGE]
