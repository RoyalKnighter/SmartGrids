"""
Einlesen, Aufbereiten und Plotten der BDEW-Standardlastprofile (Veröffentlichung 2025).

Aufbau der BDEW-CSV-Dateien
---------------------------
- Kodierung Windows-1252, Trennzeichen ";", Dezimalkomma
- Zeile 3: Monat (Januar ... Dezember), Zeile 4: Tagtyp (SA, FT, WT)
- 96 Zeilen mit Viertelstundenwerten (00:00-00:15 ... 23:45-00:00)
- Werte: Energie in kWh je Viertelstunde, normiert auf 1 Mio. kWh Jahresverbrauch

Tagtypen
--------
WT = Werktag (Mo-Fr), SA = Samstag, FT = Sonntag/Feiertag

Umrechnung
----------
Mittlere Leistung einer Viertelstunde:  P [kW] = E [kWh] / 0,25 h = 4 * E
Skalierung auf einen Jahresverbrauch:  P = P_norm * Jahresverbrauch / 1.000.000 kWh
"""

from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Lastprofil(Enum):
    """Verfügbare Profile. Der Wert ist (Dateiname, Beschreibung)."""
    HAUSHALT = ("BDEW_H25_Haushalte.csv", "H25 – Haushalt")
    PV = ("BDEW_P25_PV.csv", "P25 – Haushalt mit PV-Anlage (Netzbezug)")

    @property
    def filename(self):
        return self.value[0]

    @property
    def label(self):
        return self.value[1]


class Month(Enum):
    """Monate. Der Wert ist (Monatsnummer, Name in der BDEW-Datei)."""
    JANUARY = (1, "Januar")
    FEBRUARY = (2, "Februar")
    MARCH = (3, "März")
    APRIL = (4, "April")
    MAY = (5, "Mai")
    JUNE = (6, "Juni")
    JULY = (7, "Juli")
    AUGUST = (8, "August")
    SEPTEMBER = (9, "September")
    OCTOBER = (10, "Oktober")
    NOVEMBER = (11, "November")
    DECEMBER = (12, "Dezember")

    @property
    def number(self):
        return self.value[0]

    @property
    def label(self):
        return self.value[1]


class DayType(Enum):
    """Tagtypen der BDEW-Profile."""
    WT = "Werktag"
    SA = "Samstag"
    FT = "Sonntag/Feiertag"


class Season(Enum):
    """Jahreszeiten (meteorologische Einteilung). Der Wert ist (Name, Monate)."""
    WINTER = ("Winter", (Month.DECEMBER, Month.JANUARY, Month.FEBRUARY))
    SPRING = ("Frühling", (Month.MARCH, Month.APRIL, Month.MAY))
    SUMMER = ("Sommer", (Month.JUNE, Month.JULY, Month.AUGUST))
    AUTUMN = ("Herbst", (Month.SEPTEMBER, Month.OCTOBER, Month.NOVEMBER))

    @property
    def label(self):
        return self.value[0]

    @property
    def months(self):
        return self.value[1]


# Reihenfolge der Tage in einer Woche: 5 x Werktag, Samstag, Sonntag/Feiertag
WEEK_DAYS = [
    ("Mo", DayType.WT), ("Di", DayType.WT), ("Mi", DayType.WT),
    ("Do", DayType.WT), ("Fr", DayType.WT), ("Sa", DayType.SA), ("So", DayType.FT),
]

NORM_KWH = 1_000_000          # Normierung der BDEW-Profile: 1 Mio. kWh/a


# ---------------------------------------------------------------------------
# Einlesen
# ---------------------------------------------------------------------------

def read_bdew_csv(path):
    """
    Liest eine BDEW-Profil-CSV ein.

    Rückgabe
    --------
    DataFrame mit 96 Zeilen (Viertelstunden) und MultiIndex-Spalten (Month, DayType).
    Index: Beginn der Viertelstunde als Timedelta (0:00, 0:15, ...).
    Werte: kWh je Viertelstunde, normiert auf 1 Mio. kWh/a.
    """
    raw = pd.read_csv(path, sep=";", decimal=",", encoding="cp1252",
                      header=[2, 3], index_col=1, nrows=96)
    raw = raw.drop(columns=raw.columns[0])            # leere erste Spalte

    month_by_label = {m.label: m for m in Month}
    raw.columns = pd.MultiIndex.from_tuples(
        [(month_by_label[month], DayType[daytype]) for month, daytype in raw.columns],
        names=["month", "daytype"],
    )

    # "00:00-00:15" -> Timedelta 0:00
    raw.index = pd.to_timedelta(raw.index.str.slice(0, 5) + ":00")
    raw.index.name = "time"

    if raw.shape != (96, 36) or raw.isna().any().any():
        raise ValueError(f"{path}: unerwartete Struktur {raw.shape} oder fehlende Werte.")
    return raw.astype(float)


def load_profiles(data_dir="data", profiles=tuple(Lastprofil)):
    """
    Liest alle Profile einmal ein und gibt sie als Dictionary zurück:
        {Lastprofil.HAUSHALT: DataFrame, Lastprofil.PV: DataFrame}
    """
    data_dir = Path(data_dir)
    return {p: read_bdew_csv(data_dir / p.filename) for p in profiles}


# ---------------------------------------------------------------------------
# Aufbereitung
# ---------------------------------------------------------------------------

def get_day(profiles, profil, month, daytype, annual_kwh=NORM_KWH, unit="kW"):
    """
    Tagesverlauf (96 Viertelstundenwerte) als mittlere Leistung.

    annual_kwh : Jahresverbrauch, auf den skaliert wird (Standard: 1 Mio. kWh = unskaliert)
    unit       : "kW" oder "MW" (MW für pandapower)
    """
    energy_kwh = profiles[profil][(month, daytype)]
    power_kw = energy_kwh * 4 * annual_kwh / NORM_KWH
    return power_kw / 1000 if unit == "MW" else power_kw


def build_week(profiles, profil, month, annual_kwh=NORM_KWH, unit="kW", year=2025,
               start=None, resolution="15min"):
    """
    Erzeugt eine repräsentative Woche für einen Monat:
    5 x Werktag (Mo-Fr), 1 x Samstag, 1 x Sonntag/Feiertag.

    start      : Montag 00:00, an dem die Woche beginnen soll (z. B. der Beginn der
                 typischen Wetterwoche). Ohne Angabe: erster Montag des Monats in `year`.
    resolution : "15min" (Originalauflösung, 672 Werte) oder "1h" (Stundenmittelwerte,
                 168 Werte; Zeitstempel = Beginn der Stunde).

    Rückgabe
    --------
    pandas.Series mit DatetimeIndex, Name = Profilbezeichnung
    """
    days = [get_day(profiles, profil, month, daytype, annual_kwh, unit).to_numpy()
            for _, daytype in WEEK_DAYS]

    if start is None:
        first_day = pd.Timestamp(year=year, month=month.number, day=1)
        start = first_day + pd.Timedelta(days=(7 - first_day.weekday()) % 7)
    start = pd.Timestamp(start)
    if start.weekday() != 0:
        raise ValueError(f"start muss ein Montag sein, nicht {start:%A %d.%m.%Y}.")
    index = pd.date_range(start.normalize(), periods=7 * 96, freq="15min")

    values = pd.Series(np.concatenate(days), index=index, name=f"{profil.label} [{unit}]")
    if resolution == "1h":
        values = values.resample("1h").mean()      # Mittel der vier Viertelstunden-Leistungen
    elif resolution != "15min":
        raise ValueError('resolution muss "15min" oder "1h" sein.')
    return values


def build_weeks(profiles, profil, months=tuple(Month), annual_kwh=NORM_KWH, unit="kW", year=2025,
                starts=None, resolution="15min"):
    """
    Repräsentative Wochen für mehrere Monate: {Month: Series}.
    starts: optionales Dictionary {Month: Montag}, z. B. aus weather.select_typical_weeks
            bzw. aus den Zeitstempeln der Wetterwochen.
    """
    return {m: build_week(profiles, profil, m, annual_kwh, unit, year,
                          start=None if starts is None else starts[m], resolution=resolution)
            for m in months}


def summary(profiles, profil, annual_kwh=NORM_KWH):
    """Tabelle mit Tagesenergie [kWh/d], Maximal- und Minimalleistung [kW] je Monat und Tagtyp."""
    rows = []
    for month in Month:
        for daytype in DayType:
            p = get_day(profiles, profil, month, daytype, annual_kwh, "kW")
            rows.append({"Monat": month.label, "Tagtyp": daytype.name,
                         "Energie [kWh/d]": p.sum() / 4,
                         "P_max [kW]": p.max(), "P_min [kW]": p.min()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotten
# ---------------------------------------------------------------------------

def _week_axis(ax):
    """Beschriftet die x-Achse einer Woche mit Wochentagen."""
    ticks = [i * 24 for i in range(8)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([d for d, _ in WEEK_DAYS] + [""])
    ax.set_xticks(range(0, 169, 6), minor=True)
    ax.set_xlim(0, 168)
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.15)
    ax.set_xlabel("Wochentag (Tagesbeginn 00:00)")


def plot_week(profiles, profil, months=(Month.JANUARY, Month.APRIL, Month.JULY, Month.OCTOBER),
              annual_kwh=NORM_KWH, unit="kW", ax=None):
    """Repräsentative Wochen mehrerer Monate eines Profils in einem Diagramm."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))
    for month in months:
        week = build_week(profiles, profil, month, annual_kwh, unit)
        hours = (week.index - week.index[0]) / pd.Timedelta(hours=1)
        ax.plot(hours, week.to_numpy(), label=month.label, linewidth=1.2)
    _week_axis(ax)
    scale = "normiert auf 1 Mio. kWh/a" if annual_kwh == NORM_KWH else f"Jahresverbrauch {annual_kwh:,.0f} kWh/a".replace(",", ".")
    ax.set_title(f"{profil.label} – repräsentative Woche je Monat ({scale})")
    ax.set_ylabel(f"Mittlere Leistung [{unit}]")
    ax.legend(title="Monat", loc="upper left", bbox_to_anchor=(1.01, 1))
    return ax


def plot_day(profiles, profil, month, annual_kwh=NORM_KWH, unit="kW", ax=None):
    """Tagesverläufe der drei Tagtypen eines Monats."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    for daytype in DayType:
        p = get_day(profiles, profil, month, daytype, annual_kwh, unit)
        ax.plot(p.index / pd.Timedelta(hours=1), p.to_numpy(), label=f"{daytype.name} ({daytype.value})")
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.4)
    ax.set_title(f"{profil.label} – {month.label}")
    ax.set_xlabel("Uhrzeit [h]")
    ax.set_ylabel(f"Mittlere Leistung [{unit}]")
    ax.legend(title="Tagtyp")
    return ax


def plot_compare(profiles, profile_list, month, annual_kwh=NORM_KWH, unit="kW", ax=None):
    """Mehrere Profile für denselben Monat (repräsentative Woche) vergleichen."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))
    for profil in profile_list:
        week = build_week(profiles, profil, month, annual_kwh, unit)
        hours = (week.index - week.index[0]) / pd.Timedelta(hours=1)
        ax.plot(hours, week.to_numpy(), label=profil.label, linewidth=1.2)
    _week_axis(ax)
    ax.set_title(f"Profilvergleich – repräsentative Woche {month.label}")
    ax.set_ylabel(f"Mittlere Leistung [{unit}]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    return ax


def plot_seasons(profiles, profil, daytype=DayType.WT, annual_kwh=NORM_KWH, unit="kW"):
    """2x2-Raster: je Jahreszeit die Tagesverläufe der drei Monate für einen Tagtyp."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, season in zip(axes.flat, Season):
        for month in season.months:
            p = get_day(profiles, profil, month, daytype, annual_kwh, unit)
            ax.plot(p.index / pd.Timedelta(hours=1), p.to_numpy(), label=month.label)
        ax.set_title(season.label)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlim(0, 24)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Uhrzeit [h]")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Mittlere Leistung [{unit}]")
    fig.suptitle(f"{profil.label} – {daytype.value} nach Jahreszeit")
    fig.tight_layout()
    return fig
