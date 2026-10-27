"""
Diagramme für Task 3.

- week_axis():          einheitliche Zeitachse: Uhrzeit im 6-Stunden-Takt, senkrechte Linie
                        um 24:00 (Tageswechsel), Wochentag und Datum über jedem Tag
- plot_load_week():     Haushaltslast einer Woche
- plot_weather_week():  zweiteilig: oben Wetter, unten erzeugte Leistung
- component_overview(): Anzahl der Komponenten je Elementtabelle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.transforms import blended_transform_factory

WEEKDAYS_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


# ---------------------------------------------------------------------------
# Zeitachse
# ---------------------------------------------------------------------------

def week_axis(ax, index, label_days=True):
    """
    Formatiert eine Zeitachse für eine Woche:
    - Teilstriche und Beschriftung alle 6 Stunden (00, 06, 12, 18)
    - senkrechte Linie bei jedem Tageswechsel (24:00 bzw. 00:00)
    - Wochentag und Datum mittig über jedem Tag (wenn label_days=True)
    """
    start = pd.Timestamp(index[0]).normalize()
    end = start + pd.Timedelta(days=7)
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="y", alpha=0.4)
    ax.grid(True, axis="x", alpha=0.15)

    for d in range(8):
        ax.axvline(start + pd.Timedelta(days=d), color="black", lw=1.0, alpha=0.7)

    if label_days:
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for d in range(7):
            day = start + pd.Timedelta(days=d)
            ax.text(day + pd.Timedelta(hours=12), 1.01, f"{WEEKDAYS_DE[day.weekday()]} {day:%d.%m.}",
                    transform=trans, ha="center", va="bottom", fontsize=9)


# ---------------------------------------------------------------------------
# Last
# ---------------------------------------------------------------------------

def plot_load_week(weeks, title, ylabel="Leistung [kW]"):
    """
    Lastprofile einer Woche.
    weeks: Series (eine Kurve) oder Dictionary {Bezeichnung: Series} (mehrere Kurven)
    """
    if isinstance(weeks, pd.Series):
        weeks = {weeks.name or "Last": weeks}
    colors = ["tab:blue", "tab:purple", "tab:brown", "tab:grey"]
    fig, ax = plt.subplots(figsize=(13, 3.4))
    for (label, week), color in zip(weeks.items(), colors):
        ax.plot(week.index, week.to_numpy(), color=color, lw=1.3, label=label)
    first = next(iter(weeks.values()))
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, None)
    week_axis(ax, first.index)
    ax.set_xlabel("Uhrzeit [h]")
    ax.set_title(title, pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Wetter und Erzeugung
# ---------------------------------------------------------------------------

def plot_weather_week(w, title):
    """
    Zweiteiliges Diagramm einer Wetterwoche.

    oben:  Windgeschwindigkeit in Nabenhöhe [m/s], Lufttemperatur [°C],
           Einstrahlung auf die Modulebene [100 W/m²]
    unten: Leistung der Windenergieanlage [MW], Leistung eines PV-Moduls [100 W]

    Durch die Skalierung liegen alle Größen im selben Zahlenbereich; eine 2 auf der
    y-Achse unten bedeutet 2 MW (Wind) bzw. 200 W (PV-Modul).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1.1, 1]})
    t = w.index

    ax1.plot(t, w["wind_speed_hub"], color="tab:green", lw=1.3, label="Wind in Nabenhöhe [m/s]")
    ax1.plot(t, w["temperature_2m"], color="tab:red", lw=1.3, label="Lufttemperatur [°C]")
    ax1.plot(t, w["global_tilted_irradiance"] / 100, color="tab:orange", lw=1.3,
             label="Einstrahlung Modulebene [100 W/m²]")
    ax1.axhline(0, color="grey", lw=0.6)
    ax1.set_ylabel("Wert in der Einheit\nder Legende")
    ax1.set_title(title, pad=22)
    week_axis(ax1, t)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    ax2.plot(t, w["wind_kw"] / 1000, color="tab:green", lw=1.3, label="Windenergieanlage V90 [MW]")
    ax2.plot(t, w["pv_module_w"] / 100, color="tab:orange", lw=1.3, label="PV-Modul 460 W [100 W]")
    ax2.set_ylabel("Leistung [MW] bzw. [100 W]")
    ax2.set_ylim(0, max(2.1, (w["pv_module_w"] / 100).max() * 1.05))
    week_axis(ax2, t, label_days=False)
    ax2.set_xlabel("Uhrzeit [h]")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Netz
# ---------------------------------------------------------------------------

def component_overview(net):
    """Anzahl der Elemente je pandapower-Tabelle (nur nicht leere Tabellen)."""
    tables = ["bus", "line", "trafo", "trafo3w", "switch", "ext_grid", "load", "sgen", "gen",
              "storage", "shunt"]
    descr = {"bus": "Sammelschienen / Knoten", "line": "Leitungen", "trafo": "Zweiwicklungstransformatoren",
             "trafo3w": "Dreiwicklungstransformatoren", "switch": "Schalter", "ext_grid": "externes Netz (Slack)",
             "load": "Lasten", "sgen": "statische Erzeuger", "gen": "Generatoren (PV-Knoten)",
             "storage": "Speicher", "shunt": "Querelemente"}
    rows = [{"Tabelle": f"net.{t}", "Element": descr[t], "Anzahl": len(net[t])} for t in tables if len(net[t])]
    return pd.DataFrame(rows).set_index("Tabelle")


def table(df, decimals=1, cmap=None, vmin=None, vmax=None, int_cols=None):
    """
    Tabelle für die Anzeige im Notebook.

    Mit installiertem jinja2 wird ein formatierter pandas-Styler zurückgegeben (feste
    Nachkommastellen, optional farbig hinterlegt mit cmap/vmin/vmax). Ohne jinja2 wird die
    Tabelle gerundet als normaler DataFrame zurückgegeben, damit das Notebook trotzdem läuft.

    decimals : Nachkommastellen aller Zahlenspalten
    int_cols : Spalten ohne Nachkommastellen (z. B. Stundenzahlen)
    """
    import importlib.util
    int_cols = list(int_cols or [])
    num = [c for c in df.select_dtypes("number").columns if c not in int_cols]
    if importlib.util.find_spec("jinja2") is None:
        out = df.copy()
        out[num] = out[num].round(decimals)
        for c in int_cols:
            out[c] = out[c].round().astype("Int64")
        return out
    styler = df.style.format(f"{{:.{decimals}f}}", subset=num)
    if int_cols:
        styler = styler.format("{:.0f}", subset=int_cols)
    if cmap is not None:
        styler = styler.background_gradient(cmap=cmap, vmin=vmin, vmax=vmax, axis=None, subset=num)
    return styler
