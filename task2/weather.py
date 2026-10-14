"""
Wetterdaten (Open-Meteo) abrufen und in PV- und Windleistung umrechnen.

Ablauf
------
1. fetch_weather():        Abruf eines ganzen Jahres (stündlich) über die Open-Meteo
                           Historical Weather API, Speicherung als CSV (Rohdaten).
2. pv_power(), wind_power(): Umrechnung über die Kennlinien der gewählten Anlagen.
3. select_typical_weeks(): Wahl einer typischen Woche (Mo-So) je Monat.
4. build_weather_weeks():  Zusammenstellen der 12 Wochen und Speichern als CSV.

Aufruf als Programm (einmalig, benötigt Internet):
    python weather.py

Im Notebook werden danach nur noch die CSV-Dateien geladen (load_weather_weeks),
die Open-Meteo-Bibliotheken werden dort nicht benötigt.

Zeitkonvention
--------------
Alle Zeitstempel in der Ausgabe sind lokale Zeit (Europe/Berlin, ohne Zeitzonenangabe)
und bezeichnen den BEGINN der Stunde. Der Wert um 12:00 gilt also für 12:00-13:00.
Die Einstrahlung liefert Open-Meteo als Mittelwert der VORANGEGANGENEN Stunde; sie wird
deshalb um eine Stunde nach vorne verschoben. Wochen mit Zeitumstellung werden bei der
Auswahl übersprungen.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from profiles import Month


# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------

LATITUDE = 51.2           # Standort (Breite)  – bei Bedarf anpassen
LONGITUDE = 7.34          # Standort (Länge)
YEAR = 2025               # Wetterjahr
TIMEZONE = "Europe/Berlin"

PV_TILT_DEG = 35          # Modulneigung (0° = waagerecht)
PV_AZIMUTH_DEG = 0        # Ausrichtung (0° = Süd, -90° = Ost, 90° = West)

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Anlagendaten
# ---------------------------------------------------------------------------

# PV-Modul Trina Solar Vertex S+ TSM-NEG9RC.27, 460 W
#   P_STC: Produktseite trinasolar.com/de/NEG9RC.27 (Leistungsklasse 460 W+)
#   gamma, NOCT, Abmessungen: Datenblatt "Vertex S+ NEG9RC.27", Version TSM_DE_2024_B
#   (das Datenblatt listet die Leistungsklassen 425-450 W; die Temperaturwerte gelten
#   für die gesamte Modellfamilie und werden für 460 W übernommen)
PV_MODULE = {
    "name": "Trina Solar Vertex S+ TSM-NEG9RC.27 460 W",
    "p_stc_w": 460.0,          # Nennleistung bei STC (1000 W/m², 25 °C) [W]
    "gamma_pmax": -0.0029,     # Temperaturkoeffizient Pmax [1/K]  (-0,29 %/K)
    "noct_c": 43.0,            # NOCT [°C] (800 W/m², 20 °C Luft, 1 m/s Wind)
    "area_m2": 1.762 * 1.134,  # Modulfläche [m²]
}
PV_SYSTEM_EFFICIENCY = 0.86    # Wechselrichter, Leitungen, Verschmutzung, Mismatch (Annahme)

# Windenergieanlage Vestas V90-2.0 MW
#   Kenndaten: wind-turbine-models.com/turbines/16-vestas-v90
#   Kennlinie: OEDB-Turbinenbibliothek (windpowerlib), Eintrag "V90/2000",
#              Quelle dort ebenfalls wind-turbine-models.com
#   Oberhalb 16,5 m/s bis zur Abschaltgeschwindigkeit: Nennleistung
WIND_TURBINE = {
    "name": "Vestas V90-2.0 MW",
    "p_rated_kw": 2000.0,
    "rotor_diameter_m": 90.0,
    "hub_height_m": 105.0,     # wählbar: 80 / 95 / 105 m
    "v_cut_out": 25.0,         # Abschaltgeschwindigkeit [m/s]
    # Windgeschwindigkeit [m/s] -> Leistung [kW], Luftdichte 1,225 kg/m³
    "power_curve": {
        0.0: 0.0, 3.0: 0.0, 3.5: 42.2, 4.0: 93.3, 4.5: 145.2, 5.0: 211.3,
        5.5: 284.2, 6.0: 390.9, 6.5: 491.5, 7.0: 601.1, 7.5: 731.8,
        8.0: 884.5, 8.5: 1087.6, 9.0: 1247.1, 9.5: 1429.6, 10.0: 1594.3,
        10.5: 1742.9, 11.0: 1861.2, 11.5: 1948.1, 12.0: 1993.3, 12.5: 2000.0,
        25.0: 2000.0,
    },
}
WIND_MEASUREMENT_HEIGHT_M = 100.0   # Höhe der API-Windgeschwindigkeit
WIND_SHEAR_ALPHA = 1 / 7            # Exponent des Höhenprofils (Potenzgesetz)


# ---------------------------------------------------------------------------
# 1. Abruf
# ---------------------------------------------------------------------------

HOURLY_VARIABLES = ["temperature_2m", "global_tilted_irradiance", "shortwave_radiation", "wind_speed_100m"]


def fetch_weather(latitude=LATITUDE, longitude=LONGITUDE, year=YEAR,
                  tilt=PV_TILT_DEG, azimuth=PV_AZIMUTH_DEG, path=None, force=False):
    """
    Lädt ein ganzes Jahr stündlicher Wetterdaten und speichert sie als CSV.
    Existiert die CSV bereits, wird sie nur geladen (kein erneuter Abruf).

    Spalten: temperature_2m [°C], global_tilted_irradiance [W/m²],
             shortwave_radiation [W/m²], wind_speed_100m [m/s]
    Index:   Zeitstempel in UTC (Ende des Mittelungsintervalls, wie von der API geliefert)
    """
    path = Path(path) if path else DATA_DIR / f"weather_raw_{year}.csv"
    if path.exists() and not force:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    # Import erst hier, damit das Notebook ohne diese Bibliotheken läuft
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": HOURLY_VARIABLES,          # Reihenfolge wichtig für Variables(i)
        "tilt": tilt,
        "azimuth": azimuth,
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    response = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0]
    hourly = response.Hourly()

    index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    df = pd.DataFrame({name: hourly.Variables(i).ValuesAsNumpy() for i, name in enumerate(HOURLY_VARIABLES)},
                      index=index)
    df.index.name = "time_utc"

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"Wetterdaten gespeichert: {path} ({len(df)} Stunden, "
          f"{response.Latitude():.2f}°N {response.Longitude():.2f}°E, {response.Elevation():.0f} m)")
    return df


def to_local_hour_start(raw):
    """
    UTC-Rohdaten -> lokale Zeit (Europe/Berlin), Zeitstempel = Beginn der Stunde.
    Einstrahlungswerte (Mittel der vorangegangenen Stunde) werden um 1 h nach vorne
    verschoben, Momentanwerte (Temperatur, Wind) bleiben unverändert.
    """
    df = raw.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    for col in ["global_tilted_irradiance", "shortwave_radiation"]:
        df[col] = df[col].shift(-1)
    df.index = df.index.tz_convert(TIMEZONE)
    return df.dropna()


# ---------------------------------------------------------------------------
# 2. Umrechnung in Leistung
# ---------------------------------------------------------------------------

def pv_power(g_poa, t_air, module=PV_MODULE, system_eff=PV_SYSTEM_EFFICIENCY):
    """
    AC-Leistung je Modul [W] aus Einstrahlung auf die Modulebene g_poa [W/m²]
    und Lufttemperatur t_air [°C].

        T_Zelle = T_Luft + (NOCT - 20 °C) / 800 W/m² * G
        P       = P_STC * G / 1000 W/m² * (1 + gamma * (T_Zelle - 25 °C)) * eta_System
    """
    g = np.clip(np.asarray(g_poa, dtype=float), 0, None)
    t_cell = np.asarray(t_air, dtype=float) + (module["noct_c"] - 20.0) / 800.0 * g
    p = module["p_stc_w"] * g / 1000.0 * (1 + module["gamma_pmax"] * (t_cell - 25.0)) * system_eff
    return np.clip(p, 0, None)


def wind_at_hub(v_measured, hub_height=WIND_TURBINE["hub_height_m"],
                z_measured=WIND_MEASUREMENT_HEIGHT_M, alpha=WIND_SHEAR_ALPHA):
    """Höhenumrechnung mit dem Potenzgesetz: v_hub = v_mess * (h_hub / h_mess)^alpha."""
    return np.asarray(v_measured, dtype=float) * (hub_height / z_measured) ** alpha


def wind_power(v_hub, turbine=WIND_TURBINE):
    """Leistung [kW] aus der Windgeschwindigkeit in Nabenhöhe über die Kennlinie (linear interpoliert)."""
    v = np.asarray(v_hub, dtype=float)
    curve_v = np.array(list(turbine["power_curve"].keys()))
    curve_p = np.array(list(turbine["power_curve"].values()))
    p = np.interp(v, curve_v, curve_p)
    p[v >= turbine["v_cut_out"]] = 0.0          # Sturmabschaltung
    return p


def add_power(df, module=PV_MODULE, turbine=WIND_TURBINE):
    """Ergänzt PV- und Windleistung (absolut und normiert) im DataFrame."""
    out = df.copy()
    out["pv_module_w"] = pv_power(out["global_tilted_irradiance"], out["temperature_2m"], module)
    out["pv_kw_per_kwp"] = out["pv_module_w"] / module["p_stc_w"]        # = Leistung je kWp in kW
    out["wind_speed_hub"] = wind_at_hub(out["wind_speed_100m"], turbine["hub_height_m"])
    out["wind_kw"] = wind_power(out["wind_speed_hub"], turbine)
    out["wind_pu"] = out["wind_kw"] / turbine["p_rated_kw"]
    return out


# ---------------------------------------------------------------------------
# 3. Auswahl typischer Wochen
# ---------------------------------------------------------------------------

def _week_candidates(df_local, month):
    """Alle Montage, deren Woche (Mo 00:00 - So 23:00) vollständig im Monat liegt und 168 Stunden hat."""
    in_month = df_local[df_local.index.month == month.number]
    mondays = sorted({ts.normalize() for ts in in_month.index if ts.weekday() == 0})
    candidates = []
    for monday in mondays:
        week = df_local[(df_local.index >= monday) & (df_local.index < monday + pd.DateOffset(days=7))]
        if len(week) == 168 and (week.index.month == month.number).all():
            candidates.append(monday)
    return candidates


def select_typical_weeks(df_local, method="typical"):
    """
    Wählt je Monat eine Woche (Montag bis Sonntag).

    method="typical": die Woche, deren mittlere PV-Leistung und mittlere Windleistung
                      zusammen am wenigsten vom Monatsmittel abweichen
                      (Abweichungen jeweils relativ zum Monatsmittel, gleich gewichtet).
    method="first":   der erste vollständige Montag-Sonntag im Monat.

    Rückgabe: {Month: Timestamp des Montags 00:00 (lokale Zeit)}
    """
    starts = {}
    for month in Month:
        candidates = _week_candidates(df_local, month)
        if not candidates:
            raise ValueError(f"Keine vollständige Woche im {month.label} gefunden.")
        if method == "first":
            starts[month] = candidates[0]
            continue
        m = df_local[df_local.index.month == month.number]
        pv_mean, wind_mean = m["pv_kw_per_kwp"].mean(), m["wind_pu"].mean()

        def score(monday):
            w = df_local[(df_local.index >= monday) & (df_local.index < monday + pd.DateOffset(days=7))]
            return (abs(w["pv_kw_per_kwp"].mean() - pv_mean) / max(pv_mean, 1e-9)
                    + abs(w["wind_pu"].mean() - wind_mean) / max(wind_mean, 1e-9))

        starts[month] = min(candidates, key=score)
    return starts


# ---------------------------------------------------------------------------
# 4. Wochen zusammenstellen und speichern
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = ["month", "hour_of_week", "temperature_2m", "global_tilted_irradiance",
                  "wind_speed_100m", "wind_speed_hub", "pv_module_w", "pv_kw_per_kwp",
                  "wind_kw", "wind_pu"]


def build_weather_weeks(raw=None, method="typical", path=None, year=YEAR):
    """
    Erzeugt aus den Rohdaten 12 Wochen (je Monat eine) mit Wetter- und Leistungswerten
    und speichert sie als CSV.

    Spalten der CSV:
        time                      Beginn der Stunde, lokale Zeit (Europe/Berlin)
        month                     Monatsname des Enums Month (z. B. "JULY")
        hour_of_week              0 ... 167 (Mo 00:00 = 0)
        temperature_2m            Lufttemperatur [°C]
        global_tilted_irradiance  Einstrahlung auf die Modulebene [W/m²]
        wind_speed_100m           Windgeschwindigkeit in 100 m [m/s]
        wind_speed_hub            Windgeschwindigkeit in Nabenhöhe [m/s]
        pv_module_w               AC-Leistung je Modul [W]
        pv_kw_per_kwp             AC-Leistung je kWp installierter Leistung [kW/kWp]
        wind_kw                   Leistung einer V90-2.0 MW [kW]
        wind_pu                   Leistung bezogen auf die Nennleistung [-]
    """
    if raw is None:
        raw = fetch_weather(year=year)
    df_local = add_power(to_local_hour_start(raw))
    starts = select_typical_weeks(df_local, method)

    weeks = []
    for month, monday in starts.items():
        w = df_local[(df_local.index >= monday) & (df_local.index < monday + pd.DateOffset(days=7))].copy()
        w["month"] = month.name
        w["hour_of_week"] = np.arange(len(w))
        weeks.append(w)
    result = pd.concat(weeks)
    result.index = result.index.tz_localize(None)
    result.index.name = "time"
    result = result[OUTPUT_COLUMNS]

    path = Path(path) if path else DATA_DIR / f"weather_weeks_{year}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    result.round(4).to_csv(path)
    print(f"Typische Wochen gespeichert: {path}")
    for month, monday in starts.items():
        print(f"  {month.label:<10} {monday:%a %d.%m.%Y}")
    return result


def load_weather_weeks(path=None, year=YEAR):
    """Lädt die gespeicherten Wochen. Rückgabe: {Month: DataFrame mit 168 Stunden}."""
    path = Path(path) if path else DATA_DIR / f"weather_weeks_{year}.csv"
    df = pd.read_csv(path, index_col="time", parse_dates=True)
    return {Month[name]: g.drop(columns="month") for name, g in df.groupby("month", sort=False)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_power_curves(module=PV_MODULE, turbine=WIND_TURBINE):
    """Kennlinien: PV-Leistung über Einstrahlung (für drei Temperaturen) und Windkennlinie."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    g = np.linspace(0, 1100, 200)
    for t in [0, 15, 30]:
        ax1.plot(g, pv_power(g, np.full_like(g, t), module), label=f"Lufttemperatur {t} °C")
    ax1.set_title(f"PV-Modul: {module['name']}")
    ax1.set_xlabel("Einstrahlung auf Modulebene G [W/m²]")
    ax1.set_ylabel("AC-Leistung je Modul [W]")
    ax1.grid(True, alpha=0.4)
    ax1.legend()

    v = np.linspace(0, 27, 541)
    ax2.plot(v, wind_power(v, turbine), color="tab:green")
    ax2.plot(list(turbine["power_curve"].keys())[:-1], list(turbine["power_curve"].values())[:-1],
             "o", ms=3, color="tab:green", label="Stützstellen der Kennlinie")
    ax2.set_title(f"Windenergieanlage: {turbine['name']}")
    ax2.set_xlabel("Windgeschwindigkeit in Nabenhöhe [m/s]")
    ax2.set_ylabel("Leistung [kW]")
    ax2.grid(True, alpha=0.4)
    ax2.legend()
    fig.tight_layout()
    return fig


def plot_weather_week(weeks, month):
    """Einstrahlung, Wind und resultierende Leistungen einer Woche."""
    w = weeks[month]
    h = w["hour_of_week"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

    ax = axes[0]
    ax.plot(h, w["pv_kw_per_kwp"], color="tab:orange", label="PV [kW/kWp]")
    ax.plot(h, w["wind_pu"], color="tab:green", label="Wind [p.u. der Nennleistung]")
    ax.set_ylabel("Normierte Leistung [–]")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"Typische Woche {month.label} {w.index[0]:%d.%m.}–{w.index[-1]:%d.%m.%Y}")

    ax = axes[1]
    ax.plot(h, w["global_tilted_irradiance"], color="tab:orange", label="Einstrahlung Modulebene [W/m²]")
    ax.set_ylabel("Einstrahlung [W/m²]")
    ax2 = ax.twinx()
    ax2.plot(h, w["wind_speed_hub"], color="tab:green", label="Wind in Nabenhöhe [m/s]")
    ax2.set_ylabel("Windgeschwindigkeit [m/s]")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", bbox_to_anchor=(1.08, 1))

    for a in axes:
        a.set_xticks(range(0, 169, 24))
        a.set_xticks(range(0, 169, 6), minor=True)
        a.grid(True, which="major", alpha=0.5)
        a.grid(True, which="minor", alpha=0.15)
    axes[1].set_xticklabels(["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So", ""])
    axes[1].set_xlabel("Wochentag (Tagesbeginn 00:00)")
    axes[1].set_xlim(0, 168)
    fig.tight_layout()
    return fig


def plot_monthly_overview(weeks):
    """Mittlere PV- und Windleistung (normiert) der 12 typischen Wochen."""
    months = list(weeks)
    pv = [weeks[m]["pv_kw_per_kwp"].mean() for m in months]
    wind = [weeks[m]["wind_pu"].mean() for m in months]
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, pv, 0.4, color="tab:orange", label="PV: mittlere Leistung [kW/kWp]")
    ax.bar(x + 0.2, wind, 0.4, color="tab:green", label="Wind: mittlere Leistung [p.u.]")
    ax.set_xticks(x)
    ax.set_xticklabels([m.label for m in months], rotation=45)
    ax.set_ylabel("Mittlere normierte Leistung [–]")
    ax.set_title("Mittlere Erzeugung der typischen Wochen (Kapazitätsfaktor)")
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Programm
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw = fetch_weather()
    build_weather_weeks(raw)
