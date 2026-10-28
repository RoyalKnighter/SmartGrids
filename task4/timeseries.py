"""
Normierte Zeitreihen je Profil für die Simulation.

Für jede simulierte Woche (je Monat eine typische Wetterwoche, stündlich) wird eine Tabelle
mit einer Spalte je Profil erzeugt. Die Werte sind normiert und werden in der Simulation mit
der Skalierung des jeweiligen Elements multipliziert:

    HAUSHALT, GEWERBE, LANDWIRTSCHAFT,
    HAUSHALT_PV, HAUSHALT_PV_SPEICHER:  MW je kWh Jahresverbrauch    → × annual_kwh
    PV:                 MW je kWp                    → × kwp
    WIND:               Leistung / Nennleistung      → × p_mw (Nennleistung)
    FEST:               1                            → × p_mw
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import profiles as pr
import weather as wt
from elements import Profile
from profiles import Lastprofil, Month, DayType

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WEATHER_FILE = DATA_DIR / "weather_weeks_2025.csv"

_LOAD_PROFILE = {Profile.HAUSHALT: Lastprofil.HAUSHALT, Profile.GEWERBE: Lastprofil.GEWERBE,
                 Profile.LANDWIRTSCHAFT: Lastprofil.LANDWIRTSCHAFT, Profile.HAUSHALT_PV: Lastprofil.HAUSHALT_PV,
                 Profile.HAUSHALT_PV_SPEICHER: Lastprofil.HAUSHALT_PV_SPEICHER}


@lru_cache(maxsize=1)
def load_inputs(data_dir=str(DATA_DIR), weather_file=str(WEATHER_FILE)):
    """Lastprofile und Wetterwochen (einmal je Sitzung eingelesen)."""
    profiles = pr.load_profiles(Path(data_dir), profiles=tuple(_LOAD_PROFILE.values()))
    weather = wt.load_weather_weeks(Path(weather_file))
    return profiles, weather


def build_profile_table(months=None, weather_file=str(WEATHER_FILE)):
    """
    Normierte Zeitreihen aller Profile für die gewählten Monate.

    Rückgabe: DataFrame, Index = Zeitstempel (Beginn der Stunde, lokale Zeit), Spalten:
        month, hour_of_week, je Lastprofil eine Spalte, PV, WIND, FEST
    """
    profiles, weather = load_inputs(weather_file=weather_file)
    months = list(Month) if months is None else [months] if isinstance(months, Month) else list(months)
    parts = []
    for month in months:
        w = weather[month]
        start = w.index[0]
        df = pd.DataFrame(index=w.index)
        df["month"] = month.name
        df["hour_of_week"] = w["hour_of_week"].to_numpy()
        for prof, lp in _LOAD_PROFILE.items():
            week = pr.build_week(profiles, lp, month, annual_kwh=1, unit="MW", start=start, resolution="1h")
            df[prof.name] = week.reindex(w.index).to_numpy()
        df[Profile.PV.name] = w["pv_kw_per_kwp"].to_numpy() / 1000
        df[Profile.WIND.name] = w["wind_pu"].to_numpy()
        df[Profile.FEST.name] = 1.0
        parts.append(df)
    return pd.concat(parts)


@lru_cache(maxsize=8)
def annual_peak_mw_per_kwh(profile, year=2025):
    """
    Höchste Stundenleistung eines Jahres in MW je kWh Jahresverbrauch (alle BDEW-Lastprofile).
    Wird verwendet, um aus einer Höchstleistung (z. B. p_mw im CIGRE-Netz) den
    Jahresverbrauch zu berechnen: annual_kwh = p_max_mw / annual_peak_mw_per_kwh.

    Berücksichtigt alle Tage des Jahres mit Tagtyp (Feiertage NRW als FT) und, bei H25,
    die Dynamisierung.
    """
    profiles, _ = load_inputs()
    lp = _LOAD_PROFILE[profile]
    data = profiles[lp]
    holidays = set(wt.HOLIDAYS)
    peak = 0.0
    for day in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
        if day.date() in holidays or day.weekday() == 6:
            daytype = DayType.FT
        elif day.weekday() == 5:
            daytype = DayType.SA
        else:
            daytype = DayType.WT
        month = next(m for m in Month if m.number == day.month)
        p = data[(month, daytype)].to_numpy() * 4 / pr.NORM_KWH / 1000      # MW je kWh/a, viertelstündlich
        if lp in pr.DYNAMIC_PROFILES:
            p = p * pr.dynamization_factor(day.dayofyear)
        peak = max(peak, p.reshape(24, 4).mean(axis=1).max())                # Stundenmittel
    return peak


def annual_kwh_from_peak(profile, p_max_mw):
    """Jahresverbrauch [kWh/a], bei dem die höchste Stundenleistung des Profils p_max_mw beträgt."""
    return p_max_mw / annual_peak_mw_per_kwh(profile)
