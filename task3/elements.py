"""
Elementklassen und Aufzählungen für die Simulationssoftware.

Elemente werden über ihren Namen angesprochen. Busse werden in den Klassen ebenfalls
über den Busnamen angegeben (z. B. bus="Bus 5"), nicht über den pandapower-Index.

Beispiel:
    sim1.createElement(Load("Neubaugebiet", bus="Bus 5", profile=Profile.HAUSHALT, annual_kwh=350_000))
    sim1.createElement(Generator("PV Halle", bus="Bus 9", profile=Profile.PV, kwp=500))
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Aufzählungen
# ---------------------------------------------------------------------------

class Profile(Enum):
    """Zeitverlauf einer Last oder eines Erzeugers."""
    HAUSHALT = "Haushalt (BDEW H25)"
    GEWERBE = "Gewerbe (BDEW G25)"
    LANDWIRTSCHAFT = "Landwirtschaft (BDEW L25)"
    HAUSHALT_PV = "Haushalt mit PV, Netzbezug (BDEW P25)"
    HAUSHALT_PV_SPEICHER = "Haushalt mit PV und Speicher, Netzbezug (BDEW S25)"
    PV = "Photovoltaik (Wetter, Trina Vertex S+)"
    WIND = "Wind (Wetter, Vestas V90-2.0 MW)"
    FEST = "konstante Leistung"


# Lastprofile, die über einen Jahresverbrauch (annual_kwh) skaliert werden
ENERGY_PROFILES = {Profile.HAUSHALT, Profile.GEWERBE, Profile.LANDWIRTSCHAFT,
                   Profile.HAUSHALT_PV, Profile.HAUSHALT_PV_SPEICHER}
ENERGY_PROFILE_NAMES = {p.name for p in ENERGY_PROFILES}

# Welche Profile sind für welche Elementtabelle zulässig
LOAD_PROFILES = ENERGY_PROFILES | {Profile.FEST}
SGEN_PROFILES = {Profile.PV, Profile.WIND, Profile.FEST}

# Standard-Leistungsfaktoren
DEFAULT_COS_PHI = {Profile.HAUSHALT: 0.97, Profile.GEWERBE: 0.95, Profile.LANDWIRTSCHAFT: 0.95,
                   Profile.HAUSHALT_PV: 0.97, Profile.HAUSHALT_PV_SPEICHER: 0.97,
                   Profile.PV: 1.0, Profile.WIND: 1.0, Profile.FEST: None}


class Diagram(Enum):
    """Darstellbare Ergebnisgrößen."""
    VOLTAGE = "Spannung"
    LINE_LOADING = "Leitungsauslastung"
    TRAFO_LOADING = "Trafoauslastung"
    TRAFO_POWER = "Trafoleistung"
    ELEMENT_POWER = "Leistung von Lasten und Erzeugern"
    VOLTAGE_PROFILE = "Spannungsverlauf entlang eines Pfades"


# ---------------------------------------------------------------------------
# Elemente
# ---------------------------------------------------------------------------

@dataclass
class Bus:
    name: str
    vn_kv: float
    type: str = "b"
    zone: Optional[str] = None
    geo: Optional[tuple] = None          # (x, y)


@dataclass
class Line:
    """
    Leitung. Entweder std_type angeben oder alle Parameter
    (r_ohm_per_km, x_ohm_per_km, c_nf_per_km, max_i_ka).
    switches=True legt an beiden Enden einen Leistungsschalter an.
    """
    name: str
    from_bus: str
    to_bus: str
    length_km: float
    std_type: Optional[str] = None
    r_ohm_per_km: Optional[float] = None
    x_ohm_per_km: Optional[float] = None
    c_nf_per_km: Optional[float] = None
    max_i_ka: Optional[float] = None
    type: str = "cs"                     # "cs" Kabel, "ol" Freileitung
    parallel: int = 1
    switches: bool = False


@dataclass
class Trafo:
    """
    Zweiwicklungstransformator. Entweder std_type angeben oder die Parameter
    sn_mva, vn_hv_kv, vn_lv_kv, vk_percent, vkr_percent (weitere optional).
    """
    name: str
    hv_bus: str
    lv_bus: str
    std_type: Optional[str] = None
    sn_mva: Optional[float] = None
    vn_hv_kv: Optional[float] = None
    vn_lv_kv: Optional[float] = None
    vk_percent: Optional[float] = None
    vkr_percent: Optional[float] = None
    pfe_kw: float = 0.0
    i0_percent: float = 0.0
    shift_degree: float = 0.0
    tap_side: Optional[str] = None
    tap_neutral: Optional[int] = None
    tap_min: Optional[int] = None
    tap_max: Optional[int] = None
    tap_step_percent: Optional[float] = None
    tap_pos: Optional[int] = None


@dataclass
class ExtGrid:
    name: str
    bus: str
    vm_pu: float = 1.0
    va_degree: float = 0.0


@dataclass
class Switch:
    """Schalter an einer Leitung oder einem Trafo (element = Name der Leitung bzw. des Trafos)."""
    name: str
    bus: str
    element: str
    type: str = "CB"
    closed: bool = True


@dataclass
class Load:
    """
    Last mit Profil.
    HAUSHALT, GEWERBE, LANDWIRTSCHAFT, HAUSHALT_PV, HAUSHALT_PV_SPEICHER:
                      annual_kwh angeben, oder p_mw als Jahreshöchstleistung
                      (annual_kwh wird dann daraus berechnet)
    FEST:             p_mw (und q_mvar oder cos_phi)
    """
    name: str
    bus: str
    profile: Profile = Profile.FEST
    annual_kwh: Optional[float] = None
    p_mw: Optional[float] = None
    q_mvar: Optional[float] = None
    cos_phi: Optional[float] = None


@dataclass
class Generator:
    """
    Erzeuger (pandapower sgen) mit Profil.
    PV:   kwp (installierte Leistung), oder p_mw als Nennleistung
    WIND: p_mw = Nennleistung
    FEST: p_mw
    """
    name: str
    bus: str
    profile: Profile = Profile.FEST
    kwp: Optional[float] = None
    p_mw: Optional[float] = None
    cos_phi: Optional[float] = None
