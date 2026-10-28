"""
Simulationssoftware für Netzsimulationen mit Last- und Erzeugungsprofilen.

Nutzung (Kurzfassung, ausführlich in README.md):

    import simulation as sim
    from simulation import Profile, Diagram, Month

    sim1 = sim.init("cigre_basis", pn.create_cigre_network_mv(with_der="pv_wind"))
    sim1.assignProfilesByName()
    sim1.createElement(sim.Load("Neubau", bus="Bus 5", profile=Profile.HAUSHALT, annual_kwh=350_000))
    sim1.changeParameter("Line 1-2", "length_km", 3.5)
    sim1.runSimulation()
    sim1.createDiagram(Diagram.VOLTAGE, ["Bus 3", "Bus 11"], months=Month.JULY)

    sim1 = sim.loadSimulation("cigre_basis")
    sim2 = sim1.copy("cigre_variante")
"""

import copy
import datetime as dt
import difflib
import json
import re
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.toolbox as tb

import diagrams
import timeseries as tsm
from elements import (Profile, Diagram, Bus, Line, Trafo, ExtGrid, Switch, Load, Generator,
                      LOAD_PROFILES, SGEN_PROFILES, DEFAULT_COS_PHI, ENERGY_PROFILES, ENERGY_PROFILE_NAMES)
from profiles import Month

warnings.simplefilter(action="ignore", category=FutureWarning)

__all__ = ["init", "loadSimulation", "listSimulations", "deleteSimulation", "compare",
           "Simulation", "Profile", "Diagram", "Month",
           "Bus", "Line", "Trafo", "ExtGrid", "Switch", "Load", "Generator"]

BASE_DIR = Path(__file__).resolve().parent
SIM_DIR = BASE_DIR / "simulations"

# Tabellen, deren Elemente über Namen angesprochen werden
TABLES = ["bus", "line", "trafo", "trafo3w", "ext_grid", "switch", "load", "sgen", "gen",
          "storage", "shunt"]
TABLE_LABEL = {"bus": "Bus", "line": "Leitung", "trafo": "Trafo", "trafo3w": "Dreiwicklungstrafo",
               "ext_grid": "externes Netz", "switch": "Schalter", "load": "Last", "sgen": "Erzeuger",
               "gen": "Generator", "storage": "Speicher", "shunt": "Querelement"}
# Spalten, die Busse referenzieren (werden als Busnamen angezeigt)
BUS_COLUMNS = ["bus", "from_bus", "to_bus", "hv_bus", "lv_bus", "mv_bus"]
# Zusätzliche Spalten für Profile
PROFILE_COLUMNS = {"load": ["profile", "annual_kwh", "cos_phi"],
                   "sgen": ["profile", "kwp", "cos_phi"]}
# Spalten, die bei den Änderungsanzeigen ignoriert werden
IGNORE_COLUMNS = {"geo"}

# Nur rechnen, wenn sich etwas geändert hat:
#   sim.ONLY_RUN_IF_CHANGED = True
# runSimulation() vergleicht dann einen Fingerabdruck aus Netz, Einstellungen, Eingangsdaten und
# Programmversion mit dem des letzten gespeicherten Laufs dieser Simulation. Stimmt er überein,
# werden die gespeicherten Ergebnisse geladen, statt neu zu rechnen. runSimulation(force=True)
# rechnet immer.
ONLY_RUN_IF_CHANGED = False
CACHE_VERSION = "2"          # erhöhen, wenn sich Rechnung oder Ergebnisformat ändern
FINGERPRINT_FILE = "fingerprint.txt"

DEFAULT_SETTINGS = {
    "months": [m.name for m in Month],
    "v_min": 0.90,
    "v_max": 1.10,
    "tap": None,
}


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _data_fingerprint():
    """Hash der Eingangsdaten (Lastprofile, Wetter) im Ordner data."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted((BASE_DIR / "data").glob("*.csv")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


_DATA_FP = None


def _fingerprint(net, settings):
    """Fingerabdruck einer Rechnung: Netz (ohne Ergebnisse und Geodaten), Einstellungen, Daten, Version."""
    import hashlib
    global _DATA_FP
    if _DATA_FP is None:
        _DATA_FP = _data_fingerprint()
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode())
    h.update(_DATA_FP.encode())
    h.update(json.dumps(_jsonable(settings), sort_keys=True, default=str).encode())
    for key in sorted(net.keys()):
        df = net[key]
        if key.startswith(("res_", "_")) or not isinstance(df, pd.DataFrame) or not len(df):
            continue
        df = df.drop(columns=[c for c in ("geo",) if c in df.columns])
        h.update(key.encode())
        h.update(df.to_json(orient="split", double_precision=10, default_handler=str).encode())
    return h.hexdigest()


def _clear_folder(folder, keep_results):
    """Simulationsordner leeren; mit keep_results bleiben Ergebnisse und Fingerabdruck erhalten."""
    if not folder.exists():
        return
    if not keep_results:
        shutil.rmtree(folder)
        return
    for item in folder.iterdir():
        if item.name in ("results", FINGERPRINT_FILE):
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()

def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def _jsonable(value):
    """Wert für JSON und Anzeige aufbereiten."""
    if isinstance(value, Profile):
        return value.name
    if isinstance(value, Month):
        return value.name
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _equal(a, b):
    """Vergleich mit Toleranz und NaN-Behandlung."""
    a_na = a is None or (isinstance(a, float) and np.isnan(a)) or (a is pd.NA)
    b_na = b is None or (isinstance(b, float) and np.isnan(b)) or (b is pd.NA)
    try:
        a_na = a_na or bool(pd.isna(a))
        b_na = b_na or bool(pd.isna(b))
    except (TypeError, ValueError):
        pass
    if a_na or b_na:
        return a_na and b_na
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12))
    return a == b


def _tables(net):
    return [t for t in TABLES if t in net and len(net[t])]


def _ensure_names(net):
    """Fehlende Namen ergänzen und Namen über alle Tabellen eindeutig machen."""
    used = set()
    renamed = []
    for table in TABLES:
        if table not in net or not len(net[table]):
            continue
        if "name" not in net[table]:
            net[table]["name"] = None
        for idx in net[table].index:
            name = net[table].at[idx, "name"]
            if name is None or (isinstance(name, float) and np.isnan(name)) or str(name).strip() == "":
                name = f"{table}_{idx}"
            name = str(name)
            if name in used:
                base, k = name, 2
                while f"{base} ({table} {k})" in used:
                    k += 1
                new = f"{base} ({table} {k})" if k > 2 else f"{base} ({table})"
                if new in used:
                    new = f"{base} ({table} {idx})"
                renamed.append((name, new))
                name = new
            net[table].at[idx, "name"] = name
            used.add(name)
    return renamed


def _ensure_profile_columns(net):
    """Profilspalten anlegen; Elemente ohne Profil erhalten FEST."""
    for table, cols in PROFILE_COLUMNS.items():
        if table not in net:
            continue
        df = net[table]
        for col in cols:
            if col not in df:
                df[col] = None if col == "profile" else np.nan
        df["profile"] = df["profile"].astype(object)
        df.loc[df["profile"].isna(), "profile"] = Profile.FEST.name
        for col in cols[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)


def _profile(value):
    if isinstance(value, Profile):
        return value
    try:
        return Profile[str(value).upper()]
    except KeyError:
        raise ValueError(f"Unbekanntes Profil '{value}'. Möglich: {[p.name for p in Profile]}")


def _suggest(name, candidates):
    close = difflib.get_close_matches(str(name), [str(c) for c in candidates], n=5, cutoff=0.5)
    return f" Meinten Sie: {', '.join(close)}?" if close else ""


# ---------------------------------------------------------------------------
# Modulfunktionen
# ---------------------------------------------------------------------------

def init(name, net=None, overwrite=False, assignProfiles=False):
    """
    Neue Simulation anlegen.

    name           : Name der Simulation (zugleich Ordnername unter simulations/)
    net            : pandapower-Netz (wird kopiert) oder None für ein leeres Netz
    overwrite      : vorhandene Simulation mit diesem Namen überschreiben
    assignProfiles : True → Profile sofort über die Namensregeln zuordnen (assignProfilesByName);
                     die Zuordnung gehört dann zur Baseline und erscheint nicht als Änderung
    """
    _check_name(name)
    folder = SIM_DIR / name
    if folder.exists() and not overwrite:
        raise FileExistsError(f"Simulation '{name}' existiert bereits. "
                              f"sim.loadSimulation('{name}') zum Laden oder overwrite=True zum Überschreiben.")
    _clear_folder(folder, keep_results=ONLY_RUN_IF_CHANGED)
    net = pp.create_empty_network() if net is None else copy.deepcopy(net)
    renamed = _ensure_names(net)
    _ensure_profile_columns(net)
    if renamed:
        print(f"{len(renamed)} Elementnamen wurden eindeutig gemacht, z. B. {renamed[0][0]} → {renamed[0][1]}")
    s = Simulation(name, net)
    if assignProfiles:
        s.assignProfilesByName()
        s.baseline = copy.deepcopy(s.net)
        s.log = []
        s.meta["results_status"] = "keine"
    s.save()
    return s


def loadSimulation(name):
    """Gespeicherte Simulation laden."""
    folder = SIM_DIR / name
    if not (folder / "meta.json").exists():
        existing = [p.name for p in SIM_DIR.iterdir() if p.is_dir()] if SIM_DIR.exists() else []
        raise FileNotFoundError(f"Simulation '{name}' nicht gefunden." + _suggest(name, existing))
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    net = pp.from_json(str(folder / "net.json"))
    baseline = pp.from_json(str(folder / "baseline.json"))
    for n in (net, baseline):
        _ensure_profile_columns(n)
    results = None
    if (folder / "results").exists() and meta["meta"].get("results_status", "keine") != "keine":
        results = diagrams.Results.load(folder / "results")
    s = Simulation(name, net, baseline=baseline, settings=meta["settings"], log=meta["log"],
                   meta=meta["meta"], baseline_settings=meta.get("baseline_settings"), results=results)
    return s


def listSimulations():
    """Übersicht aller gespeicherten Simulationen."""
    rows = []
    if SIM_DIR.exists():
        for folder in sorted(SIM_DIR.iterdir()):
            f = folder / "meta.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"Name": folder.name, "Herkunft": m["meta"].get("parent") or "–",
                         "erstellt": m["meta"].get("created"), "geändert": m["meta"].get("modified"),
                         "letzter Lauf": m["meta"].get("last_run") or "–",
                         "Ergebnisse": m["meta"].get("results_status", "keine"),
                         "Änderungen": len(m["log"])})
    return pd.DataFrame(rows).set_index("Name") if rows else pd.DataFrame()


def deleteSimulation(name, confirm=False):
    """Gespeicherte Simulation löschen (nur mit confirm=True)."""
    folder = SIM_DIR / name
    if not folder.exists():
        raise FileNotFoundError(f"Simulation '{name}' nicht gefunden.")
    if not confirm:
        print(f"Zum Löschen von '{name}' bitte sim.deleteSimulation('{name}', confirm=True) aufrufen.")
        return
    shutil.rmtree(folder)
    print(f"Simulation '{name}' gelöscht.")


def compare(simulations, diagram, elements, months=None, layout="week", **kwargs):
    """Dieselbe Größe aus mehreren Simulationen in einem Diagramm vergleichen."""
    for s in simulations:
        if s.results is None:
            raise RuntimeError(f"Simulation '{s.name}' hat keine Ergebnisse. Erst runSimulation() ausführen.")
    show = kwargs.pop("show", True)
    figs = diagrams.compare(simulations, diagram, elements, months=months, layout=layout, show=show, **kwargs)
    return None if show else figs


def _check_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_\-äöüÄÖÜß ]+", str(name)):
        raise ValueError("Der Name darf nur Buchstaben, Ziffern, Leerzeichen, '-' und '_' enthalten.")


# ---------------------------------------------------------------------------
# Klasse Simulation
# ---------------------------------------------------------------------------

class Simulation:
    """Eine Netzsimulation mit Netz, Baseline, Einstellungen, Änderungsprotokoll und Ergebnissen."""

    def __init__(self, name, net, baseline=None, settings=None, log=None, meta=None,
                 baseline_settings=None, results=None):
        self.name = name
        self.net = net
        self.baseline = copy.deepcopy(net) if baseline is None else baseline
        self.settings = copy.deepcopy(DEFAULT_SETTINGS) if settings is None else settings
        self.baseline_settings = copy.deepcopy(self.settings) if baseline_settings is None else baseline_settings
        self.log = [] if log is None else log
        now = _now()
        self.meta = meta or {"created": now, "modified": now, "parent": None, "last_run": None,
                             "results_status": "keine"}
        self.results = results

    # ------------------------------------------------------------------ Darstellung
    def __repr__(self):
        counts = ", ".join(f"{len(self.net[t])} {t}" for t in _tables(self.net))
        return (f"Simulation '{self.name}' ({counts})\n"
                f"  Herkunft: {self.meta.get('parent') or '–'} | Änderungen: {len(self.log)} | "
                f"Ergebnisse: {self.meta.get('results_status')} | letzter Lauf: {self.meta.get('last_run') or '–'}")

    # ------------------------------------------------------------------ Suche
    def _find(self, name, net=None):
        """(Tabelle, Index) eines Elements über seinen Namen."""
        net = self.net if net is None else net
        for table in _tables(net):
            hits = net[table].index[net[table]["name"] == name]
            if len(hits):
                return table, hits[0]
        all_names = [n for t in _tables(net) for n in net[t]["name"]]
        raise KeyError(f"Element '{name}' nicht gefunden." + _suggest(name, all_names))

    def _bus(self, bus):
        """Busindex aus Name (oder Index)."""
        if isinstance(bus, (int, np.integer)) and bus in self.net.bus.index:
            return int(bus)
        table, idx = self._find(bus)
        if table != "bus":
            raise ValueError(f"'{bus}' ist kein Bus, sondern {TABLE_LABEL[table]}.")
        return int(idx)

    def _bus_name(self, idx, net=None):
        net = self.net if net is None else net
        try:
            return net.bus.at[int(idx), "name"]
        except (KeyError, ValueError, TypeError):
            return idx

    def _all_names(self):
        return {n for t in _tables(self.net) for n in self.net[t]["name"]}

    def _check_new_name(self, name):
        if name in self._all_names():
            raise ValueError(f"Ein Element mit dem Namen '{name}' existiert bereits.")

    # ------------------------------------------------------------------ Protokoll
    def _record(self, action, element=None, table=None, parameter=None, old=None, new=None):
        self.log.append({"zeit": _now(), "aktion": action, "element": element,
                         "tabelle": table, "parameter": parameter,
                         "alt": _jsonable(old), "neu": _jsonable(new)})
        self.meta["modified"] = _now()
        if self.results is not None:
            self.meta["results_status"] = "veraltet"

    def history(self):
        """Vollständiges Änderungsprotokoll in zeitlicher Reihenfolge."""
        if not self.log:
            return pd.DataFrame(columns=["zeit", "aktion", "element", "tabelle", "parameter", "alt", "neu"])
        return pd.DataFrame(self.log)

    # ------------------------------------------------------------------ Profile
    def _apply_profile(self, table, idx, profile, annual_kwh=None, kwp=None, p_mw=None, cos_phi=None):
        """Profil und Skalierung eines Elements setzen. p_mw ist der Bemessungs- bzw. Höchstwert."""
        profile = _profile(profile)
        df = self.net[table]
        allowed = LOAD_PROFILES if table == "load" else SGEN_PROFILES if table == "sgen" else set()
        if profile not in allowed:
            raise ValueError(f"Profil {profile.name} ist für {TABLE_LABEL.get(table, table)} nicht zulässig. "
                             f"Möglich: {sorted(p.name for p in allowed)}")
        old_p, old_q = float(df.at[idx, "p_mw"]), float(df.at[idx, "q_mvar"])
        p_ref = old_p if p_mw is None else float(p_mw)

        if cos_phi is None:
            cos_phi = df.at[idx, "cos_phi"] if "cos_phi" in df and not pd.isna(df.at[idx, "cos_phi"]) else None
        if cos_phi is None and profile in ENERGY_PROFILES and old_p > 0 and old_q != 0:
            cos_phi = old_p / np.hypot(old_p, old_q)          # Leistungsfaktor des Netzmodells beibehalten
        if cos_phi is None:
            cos_phi = DEFAULT_COS_PHI[profile]

        if profile in ENERGY_PROFILES:
            if annual_kwh is None:
                annual_kwh = df.at[idx, "annual_kwh"] if (p_mw is None and not pd.isna(df.at[idx, "annual_kwh"])) \
                    else tsm.annual_kwh_from_peak(profile, p_ref)
            df.at[idx, "annual_kwh"] = float(annual_kwh)
            df.at[idx, "p_mw"] = float(annual_kwh) * tsm.annual_peak_mw_per_kwh(profile)
        elif profile == Profile.PV:
            if kwp is None:
                kwp = df.at[idx, "kwp"] if (p_mw is None and not pd.isna(df.at[idx, "kwp"])) else p_ref * 1000
            df.at[idx, "kwp"] = float(kwp)
            df.at[idx, "p_mw"] = float(kwp) / 1000
        else:                                                   # WIND, FEST
            df.at[idx, "p_mw"] = p_ref

        df.at[idx, "profile"] = profile.name
        if cos_phi is None:                                     # FEST ohne cos φ: q_mvar unverändert
            df.at[idx, "cos_phi"] = np.nan
        else:
            df.at[idx, "cos_phi"] = float(cos_phi)
            df.at[idx, "q_mvar"] = df.at[idx, "p_mw"] * np.tan(np.arccos(float(cos_phi)))
        if table == "load" and profile not in ENERGY_PROFILES:
            df.at[idx, "annual_kwh"] = np.nan
        if table == "sgen" and profile != Profile.PV:
            df.at[idx, "kwp"] = np.nan

    def _profile_state(self, table, idx):
        df = self.net[table]
        return {c: _jsonable(df.at[idx, c]) for c in ["profile", "annual_kwh", "kwp", "cos_phi", "p_mw"] if c in df}

    def assignProfile(self, name, profile, annual_kwh=None, kwp=None, p_mw=None, cos_phi=None):
        """
        Profil eines Elements setzen.
        Ohne Skalierungsangabe wird das vorhandene p_mw als Jahreshöchstleistung (Haushalt,
        Gewerbe) bzw. Nennleistung (PV, Wind) verwendet.
        """
        table, idx = self._find(name)
        old = self._profile_state(table, idx)
        self._apply_profile(table, idx, profile, annual_kwh=annual_kwh, kwp=kwp, p_mw=p_mw, cos_phi=cos_phi)
        self._record("Profil zugeordnet", name, table, "profile", old, self._profile_state(table, idx))

    def assignProfiles(self, pattern, profile, table=None, **scaling):
        """Profil allen Elementen zuordnen, deren Name auf den regulären Ausdruck `pattern` passt."""
        tables = [table] if table else ["load", "sgen"]
        names = [n for t in tables if t in self.net for n in self.net[t]["name"] if re.search(pattern, str(n))]
        if not names:
            raise KeyError(f"Kein Element passt auf '{pattern}'.")
        for n in names:
            self.assignProfile(n, profile, **scaling)
        print(f"{len(names)} Elemente → {_profile(profile).name}")
        return names

    def assignProfilesByName(self, rules=None):
        """
        Profile über Namensmuster zuordnen. Standardregeln (CIGRE und Task-2-Ortsnetz):
            Last  'Load R…'    → HAUSHALT     Last 'HH …'   → HAUSHALT
            Last  'Load CI…'   → GEWERBE
            Erzeuger 'PV…' oder type PV → PV
            Erzeuger 'WKA…'/'WEA…' oder type WP → WIND
        Skalierung aus dem vorhandenen p_mw (Jahreshöchstleistung bzw. Nennleistung).
        rules: Liste von (Tabelle, Regex, Profile) für eigene Regeln
        """
        if rules is None:
            rules = [("load", r"^Load R\d", Profile.HAUSHALT), ("load", r"^HH ", Profile.HAUSHALT),
                     ("load", r"^Load CI\d", Profile.GEWERBE),
                     ("sgen", r"^PV", Profile.PV), ("sgen", r"^(WKA|WEA)", Profile.WIND)]
        assigned = []
        for table, pattern, profile in rules:
            if table not in self.net:
                continue
            for idx, row in self.net[table].iterrows():
                if re.search(pattern, str(row["name"])):
                    self.assignProfile(row["name"], profile)
                    assigned.append((row["name"], profile.name))
        if "sgen" in self.net and "type" in self.net.sgen:
            done = {a[0] for a in assigned}
            for idx, row in self.net.sgen.iterrows():
                if row["name"] in done:
                    continue
                prof = {"PV": Profile.PV, "WP": Profile.WIND}.get(str(row["type"]))
                if prof:
                    self.assignProfile(row["name"], prof)
                    assigned.append((row["name"], prof.name))
        result = pd.DataFrame(assigned, columns=["Element", "Profil"])
        print(f"{len(result)} Profile zugeordnet: " +
              ", ".join(f"{k}: {v}" for k, v in result["Profil"].value_counts().items()))
        return result

    # ------------------------------------------------------------------ Elemente anlegen
    def createElement(self, element):
        """Element anlegen (Bus, Line, Trafo, ExtGrid, Switch, Load, Generator)."""
        self._check_new_name(element.name)
        net = self.net
        if isinstance(element, Bus):
            idx = pp.create_bus(net, vn_kv=element.vn_kv, name=element.name, type=element.type,
                                zone=element.zone, geodata=element.geo)
            table = "bus"
        elif isinstance(element, Line):
            fb, tb_ = self._bus(element.from_bus), self._bus(element.to_bus)
            if element.std_type:
                idx = pp.create_line(net, fb, tb_, element.length_km, element.std_type, name=element.name,
                                     parallel=element.parallel)
            else:
                missing = [p for p in ("r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka")
                           if getattr(element, p) is None]
                if missing:
                    raise ValueError(f"Ohne std_type fehlen: {missing}")
                idx = pp.create_line_from_parameters(
                    net, fb, tb_, element.length_km, element.r_ohm_per_km, element.x_ohm_per_km,
                    element.c_nf_per_km, element.max_i_ka, name=element.name, parallel=element.parallel)
            net.line.at[idx, "type"] = element.type
            table = "line"
            if element.switches:
                for bus, end in ((fb, "Start"), (tb_, "Ende")):
                    sname = f"CB {element.name} {end}"
                    self._check_new_name(sname)
                    pp.create_switch(net, bus=bus, element=idx, et="l", type="CB", closed=True, name=sname,
                                     in_ka=net.line.at[idx, "max_i_ka"])
                    self._record("Element hinzugefügt", sname, "switch")
        elif isinstance(element, Trafo):
            hv, lv = self._bus(element.hv_bus), self._bus(element.lv_bus)
            tap = {k: getattr(element, k) for k in ("tap_side", "tap_neutral", "tap_min", "tap_max",
                                                    "tap_step_percent", "tap_pos")
                   if getattr(element, k) is not None}
            if element.std_type:
                idx = pp.create_transformer(net, hv, lv, element.std_type, name=element.name,
                                            tap_pos=tap.get("tap_pos", np.nan))
                for k, v in tap.items():
                    net.trafo.at[idx, k] = v
            else:
                missing = [p for p in ("sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent")
                           if getattr(element, p) is None]
                if missing:
                    raise ValueError(f"Ohne std_type fehlen: {missing}")
                idx = pp.create_transformer_from_parameters(
                    net, hv, lv, element.sn_mva, element.vn_hv_kv, element.vn_lv_kv, element.vkr_percent,
                    element.vk_percent, element.pfe_kw, element.i0_percent,
                    shift_degree=element.shift_degree, name=element.name, **tap)
            table = "trafo"
        elif isinstance(element, ExtGrid):
            idx = pp.create_ext_grid(net, self._bus(element.bus), vm_pu=element.vm_pu,
                                     va_degree=element.va_degree, name=element.name)
            table = "ext_grid"
        elif isinstance(element, Switch):
            et_table, eidx = self._find(element.element)
            et = {"line": "l", "trafo": "t"}.get(et_table)
            if et is None:
                raise ValueError("Schalter können an Leitungen und Trafos angelegt werden.")
            idx = pp.create_switch(net, bus=self._bus(element.bus), element=eidx, et=et, type=element.type,
                                   closed=element.closed, name=element.name)
            table = "switch"
        elif isinstance(element, Load):
            idx = pp.create_load(net, self._bus(element.bus), p_mw=element.p_mw or 0.0,
                                 q_mvar=element.q_mvar or 0.0, name=element.name)
            _ensure_profile_columns(net)
            self._apply_profile("load", idx, element.profile, annual_kwh=element.annual_kwh,
                                p_mw=element.p_mw, cos_phi=element.cos_phi)
            table = "load"
        elif isinstance(element, Generator):
            idx = pp.create_sgen(net, self._bus(element.bus), p_mw=element.p_mw or 0.0, q_mvar=0.0,
                                 name=element.name,
                                 type={Profile.PV: "PV", Profile.WIND: "WP"}.get(element.profile, "SGEN"))
            _ensure_profile_columns(net)
            self._apply_profile("sgen", idx, element.profile, kwp=element.kwp, p_mw=element.p_mw,
                                cos_phi=element.cos_phi)
            table = "sgen"
        else:
            raise TypeError(f"Unbekannter Elementtyp: {type(element).__name__}")
        self._record("Element hinzugefügt", element.name, table)
        return self.getElement(element.name)

    # ------------------------------------------------------------------ Elemente löschen
    def _connected(self, bus_idx):
        names = []
        for table in _tables(self.net):
            if table == "bus":
                continue
            df = self.net[table]
            mask = pd.Series(False, index=df.index)
            for col in BUS_COLUMNS:
                if col in df:
                    mask |= df[col] == bus_idx
            names += [(n, table) for n in df.loc[mask, "name"]]
        return names

    def deleteElement(self, name, cascade=False):
        """
        Element löschen. Leitungen und Trafos: zugehörige Schalter werden mitgelöscht.
        Busse: nur mit cascade=True, dann werden alle angeschlossenen Elemente mitgelöscht.
        """
        table, idx = self._find(name)
        before = {(n, t) for t in _tables(self.net) for n in self.net[t]["name"]}
        if table == "bus":
            connected = self._connected(idx)
            if connected and not cascade:
                listing = ", ".join(f"{n} ({TABLE_LABEL[t]})" for n, t in connected)
                raise RuntimeError(f"An Bus '{name}' hängen noch {len(connected)} Elemente: {listing}. "
                                   f"Mit deleteElement('{name}', cascade=True) werden sie mitgelöscht.")
            tb.drop_buses(self.net, [idx], drop_elements=True)
        elif table == "line":
            tb.drop_lines(self.net, [idx])
        elif table in ("trafo", "trafo3w"):
            tb.drop_trafos(self.net, [idx], table=table)
        else:
            self.net[table] = self.net[table].drop(idx)
        after = {(n, t) for t in _tables(self.net) for n in self.net[t]["name"]}
        removed = sorted(before - after, key=lambda x: (x[0] != name, x[0]))
        for n, t in removed:
            self._record("Element gelöscht", n, t)
        if len(removed) > 1:
            print(f"Gelöscht: {', '.join(n for n, _ in removed)}")

    # ------------------------------------------------------------------ Parameter ändern
    def changeParameter(self, name, parameter, value):
        """
        Parameter eines Elements ändern.
        - pandapower-Spalten, z. B. ("Line 1-2", "length_km", 3.5), ("S1", "closed", True)
        - Busangaben als Namen, z. B. ("Load R3", "bus", "Bus 4")
        - std_type von Leitungen/Trafos (alle Typparameter werden übernommen)
        - Profilangaben: profile, annual_kwh, kwp, cos_phi, bei Profilelementen auch p_mw
        """
        table, idx = self._find(name)
        df = self.net[table]

        if table in PROFILE_COLUMNS and parameter in ("profile", "annual_kwh", "kwp", "cos_phi", "p_mw"):
            old = self._profile_state(table, idx)
            profile = _profile(value) if parameter == "profile" else Profile[df.at[idx, "profile"]]
            kw = {"annual_kwh": None, "kwp": None, "p_mw": None, "cos_phi": None}
            if parameter != "profile":
                kw[parameter] = value
            elif table == "load":
                kw["p_mw"] = float(df.at[idx, "p_mw"])        # Höchstleistung bleibt beim Profilwechsel
            else:
                kw["p_mw"] = float(df.at[idx, "p_mw"])
            if table == "load":
                kw.pop("kwp")
            else:
                kw.pop("annual_kwh")
            self._apply_profile(table, idx, profile, **kw)
            self._record("Parameter geändert", name, table, parameter, old.get(parameter),
                         self._profile_state(table, idx).get(parameter))
            return

        if parameter == "name":
            self._check_new_name(value)
        if parameter not in df.columns:
            raise KeyError(f"{TABLE_LABEL.get(table, table)} '{name}' hat keinen Parameter '{parameter}'."
                           + _suggest(parameter, df.columns))

        old = df.at[idx, parameter]
        if parameter in BUS_COLUMNS:
            old_disp, new_idx = self._bus_name(old), self._bus(value)
            df.at[idx, parameter] = new_idx
            self._record("Parameter geändert", name, table, parameter, old_disp, self._bus_name(new_idx))
            return
        if parameter == "element" and table == "switch":
            et_table, eidx = self._find(value)
            old_disp = self.net[{"l": "line", "t": "trafo"}[df.at[idx, "et"]]].at[old, "name"]
            df.at[idx, "element"] = eidx
            df.at[idx, "et"] = {"line": "l", "trafo": "t"}[et_table]
            self._record("Parameter geändert", name, table, parameter, old_disp, value)
            return
        if parameter == "std_type" and table in ("line", "trafo"):
            pp.change_std_type(self.net, idx, value, element=table)
        else:
            if isinstance(value, Profile):
                value = value.name
            df.at[idx, parameter] = value
        self._record("Parameter geändert", name, table, parameter, old, value)

    def changeSetting(self, key, value):
        """Simulationseinstellung ändern (months, v_min, v_max, tap)."""
        if key not in DEFAULT_SETTINGS:
            raise KeyError(f"Unbekannte Einstellung '{key}'. Möglich: {list(DEFAULT_SETTINGS)}")
        value = self._normalize_setting(key, value)
        old = self.settings.get(key)
        if _jsonable(old) == _jsonable(value):
            return
        self.settings[key] = value
        self._record("Einstellung geändert", None, "settings", key, old, value)

    @staticmethod
    def _normalize_setting(key, value):
        if key == "months":
            months = [value] if isinstance(value, (Month, str)) else list(value)
            return [m.name if isinstance(m, Month) else Month[str(m).upper()].name for m in months]
        if key == "tap":
            return _jsonable(value)
        return value

    # ------------------------------------------------------------------ Netze verbinden
    def attachNetwork(self, other, at_bus, prefix=None):
        """
        Netz anhängen (z. B. das Ortsnetz aus Task 2 an einen Bus des CIGRE-Netzes).
        Der Bus des externen Netzes von `other` wird mit `at_bus` vereinigt; das externe Netz
        von `other` entfällt. Beide Busse müssen dieselbe Nennspannung haben.
        prefix: wird allen Namen des angehängten Netzes vorangestellt (gegen Namensgleichheit).
        """
        other = copy.deepcopy(other)
        if len(other.ext_grid) != 1:
            raise ValueError("Das anzuhängende Netz muss genau ein externes Netz haben.")
        at_idx = self._bus(at_bus)
        slack_idx = int(other.ext_grid.bus.iloc[0])
        if not np.isclose(other.bus.at[slack_idx, "vn_kv"], self.net.bus.at[at_idx, "vn_kv"]):
            raise ValueError(f"Nennspannungen verschieden: {other.bus.at[slack_idx, 'vn_kv']} kV "
                             f"gegenüber {self.net.bus.at[at_idx, 'vn_kv']} kV.")
        other.ext_grid = other.ext_grid.iloc[0:0]
        _ensure_names(other)
        _ensure_profile_columns(other)
        if prefix:
            for t in _tables(other):
                other[t]["name"] = [f"{prefix} {n}" for n in other[t]["name"]]
        collisions = self._all_names() & {n for t in _tables(other) for n in other[t]["name"]}
        if collisions:
            raise ValueError(f"{len(collisions)} Namen gibt es in beiden Netzen (z. B. {sorted(collisions)[:3]}). "
                             f"Bitte prefix angeben.")
        slack_name = other.bus.at[slack_idx, "name"]
        merged = tb.merge_nets(self.net, other, validate=False, merge_results=False)
        slack_new = merged.bus.index[merged.bus["name"] == slack_name][0]
        tb.fuse_buses(merged, at_idx, slack_new)
        _ensure_profile_columns(merged)
        added = sorted({(n, t) for t in _tables(merged) for n in merged[t]["name"]}
                       - {(n, t) for t in _tables(self.net) for n in self.net[t]["name"]})
        self.net = merged
        self._record("Netz angehängt", f"{len(added)} Elemente", None, "at_bus", None, at_bus)
        for n, t in added:
            self._record("Element hinzugefügt", n, t)
        print(f"{len(added)} Elemente angehängt an '{at_bus}' (Bus '{slack_name}' mit '{at_bus}' vereinigt).")

    # ------------------------------------------------------------------ Abfragen
    def _display_row(self, net, table, idx):
        row = net[table].loc[idx].to_dict()
        for col in BUS_COLUMNS:
            if col in row and not pd.isna(row[col]):
                row[col] = self._bus_name(row[col], net)
        if table == "switch" and not pd.isna(row.get("element")):
            et_table = {"l": "line", "t": "trafo", "b": "bus", "t3": "trafo3w"}.get(row["et"])
            if et_table in net and row["element"] in net[et_table].index:
                row["element"] = net[et_table].at[row["element"], "name"]
        return row

    def getElement(self, name):
        """Steckbrief eines Elements mit allen Parametern und, falls vorhanden, Ergebnissen."""
        table, idx = self._find(name)
        info = {"Tabelle": f"net.{table}", "Index": int(idx)}
        info.update({k: v for k, v in self._display_row(self.net, table, idx).items() if k not in IGNORE_COLUMNS})
        if self.results is not None:
            info.update(self.results.element_summary(name, table))
            if self.meta.get("results_status") == "veraltet":
                info["Hinweis"] = "Ergebnisse veraltet (Netz seit dem letzten Lauf geändert)"
        return pd.Series(info, name=name, dtype=object)

    def elements(self, table=None):
        """Übersicht aller Elemente mit Typ, Bus(sen) und Profil."""
        rows = []
        for t in ([table] if table else _tables(self.net)):
            df = self.net[t]
            for idx, r in df.iterrows():
                buses = [self._bus_name(r[c]) for c in BUS_COLUMNS if c in df and not pd.isna(r[c])]
                prof, scale = "", ""
                if t in PROFILE_COLUMNS:
                    prof = r["profile"]
                    if prof in ENERGY_PROFILE_NAMES:
                        scale = f"{r['annual_kwh']:,.0f} kWh/a".replace(",", ".")
                    elif prof == "PV":
                        scale = f"{r['kwp']:,.1f} kWp".replace(",", ".")
                    else:
                        scale = f"{r['p_mw']:.3f} MW"
                rows.append({"Name": r["name"], "Typ": TABLE_LABEL.get(t, t), "Tabelle": t, "Index": idx,
                             "Bus": " – ".join(map(str, buses)), "Profil": prof, "Skalierung": scale})
        return pd.DataFrame(rows).set_index("Name")

    def changedElements(self):
        """Hinzugefügte und gelöschte Elemente gegenüber der Baseline."""
        rows = []
        for t in TABLES:
            now = set(self.net[t]["name"]) if t in self.net and len(self.net[t]) else set()
            base = set(self.baseline[t]["name"]) if t in self.baseline and len(self.baseline[t]) else set()
            rows += [{"Element": n, "Typ": TABLE_LABEL[t], "Änderung": "hinzugefügt"} for n in sorted(now - base)]
            rows += [{"Element": n, "Typ": TABLE_LABEL[t], "Änderung": "gelöscht"} for n in sorted(base - now)]
        return pd.DataFrame(rows, columns=["Element", "Typ", "Änderung"])

    def changedParameters(self):
        """Geänderte Parameter (und Einstellungen) gegenüber der Baseline."""
        rows = []
        for t in TABLES:
            if t not in self.net or t not in self.baseline or not len(self.net[t]) or not len(self.baseline[t]):
                continue
            cur, base = self.net[t], self.baseline[t]
            base_by_name = {n: i for i, n in zip(base.index, base["name"])}
            cols = [c for c in cur.columns if c not in IGNORE_COLUMNS and c != "name"]
            for idx, name in zip(cur.index, cur["name"]):
                if name not in base_by_name:
                    continue
                a = self._display_row(self.baseline, t, base_by_name[name])
                b = self._display_row(self.net, t, idx)
                for c in cols:
                    old, new = a.get(c, np.nan), b.get(c, np.nan)
                    if not _equal(old, new):
                        rows.append({"Element": name, "Typ": TABLE_LABEL[t], "Parameter": c,
                                     "alt": old, "neu": new})
        for key in DEFAULT_SETTINGS:
            old, new = self.baseline_settings.get(key), self.settings.get(key)
            if _jsonable(old) != _jsonable(new):
                rows.append({"Element": "–", "Typ": "Einstellung", "Parameter": key, "alt": old, "neu": new})
        return pd.DataFrame(rows, columns=["Element", "Typ", "Parameter", "alt", "neu"])

    # ------------------------------------------------------------------ Speichern / Kopieren
    def save(self, results=True):
        """Simulation speichern (Netz, Baseline, Einstellungen, Protokoll und, mit results=True, Ergebnisse)."""
        folder = SIM_DIR / self.name
        folder.mkdir(parents=True, exist_ok=True)
        pp.to_json(self.net, str(folder / "net.json"))
        pp.to_json(self.baseline, str(folder / "baseline.json"))
        meta = {"name": self.name, "meta": self.meta, "settings": _jsonable(self.settings),
                "baseline_settings": _jsonable(self.baseline_settings), "log": self.log}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
        if results and self.results is not None:
            self.results.save(folder / "results")
        return folder

    def copy(self, new_name, overwrite=False):
        """
        Kopie der Simulation unter neuem Namen. Die Baseline der Kopie ist der aktuelle Stand
        dieser Simulation; Ergebnisse werden nicht übernommen.
        """
        _check_name(new_name)
        if (SIM_DIR / new_name).exists() and not overwrite:
            raise FileExistsError(f"Simulation '{new_name}' existiert bereits (overwrite=True zum Überschreiben).")
        _clear_folder(SIM_DIR / new_name, keep_results=ONLY_RUN_IF_CHANGED)
        now = _now()
        s = Simulation(new_name, copy.deepcopy(self.net), baseline=copy.deepcopy(self.net),
                       settings=copy.deepcopy(self.settings), baseline_settings=copy.deepcopy(self.settings),
                       log=[], meta={"created": now, "modified": now, "parent": self.name, "last_run": None,
                                     "results_status": "keine"})
        s.save()
        return s

    # ------------------------------------------------------------------ Simulation
    def _check(self):
        problems = []
        if not len(self.net.ext_grid):
            problems.append("kein externes Netz (Slack) vorhanden")
        for t in ("load", "sgen"):
            if t not in self.net or not len(self.net[t]):
                continue
            df = self.net[t]
            for idx, r in df.iterrows():
                prof = r["profile"]
                if prof in ENERGY_PROFILE_NAMES and pd.isna(r["annual_kwh"]):
                    problems.append(f"{r['name']}: Jahresverbrauch fehlt")
                if prof == "PV" and pd.isna(r["kwp"]):
                    problems.append(f"{r['name']}: kWp fehlt")
        import pandapower.topology as top
        unsupplied = top.unsupplied_buses(self.net)
        if unsupplied:
            names = [self._bus_name(b) for b in list(unsupplied)[:5]]
            print(f"Hinweis: {len(unsupplied)} Busse sind nicht versorgt (z. B. {names}); ihre Ergebnisse sind NaN.")
        if problems:
            raise RuntimeError("Simulation nicht möglich:\n- " + "\n- ".join(problems))

    def _element_matrices(self, table, profile_table):
        """Wirk- und Blindleistung je Zeitschritt für alle Elemente einer Tabelle (T × n)."""
        df = self.net[table]
        if not len(df):
            return df.index.to_numpy(), np.zeros((len(profile_table), 0)), np.zeros((len(profile_table), 0))
        prof = df["profile"].to_numpy()
        scale = np.where(np.isin(prof, list(ENERGY_PROFILE_NAMES)), df.get("annual_kwh", np.nan),
                         np.where(prof == "PV", df.get("kwp", np.nan), df["p_mw"])).astype(float)
        base = profile_table[list(prof)].to_numpy()                      # T × n normierte Werte
        p = base * scale
        p[:, ~df["in_service"].to_numpy()] = 0.0
        cos_phi = df["cos_phi"].to_numpy(dtype=float)
        tan_phi = np.tan(np.arccos(np.where(np.isnan(cos_phi), 1.0, cos_phi)))
        q = p * tan_phi
        fixed_q = np.isnan(cos_phi)                                      # FEST ohne cos φ: q aus dem Netz
        if fixed_q.any():
            q[:, fixed_q] = df["q_mvar"].to_numpy()[fixed_q]
        return df.index.to_numpy(), p, q

    def _tap_schedule(self, tap):
        """{trafo_index: {Month-Name: tap}} aus der Einstellung."""
        if tap is None:
            return {}
        trafos = {i: n for i, n in zip(self.net.trafo.index, self.net.trafo["name"])}
        with_tap = [i for i in trafos if not pd.isna(self.net.trafo.at[i, "tap_pos"])]

        def per_month(v):
            if isinstance(v, dict):
                return {Month[str(k)].name if not isinstance(k, Month) else k.name: int(x) for k, x in v.items()}
            return {m.name: int(v) for m in Month}

        if isinstance(tap, dict) and tap and all(str(k).upper() in Month.__members__ or isinstance(k, Month)
                                                 for k in tap):
            return {i: per_month(tap) for i in with_tap}
        if isinstance(tap, dict):
            sched = {}
            for name, v in tap.items():
                t, i = self._find(name)
                if t != "trafo":
                    raise ValueError(f"'{name}' ist kein Trafo.")
                sched[i] = per_month(v)
            return sched
        return {i: per_month(tap) for i in with_tap}

    def runSimulation(self, months=None, tap=None, save=True, verbose=True, force=False):
        """
        Alle Zeitschritte rechnen (Standard: 12 typische Wochen, stündlich) und speichern.

        months : Monat oder Liste von Monaten (überschreibt die Einstellung 'months')
        tap    : Stufenschalter: int (alle Trafos mit Stufenschalter), {Month: int},
                 {"Trafoname": int} oder {"Trafoname": {Month: int}}
        force  : immer rechnen, auch wenn sim.ONLY_RUN_IF_CHANGED = True ist und sich nichts
                 geändert hat
        """
        if months is not None:
            self.changeSetting("months", months)
        if tap is not None:
            self.changeSetting("tap", tap)
        self._check()
        fingerprint = _fingerprint(self.net, self.settings)
        folder = SIM_DIR / self.name
        if ONLY_RUN_IF_CHANGED and not force and save:
            fp_file = folder / FINGERPRINT_FILE
            stored = {}
            if fp_file.exists():
                try:
                    stored = json.loads(fp_file.read_text())
                except ValueError:
                    stored = {}
            if stored.get("fingerprint") == fingerprint and (folder / "results").exists():
                self.results = diagrams.Results.load(folder / "results")
                self.meta["results_status"] = "aktuell"
                self.meta["last_run"] = stored.get("last_run")
                self.save(results=False)
                if verbose:
                    print(f"Simulation '{self.name}': unverändert seit dem Lauf vom {self.meta.get('last_run')} "
                          f"– gespeicherte Ergebnisse geladen ({int(self.results.converged.sum())} von "
                          f"{len(self.results.info)} Lastflüssen konvergiert).")
                return self.results
        months = [Month[m] for m in self.settings["months"]]
        profile_table = tsm.build_profile_table(months)
        schedule = self._tap_schedule(self.settings["tap"])

        net = copy.deepcopy(self.net)
        load_idx, p_load, q_load = self._element_matrices("load", profile_table)
        sgen_idx, p_sgen, q_sgen = self._element_matrices("sgen", profile_table)

        T = len(profile_table)
        bus_idx, line_idx, trafo_idx = net.bus.index, net.line.index, net.trafo.index
        vm = np.full((T, len(bus_idx)), np.nan)
        va = np.full((T, len(bus_idx)), np.nan)
        line_load = np.full((T, len(line_idx)), np.nan)
        line_i = np.full((T, len(line_idx)), np.nan)
        line_p = np.full((T, len(line_idx)), np.nan)
        line_pl = np.full((T, len(line_idx)), np.nan)
        trafo_pl = np.full((T, len(trafo_idx)), np.nan)
        trafo_load = np.full((T, len(trafo_idx)), np.nan)
        trafo_p = np.full((T, len(trafo_idx)), np.nan)
        ext_p = np.full(T, np.nan)
        converged = np.zeros(T, dtype=bool)

        t0 = time.time()
        init = "auto"
        month_col = profile_table["month"].to_numpy()
        for k in range(T):
            if len(load_idx):
                net.load.loc[load_idx, "p_mw"] = p_load[k]
                net.load.loc[load_idx, "q_mvar"] = q_load[k]
            if len(sgen_idx):
                net.sgen.loc[sgen_idx, "p_mw"] = p_sgen[k]
                net.sgen.loc[sgen_idx, "q_mvar"] = q_sgen[k]
            for ti, plan in schedule.items():
                net.trafo.at[ti, "tap_pos"] = plan[month_col[k]]
            try:
                pp.runpp(net, init=init)
                init = "results"
                converged[k] = True
                vm[k] = net.res_bus.vm_pu.reindex(bus_idx).to_numpy()
                va[k] = net.res_bus.va_degree.reindex(bus_idx).to_numpy()
                if len(line_idx):
                    line_load[k] = net.res_line.loading_percent.reindex(line_idx).to_numpy()
                    line_i[k] = net.res_line.i_ka.reindex(line_idx).to_numpy()
                    line_p[k] = net.res_line.p_from_mw.reindex(line_idx).to_numpy()
                    line_pl[k] = net.res_line.pl_mw.reindex(line_idx).to_numpy()
                if len(trafo_idx):
                    trafo_load[k] = net.res_trafo.loading_percent.reindex(trafo_idx).to_numpy()
                    trafo_p[k] = net.res_trafo.p_hv_mw.reindex(trafo_idx).to_numpy()
                    trafo_pl[k] = net.res_trafo.pl_mw.reindex(trafo_idx).to_numpy()
                ext_p[k] = net.res_ext_grid.p_mw.sum()
            except pp.LoadflowNotConverged:
                init = "auto"

        index = profile_table.index
        names = lambda tab, idx: list(self.net[tab].loc[idx, "name"]) if len(idx) else []
        tap_info = None
        if schedule:
            tap_info = [json.dumps({self.net.trafo.at[i, "name"]: plan[m] for i, plan in schedule.items()})
                        for m in month_col]
        info = profile_table[["month", "hour_of_week"]].copy()
        info["converged"] = converged
        info["tap"] = tap_info
        self.results = diagrams.Results(
            info=info,
            vm_pu=pd.DataFrame(vm, index=index, columns=names("bus", bus_idx)),
            va_degree=pd.DataFrame(va, index=index, columns=names("bus", bus_idx)),
            line_loading=pd.DataFrame(line_load, index=index, columns=names("line", line_idx)),
            line_i_ka=pd.DataFrame(line_i, index=index, columns=names("line", line_idx)),
            line_p_mw=pd.DataFrame(line_p, index=index, columns=names("line", line_idx)),
            line_pl_mw=pd.DataFrame(line_pl, index=index, columns=names("line", line_idx)),
            trafo_pl_mw=pd.DataFrame(trafo_pl, index=index, columns=names("trafo", trafo_idx)),
            trafo_loading=pd.DataFrame(trafo_load, index=index, columns=names("trafo", trafo_idx)),
            trafo_p_mw=pd.DataFrame(trafo_p, index=index, columns=names("trafo", trafo_idx)),
            ext_grid_p_mw=pd.DataFrame({"ext_grid_p_mw": ext_p}, index=index),
            load_p_mw=pd.DataFrame(p_load, index=index, columns=names("load", load_idx)),
            sgen_p_mw=pd.DataFrame(p_sgen, index=index, columns=names("sgen", sgen_idx)),
        )
        self.meta["last_run"] = _now()
        self.meta["results_status"] = "aktuell"
        if verbose:
            bad = T - converged.sum()
            print(f"Simulation '{self.name}': {T} Lastflüsse in {time.time() - t0:.0f} s, "
                  f"nicht konvergiert: {bad}" + (" (als NaN gespeichert)" if bad else ""))
        if save:
            self.save()
            (folder / FINGERPRINT_FILE).write_text(json.dumps({"fingerprint": fingerprint,
                                                               "last_run": self.meta["last_run"]}))
        return self.results

    # ------------------------------------------------------------------ Auswertung
    def _require_results(self):
        if self.results is None:
            raise RuntimeError("Keine Ergebnisse vorhanden. Erst runSimulation() ausführen.")
        if self.meta.get("results_status") == "veraltet":
            print("Hinweis: Das Netz wurde seit dem letzten Lauf geändert – die Ergebnisse sind veraltet.")

    def summary(self, by=None):
        """
        Kennzahlen des letzten Laufs.
        by=None: Gesamtübersicht; by="month": Tabelle je Monat
        """
        self._require_results()
        return self.results.summary(self.settings.get("v_min", 0.9), self.settings.get("v_max", 1.1), by=by)

    def powerFlow(self):
        """
        Leistungsfluss je Stunde [MW]: Bezug aus dem externen Netz, Trafos (OS), Lasten und
        Erzeugung gesamt und je Profil sowie Verluste (= Bezug + Erzeugung − Lasten).
        """
        self._require_results()
        return self.results.power_flow(self.net)

    def energyBalance(self):
        """Energiebilanz je simulierter Woche (Monat) in MWh, Anteil der Erzeugung und Stunden Rückspeisung."""
        self._require_results()
        return self.results.energy_balance(self.net)

    def createDiagram(self, diagram, elements=None, months=None, layout="week", time=None,
                      title=None, limits=True, save=False, show=True, markers=None, labels=True):
        """
        Diagramm erzeugen.

        diagram : Diagram.VOLTAGE, LINE_LOADING, TRAFO_LOADING, TRAFO_POWER, ELEMENT_POWER,
                  VOLTAGE_PROFILE, POWER_FLOW, NETWORK
                  POWER_FLOW: Bezug externes Netz, Trafos, Lasten und Erzeugung je Profil, Verluste;
                  bei layout="week" zweiteilig (oben Netz und Lasten, unten Erzeugung und Verluste).
                  elements wählt Spalten aus sim.powerFlow() aus, z. B. ["Bezug externes Netz",
                  "Lasten gesamt", "Erzeugung PV"]
        elements: Liste von Namen (Busse, Leitungen, Trafos, Lasten/Erzeuger); None = alle
        months  : Monat oder Liste von Monaten (Standard: alle simulierten)
        layout  : "week" (ein Diagramm je Monat), "season" (je Jahreszeit), "overview"
                  (alle Wochen hintereinander), "boxplot" (Verteilung je Monat)
        time    : VOLTAGE_PROFILE: Zeitpunkt (Timestamp/str), "min" oder "max"
                  NETWORK: Zeitpunkt, "min", "max" (Busspannung), "max_loading" (Leitung),
                  "max_trafo" oder eine Liste davon; mit months und Stichwort je Monat ein Plan
        labels  : nur NETWORK: Namen und Werte neben die Komponenten schreiben
                  (bei NETWORK blendet elements=False die Lasten und Erzeuger aus)
        save    : Diagramme als PNG im Simulationsordner (diagrams/) speichern
        show    : Diagramme anzeigen; mit show=False werden die Figures zurückgegeben
        markers : Liste von Markierungen, je (farbe, wert, text), z. B.
                  [("orange", 0.95, "Warnschwelle"), ("red", 1.05, "obere Grenze")]
                  Zahl → waagerechte Linie; Zeitpunkt (z. B. "2025-07-23 13:00") → senkrechte Linie.
                  Optional 4. Element: Linienstil ("-", "--", ":", "-.")
        """
        self._require_results()
        out_dir = (SIM_DIR / self.name / "diagrams") if save else None
        figs = diagrams.create(self, diagram, elements, months=months, layout=layout, time=time,
                               title=title, limits=limits, out_dir=out_dir, show=show, markers=markers,
                               labels=labels)
        return None if show else figs
