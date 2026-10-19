"""
Zeitreihen an das Ortsnetz binden (Aufgabe 3) und zeitschrittweise AC-Lastflüsse
rechnen (Aufgabe 4).

Ablauf
------
1. build_timeseries(): erzeugt für jeden Monat eine Tabelle (168 Stunden) mit der Leistung
   je Haushalt, je kWp PV und der Windenergieanlage. Last- und Wetterwoche liegen auf
   denselben Kalendertagen.
2. check_timeseries(): prüft Zeitstempel, Vollständigkeit und Energiesummen.
3. run_simulation(): setzt in jedem Zeitschritt die Leistungen der Elemente, rechnet den
   Lastfluss, prüft die Konvergenz und speichert die Ergebnisse.

Vorzeichen (pandapower)
-----------------------
load: Verbraucherzählpfeilsystem, p_mw > 0 = Bezug
sgen: Erzeugerzählpfeilsystem,    p_mw > 0 = Einspeisung
res_trafo.p_hv_mw > 0: Leistung fließt vom MS- ins NS-Netz; < 0: Rückspeisung ins MS-Netz
"""

import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp

import profiles as pr
from profiles import Lastprofil, Month, Season


# ---------------------------------------------------------------------------
# Aufgabe 3: Zeitreihen
# ---------------------------------------------------------------------------

def build_timeseries(profiles, weather, annual_kwh, cos_phi_load):
    """
    Erzeugt die Zeitreihen aller Monate.

    Rückgabe: DataFrame (12 × 168 Zeilen), Index = Zeitstempel (Beginn der Stunde), Spalten:
        month, hour_of_week,
        p_hh_mw, q_hh_mvar   Wirk-/Blindleistung je Haushalt
        pv_mw_per_kwp        PV-Leistung je kWp
        wind_mw              Leistung der Windenergieanlage
    """
    tan_phi = np.tan(np.arccos(cos_phi_load))
    parts = []
    for month, w in weather.items():
        p_hh = pr.build_week(profiles, Lastprofil.HAUSHALT, month, annual_kwh=annual_kwh,
                             unit="MW", start=w.index[0], resolution="1h")
        df = pd.DataFrame({
            "month": month.name,
            "hour_of_week": w["hour_of_week"].to_numpy(),
            "p_hh_mw": p_hh.reindex(w.index).to_numpy(),
            "pv_mw_per_kwp": w["pv_kw_per_kwp"].to_numpy() / 1000,
            "wind_mw": w["wind_kw"].to_numpy() / 1000,
        }, index=w.index)
        df["q_hh_mvar"] = df["p_hh_mw"] * tan_phi
        parts.append(df)
    return pd.concat(parts)


def element_table(households, pv, idx, annual_kwh, cos_phi_load):
    """Zuordnung Element -> Bus -> Profil -> Skalierung (zur Dokumentation)."""
    rows = [{"Element": "load", "Index": l, "Bus": households.at[l, "bus"], "Name": households.at[l, "name"],
             "Profil": "H25 Haushalt", "Skalierung": f"{annual_kwh:.0f} kWh/a, cos φ = {cos_phi_load}"}
            for l in households.index]
    rows += [{"Element": "sgen", "Index": s, "Bus": pv.at[s, "bus"], "Name": f"PV an {households.at[pv.at[s, 'load'], 'name']}",
              "Profil": "PV (Wetter)", "Skalierung": f"{pv.at[s, 'kwp']:.2f} kWp, cos φ = 1"}
             for s in pv.index]
    if "wind" in idx:
        rows.append({"Element": "sgen", "Index": idx["wind"], "Bus": idx["bus1"], "Name": "WEA",
                     "Profil": "Wind (Wetter)", "Skalierung": "1 × V90-2.0 MW, cos φ = 1"})
    return pd.DataFrame(rows)


def check_timeseries(ts, households, pv, net, idx):
    """Plausibilitätsprüfungen vor der Netzeinbindung. Gibt eine Tabelle der Prüfungen zurück."""
    checks = []

    def add(name, ok, detail=""):
        checks.append({"Prüfung": name, "Ergebnis": "ok" if ok else "FEHLER", "Details": detail})

    add("12 Monate vorhanden", ts["month"].nunique() == 12, f"{ts['month'].nunique()} Monate")
    sizes = ts.groupby("month").size()
    add("je Monat 168 Stunden", (sizes == 168).all(), f"{sizes.min()}–{sizes.max()} Stunden")
    steps_ok = all((g.index[1:] - g.index[:-1] == pd.Timedelta("1h")).all() for _, g in ts.groupby("month"))
    add("lückenlose Stundenschritte", steps_ok)
    add("jede Woche beginnt montags 00:00",
        all(g.index[0].weekday() == 0 and g.index[0].hour == 0 for _, g in ts.groupby("month")))
    add("keine fehlenden Werte", not ts.isna().any().any())
    add("Leistungen nicht negativ", (ts[["p_hh_mw", "pv_mw_per_kwp", "wind_mw"]] >= 0).all().all())
    add("jeder Haushalt hat ein load-Element", households.index.isin(net.load.index).all(),
        f"{len(households)} Haushalte")
    add("jede PV-Anlage hängt am Bus ihres Haushalts",
        (pv.bus.to_numpy() == households.loc[pv.load, "bus"].to_numpy()).all() if len(pv) else True,
        f"{len(pv)} PV-Anlagen")
    if "wind" in idx:
        add("WEA an Bus 1 (20 kV)", net.sgen.at[idx["wind"], "bus"] == idx["bus1"])
    return pd.DataFrame(checks)


def energy_summary(ts, n_households, total_kwp):
    """Energie je Woche [kWh]: Last, PV, Wind und Residuallast des NS-Netzes."""
    rows = []
    for name, g in ts.groupby("month", sort=False):
        load = g["p_hh_mw"].sum() * n_households * 1000          # 1 h je Zeitschritt
        pv = g["pv_mw_per_kwp"].sum() * total_kwp * 1000
        rows.append({"Monat": Month[name].label,
                     "Last [kWh]": load, "PV [kWh]": pv, "Wind [kWh]": g["wind_mw"].sum() * 1000,
                     "Residuallast NS [kWh]": load - pv,
                     "Stunden mit PV > Last": int(((g["pv_mw_per_kwp"] * total_kwp) > (g["p_hh_mw"] * n_households)).sum())})
    return pd.DataFrame(rows).set_index("Monat")


def plot_timeseries(ts, n_households, total_kwp, months=(Month.JANUARY, Month.APRIL, Month.JULY, Month.OCTOBER)):
    """Gesamtlast, PV-Erzeugung und Residuallast des NS-Netzes sowie Windleistung je Monat."""
    fig, axes = plt.subplots(len(months), 1, figsize=(12, 2.6 * len(months)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, month in zip(axes, months):
        g = ts[ts["month"] == month.name]
        h = g["hour_of_week"]
        load = g["p_hh_mw"] * n_households * 1000
        pvp = g["pv_mw_per_kwp"] * total_kwp * 1000
        ax.plot(h, load, label="Last Haushalte [kW]", color="tab:blue")
        ax.plot(h, pvp, label="PV-Erzeugung [kW]", color="tab:orange")
        ax.plot(h, load - pvp, label="Residuallast NS (Last − PV) [kW]", color="black", lw=1)
        ax.plot(h, g["wind_mw"] * 1000 / 10, label="Wind [kW] / 10", color="tab:green", lw=1, ls="--")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{month.label}: Woche ab {g.index[0]:%d.%m.%Y}", fontsize=10)
        ax.set_ylabel("Leistung [kW]")
        ax.grid(True, alpha=0.4)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    axes[-1].set_xticks(range(0, 169, 24))
    axes[-1].set_xticklabels(["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So", ""])
    axes[-1].set_xlim(0, 168)
    axes[-1].set_xlabel("Wochentag (Tagesbeginn 00:00)")
    fig.suptitle("Zeitreihen vor der Netzeinbindung (Residuallast < 0: Rückspeisung aus dem NS-Netz)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Aufgabe 4: zeitschrittweise Lastflussrechnung
# ---------------------------------------------------------------------------

def run_simulation(net, idx, households, pv, ts, tap_pos=None, verbose=True):
    """
    Rechnet für jeden Zeitschritt einen AC-Lastfluss.

    tap_pos: None -> Tap-Position des Netzes unverändert lassen,
             int  -> feste Tap-Position,
             dict {Month: int} -> Tap-Position je Monat (für Aufgabe 5)

    Rückgabe: DataFrame mit einer Zeile je Zeitschritt. Nicht konvergierte Zeitschritte haben
              converged = False und NaN in allen Ergebnisspalten.
    """
    net = copy.deepcopy(net)                    # Ausgangsnetz bleibt unverändert
    loads = households.index.to_numpy()
    sgens_pv = pv.index.to_numpy()
    kwp = pv["kwp"].to_numpy() if len(pv) else np.array([])
    lv_buses = net.bus.index[net.bus.vn_kv < 1].to_numpy()
    trafo, eval_bus = idx["trafo"], idx["eval_bus"]
    has_wind = "wind" in idx

    rows = []
    t0 = time.time()
    first = True
    for ts_time, step in ts.iterrows():
        month = Month[step["month"]]
        if tap_pos is not None:
            net.trafo.at[trafo, "tap_pos"] = tap_pos[month] if isinstance(tap_pos, dict) else tap_pos

        net.load.loc[loads, "p_mw"] = step["p_hh_mw"]
        net.load.loc[loads, "q_mvar"] = step["q_hh_mvar"]
        if len(sgens_pv):
            net.sgen.loc[sgens_pv, "p_mw"] = step["pv_mw_per_kwp"] * kwp
        if has_wind:
            net.sgen.at[idx["wind"], "p_mw"] = step["wind_mw"]

        row = {"time": ts_time, "month": step["month"], "hour_of_week": step["hour_of_week"],
               "tap_pos": net.trafo.at[trafo, "tap_pos"],
               "p_load_mw": step["p_hh_mw"] * len(loads),
               "p_pv_mw": float(step["pv_mw_per_kwp"] * kwp.sum()),
               "p_wind_mw": step["wind_mw"] if has_wind else 0.0}
        try:
            pp.runpp(net, init="auto" if first else "results")
            first = False
            vm = net.res_bus.vm_pu
            vm_lv = vm[lv_buses]
            row.update({
                "converged": True,
                "vm_eval_pu": vm[eval_bus],
                "vm_bus1_pu": vm[idx["bus1"]],
                "vm_bus2_pu": vm[idx["bus2"]],
                "vm_lv_min_pu": vm_lv.min(), "bus_lv_min": vm_lv.idxmin(),
                "vm_lv_max_pu": vm_lv.max(), "bus_lv_max": vm_lv.idxmax(),
                "p_trafo_mw": net.res_trafo.at[trafo, "p_hv_mw"],
                "loading_trafo_pct": net.res_trafo.at[trafo, "loading_percent"],
                "loading_line_max_pct": net.res_line.loading_percent.max(),
                "p_ext_grid_mw": net.res_ext_grid.p_mw.sum(),
            })
        except pp.LoadflowNotConverged:
            first = True
            row.update({"converged": False})
        rows.append(row)

    result = pd.DataFrame(rows).set_index("time")
    if verbose:
        n_bad = (~result["converged"]).sum()
        print(f"{len(result)} Lastflüsse in {time.time() - t0:.0f} s, nicht konvergiert: {n_bad}")
    return result


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------

V_MIN, V_MAX = 0.90, 1.10       # Spannungsband nach DIN EN 50160 (±10 % U_n)


def season_of(month_name):
    month = Month[month_name]
    return next(s for s in Season if month in s.months)


def summary_by_month(res):
    """Extremwerte und Grenzwertverletzungen je Monat."""
    rows = []
    for name, g in res.groupby("month", sort=False):
        ok = g[g["converged"]]
        rows.append({
            "Monat": Month[name].label,
            "konvergiert": f"{len(ok)}/{len(g)}",
            "U_Auswertung min [pu]": ok["vm_eval_pu"].min(),
            "U_Auswertung max [pu]": ok["vm_eval_pu"].max(),
            "U_NS min [pu]": ok["vm_lv_min_pu"].min(),
            "U_NS max [pu]": ok["vm_lv_max_pu"].max(),
            "Stunden U < 0,9": int((ok["vm_lv_min_pu"] < V_MIN).sum()),
            "Stunden U > 1,1": int((ok["vm_lv_max_pu"] > V_MAX).sum()),
            "Stunden Rückspeisung": int((ok["p_trafo_mw"] < 0).sum()),
            "P_Trafo min [kW]": ok["p_trafo_mw"].min() * 1000,
            "Trafo max [%]": ok["loading_trafo_pct"].max(),
            "Leitung max [%]": ok["loading_line_max_pct"].max(),
        })
    return pd.DataFrame(rows).set_index("Monat")


def extreme_values(res, net):
    """Zeitpunkte der niedrigsten und höchsten Spannung sowie der größten Rückspeisung."""
    ok = res[res["converged"]]
    picks = {
        "niedrigste Spannung im NS-Netz": ok["vm_lv_min_pu"].idxmin(),
        "höchste Spannung im NS-Netz": ok["vm_lv_max_pu"].idxmax(),
        "größte Rückspeisung über den Trafo": ok["p_trafo_mw"].idxmin(),
        "größter Bezug über den Trafo": ok["p_trafo_mw"].idxmax(),
    }
    rows = []
    for label, t in picks.items():
        r = ok.loc[t]
        rows.append({"Ereignis": label, "Zeitpunkt": t.strftime("%a %d.%m.%Y %H:%M"),
                     "U_min [pu]": r["vm_lv_min_pu"], "an Bus": net.bus.at[int(r["bus_lv_min"]), "name"],
                     "U_max [pu]": r["vm_lv_max_pu"], "an Bus ": net.bus.at[int(r["bus_lv_max"]), "name"],
                     "U_Auswertung [pu]": r["vm_eval_pu"],
                     "P_Trafo [kW]": r["p_trafo_mw"] * 1000,
                     "Last [kW]": r["p_load_mw"] * 1000, "PV [kW]": r["p_pv_mw"] * 1000})
    return pd.DataFrame(rows).set_index("Ereignis")


def _week_axis(ax):
    ax.set_xticks(range(0, 169, 24))
    ax.set_xticks(range(0, 169, 6), minor=True)
    ax.set_xticklabels(["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So", ""])
    ax.set_xlim(0, 168)
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.15)


def plot_voltage_seasons(res, column="vm_eval_pu", title="Spannung am Auswertungsbus"):
    """Spannung über die Woche, je Jahreszeit ein Diagramm mit den drei Monaten."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True, sharey=True)
    ok = res[res["converged"]]
    # Achse auf den Datenbereich begrenzen; Grenzen nur einbeziehen, wenn sie in der Nähe liegen
    lo, hi = ok[column].min() - 0.015, ok[column].max() + 0.015
    if lo - V_MIN < 0.03:
        lo = min(lo, V_MIN - 0.01)
    if V_MAX - hi < 0.03:
        hi = max(hi, V_MAX + 0.01)
    for ax, season in zip(axes.flat, Season):
        for month in season.months:
            g = ok[ok["month"] == month.name]
            ax.plot(g["hour_of_week"], g[column], label=month.label, lw=1.2)
        ax.axhline(V_MIN, color="red", ls="--", lw=1)
        ax.axhline(V_MAX, color="red", ls="--", lw=1)
        ax.axhline(1.0, color="grey", lw=0.6)
        ax.set_title(season.label)
        ax.set_ylim(lo, hi)
        _week_axis(ax)
        ax.legend(fontsize=8, loc="lower left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Spannung [pu] (Bezug 0,4 kV)")
    for ax in axes[1]:
        ax.set_xlabel("Wochentag")
    bad = (~res["converged"]).sum()
    visible = [v for v in (V_MIN, V_MAX) if lo <= v <= hi]
    band = ("rote Linien: Spannungsband 0,9–1,1 pu" if len(visible) == 2 else
            "Spannungsband 0,9–1,1 pu" + (" teilweise" if visible else "") + " außerhalb des dargestellten Bereichs")
    fig.suptitle(f"{title} – typische Wochen ({band}"
                 + (f"; {bad} nicht konvergierte Zeitschritte ausgelassen)" if bad else ")"))
    fig.tight_layout()
    return fig


def plot_voltage_boxplot(res):
    """Verteilung von minimaler und maximaler NS-Spannung je Monat."""
    ok = res[res["converged"]]
    months = list(dict.fromkeys(ok["month"]))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    pos = np.arange(len(months))
    ax.boxplot([ok[ok["month"] == m]["vm_lv_min_pu"] for m in months], positions=pos - 0.18, widths=0.3,
               patch_artist=True, boxprops=dict(facecolor="tab:blue", alpha=0.5), medianprops=dict(color="k"))
    ax.boxplot([ok[ok["month"] == m]["vm_lv_max_pu"] for m in months], positions=pos + 0.18, widths=0.3,
               patch_artist=True, boxprops=dict(facecolor="tab:orange", alpha=0.5), medianprops=dict(color="k"))
    ax.axhline(V_MIN, color="red", ls="--", lw=1)
    ax.axhline(V_MAX, color="red", ls="--", lw=1)
    lo, hi = ok["vm_lv_min_pu"].min() - 0.015, ok["vm_lv_max_pu"].max() + 0.015
    ax.set_ylim(min(lo, V_MIN - 0.01) if lo - V_MIN < 0.03 else lo,
                max(hi, V_MAX + 0.01) if V_MAX - hi < 0.03 else hi)
    ax.set_xticks(pos)
    ax.set_xticklabels([Month[m].label for m in months], rotation=45)
    ax.set_ylabel("Spannung [pu]")
    ax.set_title("Niedrigste (blau) und höchste (orange) Spannung im NS-Netz je Stunde – Verteilung je Monat")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_trafo_power(res, months=(Month.JANUARY, Month.APRIL, Month.JULY, Month.OCTOBER)):
    """Leistung über den Transformator (negativ = Rückspeisung) und Trafoauslastung."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ok = res[res["converged"]]
    for month in months:
        g = ok[ok["month"] == month.name]
        axes[0].plot(g["hour_of_week"], g["p_trafo_mw"] * 1000, label=month.label, lw=1.2)
        axes[1].plot(g["hour_of_week"], g["loading_trafo_pct"], label=month.label, lw=1.2)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("P über Trafo (OS) [kW]")
    axes[0].set_title("Wirkleistung über den Ortsnetztransformator (> 0: Bezug aus MS, < 0: Rückspeisung)")
    axes[1].axhline(100, color="red", ls="--", lw=1)
    axes[1].set_ylabel("Trafoauslastung [%]")
    axes[1].set_title("Auslastung des Transformators (S_r = 0,4 MVA)")
    for ax in axes:
        _week_axis(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    axes[1].set_xlabel("Wochentag")
    fig.tight_layout()
    return fig


def plot_voltage_profile(net, idx, households, pv, ts, when, feeder=None):
    """
    Spannungsverlauf entlang eines Abgangs zu einem Zeitpunkt (Entfernung vom Trafo).
    when: Zeitstempel aus ts; feeder: Abgangsnummer (Standard: Abgang des Auswertungsbusses)
    """
    feeder = feeder if feeder is not None else int(households[households.bus == idx["eval_bus"]].feeder.iloc[0])
    n = copy.deepcopy(net)
    step = ts.loc[when]
    n.load.loc[households.index, "p_mw"] = step["p_hh_mw"]
    n.load.loc[households.index, "q_mvar"] = step["q_hh_mvar"]
    if len(pv):
        n.sgen.loc[pv.index, "p_mw"] = step["pv_mw_per_kwp"] * pv["kwp"].to_numpy()
    if "wind" in idx:
        n.sgen.at[idx["wind"], "p_mw"] = step["wind_mw"]
    pp.runpp(n)
    g = households[households.feeder == feeder].sort_values("distance_m")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([0] + g.distance_m.tolist(), [n.res_bus.vm_pu[idx["bus2"]]] + n.res_bus.vm_pu[g.bus].tolist(), "o-")
    with_pv = g.index.isin(pv.load) if len(pv) else np.zeros(len(g), bool)
    ax.plot(g.distance_m[with_pv], n.res_bus.vm_pu[g.bus[with_pv]], "s", color="tab:orange", label="Haushalt mit PV")
    ax.axhline(V_MAX, color="red", ls="--", lw=1)
    ax.axhline(V_MIN, color="red", ls="--", lw=1)
    ax.set_xlabel("Leitungslänge ab NS-Sammelschiene [m]")
    ax.set_ylabel("Spannung [pu]")
    ax.set_title(f"Spannung entlang Abgang {feeder} – {pd.Timestamp(when):%a %d.%m.%Y %H:%M}")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig
