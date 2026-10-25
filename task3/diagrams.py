"""
Ergebnisse einer Simulation (Klasse Results) und Diagramme.

Ergebnisse werden als breite Tabellen gehalten: Zeilen = Zeitschritte (Beginn der Stunde,
lokale Zeit), Spalten = Elementnamen. Nicht konvergierte Zeitschritte enthalten NaN.
"""

import json
import difflib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plots
from elements import Diagram
from profiles import Month, Season

TABLES = ["vm_pu", "va_degree", "line_loading", "line_i_ka", "trafo_loading", "trafo_p_mw",
          "ext_grid_p_mw", "load_p_mw", "sgen_p_mw"]


# ---------------------------------------------------------------------------
# Ergebnisse
# ---------------------------------------------------------------------------

class Results:
    """Ergebnisse eines Simulationslaufs."""

    def __init__(self, info, **tables):
        self.info = info                                    # month, hour_of_week, converged, tap
        for name in TABLES:
            setattr(self, name, tables.get(name, pd.DataFrame(index=info.index)))

    def __repr__(self):
        months = [Month[m].label for m in dict.fromkeys(self.info["month"])]
        return (f"Ergebnisse: {len(self.info)} Zeitschritte ({int(self.converged.sum())} konvergiert), "
                f"Monate: {', '.join(months)}\n"
                f"Tabellen: vm_pu, va_degree, line_loading, line_i_ka, trafo_loading, trafo_p_mw, "
                f"ext_grid_p_mw, load_p_mw, sgen_p_mw, element_p_mw, info")

    @property
    def element_p_mw(self):
        """Wirkleistung aller Lasten (Bezug) und Erzeuger (Einspeisung) [MW]."""
        return pd.concat([self.load_p_mw, self.sgen_p_mw], axis=1)

    @property
    def converged(self):
        return self.info["converged"].astype(bool)

    # ------------------------------------------------------------------ Speichern / Laden
    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.info.to_csv(folder / "info.csv")
        for name in TABLES:
            getattr(self, name).to_csv(folder / f"{name}.csv")

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        if not (folder / "info.csv").exists():
            return None
        read = lambda f: pd.read_csv(f, index_col=0, parse_dates=True)
        info = read(folder / "info.csv")
        tables = {n: read(folder / f"{n}.csv") for n in TABLES if (folder / f"{n}.csv").exists()}
        return cls(info, **tables)

    # ------------------------------------------------------------------ Kennzahlen
    def element_summary(self, name, table):
        """Kurzauswertung eines Elements (für getElement)."""
        def stats(series, unit, label):
            s = series.dropna()
            if s.empty:
                return {}
            return {f"Ergebnis {label} min [{unit}]": round(s.min(), 4),
                    f"Ergebnis {label} max [{unit}]": round(s.max(), 4),
                    f"Ergebnis {label} Mittel [{unit}]": round(s.mean(), 4)}

        if table == "bus" and name in self.vm_pu:
            return stats(self.vm_pu[name], "pu", "Spannung")
        if table == "line" and name in self.line_loading:
            out = stats(self.line_loading[name], "%", "Auslastung")
            out["Ergebnis Stunden > 100 %"] = int((self.line_loading[name] > 100).sum())
            return out
        if table == "trafo" and name in self.trafo_loading:
            out = stats(self.trafo_loading[name], "%", "Auslastung")
            out.update(stats(self.trafo_p_mw[name], "MW", "Leistung OS"))
            out["Ergebnis Stunden Rückspeisung"] = int((self.trafo_p_mw[name] < 0).sum())
            return out
        if table in ("load", "sgen"):
            df = self.load_p_mw if table == "load" else self.sgen_p_mw
            if name in df:
                out = stats(df[name], "MW", "Leistung")
                out["Ergebnis Energie [MWh]"] = round(df[name].sum(), 3)          # Stundenwerte
                return out
        return {}

    def summary(self, v_min=0.9, v_max=1.1, by=None):
        """Kennzahlen gesamt (by=None) oder je Monat (by="month")."""
        if by == "month":
            rows = []
            for m, idx in self.info.groupby("month", sort=False).groups.items():
                sub = Results(self.info.loc[idx], **{n: getattr(self, n).loc[idx] for n in TABLES})
                row = sub._kennzahlen(v_min, v_max, compact=True)
                row["Monat"] = Month[m].label
                rows.append(row)
            return pd.DataFrame(rows).set_index("Monat")
        return pd.Series(self._kennzahlen(v_min, v_max), name="Wert", dtype=object)

    def _kennzahlen(self, v_min, v_max, compact=False):
        vm, ll, tl = self.vm_pu, self.line_loading, self.trafo_loading
        ext = self.ext_grid_p_mw.iloc[:, 0] if len(self.ext_grid_p_mw.columns) else pd.Series(dtype=float)
        k = {"Zeitschritte": len(self.info),
             "konvergiert": int(self.converged.sum())}

        def where(df, fn):
            stacked = df.stack()
            if stacked.empty:
                return np.nan, "–", "–"
            (t, el) = stacked.idxmin() if fn == "min" else stacked.idxmax()
            return stacked.loc[(t, el)], el, pd.Timestamp(t).strftime("%a %d.%m. %H:%M")

        u_min, b_min, t_min = where(vm, "min")
        u_max, b_max, t_max = where(vm, "max")
        l_max, l_name, l_time = where(ll, "max")
        tr_max, tr_name, tr_time = where(tl, "max")
        k.update({
            "U min [pu]": round(u_min, 4), "U max [pu]": round(u_max, 4),
            f"Stunden U < {v_min}": int((vm.min(axis=1) < v_min).sum()),
            f"Stunden U > {v_max}": int((vm.max(axis=1) > v_max).sum()),
            "Leitung max [%]": round(l_max, 1), "Stunden Leitung > 100 %": int((ll.max(axis=1) > 100).sum()),
            "Trafo max [%]": round(tr_max, 1), "Stunden Trafo > 100 %": int((tl.max(axis=1) > 100).sum()),
            "Stunden Rückspeisung ins vorgelagerte Netz": int((ext < 0).sum()),
            "max. Rückspeisung [MW]": round(max(0.0, -ext.min()), 3) if len(ext.dropna()) else np.nan,
            "Energie Lasten [MWh]": round(self.load_p_mw.sum().sum(), 1),
            "Energie Erzeuger [MWh]": round(self.sgen_p_mw.sum().sum(), 1),
        })
        if not compact:
            k.update({"U min bei": f"{b_min}, {t_min}", "U max bei": f"{b_max}, {t_max}",
                      "Leitung max bei": f"{l_name}, {l_time}", "Trafo max bei": f"{tr_name}, {tr_time}"})
        return k


# ---------------------------------------------------------------------------
# Diagramme
# ---------------------------------------------------------------------------

def _source(results, diagram):
    """(Tabelle, y-Beschriftung, Grenzlinien, Elementart) für eine Diagrammart."""
    if diagram in (Diagram.VOLTAGE, Diagram.VOLTAGE_PROFILE):
        return results.vm_pu, "Spannung [pu]", None, "Bus"
    if diagram == Diagram.LINE_LOADING:
        return results.line_loading, "Auslastung [%]", [100], "Leitung"
    if diagram == Diagram.TRAFO_LOADING:
        return results.trafo_loading, "Auslastung [%]", [100], "Trafo"
    if diagram == Diagram.TRAFO_POWER:
        return results.trafo_p_mw, "Wirkleistung OS [MW] (< 0: Rückspeisung)", [0], "Trafo"
    if diagram == Diagram.ELEMENT_POWER:
        return results.element_p_mw, "Wirkleistung [MW]", None, "Last/Erzeuger"
    raise ValueError(f"Unbekannte Diagrammart: {diagram}")


def _select(df, elements, kind):
    if elements is None:
        return list(df.columns)
    if isinstance(elements, str):
        elements = [elements]
    missing = [e for e in elements if e not in df.columns]
    if missing:
        hints = [difflib.get_close_matches(m, df.columns, n=3, cutoff=0.5) for m in missing]
        msg = "; ".join(f"'{m}'" + (f" (meinten Sie {', '.join(h)}?)" if h else "") for m, h in zip(missing, hints))
        raise KeyError(f"Für dieses Diagramm werden Elemente vom Typ {kind} erwartet. Nicht gefunden: {msg}")
    return list(elements)


def _months(results, months):
    simulated = list(dict.fromkeys(results.info["month"]))
    if months is None:
        return [Month[m] for m in simulated]
    months = [months] if isinstance(months, (Month, str)) else list(months)
    months = [m if isinstance(m, Month) else Month[str(m).upper()] for m in months]
    missing = [m.label for m in months if m.name not in simulated]
    if missing:
        raise ValueError(f"Monate nicht simuliert: {missing}. Simuliert: {[Month[m].label for m in simulated]}")
    return months


def _limit_lines(ax, diagram, limits, sim_settings):
    if not limits:
        return
    if diagram in (Diagram.VOLTAGE, Diagram.VOLTAGE_PROFILE):
        for v in (sim_settings.get("v_min", 0.9), sim_settings.get("v_max", 1.1)):
            ax.axhline(v, color="red", ls="--", lw=1)
    elif diagram in (Diagram.LINE_LOADING, Diagram.TRAFO_LOADING):
        ax.axhline(100, color="red", ls="--", lw=1)
    elif diagram == Diagram.TRAFO_POWER:
        ax.axhline(0, color="black", lw=0.8)


def parse_markers(markers):
    """
    Markierungen vereinheitlichen. Jede Markierung ist
        (farbe, wert, text)                 oder  {"color": ..., "value": ..., "text": ...}
    Optional als 4. Element bzw. Schlüssel "style": Linienstil (Standard "-.").

    wert = Zahl                     → waagerechte Linie (z. B. 0.95 pu, 80 %)
    wert = Zeitpunkt (str/Timestamp) → senkrechte Linie (nur in Zeitdiagrammen)
    """
    if markers is None:
        return []
    if isinstance(markers, (tuple, dict)):
        markers = [markers]
    out = []
    for m in markers:
        if isinstance(m, dict):
            color, value, text, style = m.get("color", "black"), m["value"], m.get("text", ""), m.get("style", "-.")
        else:
            if len(m) < 2:
                raise ValueError("Markierung als (farbe, wert, text) angeben.")
            color, value = m[0], m[1]
            text = m[2] if len(m) > 2 else ""
            style = m[3] if len(m) > 3 else "-."
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            out.append({"kind": "h", "color": color, "value": float(value), "text": text, "style": style})
        else:
            try:
                t = pd.Timestamp(value)
            except (ValueError, TypeError):
                raise ValueError(f"Wert der Markierung '{value}' ist weder Zahl noch Zeitpunkt.")
            out.append({"kind": "v", "color": color, "value": t, "text": text, "style": style})
    return out


def _draw_markers(ax, markers, mode="time", index=None, month=None, info=None):
    """
    Markierungen zeichnen.
    mode: "time" (x = Zeitstempel), "hour" (x = Stunde der Woche, Monat `month`),
          "position" (x = laufende Nummer, Zeitstempel in `index`), "none" (nur waagerecht)
    """
    from matplotlib.transforms import blended_transform_factory
    for m in markers:
        if m["kind"] == "h":
            ax.axhline(m["value"], color=m["color"], ls=m["style"], lw=1.4, zorder=3)
            if m["text"]:
                ax.text(0.995, m["value"], m["text"], color=m["color"], fontsize=8, ha="right", va="bottom",
                        transform=blended_transform_factory(ax.transAxes, ax.transData), zorder=4,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
            continue
        t = m["value"]
        x = None
        if mode == "time" and index is not None and len(index) and index[0] <= t <= index[-1]:
            x = t
        elif mode == "hour" and info is not None and t in info.index and info.at[t, "month"] == month:
            x = info.at[t, "hour_of_week"]
        elif mode == "position" and index is not None and t in index:
            x = index.get_loc(t)
        if x is None:
            continue
        ax.axvline(x, color=m["color"], ls=m["style"], lw=1.4, zorder=3)
        if m["text"]:
            ax.text(x, 0.98, " " + m["text"], color=m["color"], fontsize=8, ha="left", va="top", rotation=90,
                    transform=blended_transform_factory(ax.transData, ax.transAxes), zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))


def _hour_axis(ax):
    """Achse für Stunde der Woche (0…168): 6-h-Teilung, Tageslinien, Wochentage."""
    ax.set_xlim(0, 168)
    ax.set_xticks(range(0, 169, 6))
    ax.set_xticklabels([f"{h % 24:02d}" for h in range(0, 169, 6)], fontsize=7)
    for d in range(8):
        ax.axvline(d * 24, color="black", lw=1, alpha=0.7)
    for d, name in enumerate(plots.WEEKDAYS_DE):
        ax.text(d * 24 + 12, 1.01, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3)


def _finish(fig, out_dir, filename, show):
    fig.tight_layout()
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / filename, dpi=120, bbox_inches="tight")
    if show:
        plt.show()


def create(sim, diagram, elements=None, months=None, layout="week", time=None, title=None,
           limits=True, out_dir=None, show=True, markers=None):
    """Diagramm(e) einer Simulation. Rückgabe: Liste der Figures."""
    res = sim.results
    if diagram == Diagram.VOLTAGE_PROFILE:
        return [_voltage_profile(sim, elements, time, title, limits, out_dir, show, markers)]
    df, ylabel, _, kind = _source(res, diagram)
    cols = _select(df, elements, kind)
    months = _months(res, months)
    figs = []
    markers = parse_markers(markers)
    label = title or f"{diagram.value} – Simulation '{sim.name}'"

    if layout == "week":
        for m in months:
            idx = res.info.index[res.info["month"] == m.name]
            fig, ax = plt.subplots(figsize=(13, 4))
            for c in cols:
                ax.plot(idx, df.loc[idx, c], lw=1.3, label=c)
            _limit_lines(ax, diagram, limits, sim.settings)
            _draw_markers(ax, markers, "time", index=idx)
            plots.week_axis(ax, idx)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Uhrzeit [h]")
            ax.set_title(f"{label} – {m.label} ({idx[0]:%d.%m.}–{idx[-1]:%d.%m.%Y})", pad=22)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
            _finish(fig, out_dir, f"{diagram.name}_week_{m.name}.png", show)
            figs.append(fig)

    elif layout == "season":
        seasons = [s for s in Season if any(m in months for m in s.months)]
        fig, axes = plt.subplots(len(seasons), 1, figsize=(13, 3.6 * len(seasons)), sharey=True, squeeze=False)
        for ax, s in zip(axes[:, 0], seasons):
            for m in [m for m in s.months if m in months]:
                sel = res.info["month"] == m.name
                for c in cols:
                    ax.plot(res.info.loc[sel, "hour_of_week"], df.loc[sel, c], lw=1.2,
                            label=f"{m.label} – {c}" if len(cols) > 1 else m.label)
            _limit_lines(ax, diagram, limits, sim.settings)
            for m in [m for m in s.months if m in months]:
                _draw_markers(ax, [k for k in markers if k["kind"] == "v"], "hour", month=m.name, info=res.info)
            _draw_markers(ax, [k for k in markers if k["kind"] == "h"], "none")
            _hour_axis(ax)
            ax.set_ylabel(ylabel)
            ax.set_title(s.label, pad=16)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        axes[-1, 0].set_xlabel("Uhrzeit [h]")
        fig.suptitle(label)
        _finish(fig, out_dir, f"{diagram.name}_season.png", show)
        figs.append(fig)

    elif layout == "overview":
        sel = res.info["month"].isin([m.name for m in months]).to_numpy()
        sub, info = df.loc[sel, cols], res.info.loc[sel]
        x = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(15, 4.5))
        for c in cols:
            ax.plot(x, sub[c].to_numpy(), lw=0.9, label=c)
        _limit_lines(ax, diagram, limits, sim.settings)
        _draw_markers(ax, markers, "position", index=info.index)
        starts = np.flatnonzero(info["hour_of_week"].to_numpy() == 0)
        for s0 in starts:
            ax.axvline(s0, color="black", lw=1, alpha=0.7)
            ax.text(s0 + 84, 1.01, Month[info["month"].iloc[s0]].label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(starts)
        ax.set_xticklabels([pd.Timestamp(t).strftime("%d.%m.") for t in info.index[starts]], fontsize=8)
        ax.set_xlim(0, len(sub))
        ax.set_xlabel("Beginn der typischen Woche (je Monat 168 Stunden)")
        ax.set_ylabel(ylabel)
        ax.set_title(label, pad=18)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        _finish(fig, out_dir, f"{diagram.name}_overview.png", show)
        figs.append(fig)

    elif layout == "boxplot":
        fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(months) * len(cols)), 4.5))
        width = 0.8 / len(cols)
        for j, c in enumerate(cols):
            data = [df.loc[res.info["month"] == m.name, c].dropna() for m in months]
            pos = np.arange(len(months)) + (j - (len(cols) - 1) / 2) * width
            bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                            medianprops=dict(color="black"))
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][j % 10]
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.6)
            ax.plot([], [], "s", color=color, alpha=0.6, label=c)
        _limit_lines(ax, diagram, limits, sim.settings)
        _draw_markers(ax, markers, "none")
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([m.label for m in months], rotation=45)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{label} – Verteilung je Monat")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        _finish(fig, out_dir, f"{diagram.name}_boxplot.png", show)
        figs.append(fig)
    else:
        raise ValueError('layout muss "week", "season", "overview" oder "boxplot" sein.')
    return figs


def _voltage_profile(sim, buses, time, title, limits, out_dir, show, markers=None):
    """Spannung entlang eines Pfades (Busse in Reihenfolge) zu einem Zeitpunkt."""
    res, net = sim.results, sim.net
    if not buses or len(buses) < 2:
        raise ValueError("Für VOLTAGE_PROFILE eine Liste von mindestens zwei Busnamen in Pfadreihenfolge angeben.")
    cols = _select(res.vm_pu, buses, "Bus")
    vm = res.vm_pu[cols]
    if time is None or time == "min":
        t = vm.min(axis=1).idxmin()
    elif time == "max":
        t = vm.max(axis=1).idxmax()
    else:
        t = pd.Timestamp(time)
        if t not in vm.index:
            raise KeyError(f"Zeitpunkt {t} nicht in den Ergebnissen.")
    # Entfernung entlang des Pfades über die Leitungslängen
    name_to_idx = dict(zip(net.bus["name"], net.bus.index))
    dist = [0.0]
    for a, b in zip(cols[:-1], cols[1:]):
        ia, ib = name_to_idx[a], name_to_idx[b]
        l = net.line[((net.line.from_bus == ia) & (net.line.to_bus == ib)) |
                     ((net.line.from_bus == ib) & (net.line.to_bus == ia))]
        dist.append(dist[-1] + (float(l.length_km.iloc[0]) if len(l) else 0.0))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dist, vm.loc[t].to_numpy(), "o-")
    for d, c, v in zip(dist, cols, vm.loc[t]):
        ax.annotate(c, (d, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    _limit_lines(ax, Diagram.VOLTAGE_PROFILE, limits, sim.settings)
    _draw_markers(ax, parse_markers(markers), "none")
    ax.set_xlabel("Leitungslänge entlang des Pfades [km]")
    ax.set_ylabel("Spannung [pu]")
    ax.set_title(title or f"Spannungsverlauf – Simulation '{sim.name}' – {pd.Timestamp(t):%a %d.%m.%Y %H:%M}")
    ax.grid(True, alpha=0.4)
    _finish(fig, out_dir, f"VOLTAGE_PROFILE_{pd.Timestamp(t):%Y%m%d_%H}.png", show)
    return fig


def compare(simulations, diagram, elements, months=None, layout="week", title=None, limits=True, show=True,
            markers=None):
    """Dieselbe Größe mehrerer Simulationen in einem Diagramm (layout "week" oder "overview")."""
    base = simulations[0]
    df0, ylabel, _, kind = _source(base.results, diagram)
    cols = _select(df0, elements, kind)
    months = _months(base.results, months)
    figs = []
    markers = parse_markers(markers)
    styles = ["-", "--", ":", "-."]
    if layout == "week":
        for m in months:
            fig, ax = plt.subplots(figsize=(13, 4))
            idx = None
            for k, s in enumerate(simulations):
                df, *_ = _source(s.results, diagram)
                sel = s.results.info.index[s.results.info["month"] == m.name]
                idx = sel
                for c in _select(df, cols, kind):
                    ax.plot(sel, df.loc[sel, c], ls=styles[k % 4], lw=1.3, label=f"{s.name}: {c}")
            _limit_lines(ax, diagram, limits, base.settings)
            _draw_markers(ax, markers, "time", index=idx)
            plots.week_axis(ax, idx)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Uhrzeit [h]")
            ax.set_title(f"{title or 'Vergleich ' + diagram.value} – {m.label}", pad=22)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
            _finish(fig, None, "", show)
            figs.append(fig)
    elif layout == "overview":
        fig, ax = plt.subplots(figsize=(15, 4.5))
        for k, s in enumerate(simulations):
            df, *_ = _source(s.results, diagram)
            sel = s.results.info["month"].isin([m.name for m in months]).to_numpy()
            for c in _select(df, cols, kind):
                ax.plot(np.arange(sel.sum()), df.loc[sel, c].to_numpy(), ls=styles[k % 4], lw=0.9,
                        label=f"{s.name}: {c}")
        _limit_lines(ax, diagram, limits, base.settings)
        info = base.results.info[base.results.info["month"].isin([m.name for m in months])]
        _draw_markers(ax, markers, "position", index=info.index)
        starts = np.flatnonzero(info["hour_of_week"].to_numpy() == 0)
        for s0 in starts:
            ax.axvline(s0, color="black", lw=1, alpha=0.7)
            ax.text(s0 + 84, 1.01, Month[info["month"].iloc[s0]].label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(starts)
        ax.set_xticklabels([pd.Timestamp(t).strftime("%d.%m.") for t in info.index[starts]], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"Vergleich {diagram.value}", pad=18)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        _finish(fig, None, "", show)
        figs.append(fig)
    else:
        raise ValueError('Für den Vergleich layout "week" oder "overview" verwenden.')
    return figs
