"""
Netzplan mit Lastflussergebnissen zu einem Zeitpunkt.

Darstellung:
    Bus          Kreis, Füllfarbe nach Spannung 0,8 … 1,2 pu
                 (lila – blau – grün bei 1,0 pu – gelb – rot)
    Leitung      Linie, Farbe nach Auslastung 0 … 100 % (grün – gelb – rot, schwarz über 100 %)
    Trafo        zwei Kreise auf einer grauen Linie zwischen OS- und US-Bus,
                 Kreisfarbe nach Auslastung wie bei den Leitungen
    Schalter     Quadrat auf der Leitung bzw. dem Trafo, nahe dem Bus, an dem er sitzt;
                 schwarz = geschlossen, weiß = offen
    ext. Netz    schraffiertes Quadrat über dem Anschlussbus
    Last         Pfeil vom Bus weg
    Erzeuger     Kreis mit G
Lasten und Erzeuger werden in die freien Richtungen um ihren Bus verteilt; daneben stehen
Name, Profil und die Leistung zum gewählten Zeitpunkt.
Neben jeder Komponente steht ihr Name, darunter der Ergebniswert.

Die Lage der Busse kommt aus den Geodaten des Netzes (net.bus.geo bzw. net.bus_geodata).
Busse ohne Geodaten werden automatisch angeordnet (Kamada-Kawai-Layout über networkx).

Verwendung:
    sim1.createDiagram(Diagram.NETWORK, time="2025-01-15 18:00")
    sim1.createDiagram(Diagram.NETWORK, time=["min", "max_loading"])
    netplan.plot_static(net)            # nach pp.runpp(net)
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Rectangle
from matplotlib.cm import ScalarMappable

V_RANGE = (0.8, 1.2)
LOADING_RANGE = (0.0, 100.0)

VOLTAGE_CMAP = LinearSegmentedColormap.from_list(
    "spannung", [(0.0, "purple"), (0.25, "blue"), (0.5, "green"), (0.75, "yellow"), (1.0, "red")])
LOADING_CMAP = LinearSegmentedColormap.from_list(
    "auslastung", [(0.0, "green"), (0.5, "yellow"), (1.0, "red")])
LOADING_CMAP.set_over("black")
NO_RESULT = "lightgray"

PROFILE_LABEL = {"HAUSHALT": "Haushalt", "GEWERBE": "Gewerbe", "LANDWIRTSCHAFT": "Landwirtschaft",
                 "HAUSHALT_PV": "Haushalt mit PV", "HAUSHALT_PV_SPEICHER": "Haushalt mit PV + Speicher",
                 "PV": "PV", "WIND": "Wind", "FEST": "konstant"}
SGEN_TYPE_LABEL = {"PV": "PV", "WP": "Wind", "Wind": "Wind"}


def _profile_label(net, table, idx):
    """Profil eines Elements als Text; ohne Profilspalte aus Name bzw. Typ abgeleitet."""
    r = net[table].loc[idx]
    prof = r.get("profile") if "profile" in net[table] else None
    if isinstance(prof, str) and prof:
        return PROFILE_LABEL.get(prof, prof)
    name = str(r.get("name", ""))
    if table == "load":
        if name.startswith("Load R") or name.startswith("HH "):
            return "Haushalt"
        if name.startswith("Load CI"):
            return "Gewerbe"
        return ""
    return SGEN_TYPE_LABEL.get(str(r.get("type", "")), "")


# Zeitpunkte, die über ein Stichwort gewählt werden können
TIME_KEYS = {
    "min": "niedrigste Busspannung",
    "max": "höchste Busspannung ohne Slack-Bus",
    "max_loading": "höchste Leitungsauslastung",
    "max_trafo": "höchste Trafoauslastung",
}


# ---------------------------------------------------------------------------
# Koordinaten
# ---------------------------------------------------------------------------

def _geo_point(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, dict) and "coordinates" in value:
        x, y = value["coordinates"][:2]
        return float(x), float(y)
    return None


def bus_positions(net):
    """{Busindex: (x, y)} aus den Geodaten; fehlende Busse werden automatisch angeordnet."""
    pos = {}
    if "geo" in net.bus:
        for idx, g in net.bus["geo"].items():
            p = _geo_point(g)
            if p is not None:
                pos[idx] = p
    if not pos and "bus_geodata" in net and len(net.bus_geodata):
        pos = {idx: (float(r.x), float(r.y)) for idx, r in net.bus_geodata.iterrows()
               if not (pd.isna(r.x) or pd.isna(r.y))}
    missing = [b for b in net.bus.index if b not in pos]
    if missing:
        pos.update(_auto_layout(net, pos, missing))
    return pos


def _auto_layout(net, known, missing):
    """Kamada-Kawai-Layout für Busse ohne Geodaten; bekannte Busse bleiben fest."""
    import networkx as nx
    import pandapower.topology as top
    g = nx.Graph(top.create_nxgraph(net, respect_switches=False, include_out_of_service=True))
    if known:
        xs, ys = zip(*known.values())
        scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) / 2
        center = np.array([np.mean(xs), np.mean(ys)])
    else:
        scale, center = 5.0, np.array([0.0, 0.0])
    layout = nx.kamada_kawai_layout(g, weight=None, scale=scale, center=center)
    if known:                                       # Unbekannte Busse neben ihrem nächsten bekannten Nachbarn
        out = {}
        for b in missing:
            p = np.array(layout[b])
            try:
                lengths = nx.single_source_shortest_path_length(g, b)
                anchors = [n for n in lengths if n in known]
            except nx.NetworkXError:
                anchors = []
            if anchors:
                a = min(anchors, key=lambda n: lengths[n])
                shift = np.array(known[a]) - np.array(layout[a])
                p = p + shift
            out[b] = tuple(p)
        return out
    return {b: tuple(layout[b]) for b in missing}


# ---------------------------------------------------------------------------
# Zeichnen
# ---------------------------------------------------------------------------

def _values(source, names):
    if source is None:
        return {}
    if isinstance(source, pd.Series):
        source = source.to_dict()
    return {n: source.get(n, np.nan) for n in names}


def _loading_color(v):
    if v is None or pd.isna(v):
        return NO_RESULT
    if v > LOADING_RANGE[1]:
        return "black"
    return LOADING_CMAP(Normalize(*LOADING_RANGE)(v))


def _voltage_color(v, v_range):
    if v is None or pd.isna(v):
        return NO_RESULT
    return VOLTAGE_CMAP(Normalize(*v_range, clip=True)(v))


def _label(ax, x, y, text, fontsize, ha="left", va="bottom", color="black"):
    ax.text(x, y, text, fontsize=fontsize, ha=ha, va=va, color=color, zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6))


def draw(net, vm_pu=None, line_loading=None, trafo_loading=None, load_p_mw=None, sgen_p_mw=None,
         title=None, labels=True, values=True, elements=True, v_range=V_RANGE, ax=None, figsize=(14, 12)):
    """
    Netzplan zeichnen.

    vm_pu, line_loading, trafo_loading, load_p_mw, sgen_p_mw :
              Series oder dict, Schlüssel = Elementname
    labels  : Namen (und bei Lasten/Erzeugern das Profil) neben die Komponenten schreiben
    values  : Ergebniswert unter den Namen schreiben
    elements: Lasten (Pfeil) und Erzeuger (Kreis mit G) einzeichnen
    Rückgabe: Figure
    """
    pos = bus_positions(net)
    pts = np.array(list(pos.values()))
    extent = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0)
    r_bus = 0.017 * extent                          # Radius der Buskreise
    r_tr = 0.75 * r_bus                             # Radius der Trafokreise
    sw = 0.9 * r_bus                                # Kantenlänge der Schalter
    fs = 8 if labels else 0
    fs_small = 7

    bus_name = net.bus["name"].to_dict()
    vm = _values(vm_pu, net.bus["name"])
    ll = _values(line_loading, net.line["name"]) if len(net.line) else {}
    tl = _values(trafo_loading, net.trafo["name"]) if len(net.trafo) else {}
    lp = _values(load_p_mw, net.load["name"]) if len(net.load) else {}
    gp = _values(sgen_p_mw, net.sgen["name"]) if len(net.sgen) else {}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ---------------- Leitungen
    line_geom = {}
    for idx, r in net.line.iterrows():
        a, b = np.array(pos[r.from_bus]), np.array(pos[r.to_bus])
        line_geom[idx] = (a, b)
        color = _loading_color(ll.get(r["name"]))
        ls = "-" if r.in_service else ":"
        ax.plot(*zip(a, b), color=color, lw=3.2, ls=ls, solid_capstyle="round", zorder=2)
        if labels:
            mid, d = (a + b) / 2, b - a
            n = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1)        # Normale zur Leitung
            # Beschriftung rechts von senkrechten und unter waagerechten Leitungen,
            # damit sie nicht mit den Busnamen (links oben am Bus) zusammenstößt
            if abs(n[0]) > abs(n[1]):
                n = n if n[0] > 0 else -n
            else:
                n = n if n[1] < 0 else -n
            txt = r["name"]
            v = ll.get(r["name"])
            if values and v is not None and not pd.isna(v):
                txt += f"\n{v:.1f} %"
            p = mid + n * 0.6 * r_bus
            _label(ax, *p, txt, fs_small, ha="left" if n[0] > 0.3 else "right" if n[0] < -0.3 else "center",
                   va="top" if n[1] < -0.3 else "bottom" if n[1] > 0.3 else "center")

    # ---------------- Trafos
    trafo_geom = {}
    for idx, r in net.trafo.iterrows():
        a, b = np.array(pos[r.hv_bus]), np.array(pos[r.lv_bus])
        trafo_geom[idx] = (a, b)
        d = b - a
        L = np.hypot(*d) or 1
        u = d / L
        mid = (a + b) / 2
        c1, c2 = mid - u * 0.55 * r_tr, mid + u * 0.55 * r_tr
        ax.plot(*zip(a, c1 - u * r_tr), color="gray", lw=2, zorder=2)
        ax.plot(*zip(c2 + u * r_tr, b), color="gray", lw=2, zorder=2)
        color = _loading_color(tl.get(r["name"]))
        for c in (c1, c2):
            ax.add_patch(Circle(c, r_tr, facecolor="none", edgecolor=color, lw=2.6, zorder=5))
        if labels:
            n = np.array([-u[1], u[0]])
            if n[0] < 0:
                n = -n
            txt = r["name"]
            v = tl.get(r["name"])
            if values and v is not None and not pd.isna(v):
                txt += f"\n{v:.1f} %"
            p = mid + n * 2.0 * r_tr
            _label(ax, *p, txt, fs_small, ha="left" if n[0] >= 0 else "right", va="center")

    # ---------------- Schalter
    for idx, r in net.switch.iterrows():
        bus = np.array(pos[r.bus])
        if r.et == "l" and r.element in line_geom:
            a, b = line_geom[r.element]
            other = b if np.allclose(a, bus) else a
        elif r.et == "t" and r.element in trafo_geom:
            a, b = trafo_geom[r.element]
            other = b if np.allclose(a, bus) else a
        elif r.et == "b" and r.element in pos:
            other = np.array(pos[r.element])
            ax.plot(*zip(bus, other), color="gray", lw=1.5, zorder=2)
        else:
            continue
        d = other - bus
        L = np.hypot(*d) or 1
        if r.et == "b":
            p = bus + d / 2
        else:
            p = bus + d / L * min(0.3 * L, 2.6 * r_bus)
        face = "black" if r.closed else "white"
        ax.add_patch(Rectangle(p - sw / 2, sw, sw, facecolor=face, edgecolor="black", lw=1.2, zorder=6))
        if labels and r["name"]:
            _label(ax, p[0] + 0.6 * sw, p[1] + 0.6 * sw, str(r["name"]), fs_small - 1.5, ha="left", va="bottom",
                   color="dimgray")

    # ---------------- externes Netz
    for idx, r in net.ext_grid.iterrows():
        b = np.array(pos[r.bus])
        size = 2.4 * r_bus
        p = b + np.array([0, 2.6 * r_bus])
        ax.plot([b[0], b[0]], [b[1], p[1]], color="gray", lw=2, zorder=2)
        ax.add_patch(Rectangle((p[0] - size / 2, p[1]), size, size, facecolor="white", edgecolor="black",
                               hatch="xxx", lw=1.2, zorder=6))
        if labels:
            _label(ax, p[0] + 0.7 * size, p[1] + size / 2, r["name"] or "Externes Netz", fs_small, va="center")

    # ---------------- Busse
    # Richtungen der angeschlossenen Leitungen/Trafos je Bus, damit der Name in eine freie Ecke kommt
    edges = {b: [] for b in pos}
    for a_b in [(r.from_bus, r.to_bus) for _, r in net.line.iterrows()] + \
               [(r.hv_bus, r.lv_bus) for _, r in net.trafo.iterrows()]:
        for b, o in (a_b, a_b[::-1]):
            d = np.array(pos[o]) - np.array(pos[b])
            edges[b].append(np.arctan2(d[1], d[0]))
    for _, r in net.ext_grid.iterrows():
        edges[r.bus].append(np.pi / 2)
    corners = [(3 * np.pi / 4, "right", "bottom", (-1, 1)), (np.pi / 4, "left", "bottom", (1, 1)),
               (-3 * np.pi / 4, "right", "top", (-1, -1)), (-np.pi / 4, "left", "top", (1, -1))]

    def free_corner(b):
        if not edges[b]:
            return corners[0]
        gap = lambda ang: min(abs((ang - e + np.pi) % (2 * np.pi) - np.pi) for e in edges[b])
        return max(corners, key=lambda c: gap(c[0]) + (0.05 if c is corners[0] else 0))

    attached = {b: [] for b in pos}                                  # (Tabelle, Index) je Bus
    if elements:
        for table in ("load", "sgen"):
            if table in net and len(net[table]):
                for idx, r in net[table].iterrows():
                    attached[r.bus].append((table, idx))

    def gap_to(ang, occupied):
        if not occupied:
            return np.pi
        return min(abs((ang - e + np.pi) % (2 * np.pi) - np.pi) for e in occupied)

    for idx, r in net.bus.iterrows():
        x, y = pos[idx]
        v = vm.get(r["name"])
        ax.add_patch(Circle((x, y), r_bus, facecolor=_voltage_color(v, v_range), edgecolor="black",
                            lw=1.0, zorder=7, alpha=1 if r.in_service else 0.4))
        corner = free_corner(idx)
        if labels:
            txt = r["name"]
            if values and v is not None and not pd.isna(v):
                txt += f"\n{v:.3f} pu"
            _, ha, va, (sx, sy) = corner
            _label(ax, x + sx * 0.9 * r_bus, y + sy * 0.9 * r_bus, txt, fs, ha=ha, va=va)

        # Lasten und Erzeuger in die freien Richtungen um den Bus verteilen
        occupied = list(edges[idx]) + ([corner[0]] if labels else [])
        candidates = np.deg2rad(np.arange(-180, 180, 15))
        for table, el in attached[idx]:
            ang = max(candidates, key=lambda a: (round(gap_to(a, occupied), 3), -abs(a + np.pi / 2)))
            occupied.append(ang)
            _draw_element(ax, net, table, el, np.array([x, y]), ang, r_bus,
                          (lp if table == "load" else gp).get(net[table].at[el, "name"]),
                          labels, values, fs_small)

    # ---------------- Achsen und Farbskalen
    pad = 4 * r_bus * (1.6 if elements else 1.0)                    # Platz für Lasten/Erzeuger am Rand
    ax.set_xlim(pts[:, 0].min() - pad * 1.8, pts[:, 0].max() + pad * 1.8)
    ax.set_ylim(pts[:, 1].min() - pad * 1.3, pts[:, 1].max() + pad * 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    cax_v = div.append_axes("right", size="2.2%", pad=0.15)
    cax_l = div.append_axes("right", size="2.2%", pad=0.75)
    sm_v = ScalarMappable(norm=Normalize(*v_range), cmap=VOLTAGE_CMAP)
    sm_l = ScalarMappable(norm=Normalize(*LOADING_RANGE), cmap=LOADING_CMAP)
    fig.colorbar(sm_v, cax=cax_v).set_label("Busspannung [pu]")
    fig.colorbar(sm_l, cax=cax_l, extend="max").set_label("Auslastung Leitungen und Trafos [%]  (schwarz: > 100 %)")
    if title:
        ax.set_title(title, fontsize=12)
    _legend(ax)
    return fig


def _draw_element(ax, net, table, idx, bus_xy, ang, r_bus, p_mw, labels, values, fontsize):
    """Last als Pfeil vom Bus weg, Erzeuger als Kreis mit G, verbunden über eine kurze Linie."""
    u = np.array([np.cos(ang), np.sin(ang)])
    r = net[table].loc[idx]
    alpha = 1.0 if r.in_service else 0.35
    reach = 3.4 * r_bus                                               # Abstand Busmitte → Symbolende
    if table == "load":
        start, end = bus_xy + u * r_bus, bus_xy + u * reach
        ax.annotate("", xy=end, xytext=start, zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4, mutation_scale=12, alpha=alpha,
                                    shrinkA=0, shrinkB=0))
        tip = end
    else:
        r_g = 0.7 * r_bus
        center = bus_xy + u * (reach - r_g)
        ax.plot(*zip(bus_xy + u * r_bus, center - u * r_g), color="black", lw=1.2, zorder=4, alpha=alpha)
        ax.add_patch(Circle(center, r_g, facecolor="white", edgecolor="black", lw=1.3, zorder=6, alpha=alpha))
        ax.text(*center, "G", ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=7,
                alpha=alpha)
        tip = center + u * r_g
    if not labels:
        return
    lines = [str(r["name"])]
    prof = _profile_label(net, table, idx)
    if prof:
        lines.append(prof)
    if values and p_mw is not None and not pd.isna(p_mw):
        lines.append(f"{p_mw * 1000:.0f} kW" if abs(p_mw) < 1 else f"{p_mw:.2f} MW")
    p = tip + u * 0.35 * r_bus
    ha = "left" if u[0] > 0.3 else "right" if u[0] < -0.3 else "center"
    va = "bottom" if u[1] > 0.3 else "top" if u[1] < -0.3 else "center"
    _label(ax, *p, "\n".join(lines), fontsize - 1, ha=ha, va=va, color="dimgray")


def _legend(ax):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="green", markeredgecolor="black", markersize=10,
               label="Bus (Farbe = Spannung)"),
        Line2D([], [], color="green", lw=3, label="Leitung (Farbe = Auslastung)"),
        Line2D([], [], marker="$⦾$", ls="", color="green", markersize=13,
               label="Trafo (Farbe = Auslastung)"),
        Line2D([], [], marker="s", ls="", markerfacecolor="black", markeredgecolor="black", label="Schalter geschlossen"),
        Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor="black", label="Schalter offen"),
        Line2D([], [], marker=r"$\rightarrow$", ls="", color="black", markersize=14, label="Last"),
        Line2D([], [], marker=r"$\mathbf{G}$", ls="", color="black", markersize=8, label="Erzeuger"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=7, fontsize=8, frameon=False)


# ---------------------------------------------------------------------------
# Zeitpunkte und Aufrufe
# ---------------------------------------------------------------------------

def resolve_time(results, time, months=None, net=None):
    """
    Zeitpunkt bestimmen.
    time: Timestamp/str oder Stichwort "min", "max", "max_loading", "max_trafo"
    months: Stichwortsuche auf diese Monate beschränken (Liste von Month oder None)
    """
    if time is None:
        time = "max_loading"
    if isinstance(time, str) and time in TIME_KEYS:
        sel = np.ones(len(results.info), dtype=bool)
        if months:
            sel = results.info["month"].isin([m.name for m in months]).to_numpy()
        vm, ll, tl = results.vm_pu[sel], results.line_loading[sel], results.trafo_loading[sel]
        if time == "min":
            return vm.min(axis=1).idxmin(), TIME_KEYS[time]
        if time == "max":
            if net is not None and len(net.ext_grid):          # Slack-Bus hat feste Spannung → ausnehmen
                slack = set(net.bus.loc[net.ext_grid.bus, "name"])
                vm = vm[[c for c in vm.columns if c not in slack]]
            return vm.max(axis=1).idxmax(), TIME_KEYS[time]
        if time == "max_loading":
            return ll.max(axis=1).idxmax(), TIME_KEYS[time]
        return tl.max(axis=1).idxmax(), TIME_KEYS[time]
    t = pd.Timestamp(time)
    if t not in results.info.index:
        near = results.info.index[np.argmin(np.abs(results.info.index - t))]
        raise KeyError(f"Zeitpunkt {t:%d.%m.%Y %H:%M} wurde nicht simuliert. "
                       f"Simuliert wird je Monat eine Woche, nächster Zeitpunkt: {near:%a %d.%m.%Y %H:%M}")
    return t, None


def plot_time(sim, time=None, months=None, title=None, labels=True, values=True, elements=True):
    """Netzplan einer Simulation zu einem Zeitpunkt (siehe resolve_time)."""
    res = sim.results
    t, reason = resolve_time(res, time, months, sim.net)
    label = title or f"Netzplan – Simulation '{sim.name}'"
    sub = f"{pd.Timestamp(t):%a %d.%m.%Y %H:%M}" + (f" ({reason})" if reason else "")
    return draw(sim.net, res.vm_pu.loc[t], res.line_loading.loc[t], res.trafo_loading.loc[t],
                res.load_p_mw.loc[t], res.sgen_p_mw.loc[t],
                title=f"{label}\n{sub}", labels=labels, values=values, elements=elements), t


def plot_static(net, title="Netzplan – statischer Lastfluss", labels=True, values=True, elements=True):
    """Netzplan mit den Ergebnissen von pp.runpp(net)."""
    names = lambda t: net[t]["name"]
    vm = pd.Series(net.res_bus.vm_pu.to_numpy(), index=names("bus"))
    ll = pd.Series(net.res_line.loading_percent.to_numpy(), index=names("line")) if len(net.line) else None
    tl = pd.Series(net.res_trafo.loading_percent.to_numpy(), index=names("trafo")) if len(net.trafo) else None
    lp = pd.Series(net.res_load.p_mw.to_numpy(), index=names("load")) if len(net.load) else None
    gp = pd.Series(net.res_sgen.p_mw.to_numpy(), index=names("sgen")) if len(net.sgen) else None
    fig = draw(net, vm, ll, tl, lp, gp, title=title, labels=labels, values=values, elements=elements)
    plt.show()
    return fig
