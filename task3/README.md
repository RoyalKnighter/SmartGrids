# Task 3 – Simulationssoftware

`simulation.py` ist eine kleine Simulationsumgebung auf Basis von pandapower. Damit lassen sich Netze mit Last- und Erzeugungsprofilen über die 12 typischen Wochen des Jahres simulieren. Simulationen lassen sich einfach ändern, speichern, laden, kopieren, vergleichen und auswerten.

Ein lauffähiges Beispiel mit allen Funktionen steht in **`Simulation_Beispiel.ipynb`**.

---

## Schnellstart

```python
import pandapower.networks as pn
import simulation as sim
from simulation import Profile, Diagram, Month

sim1 = sim.init("cigre_basis", pn.create_cigre_network_mv(with_der="pv_wind"), assignProfiles=True)

sim1.createElement(sim.Load("Neubaugebiet", bus="Bus 10", profile=Profile.HAUSHALT, annual_kwh=1_500_000))
sim1.changeParameter("Line 1-2", "length_km", 3.2)
sim1.deleteElement("PV 11")

sim1.runSimulation()                     # rechnet und speichert
sim1.summary()
sim1.createDiagram(Diagram.VOLTAGE, ["Bus 6", "Bus 11"], months=Month.JULY)

sim2 = sim1.copy("cigre_variante")
sim1 = sim.loadSimulation("cigre_basis")
```

---

## Dateien

| Datei | Inhalt |
|---|---|
| `simulation.py` | Klasse `Simulation` und Modulfunktionen (`init`, `loadSimulation`, `listSimulations`, `deleteSimulation`, `compare`) |
| `elements.py` | Elementklassen (`Bus`, `Line`, `Trafo`, `ExtGrid`, `Switch`, `Load`, `Generator`) und Enums `Profile`, `Diagram` |
| `timeseries.py` | normierte Zeitreihen je Profil, Skalierung über die Jahreshöchstleistung |
| `netplan.py` | Netzplan mit Ergebnissen (Diagrammart `NETWORK`, `plot_static`) |
| `diagrams.py` | Klasse `Results` (Ergebnisse, Kennzahlen, Speichern/Laden) und alle Diagramme |
| `profiles.py`, `weather.py` | Lastprofile (BDEW H25, G25, L25, P25, S25) und Wetter/PV/Wind (aus Task 2) |
| `plots.py` | Zeitachse der Diagramme (6-h-Takt, Tageslinien, Datum) |
| `data/` | BDEW-Profile und Wetterwochen |
| `simulations/` | gespeicherte Simulationen (wird automatisch angelegt) |

---

## Profile

Jede Last und jeder Erzeuger hat ein Profil. Es bestimmt den Leistungsverlauf über die Zeit:

| `Profile` | Element | Skalierung | Leistung im Zeitschritt t |
|---|---|---|---|
| `HAUSHALT` | `Load` | `annual_kwh` | BDEW H25 (dynamisiert) × Jahresverbrauch |
| `GEWERBE` | `Load` | `annual_kwh` | BDEW G25 × Jahresverbrauch |
| `LANDWIRTSCHAFT` | `Load` | `annual_kwh` | BDEW L25 × Jahresverbrauch |
| `HAUSHALT_PV` | `Load` | `annual_kwh` | BDEW P25 (dynamisiert): **Netzbezug** eines Haushalts mit PV × Jahresverbrauch |
| `HAUSHALT_PV_SPEICHER` | `Load` | `annual_kwh` | BDEW S25 (dynamisiert): **Netzbezug** eines Haushalts mit PV und Speicher × Jahresverbrauch |
| `PV` | `Generator` | `kwp` | PV-Leistung je kWp aus den Wetterdaten × kWp |
| `WIND` | `Generator` | `p_mw` (Nennleistung) | normierte V90-Kennlinie aus den Wetterdaten × Nennleistung |
| `FEST` | `Load`, `Generator` | `p_mw` | konstant |

**Dynamisierung:** H25, P25 und S25 werden dynamisiert, G25 und L25 nicht (so steht es im jeweiligen Dateikopf).

**P25 und S25 sind Bezugsprofile:** Sie beschreiben, was ein Haushalt mit eigener PV-Anlage (und Speicher) **aus dem Netz bezieht**, nachdem der Eigenverbrauch abgezogen ist. Eine Einspeisung ins Netz ist darin nicht enthalten. Für Untersuchungen zu Rückspeisung und Spannungsanhebung braucht man deshalb `HAUSHALT` plus einen `Generator` mit `Profile.PV`. P25/S25 an derselben Last zusätzlich mit einem PV-Generator zu kombinieren, würde die PV doppelt zählen.

**Blindleistung:** über `cos_phi`. Die Standardwerte sind Haushalt 0,97, Gewerbe 0,95 sowie PV und Wind 1,0. Wird einer Last aus einem Netzmodell (z. B. CIGRE) ein Profil zugeordnet, bleibt ihr bisheriger Leistungsfaktor erhalten. Bei `FEST` ohne `cos_phi` bleibt `q_mvar` konstant.

**Skalierung über die Jahreshöchstleistung:** Gibt man einer Haushalts- oder Gewerbelast statt `annual_kwh` eine Leistung `p_mw`, wird diese als **höchste Stundenleistung des Jahres** interpretiert. Daraus wird der Jahresverbrauch berechnet:

- 1 MW Höchstleistung entspricht etwa 4,4 GWh/a (H25), 3,7 GWh/a (G25), 4,3 GWh/a (L25), 3,1 GWh/a (P25) bzw. 2,0 GWh/a (S25).
- So werden die CIGRE-Lasten (`p_mw` = Bemessungsleistung) automatisch skaliert.
- Gespeichert wird in `net.load.p_mw` immer diese Höchstleistung. Ein einzelner `pp.runpp` auf dem Netz rechnet also den Starklastfall.

**Speicherort:** Profil und Skalierung stehen als zusätzliche Spalten in `net.load` und `net.sgen` (`profile`, `annual_kwh`, `kwp`, `cos_phi`). Das Netz bleibt ein normales pandapower-Netz.

---

## Funktionen

### Simulation anlegen, laden, kopieren

| Aufruf | Beschreibung |
|---|---|
| `sim.init(name, net=None, overwrite=False, assignProfiles=False)` | Neue Simulation. `net` wird kopiert (auch `andereSim.net`), `None` ergibt ein leeres Netz. Fehlende oder doppelte Namen werden ergänzt bzw. eindeutig gemacht. `assignProfiles=True` ordnet die Profile sofort zu; sie gehören dann zur Baseline. |
| `sim.loadSimulation(name)` | Lädt Netz, Baseline, Einstellungen, Protokoll und Ergebnisse. |
| `sim.listSimulations()` | Tabelle aller gespeicherten Simulationen. |
| `sim.deleteSimulation(name, confirm=True)` | Löscht eine gespeicherte Simulation. |
| `sim1.copy(neuerName, overwrite=False)` | Kopie von Netz und Einstellungen. **Baseline der Kopie = aktueller Stand von `sim1`.** Ergebnisse werden nicht kopiert. |
| `sim1.save()` | Speichert; `runSimulation` speichert automatisch. |

### Netz ändern

| Aufruf | Beschreibung |
|---|---|
| `sim1.createElement(element)` | Legt ein Element an (siehe Elementklassen). |
| `sim1.deleteElement(name, cascade=False)` | Löscht ein Element. Bei Leitungen und Trafos werden die zugehörigen Schalter mitgelöscht. Busse mit angeschlossenen Elementen nur mit `cascade=True`. |
| `sim1.changeParameter(name, parameter, wert)` | Ändert einen Parameter: jede pandapower-Spalte, Busse als Namen (`"bus"`, `"from_bus"`, …), `std_type` (übernimmt alle Typwerte) sowie `profile`, `annual_kwh`, `kwp`, `cos_phi` und `p_mw`. |
| `sim1.assignProfile(name, profile, annual_kwh=, kwp=, p_mw=, cos_phi=)` | Profil eines Elements setzen. |
| `sim1.assignProfiles(regex, profile, **skalierung)` | Profil allen Elementen zuordnen, deren Name auf den regulären Ausdruck passt, z. B. `assignProfiles(r"^HH", Profile.HAUSHALT, annual_kwh=3500)`. |
| `sim1.assignProfilesByName(rules=None)` | Zuordnung über Namensregeln (CIGRE: `Load R…`, `Load CI…`, `PV…`, `WKA…`; Ortsnetz Task 2: `HH …`). |
| `sim1.attachNetwork(net, at_bus, prefix=None)` | Hängt ein Netz an einen Bus an, z. B. das Ortsnetz aus Task 2 an das CIGRE-Netz. Der Slack-Bus des angehängten Netzes wird mit `at_bus` vereinigt, sein externes Netz entfällt. `prefix` wird allen Namen des angehängten Netzes vorangestellt. |
| `sim1.changeSetting(name, wert)` | Einstellung ändern (siehe unten). |

### Abfragen

| Aufruf | Rückgabe |
|---|---|
| `sim1.getElement(name)` | Steckbrief: Tabelle, Index, alle Parameter (Busse als Namen), Profil. Nach einem Lauf zusätzlich Min./Max./Mittel der Ergebnisgröße, bei Leitungen auch die Stunden über 100 %, bei Trafos die Stunden mit Rückspeisung, bei Lasten/Erzeugern die Energie. |
| `sim1.elements(table=None)` | Übersicht aller Elemente mit Typ, Bus(sen), Profil und Skalierung. |
| `sim1.changedParameters()` | Geänderte Parameter und Einstellungen gegenüber der Baseline (Element, Typ, Parameter, alt, neu). |
| `sim1.changedElements()` | Hinzugefügte und gelöschte Elemente gegenüber der Baseline. |
| `sim1.history()` | Jede einzelne Änderung mit Zeitstempel. |

`changedParameters()` und `changedElements()` vergleichen den aktuellen Stand mit der Baseline. Eine Änderung, die wieder zurückgenommen wurde, erscheint dort also nicht mehr. `history()` zeigt dagegen alle Schritte.

### Rechnen

```python
sim1.runSimulation(months=None, tap=None)
```

- **`months`:** ein Monat oder eine Liste. Der Standard sind alle 12, also 2016 Lastflüsse. Die Angabe wird als Einstellung gespeichert.
- **`tap`:** Stufenschalterstellung, gültig für alle Trafos mit Stufenschalter:
  - `int`: eine feste Stufe
  - `{Month: int}`: eine Stufe je Monat
  - `{"Trafoname": int}` oder `{"Trafoname": {Month: int}}`: einzelne Trafos

**Ablauf:**

1. Netz prüfen: externes Netz vorhanden, Skalierungen vollständig, nicht versorgte Busse melden.
2. Zeitreihen erzeugen. Die Lastwochen liegen auf denselben Kalendertagen wie die Wetterwochen.
3. Je Stunde die Leistungen setzen, ggf. den Stufenschalter stellen, `pp.runpp` rechnen und die Konvergenz prüfen. Nicht konvergierte Schritte werden als NaN gespeichert.
4. Ergebnisse speichern.

**Ergebnisse** stehen danach in `sim1.results`. Es sind Tabellen mit Zeitschritten als Zeilen und Elementnamen als Spalten:

| Tabelle | Inhalt |
|---|---|
| `vm_pu`, `va_degree` | Spannung aller Busse (Betrag in pu, Winkel in Grad) |
| `line_loading`, `line_i_ka` | Auslastung [%] und Strom [kA] aller Leitungen |
| `trafo_loading`, `trafo_p_mw` | Auslastung [%] und Wirkleistung auf der OS-Seite [MW] aller Trafos (< 0: Rückspeisung) |
| `ext_grid_p_mw` | Wirkleistung am externen Netz (< 0: Rückspeisung ins vorgelagerte Netz) |
| `load_p_mw`, `sgen_p_mw`, `element_p_mw` | eingesetzte Leistung der Lasten, Erzeuger bzw. beider |
| `info` | Monat, Stunde der Woche, Konvergenz, Stufenstellung |

### Auswerten

| Aufruf | Beschreibung |
|---|---|
| `sim1.summary()` | Kennzahlen: Spannungsbereich (mit Bus und Zeitpunkt), Stunden außerhalb des Spannungsbands, max. Leitungs-/Trafoauslastung, Rückspeisung, Energie |
| `sim1.summary(by="month")` | dieselben Kennzahlen je Monat |
| `sim1.powerFlow()` | Leistungsfluss je Stunde [MW]: Bezug externes Netz, Trafos (OS), Lasten und Erzeugung gesamt und je Profil, Verluste (= Bezug + Erzeugung − Lasten) |
| `sim1.energyBalance()` | Energie je simulierter Woche [MWh] aus `powerFlow()`, Anteil der Erzeugung an der Last, Stunden mit Rückspeisung |
| `sim1.createDiagram(diagram, elements, months=None, layout="week", time=None, title=None, limits=True, save=False, show=True, markers=None)` | Diagramme (siehe unten) |
| `sim.compare([sim1, sim2], diagram, elements, months=None, layout="week", markers=None)` | dieselbe Größe aus mehreren Simulationen in einem Diagramm |

**Diagrammarten (`Diagram`):**

| `Diagram` | Größe | `elements` | Grenzlinien |
|---|---|---|---|
| `VOLTAGE` | Busspannung [pu] | Busse | `v_min`, `v_max` |
| `LINE_LOADING` | Leitungsauslastung [%] | Leitungen | 100 % |
| `TRAFO_LOADING` | Trafoauslastung [%] | Trafos | 100 % |
| `TRAFO_POWER` | Wirkleistung Trafo OS [MW] | Trafos | 0 |
| `ELEMENT_POWER` | Leistung von Lasten/Erzeugern [MW] | Lasten, Erzeuger | – |
| `VOLTAGE_PROFILE` | Spannung entlang eines Pfades zu einem Zeitpunkt | Busse in Pfadreihenfolge | `v_min`, `v_max` |
| `NETWORK` | Netzplan zu einem Zeitpunkt (Spannung und Auslastung farbig) | – (Zeitpunkt über `time`) | – |
| `POWER_FLOW` | Leistungsfluss [MW]: externes Netz, Trafos, Lasten und Erzeuger je Profil, Verluste | Spaltennamen aus `sim1.powerFlow()`, z. B. `["Bezug externes Netz", "Erzeugung PV"]` | 0 |

**Netzplan (`NETWORK`):** Busse sind Kreise, Leitungen Linien, Trafos zwei Kreise auf einer grauen Linie zwischen OS- und US-Bus. Schalter sind Quadrate auf der Leitung nahe ihrem Bus: schwarz = geschlossen, weiß = offen. Das externe Netz ist ein schraffiertes Quadrat. Neben jeder Komponente stehen Name und Ergebniswert.

| Farbe | Skala |
|---|---|
| Bus | Spannung 0,8 … 1,2 pu: lila – blau – grün (1,0 pu) – gelb – rot |
| Leitung, Trafo | Auslastung 0 … 100 %: grün – gelb – rot, schwarz über 100 % |

| `time` | Zeitpunkt |
|---|---|
| `"2025-07-23 13:00"` | fester Zeitpunkt (muss in einer simulierten Woche liegen) |
| `"min"` / `"max"` | niedrigste / höchste Busspannung (bei `"max"` ohne Slack-Bus) |
| `"max_loading"` / `"max_trafo"` | höchste Leitungs- / Trafoauslastung (Standard: `"max_loading"`) |
| Liste | ein Plan je Eintrag |

Mit `months` und einem Stichwort entsteht je Monat ein Plan. `labels=False` blendet Namen und Werte aus, zum Beispiel bei großen Netzen. Die Lage der Busse kommt aus den Geodaten des Netzes. Busse ohne Geodaten (z. B. nach `attachNetwork`) werden automatisch angeordnet. Für einen einzelnen Lastfluss ohne Simulation: `netplan.plot_static(net)` nach `pp.runpp(net)`.

```python
sim1.createDiagram(Diagram.NETWORK, time="2025-07-23 13:00")
sim1.createDiagram(Diagram.NETWORK, time="min", months=[Month.JANUARY, Month.JULY])
```

Bei `POWER_FLOW` mit `layout="week"` besteht jedes Diagramm aus zwei Teilen: oben das externe Netz, die Trafos und die Lasten, unten die Erzeugung und die Verluste mit eigener Skala. Die Erzeugung ist im CIGRE-Netz viel kleiner als die Last und wäre sonst nicht zu erkennen. Ein positiver Bezug bedeutet Leistung aus dem vorgelagerten Netz, ein negativer Bezug Rückspeisung.

```python
sim1.createDiagram(Diagram.POWER_FLOW, months=Month.JULY)                       # alles
sim1.createDiagram(Diagram.POWER_FLOW, ["Bezug externes Netz", "Erzeugung gesamt"], layout="overview")
```

**Auswahl:**

- **`months`:** wählt Woche bzw. Monat, z. B. `Month.JULY` oder `[Month.JANUARY, Month.JULY]`. Ohne Angabe werden alle simulierten Monate gezeigt.
- **`layout`:**

  | `layout` | Darstellung |
  |---|---|
  | `"week"` | ein Diagramm je Monat; Uhrzeit im 6-h-Takt, senkrechte Linie bei 24 Uhr, Wochentag und Datum über jedem Tag |
  | `"season"` | je Jahreszeit ein Teildiagramm, die Monate übereinander |
  | `"overview"` | alle gewählten Wochen hintereinander |
  | `"boxplot"` | Verteilung je Monat |

- **`time`** (nur für `VOLTAGE_PROFILE`): ein Zeitstempel, `"min"` (Zeitpunkt der niedrigsten Spannung) oder `"max"`.
- **`save=True`:** speichert die Diagramme als PNG in `simulations/<name>/diagrams/`.
- **`markers`:** Liste eigener Markierungen, jede als `(farbe, wert, text)` oder `{"color": …, "value": …, "text": …}`. Optional gibt ein 4. Element bzw. `"style"` den Linienstil an (`"-"`, `"--"`, `":"`, `"-."`, Standard `"-."`).

  | `wert` | Darstellung | Layouts |
  |---|---|---|
  | Zahl | waagerechte Linie, Text rechts darüber | alle |
  | Zeitpunkt, z. B. `"2025-07-23 13:00"` | senkrechte Linie, Text senkrecht daneben | `week`, `season`, `overview` (nur wenn der Zeitpunkt in der gezeigten Woche liegt) |

  ```python
  sim1.createDiagram(Diagram.VOLTAGE, ["Bus 6", "Bus 11"], months=Month.JULY, markers=[
      ("orange", 0.95, "Warnschwelle 0,95 pu"),
      ("green", 1.00, "Nennspannung", ":"),
      ("purple", "2025-07-23 13:00", "PV-Spitze"),
  ])
  ```

### Einstellungen

| Einstellung | Standard | Bedeutung |
|---|---|---|
| `months` | alle 12 | simulierte Monate |
| `v_min`, `v_max` | 0,90 / 1,10 pu | Spannungsband für Kennzahlen und Diagramme |
| `tap` | `None` | Stufenschalterstellung (siehe `runSimulation`) |

Einstellungen werden mit kopiert. Änderungen erscheinen in `changedParameters()`.

---

## Elementklassen

Busse werden immer über ihren **Namen** angegeben. Alle Namen müssen im Netz eindeutig sein.

```python
sim.Bus("Bus X", vn_kv=20.0)
sim.Line("Line 9-X", from_bus="Bus 9", to_bus="Bus X", length_km=0.8,
         std_type="NA2XS2Y 1x185 RM/25 12/20 kV", switches=True)          # oder r/x/c/max_i_ka
sim.Trafo("ONS 1", hv_bus="Bus 5", lv_bus="NS 1", std_type="0.4 MVA 20/0.4 kV", tap_pos=0)
sim.ExtGrid("Netz", bus="Bus 0", vm_pu=1.03)
sim.Switch("S4", bus="Bus 9", element="Line 9-X", type="LBS", closed=False)
sim.Load("Wohngebiet", bus="Bus 3", profile=Profile.HAUSHALT, annual_kwh=500_000)
sim.Load("Halle", bus="Bus 9", profile=Profile.GEWERBE, p_mw=0.4)           # p_mw = Jahreshöchstleistung
sim.Load("Pumpe", bus="Bus 9", profile=Profile.FEST, p_mw=0.1, cos_phi=0.85)
sim.Load("Hof", bus="Bus 13", profile=Profile.LANDWIRTSCHAFT, annual_kwh=200_000)
sim.Load("Siedlung mit PV", bus="Bus 4", profile=Profile.HAUSHALT_PV, annual_kwh=500_000)
sim.Load("Siedlung mit PV+Speicher", bus="Bus 5", profile=Profile.HAUSHALT_PV_SPEICHER, annual_kwh=400_000)
sim.Generator("PV Dach", bus="Bus 4", profile=Profile.PV, kwp=250)
sim.Generator("WEA 2", bus="Bus 13", profile=Profile.WIND, p_mw=2.0)
```

---

## Speicherformat

Jede Simulation liegt in `simulations/<name>/`:

| Datei | Inhalt |
|---|---|
| `meta.json` | Name, Herkunft, Zeitstempel, Einstellungen, Änderungsprotokoll, Ergebnisstatus |
| `net.json` | aktuelles Netz (`pandapower.to_json`, inkl. Profilspalten) |
| `baseline.json` | Netz zum Zeitpunkt von `init` bzw. `copy` |
| `results/*.csv` | Ergebnistabellen |
| `diagrams/*.png` | mit `save=True` gespeicherte Diagramme |

Eine Simulation mit 12 Wochen belegt beim CIGRE-Netz etwa 4 MB (vor allem die Ergebnistabellen). Wenn die Simulationen nicht ins Repository sollen, gehört `task3/simulations/` in die `.gitignore`.

---

## Grenzen

- Das Profil ist je Element gleich. Lastspitzen einzelner Kunden sind in den Standardlastprofilen nicht enthalten.
- Die Auflösung ist stündlich (Wetterdaten), mit einer typischen Woche je Monat.
- Der Lastfluss ist symmetrisch, die Erzeuger speisen ohne Blindleistungsregelung ein.
- Der Stufenschalter wird als feste Stellung je Monat vorgegeben. Ein automatischer Regler (Aufgabe 5 aus Task 2) lässt sich über `tap` vorbereiten.
