# Task 4 – Erneuerbare Energien im CIGRE-Mittelspannungsnetz und im Ortsnetz

Bearbeitung des Arbeitsauftrags W4 (`W4_Arbeitsauftrag.ipynb`), Modul Smart Grids, TH Köln.

## Hinweis
Da die Abgabe mit den ausgeführten Simulationen zu groß wäre sind diese incht enthalten in der abgabe.
Wenn die Notebooks aber ausgeführt werden, werden aber alle Simulationen durchlaufen, welches aber länger dauert.
Die aktuelle version inklusive ausgeführten Simulationen: https://github.com/RoyalKnighter/SmartGrids

## Ordner

| Ordner | Inhalt |
|---|---|
| `notebooks/` | alle Notebooks (Reihenfolge unten) |
| `src/` | Python-Module (Simulationsumgebung, Auswertungen); Beschreibung in `src/README.md` |
| `data/` | BDEW-Standardlastprofile und Wetterwochen 2025 |
| `bilder/` | Netzpläne aus der pandapower-Dokumentation |
| `simulations/` | gespeicherte Zeitreihen; entsteht beim Ausführen und gehört nicht ins Repository |
| `ergebnisse/` | vollständige Spannungstabelle aller Busse (entsteht in Teil 7) |

## Notebooks

Die Notebooks bauen aufeinander auf und werden in dieser Reihenfolge ausgeführt (jeweils im Ordner `notebooks/`):

| Nr. | Notebook | Aufgabe |
|---|---|---|
| 0 | `Task4-Datengrundlage.ipynb` | Lastprofile, Wetter, Anlagen, PV-Haushalte, **zentrales Quellenverzeichnis** |
| 1 | `Task4-1_Baseline.ipynb` | 1. Baseline und Modellverständnis |
| 2 | `Task4-2_Ortsnetz.ipynb` | 2. Niederspannungs-Ortsnetz und Wahl des Anschlussbusses |
| 3 | `Task4-3_Ortsnetz_Auslegung.ipynb` | 3. Spannungsband, Auslegung des Ortsnetzes, Schalterstellungen |
| 4 | `Task4-3b_Regelung.ipynb` | 3. Regelbarer Ortsnetztransformator, Backup-Plan mit S1–S3 |
| 5 | `Task4-4a_Erzeugung_Anschluss.ipynb` | 4. Windpark und PV-Anlage: Anschlussbus, Kombination, (n-1) |
| 6 | `Task4-4b_Szenarien.ipynb` | 4. PV-Haushalte im Ortsnetz, alle Szenarien in allen Schalterstellungen, hohe/niedrige Situationen, Blindleistung |
| 7 | `Task4-5_Lastflussumkehr.ipynb` | 5. Lastflussumkehr |
| 8 | `Task4-6_Leitung_3-8.ipynb` | 6. Engpass auf Line 3-8 und Kapazitätserhöhung |
| 9 | `Task4-7_Methodik.ipynb` | 7. Methodik, Konvergenz, Spannungen an allen Bussen, Modellgrenzen |
| 10 | `Task4-8_Leitfragen_Ausblick_Fazit.ipynb` | 8. Leitfragen, 9. Ausblick Smart Grid, Fazit |
| – | `Simulation_Beispiel.ipynb` | Beispiel für die Simulationsumgebung (nicht Teil der Auswertung) |

Ein vollständiger Durchlauf aller Zeitreihen dauert mehrere Stunden. Danach liegen die Ergebnisse in `simulations/`, und weitere Durchläufe laden sie aus dem Speicher, solange sich Netz, Einstellungen und Daten nicht ändern (`sim.ONLY_RUN_IF_CHANGED = True`).

## Voraussetzungen

Python 3.10 oder neuer mit `pandapower` (3.x), `numpy`, `pandas`, `matplotlib`, `networkx`, `jupyter`.

---

Dies ist die arbeit von Tinus
