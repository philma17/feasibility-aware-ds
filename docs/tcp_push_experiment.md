# Reproduzierbarer TCP-Schubs in Five Obstacle

Arbeitszweig: `dof`, Ausgangscommit `a466106`. Bereits vorhandene lokale
Änderungen bleiben erhalten; ihr genauer Ausgangsstand liegt in
`results/tcp_push_2026-11-19/provenance/baseline_worktree.patch`.
Im Zielrepository sowie `/`, `/home` und `/home/philipp` wurde keine
`AGENTS.md` gefunden; auch die Git-Historie enthält dort keine solche Datei.
Die `AGENTS.md` des separat untersuchten Vorgängers wurde gelesen; dort wurde
nichts geändert.

## Historische Bestandsaufnahme

- `git log --all -S 'apply_tcp_force' -- sim` und `git blame` führen zur
  Einführung mit `b634855e3b3295c12df5c390f53fb4c6f82e8bbe`,
  „Add PIC and position-servo simulation entry point“. Die ursprünglichen
  Zeilen 226–231 in `sim/controllers/pic.py` bilden bereits eine Weltkraft am
  TCP auf eine Kraft am Körper-COM plus `(p_tcp - xipos) × F` ab.
- `_run_source(..., pulses=...)` summierte überlappende rechteckige Pulse in
  Newton mit den Feldern `t_start_s`, `duration_s`, `force_N`. Der PIC-Callback
  setzte die Kraft vor `mj_step`. Der nominelle PIC hatte einen Physikschritt
  pro Reglerintervall; Servo hatte keinen angeschlossenen Kraftpfad. Alle
  `xfrc_applied`-Lasten wurden pauschal gelöscht. Der Angriffspunkt wurde im
  äußeren Reglerintervall ermittelt, nicht für weitere Substeps aktualisiert.
- `disturbance.enabled`/`disturbance.pulses` wurden in `pic.run` ausgewählt;
  `_run_source.observe` plante die wirksame Summe. `command` wendete sie
  physikalisch an. `traj.F_ext`, `P_ext=F_ext·xd` und die Energiebilanz
  protokollierten sie. **`F_ext` ist kein Eingang des PIC-Gesetzes.**
  `PICController.step(xd, f, dt)` erhält gemessene TCP-Geschwindigkeit, am
  gemessenen TCP ausgewertetes DS-Feld und Zeitschritt; sein interner Zustand
  sind Tank und Dämpfungsbasis. Erst die Physik verändert die Rückführung.
  Die Reglerkraft `F`, `tau_cmd` und `tau_applied` sind getrennte Kanäle.
- Die nominalen Konfigurationen einschließlich FR3 und J3+J5 haben
  `disturbance.enabled: false`, `pulses: []`, DS-Faktor 1; für Five Obstacle
  gilt aktuell `time_rate: 1.0`, `dt: 0.001`, Horizont 20 s, Zielradius 1 mm,
  PIC-λ `[1600,1600,1600]`, Tankinitialisierung 5 J und originale Momentlimits.
  Im Zielrepository sind weder aktivierte historische Konfigurationen noch
  historische Puls-Rohdaten bzw. entsprechende Tests belegt. Die damalige
  Docstring-Referenz `tests/test_sim_pic.py` bezeichnet keine vorhandene Datei.
- `asset_manifest.json:source_checkout` nennt die übernommene Quelle
  `LfD_sim@63e07b4f22240790818a0d884cb634e3b2a90054`, Branch
  `LfD/failure-report`. Die lokal archivierte Quelle
  `/home/philipp/erkenntnisarchiv_2026-11-19/sources/masterarbeit_windows/lfd-simulation/pic/sim_pic.py`
  und `/home/philipp/lfd-sim-pic/pic/sim_pic.py` enthalten dieselbe
  `apply_tcp_force`-Funktion (AST-Vergleich ohne Docstring exakt gleich;
  `provenance/source_comparison.json`). Das ist eine identifizierte
  Vorgängerimplementierung, keine Behauptung über einen unbelegten direkten
  Git-Import aus dem separaten Checkout.
- Im lokalen Vorgänger ist die Funktion schon mit `58599dc`,
  „Initial commit: PIC-DS prototype“, enthalten. Dessen README beschreibt
  den Vergleich Servo/PIC/PIC+pulse und einen gewählten, unvalidierten
  5-N-Puls `[0,3,4]`, 1.00–1.05 s. `tests/test_pic.py:193–201`
  (`test_3_push_recovers`) prüft sichtbare Auslenkung, Zielerreichen und
  Pulsfenster; `:227–245` vergleicht `xfrc` mit `Jp.T @ F`.
  `configs/picds.yaml:67–75` aktiviert diesen Puls. Der
  `EXPERIMENT_LOG.md:16–29` dokumentiert einen LASA/Sine-Einzelstart:
  bei nominaler Reibung PIC+pulse kein Ziel, reibungsfrei Zielzeit 6.802 s.
  README und Log sind **dokumentierte historische Ergebnisse**; die
  referenzierten Rohdateien fehlen im lokalen Vorgänger-`results/` und wurden
  hier nicht als neue Messung oder Five-Obstacle-Versuch verwendet.
- Belegter früher Zweck: Demonstration und Test einer externen Störung mit
  Auslenkung/Erholung in einem PIC-Einzelstart. **Für den konkreten Anlass
  der Übernahme in dieses Repository: ursprünglicher Zweck offen.**
  Die `git log -S`- und Blame-Ausgaben sind unter `provenance/` abgelegt.

## Gemeinsamer Kraftpfad

`sim/disturbance.py:TCPForceEnvironment` wird vom gemeinsamen Scheduler in
`sim/sim.py` bei jedem Physikschritt aufgerufen. PIC und Servo verwenden
denselben Hook. Der Puls ist eine konstante Weltkraft in Newton am jeweils
aktuellen TCP. `mj_kinematics` und `mj_comPos` aktualisieren TCP und COM vor
jedem Schritt. Die COM-Last enthält zusätzlich das Hebelarmmoment.

Der Hook addiert die eigene Last zur vorhandenen Last und stellt nach dem
Schritt exakt deren vorherigen Wert wieder her, auch bei einer Ausnahme.
Er löscht keine Last anderer Körper und akkumuliert den Puls nicht über
Substeps. Beginn und Dauer müssen ganzzahlige Vielfache des Physikzeitschritts
sein; die Aktivität wird mit ganzzahligen Schrittindizes auf `[a,b)` bestimmt.
`physics_F_ext` ist die Störkraft für das folgende Physikintervall, der letzte
Datensatz hat keine Folgewirkung. Nach Pulsende ist dieser Beitrag null.

Die offizielle [MuJoCo-API-Dokumentation](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata)
beschreibt `xfrc_applied` als kartesische Kraft/Drehmoment-Eingabe und `xipos`
als Weltposition des Körper-COM. Die Äquivalenz des verwendeten Ansatzes mit
`JpᵀF` wurde zusätzlich numerisch für alle registrierten Robotermodelle geprüft.

PIC-Gesetz, Feld, ideale Sollbahn, Tankinitialisierung, Gains und Momentlimits
bleiben unverändert. Es wird weder `qpos`, `qvel` noch ein TCP-Sollwert
verschoben. Ein zusätzlicher SVD-Logkanal enthält den kleinsten Singulärwert
von `Jp` der tatsächlich beweglichen Achsen. Optionale teure Standard-
Trackingmetriken können im Studienrunner deaktiviert werden; dies verändert
keinen Simulationsschritt. Die geforderten Paar-, Kontur- und Erholungsmetriken
werden aus den vollständigen Rohtrajektorien berechnet. Die öffentliche
Standardausführung berechnet ihre bisherigen Trackingmetriken weiterhin.

## Versuchsregel und Auswertung

Die fünf Pilotstarts verwenden Seed `20261119`; die 100 Hauptstarts behalten
die bisherigen Start-IDs und Seed `0`. Alle geplanten IDs bleiben erhalten.
Zeitpunkt und Richtung hängen ausschließlich von der ungestörten idealen
DS-Bahn ab: erste Zielannäherung bis zum ursprünglichen 1-mm-Eintritt,
ungefähr halbe Bogenlänge. Ist der nächste erreichbare Konturpunkt zu
tangential (`|cos| > 0.5`), werden Punkte im mittleren Abschnitt 30–70 % in
aufsteigender Entfernung zur halben Bogenlänge geprüft. Gleichstand bevorzugt
den früheren Zeitpunkt. Die Verbindung muss innerhalb X ohne Hindernis und
bis unmittelbar vor der Kontur bei B≤0 verlaufen. Fehlende Richtungen werden
als nicht planbar gekennzeichnet.

Die Pilot-Kandidaten, Auswahlregel, das Zielband 1–4 mm und eine minimale
seitliche Auslenkung von 2 mm stehen vor der Kalibrierung in
`pilot_plan.json`. Die ausgewählte Regel wird vor den gestörten Hauptläufen
in `protocol_frozen.json` gespeichert. Pro Start haben beide Roboter exakt
Beginntakt, Dauer und denselben Weltkraftvektor. Im Hauptversuch erfolgt
keine fallweise Nachkalibrierung.

Die grüne Kontur wird aus dem zu `model_f.pth` gehörenden, hashgeprüften
`model_b.pth` auf einem 1001×1001-Gitter extrahiert (Rasterweite 0.2 mm).
Abstände sind euklidische Abstände zu den interpolierten Kontursegmenten,
keine Barrierwerte; B selbst wird separat direkt durch das originale
float32-Netz ausgewertet. Es handelt sich um eine numerische Kontur und um
Prüfungen an allen gespeicherten Zuständen, nicht um kontinuierliche Beweise.
Die Hindernisse sind die Kreise aus der Szenariometadatei, unabhängig von B.

Zeitphasen für Zustandsereignisse: vor `t<a`, während `a≤t≤b` einschließlich
Pulsende, nach `t>b`. Der Kraftbeitrag wirkt auf `[a,b)`.
Box-Exit, B>0 innerhalb X und Hinderniseintritt erhalten eigene Zähler und
erste Zeitpunkte; Kontrollen werden ebenso ausgewertet. Auslenkung am
Pulsende wird gegen die zeitgleiche eigene ungestörte TCP-Bahn gemessen.
Zielerholung ist der erste Eintritt ab Pulsende mit mindestens 0.5 s
ununterbrochenem Aufenthalt im ursprünglichen 1-mm-Ziel. Erfolgt dieser nicht
innerhalb 20 s, bleibt die Erholungszeit fehlend und wird rechtszensiert mit
angegebener Beobachtungsdauer. Das ursprüngliche Endzustandskriterium wird
zusätzlich unverändert berichtet.

Momentensättigung wird aus `|tau_cmd| > tau_limit + 1e-12` bestimmt;
Gelenknamen, Zeitintervalle und Gesamtdauern bleiben nachvollziehbar.
Der kleinste translatorische Singulärwert wird unmittelbar vor dem Puls und
ab Pulsende bis zum bestätigten Haltezeiteintritt bzw. Horizont gemessen.

## Reproduktion

Die Umgebung ist `/home/philipp/lfd_env/bin/python`; die Versionen stehen in
der Provenienz. Keine wissenschaftliche Interpretation wird vorgenommen.
Die bestehende `model_f.pth` muss vor Übernahme eines Barrier-Modells mit
dessen benachbarter `model_f.pth` exakt übereinstimmen. Für diesen Versuch
war die Quelle `/home/philipp/S2NNDS/models_verified/2D_Shapes/Five_Obstacle_DS/`;
der F-Hash ist `fbcd10d2ff7dfa9c34a24366fa6992df54aa61f89b45f1c7b36a1dee85c0e82c`.
`model_b.pth` liegt danach lokal neben dem F-Modell; sein Hash wird gespeichert.

```bash
cd /home/philipp/feasibility-aware-ds
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/philipp/lfd_env/bin/python
OUT=results/tcp_push_repeat_20261119
$PY -B -m tools.tcp_push plan --output "$OUT" --barrier-model /home/philipp/S2NNDS/models_verified/2D_Shapes/Five_Obstacle_DS/model_b.pth
$PY -B -m tools.tcp_push audit --output "$OUT"
$PY -B -m tools.tcp_push pilot --output "$OUT" --workers 3
$PY -B -m tools.tcp_push freeze --output "$OUT"
$PY -B -m tools.tcp_push main --output "$OUT" --workers 4
$PY -B -m tools.tcp_push report --output "$OUT"
$PY -B -m tools.tcp_push_checks configs --output "$OUT"
$PY -B -m tools.verify_tcp_push --output "$OUT" --legacy-pic results/tcp_push_2026-11-19/provenance/pic_before.py
$PY -B -m unittest discover -s tests -v
```

Alle Stufen akzeptieren `--output PATH`; der Standardordner ist
`results/tcp_push_2026-11-19`. Bereits abgeschlossene Runs werden gelesen,
niemals neu berechnet und überschrieben. Für eine unabhängige Wiederholung
einen neuen Ordner wählen. Kontrollen bleiben in ihren ursprünglichen Ordnern;
der Audit verzeichnet Quelle und Hash jedes verwendeten Runs.


## Kontrollaudit und Pilotrevision vor den Hauptläufen

Der Audit akzeptiert 100 Kontrollen aus `results/five_obstacle_pic_fr3_100`
und 50 aus `results/five_obstacle_pic_fr3_fixed_j3_j5_50`. Die 100er-Serie
`five_obstacle_pic_fr3_fixed_j3_j5_100` hat mit `time_rate: demo` effektiv
0.32661202093817854/s statt 1/s und wird ausgeschlossen. Die ersten 50
Startpunkte stimmen trotz verschiedener Samplinganzahl exakt mit den 100er-
Starts überein (NumPy-Ziehungspräfix). Für IDs 050–099 entstehen 50 neue
Kontrollen. Geprüft werden effektive Konfiguration, Startpunkt, gespeicherte
Anfangsgelenke, Anfangsgeschwindigkeit null, Kraft null, Zeitraster sowie die
Hashes der tatsächlich verwendeten Netz-, Metadaten-, MJCF- und Meshdateien.
Unterschiede allein am Inventar, Quellhinweisen und unbenutzten Varianten-XMLs
werden einzeln protokolliert. Alle dynamisch verwendeten Eingaben müssen
übereinstimmen. Die gesamte vollständige Inventarprüfung bleibt zusätzlich
unter `provenance/control_audit_full_inventory_initial.json` erhalten.

Die konstante Pilotstärke deckt das Zielband nicht bei allen Starts ab:
80 N ergeben bei Piloten 000–002 ungefähr 2.2–3.3 mm, bei 003–004 etwa
5.6–6.2 mm Abstand am Pulsende. 120 N bringen 003–004 auf 2.4–3.0 mm,
überschreiten bei 000–002 jedoch teilweise B=0. Die vollständige Kandidaten-
Tabelle enthält 40,80,…,320 N bei jeweils 150 ms; keine Kandidaten werden
nachträglich gelöscht.

Die Pilotphase prüft deshalb vor den gestörten Hauptläufen eine geometrische
Stärkeregel. `d` ist der Konturabstand der **idealen** ungestörten DS-Bahn am
vorgeplanten Pulsbeginn in mm. Die Formel lautet
`F_N = clip(40 * floor((d - 2.5)/(40*g) + 0.5), 40, 320)`.
Die Pilotdaten prüfen `g ∈ {0.065,0.070,0.075,0.080,0.085} mm/N`.
Auswahl: zuerst größtmögliche Zahl bestandener Pilotvarianten, dann kleinste
summierte quadratische Abweichung zur Bandmitte 2.5 mm; bei Gleichstand
Nähe zu 0.075 mm/N. Nur Pilotmessungen gehen in diese Kalibrierung ein.
Die resultierenden Pulse aller Haupt-IDs werden vorab in
`pulse_plan_frozen.json` und `frozen_pulses.csv` gespeichert. Beide Roboter
erhalten je ID denselben Vektor. Die vorläufige konstante Auswahl bleibt
unverändert als `protocol_constant_candidate.json` erhalten.

**Grenze der Kalibrierung:** Pilot 003 verlässt bereits vor dem Puls die
Modellbox X: FR3 in 810, J3+J5 in 731 gespeicherten Zuständen. Genau dieselben
Vorereignisse sind in den ungestörten Kontrollen vorhanden. Ein erst später
angesetzter Puls kann die Bedingung „vor und während vollständig in X“ für
diesen Start nicht erfüllen. Dieser Pilot bleibt mit beiden Varianten im
Nenner. Die Pilotplots und Tabellen zeigen den konkreten Grund; eine
vollständig bestandene Fünf-Start-Kalibrierung wird nicht behauptet.
Die 100 geplanten Haupt-IDs werden mit der dokumentierten eingefrorenen Regel
vollständig untersucht, einschließlich solcher Vorereignisse.

Die Konturziele sind zusätzlich für alle 105 Starts und beide Varianten mit
`ik_pose` auf Erreichbarkeit innerhalb der ursprünglichen Gelenkgrenzen
geprüft (210/210 Positionslösungen, Positionsfehler <1 µm). Das belegt
geometrische Erreichbarkeit, keine dynamische Erreichbarkeitsgarantie.


Die endgültige Pilotkalibrierung wählt `g=0.070 mm/N`. Zuvor wurde dieselbe
Formel mit dem idealen Abstand am Pulsende geprüft; sie verfehlte bei Pilot
004 das Band. Diese verworfene Pilotregel bleibt einschließlich Plan und
Tabellen als `*_distance_end_candidate.*` erhalten. Erst danach, weiterhin
vor jedem gestörten Hauptlauf, wurde der ideale Abstand am Pulsbeginn
verwendet. Auch diese Revision ist in `protocol_frozen.json` beschrieben.
Die endgültige Regel bringt alle zehn Pilotvarianten am Pulsende auf
1–4 mm Innenabstand. Acht erfüllen zusätzlich die vollständige Vor-/Während-
Bedingung; die beiden Varianten von Pilot 003 scheitern an ihrem vorherigen
Box-Exit. Die Hauptplanung enthält 47 IDs mit 80 N, 50 mit 120 N und 3 mit
160 N. Diese Zuordnung wurde vor dem Start gespeichert.

`B` und `X` beziehen sich auf die normierten, projizierten XY-Koordinaten des
zweidimensionalen Modells. Welt-Z bleibt in den Rohtrajektorien gespeichert.
Die seitliche Auslenkung ist zusätzlich vorzeichenbehaftet in der
Bewegungsebene gespeichert (`signed_lateral_deflection_end_mm`); die Norm
`lateral_deflection_end_mm` umfasst die vollständige orthogonale Komponente
zur gemeinsamen lokalen idealen Bahntangente am Pulsbeginn, einschließlich
eines möglichen Z-Anteils. Die volle Auslenkung ist der zeitgleiche Abstand
zur eigenen Kontrolle am Pulsende.

Weitere Prüfkommandos:

```bash
$PY -B -m tools.tcp_push_checks boundary
$PY -B -m tools.tcp_push_checks legacy
$PY -B -m tools.tcp_push_checks configs
```

`legacy_window_comparison.json` vergleicht die historischen Gleitkomma-
Zeitfenster mit der ganzzahligen Aktivierung für alle 105 geplanten Starts.
Bei 104 stimmen sie in 150 Intervallen überein. Für `start_021` aktiviert
der alte Vergleich wegen Rundung 151 Intervalle. Die neue Umsetzung führt
für jede ID genau den geplanten 150-ms-Puls aus. Das ist eine dokumentierte
Korrektur des Zeitfensters; der COM-Kraftansatz selbst bleibt äquivalent.


Die Abschlussauswertung enthält außerdem die seitliche Komponente relativ
zur **eigenen Kontrollbahntangente am Pulsende**
(`lateral_own_control_tangent_end_mm` und vorzeichenbehaftete Variante),
den geometrischen Abstand zur Kontroll-Polylinie und den Welt-Z-Anteil.
Die lokalen Kontrolltangenten vor dem Puls und am Pulsende stehen als
Weltvektoren in der CSV; der Kraftwinkel an der Kontrollbahn wird separat
von der idealen Planungstangente ausgewiesen. Die Hauptplots zeigen die
lokale Kontrolltangente schwarz und die ideale Planungstangente grau.

Die Kontroll-Singulärwerte werden für ältere Kontrollen ohne diesen Kanal
aus jedem gespeicherten Gelenkzustand im jeweiligen Erholungsfenster neu
berechnet; `control_sigma_source` kennzeichnet dies. Regler-Sättigung bezieht
sich auf `tau_cmd` und die unveränderten Aktuatormomentlimits, nicht auf eine
Summe mit äußeren Lasten. Die Szenario-Hindernisse werden geometrisch nach
den Metadaten geprüft.

```bash
$PY -B -m tools.tcp_push_checks geometry
# Optionaler einzelner Replay über die öffentliche Simulations-CLI:
$PY -B main.py results/tcp_push_2026-11-19/main/start_000/FR3/force_80N/config_resolved.yaml \
  --output results/tcp_push_replay_start_000_FR3
```

Die erzeugten `config_resolved.yaml`-Dateien enthalten die tatsächlich
verwendete Anfangsgelenkkonfiguration explizit. Die ursprüngliche
Konfiguration einschließlich einer gegebenenfalls ausgeführten IK bleibt
im jeweiligen Manifest erhalten. F-/B-Netze, Szenariometadaten,
Konfigurationen und Paketversionen sind zusätzlich unter
`provenance/model_inputs/` bzw. `provenance/versions.json` gesichert.

## Ergebnisse der vollständigen Hauptserie

Alle 100 geplanten IDs wurden für beide Varianten ausgewertet: **200/200
gestörte Läufe, 200/200 passende Kontrollen; keine fehlende Haupt-ID**.
Jeder Hauptlauf enthält 20 001 Zustände und 20 000 Physikintervalle.
150 Kontrollen wurden nach dem Audit übernommen, 50 neu simuliert.

| Beobachtung | FR3 | FR3_fixed_J3_J5 |
|---|---:|---:|
| Auswertbare gestörte Läufe | 100/100 | 100/100 |
| Ursprüngliches Zielkriterium am Ende | 97/100 | 100/100 |
| Zielkriterium der Kontrollen | 100/100 | 100/100 |
| 0.5-s-Zieleintritt ab Pulsende beobachtet | 97/100 | 100/100 |
| Median Zeit ab Pulsende bis zum Halte-Eintrittsbeginn | 1.851 s (n=97) | 1.868 s (n=100) |
| Median gepaarte Zeitdifferenz zur Kontrolle | −0.086 s (n=97) | −0.078 s (n=100) |
| Median volle Auslenkung am Pulsende | 9.408 mm | 9.026 mm |
| Median seitlich zur eigenen Kontrolltangente | 7.943 mm | 7.895 mm |
| Innenabstand im vorab festgelegten 1–4-mm-Zielband am Pulsende | 76/100 | 86/100 |
| Box-Exit vor dem Puls, ebenso in Kontrolle | 6/100 | 5/100 |
| Box-Exit während / nach Puls | 0 / 0 | 0 / 0 |
| B>0 innerhalb X während / nach Puls | 3 / 3 | 0 / 0 |
| Geometrischer Hinderniseintritt vor / während / nach Puls | 0 / 0 / 0 | 0 / 0 / 0 |
| Läufe mit Regler-Momentensättigung | 53/100 | 0/100 |
| Kleinster σ vor Puls über alle Starts | 0.247329 m/rad | 0.247762 m/rad |
| Kleinster σ im Erholungsfenster über alle Starts | 0.244970 m/rad | 0.248584 m/rad |

Bei FR3 sind **013, 045 und 047** die drei Fälle mit B>0 während und nach
dem Puls sowie ohne Zieleintritt bis 20 s. Sie erhielten gemäß der vorab
gespeicherten Regel jeweils 160 N. Ihre Erholungszeiten bleiben fehlend mit
`recovery_censored: true` und individuell angegebener Beobachtungsdauer.
Sie bleiben im Nenner 100; die bedingten Zeitmediane sind keine Ersetzung
der zensierten Zeiten. Die ungestörten Kontrollen dieser IDs erreichen das
Ziel und zeigen während und nach dem Pulsvergleichsfenster kein B>0.

FR3 sättigt J5 in 53 Läufen, J1 und J3 jeweils in den drei oben genannten
Läufen. Summiert über alle FR3-Hauptläufe: J1 **0.436 s**, J3 **0.424 s**,
J5 **7.423 s**; andere Achsen 0 s. Jede einzelne Dauer und jedes zusammenhängende
Sättigungsintervall stehen mit ursprünglichem Gelenknamen in der CSV.
Alle Kontrollen haben 0 Sättigungsdauer.

Der Kraftpfad ist für **200/200** Läufe geprüft: exakt 150 aktive
Physikintervalle, maximale Impulsabweichung **7.11·10⁻¹⁴ N·s**, identische
Regler- und Physik-TCP-Logs, Kraft und eigener Hebelarmmoment-Beitrag nach
Pulsende exakt null. Alle 200 Paare stimmen bis einschließlich Pulsbeginn
in P/Q/QD exakt überein. Die Kontrollen und gestörten Läufe haben unmittelbar
vor Pulsbeginn ebenfalls exakt gleiche translatorische Singulärwerte.

Der Vergleich der alten und gemeinsamen Umsetzung ist für den tatsächlich
verwendeten Puls von Start 000 bei beiden Robotern in allen verglichenen
Kanälen exakt gleich. Bei Start 021 wirkt im alten Gleitkommafenster ein
zusätzlicher Schritt; die numerischen Zustands- und Impulsunterschiede sind
in `legacy_actual_pulses.json` getrennt dokumentiert.

Die Prüfung an den tatsächlichen Kontrollposen findet für alle 210
Pilot-/Hauptstart-Varianten einen Schnitt des Kraftstrahls mit B=0 in X und
eine lokale IK-Lösung aus der aktuellen Gelenkpose. Keine Richtung liegt
innerhalb 30° zur tatsächlichen lokalen Kontrolltangente. Die vollständigen
Winkelwerte stehen in `actual_geometry_audit.csv`; die vorab verwendete
Planungsregel zur idealen Tangente wird dadurch nicht geändert.

Ein Kontrollprozess endete mit Signal 15; die Ursache ist nicht belegt.
Der unvollständige Versuch von `start_073` wurde unter dessen `attempts/`
archiviert und mit identischen Eingaben neu ausgeführt. Abgeschlossene
Trajektorien wurden nicht überschrieben. Die Quellen der anfänglichen
Pilotkohorte und Unterschiede beim Zeitpunkt der Quellhash-Aufnahme sind
in `provenance/pilot_source_note.md` offen dokumentiert. Für die gestörten
Hauptläufe passen alle 19 gespeicherten Quellhashes zu erhaltenen Dateien
(`provenance/main_loaded_source_validation.json`).

Die anfängliche Suite bestand aus 61 Tests; nach der Umsetzung bestehen
**68 Tests** (`provenance/final_all_tests.log`, 250.251 s unter paralleler
Simulationslast). Kraftäquivalenz für alle registrierten Modelle, Substeps,
bewegter TCP, zusätzliche Lasten, Fehler-Cleanup, Servo-Hook, TCP-Offset,
Cache- und Nominalgleichheit sind abgedeckt. Alle geforderten Hauptdaten
liegen vor; historische Pulsrohdaten des Vorgängers fehlen weiterhin.
Es wird keine wissenschaftliche Interpretation aus diesen Beobachtungen
abgeleitet.

## Ergebnisdateien

- [CSV aller 200 Start-/Variantenfälle](../results/tcp_push_2026-11-19/per_start_variant.csv)
- Je Start und Variante eine eigene `main/start_NNN/ROBOT/summary.csv`;
  Rohtrajektorie, Controllerdiagnostik, Manifest und explizite Replay-Konfiguration
  liegen darunter im Kraftordner.
- [Zusammenfassung](../results/tcp_push_2026-11-19/summary.json),
  [eingefrorene Pulse](../results/tcp_push_2026-11-19/frozen_pulses.csv),
  [Kontrollaudit](../results/tcp_push_2026-11-19/control_audit.csv)
- [XY-Vergleich Start 000](../results/tcp_push_2026-11-19/comparisons/start_000.png)
  und [XY-Vergleich Start 013](../results/tcp_push_2026-11-19/comparisons/start_013.png)
- Alle 100 XY-Paarplots: `results/tcp_push_2026-11-19/comparisons/`.
- [Zeitverlauf Start 013](../results/tcp_push_2026-11-19/start_013_recovery.png):
  20 s Zielabstand, B, äußere Kraft getrennt von PIC-Kraft und translatorischer
  Singulärwert; die Störung endet nach 150 ms.
- [Pilot 003: früher Box-Exit vergrößert](../results/tcp_push_2026-11-19/pilot_003_before_pulse_box_exit_zoom.png)
- [Vollständige Physikprüfung](../results/tcp_push_2026-11-19/physics_validation.json)
  und `physics_validation.csv`; [Artefakthashes](../results/tcp_push_2026-11-19/artifact_index.json).

Die ergänzenden Abschlussprüfungen sind ebenfalls reproduzierbar:

```bash
$PY -B -m tools.tcp_push_checks validate
$PY -B -m tools.tcp_push_checks recovery --start-id start_013
```


Auch ein vollständiger **20-s-Nominallauf** von Start 000 wurde mit der
aktuellen gemeinsamen Implementierung erneut ausgeführt: Störung deaktiviert,
die Pulsliste ausdrücklich trotzdem befüllt. P, Q, QD, PIC-Kraft, F_ext,
tau_cmd, tau_applied, Tankzustand und qacc stimmen über alle 20 001 Zustände
bzw. ihre Intervallkanäle exakt mit der vorhandenen ungestörten FR3-Kontrolle
überein. F_ext bleibt überall null. Ergebnis und explizite Konfiguration:
`nominal_replay_verification.json`; zusätzliche Verifikationstrajektorie unter
`provenance/nominal_replay_start_000_FR3.npz`.

```bash
$PY -B -m tools.tcp_push_checks nominal
```
