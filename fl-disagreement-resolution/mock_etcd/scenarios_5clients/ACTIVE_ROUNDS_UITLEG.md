# Wat betekent `active_rounds`?

## Uitleg

`active_rounds` bepaalt **WANNEER** een disagreement actief is tijdens de federated learning rounds.

### Voorbeelden:

#### 1. PERMANENTE Disagreement
```json
{
  "type": "inbound",
  "target": "client_1",
  "active_rounds": {
    "start": 1,
    "end": null
  }
}
```
**Betekenis:** Client 0 sluit client 1 uit vanaf round 1, en dit blijft **voor altijd** actief (null = geen einde).

#### 2. TIJDELIJKE Disagreement
```json
{
  "type": "inbound",
  "target": "client_1",
  "active_rounds": {
    "start": 1,
    "end": 3
  }
}
```
**Betekenis:** Client 0 sluit client 1 uit **alleen** tijdens rounds 1, 2, en 3. Na round 3 stopt de exclusion automatisch.

#### 3. Laat Startende Disagreement
```json
{
  "type": "inbound",
  "target": "client_1",
  "active_rounds": {
    "start": 5,
    "end": 10
  }
}
```
**Betekenis:** Client 0 sluit client 1 uit **alleen** tijdens rounds 5, 6, 7, 8, 9, en 10.

## Waarom is dit belangrijk?

**Scenario 1 (permanent):**
- Client 0 sluit client 1 uit vanaf round 1 -> **voor altijd**
- Track `track_0_no1` blijft bestaan in alle rounds

**Scenario 4 (tijdelijk):**
- Client 0 sluit client 1 uit alleen in rounds 1-3
- Track `track_0_no1` bestaat alleen in rounds 1-3
- Vanaf round 4 gebruiken alle clients de `global` track

## Conclusie

**Niet alle scenario's zijn uniek!** Er zijn **12 scenario's die echt identiek zijn** aan andere scenario's:

### Groep 1: Ring Patroon (8 identieke scenario's)
- scenario8, scenario10, scenario11, scenario12, scenario24, scenario26, scenario27, scenario32
- **Allemaal:** 5-client ring (0->1->2->3->4->0), allemaal permanent

### Groep 2: Geen Disagreements (4 identieke scenario's)
- scenario0, scenario7, scenario13, scenario20, scenario25
- **Allemaal:** Geen disagreements, zelfde expected_tracks, zelfde validation_rules

### Groep 3: Ring Patroon Variant (3 identieke scenario's)
- scenario9, scenario22, scenario23
- **Allemaal:** 5-client ring (0->1->2->3->4->0), allemaal permanent

## Wat betekent dit voor jouw experimenten?

Je hebt **20 unieke scenario's** en **15 duplicaten** (12 echt identiek + 3 die alleen verschillen in naam/beschrijving).

**Aanbeveling:** Als je alle scenario's wilt testen, kun je de duplicaten overslaan om tijd te besparen. Of je houdt ze voor traceability naar de originele 10-client scenario's.
