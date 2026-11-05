# Airline RAG Bonus Results

Embeddings: `text-embedding-3-small`. Corpus: 6 documents. Bonus A uses characters, unlike the starter's token-aware splitter. Checklist coverage is a limited diagnostic; inspect the displayed evidence.

Fixed: 104 chunks; target=500 characters; overlap=0; longest=937; oversized=6.

Recursive: 65 chunks; target=800 characters; overlap=150; longest=798; oversized=0.

CharacterTextSplitter uses its default paragraph separator. A paragraph exceeding 500 characters can remain oversized; 500 is a target, not a guaranteed hard limit.

# Bonus A — Chunking Strategy Comparison

## Query 5

How should ground staff communicate a delay of more than 2 hours to connecting passengers?

### Fixed — top 3

#### Rank 1: fixed-079

Source: `airline_kb/IROPS_Procedures.txt`; start character: 2444; length: 468 characters; initial vector rank: 1; cosine distance=0.2434 (lower is better).

```text
For delays exceeding 2 hours, ground staff must also directly notify connecting passengers at gates. A list of connecting passenger gate locations is available via the Passenger Connect tool in the Operations Portal. Ground staff must personally approach each connecting gate and coordinate with the receiving agent.

=============================================================
SECTION 3: SEVERITY LEVELS
=============================================================
```

#### Rank 2: fixed-045

Source: `airline_kb/Delay_Classification.txt`; start character: 2901; length: 276 characters; initial vector rank: 2; cosine distance=0.3424 (lower is better).

```text
Step 4: Gate agents must update the passenger announcement every 15 minutes (not 30 minutes as in standard delays) once an ECT has been missed. Passengers waiting more than 3 hours from original departure time must be offered the option to deplane if the aircraft has boarded.
```

#### Rank 3: fixed-078

Source: `airline_kb/IROPS_Procedures.txt`; start character: 2115; length: 327 characters; initial vector rank: 3; cosine distance=0.3517 (lower is better).

```text
For passenger-facing communications, the gate agent is responsible for making public address announcements at the following intervals:
- Initial delay announcement: Within 15 minutes of known delay
- Subsequent updates: Every 30 minutes until departure or cancellation
- Cancellation announcement: Immediately upon confirmation
```

Evidence checklist: 4/4. Found: Initial announcement and continuing updates; Special procedure for delays over two hours; Passenger Connect identifies connecting gates; Personal gate visit and receiving-agent coordination. Not found: none.

### Recursive — top 3

#### Rank 1: recursive-048

Source: `airline_kb/IROPS_Procedures.txt`; start character: 2115; length: 797 characters; initial vector rank: 1; cosine distance=0.2688 (lower is better).

```text
For passenger-facing communications, the gate agent is responsible for making public address announcements at the following intervals:
- Initial delay announcement: Within 15 minutes of known delay
- Subsequent updates: Every 30 minutes until departure or cancellation
- Cancellation announcement: Immediately upon confirmation

For delays exceeding 2 hours, ground staff must also directly notify connecting passengers at gates. A list of connecting passenger gate locations is available via the Passenger Connect tool in the Operations Portal. Ground staff must personally approach each connecting gate and coordinate with the receiving agent.

=============================================================
SECTION 3: SEVERITY LEVELS
=============================================================
```

#### Rank 2: recursive-032

Source: `airline_kb/Delay_Classification.txt`; start character: 5314; length: 787 characters; initial vector rank: 2; cosine distance=0.3622 (lower is better).

```text
When weather delays are anticipated more than 4 hours in advance (e.g., based on meteorological forecasts), the OCC Weather Desk must issue a Weather Advisory to all affected stations. Gate agents receiving a Weather Advisory should begin proactive passenger communication and coordinate with Customer Service for early rebooking options.

=============================================================
SECTION 5: PASSENGER NOTIFICATION STANDARDS
=============================================================

All delays require passenger notification within the following timeframes:
- At gate: Within 15 minutes of known delay, then every 30 minutes
- App/SMS: Automated push notifications triggered by SABRE status change
- Email: Automated notification for delays exceeding 60 minutes
```

#### Rank 3: recursive-028

Source: `airline_kb/Delay_Classification.txt`; start character: 2648; length: 785 characters; initial vector rank: 3; cosine distance=0.3702 (lower is better).

```text
Step 2: The Duty Manager contacts MCC for a revised ECT and determines if an aircraft swap is now required.

Step 3: If the revised ECT adds more than 60 minutes beyond the original estimate, the Duty Manager must notify the Customer Service Director.

Step 4: Gate agents must update the passenger announcement every 15 minutes (not 30 minutes as in standard delays) once an ECT has been missed. Passengers waiting more than 3 hours from original departure time must be offered the option to deplane if the aircraft has boarded.

Step 5: Passengers who have boarded and waited more than 3 hours on the aircraft without takeoff are subject to the DOT Tarmac Delay Rule. In this case, the Captain must be immediately informed and the OCC must initiate the Tarmac Delay Contingency Plan.
```

Evidence checklist: 4/4. Found: Initial announcement and continuing updates; Special procedure for delays over two hours; Passenger Connect identifies connecting gates; Personal gate visit and receiving-agent coordination. Not found: none.

### Qualitative assessment

Both strategies cover the same number of checklist items. Fixed uniquely covers: no additional checklist items. Recursive uniquely covers: no additional checklist items. Both miss: no checklist items. The complete chunks also show how much surrounding policy context is preserved.

## Query 7

Compare the rebooking options available to elite status passengers versus standard passengers during a weather cancellation.

### Fixed — top 3

#### Rank 1: fixed-034

Source: `airline_kb/Customer_Rebooking_Policy.txt`; start character: 4672; length: 292 characters; initial vector rank: 1; cosine distance=0.3362 (lower is better).

```text
For weather cancellations, hotel accommodation is NOT required by DOT regulations and is NOT provided to Standard or Silver Elite passengers by default.

Gold Elite and Platinum Elite passengers retain their hotel accommodation benefit even during weather events, as per their tier agreement.
```

#### Rank 2: fixed-036

Source: `airline_kb/Customer_Rebooking_Policy.txt`; start character: 5379; length: 364 characters; initial vector rank: 2; cosine distance=0.3638 (lower is better).

```text
All change fees and fare differences are waived during weather cancellations for all passengers regardless of tier, as long as the rebooked travel is within 14 days of the original departure date.

=============================================================
SECTION 4: REBOOKING PROCESS — AGENT STEPS
=============================================================
```

#### Rank 3: fixed-026

Source: `airline_kb/Customer_Rebooking_Policy.txt`; start character: 1395; length: 484 characters; initial vector rank: 3; cosine distance=0.3702 (lower is better).

```text
Rebooking window: 24 hours from original departure
Available flights: Any available seat in the same cabin class
Seat upgrade on rebook: Not eligible
Hotel accommodation: Not provided unless overnight delay is carrier-caused
Meal voucher: $15 for delays 1–2 hours; $25 for delays 2–4 hours; $35 for delays >4 hours
Change fee waiver: Yes, for the rebooked flight only
Refund option: Full refund to original payment method if passenger chooses not to travel

---

Tier 2 — Silver Elite
```

Evidence checklist: 2/6. Found: Weather hotel exceptions by tier; Weather fee AND fare-difference waiver for all tiers. Not found: Weather rebooking priority by status; All four status-tier rebooking windows; Gold partner-airline condition; Platinum any-carrier condition.

### Recursive — top 3

#### Rank 1: recursive-020

Source: `airline_kb/Customer_Rebooking_Policy.txt`; start character: 4032; length: 792 characters; initial vector rank: 1; cosine distance=0.2985 (lower is better).

```text
=============================================================
SECTION 3: WEATHER CANCELLATION SPECIFIC RULES
=============================================================

Weather cancellations are treated as uncontrollable events. The following rules apply:

3.1 Automatic Rebooking

All passengers are automatically rebooked on the next available flight regardless of status. The SABRE Rebooking Engine runs the following priority queue:
  Priority 1: Platinum Elite passengers
  Priority 2: Gold Elite passengers
  Priority 3: Silver Elite passengers
  Priority 4: Standard passengers (sorted by check-in time)

3.2 Hotel Accommodation

For weather cancellations, hotel accommodation is NOT required by DOT regulations and is NOT provided to Standard or Silver Elite passengers by default.
```

#### Rank 2: recursive-021

Source: `airline_kb/Customer_Rebooking_Policy.txt`; start character: 4826; length: 749 characters; initial vector rank: 2; cosine distance=0.3127 (lower is better).

```text
Gold Elite and Platinum Elite passengers retain their hotel accommodation benefit even during weather events, as per their tier agreement.

Gate agents must explain this distinction clearly and courteously. A suggested script is:
"I understand this is frustrating. Unfortunately, weather events are outside our control, and FAA regulations don't require airlines to provide hotel accommodations in these cases. However, I can provide you with [applicable meal voucher] and help rebook you on our next available flight."

3.3 Fee Waivers During Weather

All change fees and fare differences are waived during weather cancellations for all passengers regardless of tier, as long as the rebooked travel is within 14 days of the original departure date.
```

#### Rank 3: recursive-014

Source: `airline_kb/Customer_Rebooking_Policy.txt`; start character: 695; length: 698 characters; initial vector rank: 3; cosine distance=0.3670 (lower is better).

```text
For conditions 1 and 4, rebooking is initiated automatically by the SABRE Rebooking Engine within 30 minutes of the triggering event. Passengers receive an SMS and email with their new itinerary. They may accept or decline via the app or by contacting a gate agent.

For conditions 2 and 3, gate agents initiate manual rebooking through the Customer Service Portal.

=============================================================
SECTION 2: ELITE STATUS TIERS AND REBOOKING BENEFITS
=============================================================

The airline's loyalty program has four tiers. Benefits during irregular operations differ significantly across tiers.

---

Tier 1 — Standard (No Status)
```

Evidence checklist: 3/6. Found: Weather rebooking priority by status; Weather hotel exceptions by tier; Weather fee AND fare-difference waiver for all tiers. Not found: All four status-tier rebooking windows; Gold partner-airline condition; Platinum any-carrier condition.

### Qualitative assessment

Recursive retrieves more of the required evidence within the top three. Fixed uniquely covers: no additional checklist items. Recursive uniquely covers: Weather rebooking priority by status. Both miss: All four status-tier rebooking windows; Gold partner-airline condition; Platinum any-carrier condition. The complete chunks also show how much surrounding policy context is preserved.

## Query 10

How does crew rest regulation compliance affect flight scheduling recovery during a major hub disruption?

### Fixed — top 3

#### Rank 1: fixed-013

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 4599; length: 520 characters; initial vector rank: 1; cosine distance=0.2625 (lower is better).

```text
During a major hub disruption (IROPS Level 3), crew rest violations are a common reason flights cannot be recovered:
- If inbound crew has been delayed by weather for 4+ hours, their FDP clock may already be close to maximum, leaving no time for the return or onward flight.
- Reserve crew must be activated. Reserve pools are maintained at each hub.
- If local reserve is exhausted, the OCC Crew Desk contacts regional crew bases for reassignment. Positioning flights may be required to bring crew to the disrupted hub.
```

#### Rank 2: fixed-018

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 6280; length: 302 characters; initial vector rank: 2; cosine distance=0.3010 (lower is better).

```text
In a hub meltdown scenario (IROPS Level 3), crew rest constraints cascade:
- Early morning disruptions create afternoon rest violations
- Crew who rest during the day cannot be used for night operations
- Recovery often requires 24–36 hours for the schedule to fully normalize as crew rest cycles reset
```

#### Rank 3: fixed-012

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 4335; length: 262 characters; initial vector rank: 3; cosine distance=0.3160 (lower is better).

```text
During an IROPS event, crew rest compliance is frequently the binding constraint on recovery planning. Crewmembers who have reached or are close to their maximum FDP cannot be assigned additional flying, even if the aircraft is ready.

5.1 Impact on Hub Recovery
```

Evidence checklist: 3/6. Found: FDP limits constrain recovery despite aircraft readiness; Local and regional reserve-crew recovery; Cascading recovery can take 24–36 hours. Not found: Earliest legal crew time communicated to OCC; OCC coordinates aircraft, crew, passenger, and gate recovery; Two-hour recovery reviews and station confirmation.

### Recursive — top 3

#### Rank 1: recursive-008

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 4571; length: 586 characters; initial vector rank: 1; cosine distance=0.2397 (lower is better).

```text
5.1 Impact on Hub Recovery

During a major hub disruption (IROPS Level 3), crew rest violations are a common reason flights cannot be recovered:
- If inbound crew has been delayed by weather for 4+ hours, their FDP clock may already be close to maximum, leaving no time for the return or onward flight.
- Reserve crew must be activated. Reserve pools are maintained at each hub.
- If local reserve is exhausted, the OCC Crew Desk contacts regional crew bases for reassignment. Positioning flights may be required to bring crew to the disrupted hub.

5.2 Planning Guidance for Schedulers
```

#### Rank 2: recursive-011

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 6257; length: 772 characters; initial vector rank: 2; cosine distance=0.2788 (lower is better).

```text
5.4 Cascading Effects

In a hub meltdown scenario (IROPS Level 3), crew rest constraints cascade:
- Early morning disruptions create afternoon rest violations
- Crew who rest during the day cannot be used for night operations
- Recovery often requires 24–36 hours for the schedule to fully normalize as crew rest cycles reset

This is why IROPS recovery planning for crew begins as soon as a disruption is detected, not after it peaks. Proactive rest management — protecting certain crews from early FDP consumption — is the most effective tool for minimizing the schedule impact of major hub disruptions.

=============================================================
SECTION 6: CREW NOTIFICATION REQUIREMENTS
=============================================================
```

#### Rank 3: recursive-009

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 5121; length: 607 characters; initial vector rank: 3; cosine distance=0.3520 (lower is better).

```text
5.2 Planning Guidance for Schedulers

When building IROPS recovery pairings, Crew Scheduling must check ALL of the following in CrewTrac:
  1. Remaining FDP availability for each crew at the disrupted station
  2. Rest period status (is any crew currently "in rest"?)
  3. Cumulative 28-day and 365-day flight time tracking
  4. Reserve crew availability by crew base and equipment type

A crew member currently "in rest" cannot be called for duty regardless of operational need unless the full minimum rest period has elapsed. Interrupting crew rest is a federal violation.

5.3 Communication to Operations
```

Evidence checklist: 2/6. Found: Local and regional reserve-crew recovery; Cascading recovery can take 24–36 hours. Not found: FDP limits constrain recovery despite aircraft readiness; Earliest legal crew time communicated to OCC; OCC coordinates aircraft, crew, passenger, and gate recovery; Two-hour recovery reviews and station confirmation.

### Qualitative assessment

Fixed retrieves more of the required evidence within the top three. Fixed uniquely covers: FDP limits constrain recovery despite aircraft readiness. Recursive uniquely covers: no additional checklist items. Both miss: Earliest legal crew time communicated to OCC; OCC coordinates aircraft, crew, passenger, and gate recovery; Two-hour recovery reviews and station confirmation. The complete chunks also show how much surrounding policy context is preserved.

## Recommendation

The strategies tie on total checklist coverage; this sample does not establish a winner. I would retain recursive splitting as a provisional default for its overlap and fallback boundaries, pending evaluation on more queries. Recursive splitting provides overlap and fallback boundaries, while smaller fixed chunks may focus retrieval but separate related rules. Validate the choice on more queries, especially policy exceptions and cross-document synthesis. This comparison changes size, overlap, and splitting method together, so it does not isolate the effect of overlap alone.

Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`. LLM: `gpt-4o-mini`.

# Bonus B — Cross-Encoder Reranking

Retrieve k=8 from the recursive index. BEFORE uses vector ranks 1–3; AFTER uses the cross-encoder's top 3 from the same pool. Both answers use the same model, temperature=0, prompt, and context count. Distances and reranker scores are different scales. Pairs longer than the model's 512-token limit are truncated. The answer comparison is LLM-assisted qualitative feedback, not a correctness guarantee.

## Query 5

How should ground staff communicate a delay of more than 2 hours to connecting passengers?

### Before reranking — all 8 candidates

#### Rank 1: recursive-048

Source: `airline_kb/IROPS_Procedures.txt`; start character: 2115; length: 797 characters; initial vector rank: 1; cosine distance=0.2688 (lower is better).

```text
For passenger-facing communications, the gate agent is responsible for making public address announcements at the following intervals:
- Initial delay announcement: Within 15 minutes of known delay
- Subsequent updates: Every 30 minutes until departure or cancellation
- Cancellation announcement: Immediately upon confirmation

For delays exceeding 2 hours, ground staff must also directly notify connecting passengers at gates. A list of connecting passenger gate locations is available via the Passenger Connect tool in the Operations Portal. Ground staff must personally approach each connecting gate and coordinate with the receiving agent.

=============================================================
SECTION 3: SEVERITY LEVELS
=============================================================
```

#### Rank 2: recursive-032

Source: `airline_kb/Delay_Classification.txt`; start character: 5314; length: 787 characters; initial vector rank: 2; cosine distance=0.3622 (lower is better).

```text
When weather delays are anticipated more than 4 hours in advance (e.g., based on meteorological forecasts), the OCC Weather Desk must issue a Weather Advisory to all affected stations. Gate agents receiving a Weather Advisory should begin proactive passenger communication and coordinate with Customer Service for early rebooking options.

=============================================================
SECTION 5: PASSENGER NOTIFICATION STANDARDS
=============================================================

All delays require passenger notification within the following timeframes:
- At gate: Within 15 minutes of known delay, then every 30 minutes
- App/SMS: Automated push notifications triggered by SABRE status change
- Email: Automated notification for delays exceeding 60 minutes
```

#### Rank 3: recursive-028

Source: `airline_kb/Delay_Classification.txt`; start character: 2648; length: 785 characters; initial vector rank: 3; cosine distance=0.3702 (lower is better).

```text
Step 2: The Duty Manager contacts MCC for a revised ECT and determines if an aircraft swap is now required.

Step 3: If the revised ECT adds more than 60 minutes beyond the original estimate, the Duty Manager must notify the Customer Service Director.

Step 4: Gate agents must update the passenger announcement every 15 minutes (not 30 minutes as in standard delays) once an ECT has been missed. Passengers waiting more than 3 hours from original departure time must be offered the option to deplane if the aircraft has boarded.

Step 5: Passengers who have boarded and waited more than 3 hours on the aircraft without takeoff are subject to the DOT Tarmac Delay Rule. In this case, the Captain must be immediately informed and the OCC must initiate the Tarmac Delay Contingency Plan.
```

#### Rank 4: recursive-026

Source: `airline_kb/Delay_Classification.txt`; start character: 1468; length: 557 characters; initial vector rank: 4; cosine distance=0.3989 (lower is better).

```text
Step 2: Enter delay code 41 (Aircraft Defect) or 43 (Aircraft Scheduled Maintenance) in SABRE provisionally. The MCC will confirm or update the code once diagnosis is complete.

Step 3: Announce to passengers at the gate: "We are currently experiencing a maintenance review on your aircraft. We expect an update within 15 minutes. We apologize for the inconvenience."

Step 4: Do NOT speculate on the nature of the mechanical issue or estimated repair time to passengers until the MCC provides an official estimate.

2.2 Escalation — Delay Beyond 45 Minutes
```

#### Rank 5: recursive-042

Source: `airline_kb/Ground_Operations.txt`; start character: 5367; length: 743 characters; initial vector rank: 5; cosine distance=0.4079 (lower is better).

```text
For misconnected passengers:
Step 1: Gate agent on the departing flight checks the Passenger Connect list 20 minutes before scheduled departure.
Step 2: If any connecting passengers are "at risk," the agent holds the flight a maximum of 8 minutes (domestic) or 12 minutes (international) if the inbound aircraft is on final approach.
Step 3: Passengers who misconnect are directed to the Customer Service Desk, not rebooked at the departure gate.

Holding a flight beyond the hold-time limits requires OCC Duty Manager approval and must be logged in SABRE with justification.

=============================================================
SECTION 6: RAMP SPEED AND SAFETY STANDARDS
=============================================================
```

#### Rank 6: recursive-027

Source: `airline_kb/Delay_Classification.txt`; start character: 1985; length: 770 characters; initial vector rank: 6; cosine distance=0.4163 (lower is better).

```text
2.2 Escalation — Delay Beyond 45 Minutes

If the MCC estimates a repair exceeding 45 minutes:
- Gate agent contacts OCC to assess aircraft swap feasibility
- Customer Service Manager is notified to authorize meal vouchers ($15 for delays 1–2 hours, $25 for 2–4 hours)
- Passengers may be offered rebooking on alternative flights (see Customer Rebooking Policy)

2.3 Escalation — Delay Beyond Originally Estimated Time

If a mechanical delay extends beyond the MCC's originally communicated estimate, the following escalation steps apply:

Step 1: The gate agent immediately notifies the OCC Duty Manager that the estimated completion time (ECT) has been missed.

Step 2: The Duty Manager contacts MCC for a revised ECT and determines if an aircraft swap is now required.
```

#### Rank 7: recursive-025

Source: `airline_kb/Delay_Classification.txt`; start character: 793; length: 673 characters; initial vector rank: 7; cosine distance=0.4299 (lower is better).

```text
Accurate delay coding is required for DOT reporting compliance. Intentional miscoding is a terminable offense.

=============================================================
SECTION 2: MECHANICAL DELAY PROCEDURES
=============================================================

2.1 Initial Response (0–15 minutes)

When a mechanical issue is identified by the flight crew or maintenance team, the gate agent must:

Step 1: Immediately notify the Maintenance Control Center (MCC) via the Operations Portal using the AOG (Aircraft on Ground) notification form. Include: tail number, departure gate, scheduled departure time, and symptom description provided by the flight crew.
```

#### Rank 8: recursive-029

Source: `airline_kb/Delay_Classification.txt`; start character: 3435; length: 557 characters; initial vector rank: 8; cosine distance=0.4309 (lower is better).

```text
2.4 Documentation for Mechanical Delays

Required documentation for all mechanical delays:
- MCC Diagnosis Report (generated by MCC, attached to flight record)
- Delay code confirmation in SABRE
- Meal/voucher log (if issued)
- Passenger complaint log (if applicable)
- If an ECT was missed: Escalation notification timestamps in Operations Portal

=============================================================
SECTION 3: ATC-RELATED DELAY AND CANCELLATION PROCEDURES
=============================================================

3.1 Ground Stop Procedures
```

Baseline answer uses the first 3 chunks.

Evidence checklist: 4/4. Found: Initial announcement and continuing updates; Special procedure for delays over two hours; Passenger Connect identifies connecting gates; Personal gate visit and receiving-agent coordination. Not found: none.

### After reranking — top 3

#### Rank 1: recursive-048

Source: `airline_kb/IROPS_Procedures.txt`; start character: 2115; length: 797 characters; initial vector rank: 1; cosine distance=0.2688 (lower is better); cross-encoder score=6.9520 (higher is better).

```text
For passenger-facing communications, the gate agent is responsible for making public address announcements at the following intervals:
- Initial delay announcement: Within 15 minutes of known delay
- Subsequent updates: Every 30 minutes until departure or cancellation
- Cancellation announcement: Immediately upon confirmation

For delays exceeding 2 hours, ground staff must also directly notify connecting passengers at gates. A list of connecting passenger gate locations is available via the Passenger Connect tool in the Operations Portal. Ground staff must personally approach each connecting gate and coordinate with the receiving agent.

=============================================================
SECTION 3: SEVERITY LEVELS
=============================================================
```

#### Rank 2: recursive-032

Source: `airline_kb/Delay_Classification.txt`; start character: 5314; length: 787 characters; initial vector rank: 2; cosine distance=0.3622 (lower is better); cross-encoder score=-0.5125 (higher is better).

```text
When weather delays are anticipated more than 4 hours in advance (e.g., based on meteorological forecasts), the OCC Weather Desk must issue a Weather Advisory to all affected stations. Gate agents receiving a Weather Advisory should begin proactive passenger communication and coordinate with Customer Service for early rebooking options.

=============================================================
SECTION 5: PASSENGER NOTIFICATION STANDARDS
=============================================================

All delays require passenger notification within the following timeframes:
- At gate: Within 15 minutes of known delay, then every 30 minutes
- App/SMS: Automated push notifications triggered by SABRE status change
- Email: Automated notification for delays exceeding 60 minutes
```

#### Rank 3: recursive-028

Source: `airline_kb/Delay_Classification.txt`; start character: 2648; length: 785 characters; initial vector rank: 3; cosine distance=0.3702 (lower is better); cross-encoder score=-1.1365 (higher is better).

```text
Step 2: The Duty Manager contacts MCC for a revised ECT and determines if an aircraft swap is now required.

Step 3: If the revised ECT adds more than 60 minutes beyond the original estimate, the Duty Manager must notify the Customer Service Director.

Step 4: Gate agents must update the passenger announcement every 15 minutes (not 30 minutes as in standard delays) once an ECT has been missed. Passengers waiting more than 3 hours from original departure time must be offered the option to deplane if the aircraft has boarded.

Step 5: Passengers who have boarded and waited more than 3 hours on the aircraft without takeoff are subject to the DOT Tarmac Delay Rule. In this case, the Captain must be immediately informed and the OCC must initiate the Tarmac Delay Contingency Plan.
```

Evidence checklist: 4/4. Found: Initial announcement and continuing updates; Special procedure for delays over two hours; Passenger Connect identifies connecting gates; Personal gate visit and receiving-agent coordination. Not found: none.

### Answer before reranking

Ground staff must directly notify connecting passengers at gates for delays exceeding 2 hours. They should personally approach each connecting gate and coordinate with the receiving agent. A list of connecting passenger gate locations is available via the Passenger Connect tool in the Operations Portal.

SOURCES: airline_kb/IROPS_Procedures.txt

### Answer after reranking

Ground staff must directly notify connecting passengers at gates for delays exceeding 2 hours. They should personally approach each connecting gate and coordinate with the receiving agent. A list of connecting passenger gate locations is available via the Passenger Connect tool in the Operations Portal.

SOURCES: airline_kb/IROPS_Procedures.txt

### Impact on answer quality (LLM-assisted)

Both the BEFORE and AFTER answers accurately convey the requirement for ground staff to notify connecting passengers at gates for delays exceeding 2 hours, including the need to coordinate with the receiving agent and utilize the Passenger Connect tool. However, both answers lack mention of the specific communication protocol that requires initial delay announcements within 15 minutes and subsequent updates every 30 minutes, which is crucial for comprehensive passenger communication during IROPS. Additionally, both answers fail to address the broader context of IROPS declaration and escalation, which is relevant to understanding the operational environment in which these actions take place. The source citation is consistent in both answers, but since neither answer improves upon the other in terms of detail or context, the quality remains unchanged. Therefore, the AFTER answer is unchanged compared to the BEFORE answer.

## Query 10

How does crew rest regulation compliance affect flight scheduling recovery during a major hub disruption?

### Before reranking — all 8 candidates

#### Rank 1: recursive-008

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 4571; length: 586 characters; initial vector rank: 1; cosine distance=0.2397 (lower is better).

```text
5.1 Impact on Hub Recovery

During a major hub disruption (IROPS Level 3), crew rest violations are a common reason flights cannot be recovered:
- If inbound crew has been delayed by weather for 4+ hours, their FDP clock may already be close to maximum, leaving no time for the return or onward flight.
- Reserve crew must be activated. Reserve pools are maintained at each hub.
- If local reserve is exhausted, the OCC Crew Desk contacts regional crew bases for reassignment. Positioning flights may be required to bring crew to the disrupted hub.

5.2 Planning Guidance for Schedulers
```

#### Rank 2: recursive-011

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 6257; length: 772 characters; initial vector rank: 2; cosine distance=0.2788 (lower is better).

```text
5.4 Cascading Effects

In a hub meltdown scenario (IROPS Level 3), crew rest constraints cascade:
- Early morning disruptions create afternoon rest violations
- Crew who rest during the day cannot be used for night operations
- Recovery often requires 24–36 hours for the schedule to fully normalize as crew rest cycles reset

This is why IROPS recovery planning for crew begins as soon as a disruption is detected, not after it peaks. Proactive rest management — protecting certain crews from early FDP consumption — is the most effective tool for minimizing the schedule impact of major hub disruptions.

=============================================================
SECTION 6: CREW NOTIFICATION REQUIREMENTS
=============================================================
```

#### Rank 3: recursive-009

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 5121; length: 607 characters; initial vector rank: 3; cosine distance=0.3520 (lower is better).

```text
5.2 Planning Guidance for Schedulers

When building IROPS recovery pairings, Crew Scheduling must check ALL of the following in CrewTrac:
  1. Remaining FDP availability for each crew at the disrupted station
  2. Rest period status (is any crew currently "in rest"?)
  3. Cumulative 28-day and 365-day flight time tracking
  4. Reserve crew availability by crew base and equipment type

A crew member currently "in rest" cannot be called for duty regardless of operational need unless the full minimum rest period has elapsed. Interrupting crew rest is a federal violation.

5.3 Communication to Operations
```

#### Rank 4: recursive-010

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 5697; length: 581 characters; initial vector rank: 4; cosine distance=0.3676 (lower is better).

```text
5.3 Communication to Operations

When crew rest limits prevent a scheduled flight, the Crew Scheduling Supervisor notifies the OCC Duty Manager with:
- Affected flight(s)
- Earliest time any available crew can be legal (factoring rest requirements)
- Whether reserve crew can cover the flight and ETD (Estimated Time of Departure)

This notification allows the OCC to determine if the flight should be delayed, swapped to a different aircraft, or cancelled. The decision authority rests with the OCC Duty Manager in consultation with the Hub Station Manager.

5.4 Cascading Effects
```

#### Rank 5: recursive-007

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 3794; length: 775 characters; initial vector rank: 5; cosine distance=0.3892 (lower is better).

```text
FAA Part 117 sets the following limits on cumulative flight time (block hours):

- 100 hours in any 672 consecutive hours (28 days)
- 1,000 hours in any 365 consecutive calendar days

Crews approaching 90 hours in a 28-day window are flagged in CrewTrac with a "soft limit" alert. Schedulers may not assign flights that would push the crewmember beyond 100 hours.

=============================================================
SECTION 5: IROPS RECOVERY AND CREW REST COMPLIANCE
=============================================================

During an IROPS event, crew rest compliance is frequently the binding constraint on recovery planning. Crewmembers who have reached or are close to their maximum FDP cannot be assigned additional flying, even if the aircraft is ready.
```

#### Rank 6: recursive-001

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 0; length: 700 characters; initial vector rank: 6; cosine distance=0.3930 (lower is better).

```text
CREW REST AND DUTY TIME REGULATIONS
Flight Operations — Crew Scheduling Division
Reference: FAA Part 117 (Flightcrew Member Duty and Rest Requirements)
Version 4.0 | Effective: January 1, 2025

=============================================================
SECTION 1: OVERVIEW AND REGULATORY BASIS
=============================================================

This document summarizes the airline's operational interpretation of FAA Part 117, which governs rest and duty requirements for all Part 121 flightcrew members (Captains and First Officers). These are federal minimums; this airline's CBA (Collective Bargaining Agreement) may impose additional restrictions for certain crew classifications.
```

#### Rank 7: recursive-004

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 2152; length: 519 characters; initial vector rank: 7; cosine distance=0.4130 (lower is better).

```text
Augmented crew rules apply only when the rest facility meets FAA standards (bunk rest facility, not a passenger seat).

=============================================================
SECTION 3: MANDATORY REST REQUIREMENTS
=============================================================

3.1 Minimum Rest Between Duty Periods

After any flight duty period, a crewmember must receive a minimum rest period before the next FDP. Rest is measured from off-duty time (block-in plus transit time to rest facility) to report time.
```

#### Rank 8: recursive-006

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 3003; length: 789 characters; initial vector rank: 8; cosine distance=0.4197 (lower is better).

```text
3.2 Reduced Rest Authorization

Reduced rest (minimum 9 hours) may be granted in operational necessity situations, but only under ALL of the following conditions:
- The rest period occurs at a layover hotel designated as an approved rest facility
- The crewmember is not in their physiological night at the start of the reduced rest period
- The airline provides transportation to and from the rest facility
- No more than 2 consecutive reduced-rest periods are scheduled in a pairing

The Crew Scheduling Supervisor (not a line scheduler) must personally approve any reduced-rest authorization and document it in CrewTrac.

=============================================================
SECTION 4: CUMULATIVE FLIGHT TIME LIMITS
=============================================================
```

Baseline answer uses the first 3 chunks.

Evidence checklist: 2/6. Found: Local and regional reserve-crew recovery; Cascading recovery can take 24–36 hours. Not found: FDP limits constrain recovery despite aircraft readiness; Earliest legal crew time communicated to OCC; OCC coordinates aircraft, crew, passenger, and gate recovery; Two-hour recovery reviews and station confirmation.

### After reranking — top 3

#### Rank 1: recursive-008

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 4571; length: 586 characters; initial vector rank: 1; cosine distance=0.2397 (lower is better); cross-encoder score=6.9368 (higher is better).

```text
5.1 Impact on Hub Recovery

During a major hub disruption (IROPS Level 3), crew rest violations are a common reason flights cannot be recovered:
- If inbound crew has been delayed by weather for 4+ hours, their FDP clock may already be close to maximum, leaving no time for the return or onward flight.
- Reserve crew must be activated. Reserve pools are maintained at each hub.
- If local reserve is exhausted, the OCC Crew Desk contacts regional crew bases for reassignment. Positioning flights may be required to bring crew to the disrupted hub.

5.2 Planning Guidance for Schedulers
```

#### Rank 2: recursive-011

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 6257; length: 772 characters; initial vector rank: 2; cosine distance=0.2788 (lower is better); cross-encoder score=5.2684 (higher is better).

```text
5.4 Cascading Effects

In a hub meltdown scenario (IROPS Level 3), crew rest constraints cascade:
- Early morning disruptions create afternoon rest violations
- Crew who rest during the day cannot be used for night operations
- Recovery often requires 24–36 hours for the schedule to fully normalize as crew rest cycles reset

This is why IROPS recovery planning for crew begins as soon as a disruption is detected, not after it peaks. Proactive rest management — protecting certain crews from early FDP consumption — is the most effective tool for minimizing the schedule impact of major hub disruptions.

=============================================================
SECTION 6: CREW NOTIFICATION REQUIREMENTS
=============================================================
```

#### Rank 3: recursive-009

Source: `airline_kb/Crew_Rest_Regulations.txt`; start character: 5121; length: 607 characters; initial vector rank: 3; cosine distance=0.3520 (lower is better); cross-encoder score=2.4634 (higher is better).

```text
5.2 Planning Guidance for Schedulers

When building IROPS recovery pairings, Crew Scheduling must check ALL of the following in CrewTrac:
  1. Remaining FDP availability for each crew at the disrupted station
  2. Rest period status (is any crew currently "in rest"?)
  3. Cumulative 28-day and 365-day flight time tracking
  4. Reserve crew availability by crew base and equipment type

A crew member currently "in rest" cannot be called for duty regardless of operational need unless the full minimum rest period has elapsed. Interrupting crew rest is a federal violation.

5.3 Communication to Operations
```

Evidence checklist: 2/6. Found: Local and regional reserve-crew recovery; Cascading recovery can take 24–36 hours. Not found: FDP limits constrain recovery despite aircraft readiness; Earliest legal crew time communicated to OCC; OCC coordinates aircraft, crew, passenger, and gate recovery; Two-hour recovery reviews and station confirmation.

### Answer before reranking

Crew rest regulation compliance significantly affects flight scheduling recovery during a major hub disruption by imposing constraints that can hinder the ability to recover flights. When crew members are delayed, particularly if inbound crew has been delayed by weather for 4 or more hours, their Flight Duty Period (FDP) clock may approach its maximum limit, leaving insufficient time for subsequent flights. This necessitates the activation of reserve crew, which are maintained at each hub. If local reserves are exhausted, the Operations Control Center (OCC) Crew Desk must contact regional crew bases for reassignment, potentially requiring positioning flights to bring crew to the disrupted hub.

Moreover, crew rest violations can lead to cascading effects, where early disruptions can create rest violations later in the day, limiting the availability of crew for night operations. Recovery from such disruptions often requires 24 to 36 hours for the schedule to normalize as crew rest cycles reset. Therefore, proactive management of crew rest is essential for minimizing the impact of major hub disruptions on the flight schedule.

SOURCES: airline_kb/Crew_Rest_Regulations.txt

### Answer after reranking

Crew rest regulation compliance significantly affects flight scheduling recovery during a major hub disruption by imposing constraints that can hinder the ability to recover flights. When crew members are delayed, particularly by weather, their Flight Duty Period (FDP) may approach its maximum limit, leaving insufficient time for subsequent flights. This necessitates the activation of reserve crew, which must be managed carefully to ensure compliance with rest regulations. If local reserves are exhausted, crew must be reassigned from regional bases, potentially requiring positioning flights.

Moreover, disruptions can lead to cascading effects on crew rest cycles, where early morning disruptions can create violations in the afternoon, and crews resting during the day cannot be utilized for night operations. This complicates recovery efforts, often requiring 24–36 hours for the schedule to normalize as crew rest cycles reset. Therefore, proactive management of crew rest is essential for minimizing the impact of major disruptions on flight schedules.

SOURCES: airline_kb/Crew_Rest_Regulations.txt

### Impact on answer quality (LLM-assisted)

Both the BEFORE and AFTER answers correctly identify that crew rest regulation compliance significantly impacts flight scheduling recovery during major hub disruptions, emphasizing the constraints imposed by crew rest requirements. However, the AFTER answer lacks specific details about the operational procedures for managing crew rest compliance during IROPS, such as the necessity for the Crew Scheduling Supervisor to notify the OCC Duty Manager about affected flights and the earliest time any available crew can be legal. Additionally, while both answers mention the activation of reserve crews, the AFTER answer does not explicitly state that reserve pools are maintained at each hub, which is a concrete detail from the reference documents. Both answers cite the same source, but the AFTER answer does not synthesize the crew constraints with OCC recovery actions as effectively as it could. Overall, the AFTER answer is **worse** than the BEFORE answer because it omits critical operational details and lacks a comprehensive synthesis of the information provided in the reference documents.
