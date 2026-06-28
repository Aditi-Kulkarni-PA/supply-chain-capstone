# Delay Simulation Agent — Workflow & Design

## Simulation / What if Agent

### Example Queries

"Simulate delays if weather changes to Stormy across East regions"

"What if all Same Day deliveries in the North region face Foggy weather — simulate the impact"

"Simulate delays if we replace all Bike assignments on routes over 100km with Vans"

## Flow Diagram

```perl
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                   │
│  e.g. "Simulate delays if weather changes to Stormy in East"        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     delivery_app.py                                 │
│  Freshness check: is daily_delivery_delay_prediction_meta.json      │
│  < 1 hour old?                                                      │
│    YES → prepend [SYSTEM: … FRESH]                                  │
│    NO  → prepend [SYSTEM: … NOT FRESH]                              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              master_expert (orchestrator agent)                     │
│                                                                     │
│  Section 3 pre-requisite check:                                     │
│    NOT FRESH? → call predict_delivery_delays_tool first             │
│    FRESH?     → skip predict, go straight to simulate               │
│                                                                     │
│  Calls delay_simulations_tool (wraps simulation agent)              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│          delay_simulation_agent  (delay_simulation.md)              │
│                                                                     │
│  Translates natural-language query into:                            │
│    scenario = "Weather changes to Stormy in East"                   │
│    filters  = {"region": "east"}                                    │
│    changes  = {"weather_condition": "stormy"}                       │
│                                                                     │
│  Calls simulate_order_delays tool exactly once                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│         simulate_order_delays  (tools/simulate_delays.py)           │
│                                                                     │
│  Step 1: Parse & validate JSON inputs                               │
│                                                                     │
│  Step 2: Read daily_delivery_delay_prediction.csv                   │
│          (1346 delayed orders from latest predict run)              │
│                                                                     │
│  Step 3: Build filter mask                                          │
│          region == "east"  →  241 matching rows                     │
│                                                                     │
│  Step 4: Apply column changes                                       │
│          Set weather_condition = "stormy" on those 241 rows         │
│                                                                     │
│  Step 5: Severity lookup from SQLite DB                             │
│          ┌─────────────────────────────────────────────┐            │
│          │  Lookup priority:                           │            │
│          │  1. mode + weather  (hist_summary_by_mode_  │            │
│          │     weather)  ← uses filter_ctx if avail    │            │
│          │  2. weather + vehicle (hist_summary_by_     │            │
│          │     weather_vehicle)                        │            │
│          │  3. mode + vehicle  (averaged from two      │            │
│          │     single-dim tables)                      │            │
│          │  4. Single-dim fallback (highest delay_rate)│            │
│          └─────────────────────────────────────────────┘            │
│          Returns: {delay_rate: 0.416, fracs: [0.08, 0.63, 0.29]}    │
│                                                                     │
│  Step 6: Assign new severity labels proportionally                  │
│          Short: 8%  Medium: 63%  Long: 29%  (across 241 rows)       │
│                                                                     │
│  Step 7: Save simulation_delivery_delays.csv                        │
│                                                                     │
│  Step 8: Return Markdown summary + top N rows (SC_MCP_DISPLAY_ROWS) │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│          delay_simulation_agent  (post-processing)                  │
│                                                                     │
│  Enriches EVERY row with simulate_delay_reason inferred from        │
│  scenario + original_severity + simulated_severity                  │
│                                                                     │
│  Returns SimulationsList (list[SimulateDelays])                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              master_expert  (post-processing)                       │
│                                                                     │
│  Copies simulations → MasterOutput.simulate_rows                    │
│  Calls format_summary_tool → MasterOutput.simulate_summary          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      delivery_app.py  (Gradio)                      │
│                                                                     │
│  simulate_summary → Simulation tab (Markdown)                       │
│  simulate_rows → DataFrame table                                    │
│  Saves simulate_delays_latest.csv to output/                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Changes Made (4 Files)

| # | File | What changed |
|---|---|---|
| 1 | `tools/simulate_delays.py` | Complete rewrite. New signature (`scenario`, `filters`, `changes`). Reads real delayed-orders CSV, filters rows by any combo of `region/mode/vehicle/weather/partner/package/distance`, applies column changes, looks up historical severity distribution from `hist_summary_by_*` SQLite tables, reassigns severity labels proportionally, saves simulation CSV, returns Markdown summary + top rows. Row cap uses `SC_MCP_DISPLAY_ROWS` env var (default 50). |
| 2 | `delivery_agents.py` | Updated **SimulateDelays** Pydantic model. Replaced `order_id`, `simulate_weather`, `simulate_traffic`, `order_date`, `delivery_date`, `simulate_delay_hours` with `delivery_id`, `delivery_partner`, `delivery_mode`, `region`, `weather_condition`, `vehicle_type`, `distance_km`, `original_severity`, `simulated_severity`, `simulate_delay_reason`. |
| 3 | `delay_simulation.md` | Rewrote agent prompt. Documented the **3-arg tool signature**, JSON parameter format with examples, valid values for all columns (lowercase), removed traffic references. |
| 4 | `master_expert.md` | Updated Section 3. Added **FRESH/NOT FRESH prerequisite check** (same as diagnosis). Added valid simulation parameter reference. Updated tool description from `"weather/traffic"` to `"weather/vehicle/mode"`. |



### Key Design Decisions

| Design Decision | Description |
|---|---|
| No ML re-run | Severity is inferred from **historical summary tables already in the DB**, not from re-running the prediction pipeline. |
| Combined-table priority | Lookup priority: `mode+weather → weather+vehicle → mode+vehicle (averaged) → single-dimension fallback`. |
| Filter context | Filter values (not being changed) are passed to the lookup to enable more specific combined-table matches (e.g., filtering by `delivery_mode=same day` + changing `weather=foggy` → uses the `mode_weather` combined table). |
| Prerequisite check | Simulation now has the same **FRESH/NOT FRESH check** as diagnosis, since it reads the delayed-orders CSV that predict produces. |
