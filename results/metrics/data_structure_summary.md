# Data Structure Summary

## Raw CSV Files

Six CSV files from [MARKlab dataset (Zenodo)](https://doi.org/10.5281/zenodo.15420422), placed under `data/raw/`:

| File | Size | Rows (full or sampled) | Measurement Type |
|------|------|------------------------|------------------|
| `figure4_latency.csv` | 201 MB | 10,000 sampled *(full file not loaded)* | ICMP ping latency |
| `figure5_packet_loss.csv` | 10 MB | 95,016 | ICMP ping (packet loss) |
| `figure6_throughput.csv` | 0.2 MB | 1,514 | Upload/download throughput |
| `figure7_current.csv` | 0.2 MB | 1,514 | **(see note below)** |
| `figure8_power_idle.csv` | 1.85 GB | 10,000 sampled *(full file not loaded)* | Idle-state power |
| `figure8_power_upload.csv` | 55 MB | 334,516 | Upload power consumption |

Large files (>100 MB) were inspected by reading only the first 10,000 rows to avoid excessive memory use. Full row counts for those files have not been measured.

---

## Common Columns (all 6 files)

These 12 columns are present in every CSV:

| Column | Description | Type |
|--------|-------------|------|
| `id` | Measurement session identifier | int |
| `run` | Sub-index within a session | int |
| `node_name` | Hardware node identifier | str |
| `modem_name` | Modem model (e.g., `Quectel_BG96`, `Quectel_EC21`) | str |
| `location` | City and country (e.g., `Zagreb, Croatia`) | str |
| `mcc` | Mobile Country Code | int |
| `country` | Full country name | str |
| `iso_code` | ISO 3166-1 alpha-2 country code | str |
| `operator_anon` | Anonymized operator name (`Op A` through `Op D`) | str |
| `rat` | Radio Access Technology code | int |
| `rat_name` | Human-readable RAT name | str |
| `timestamp` | Unix epoch timestamp | float |

---

## RAT Labels (Target Variable)

The RAT classification target is available in two columns:

| `rat` (int) | `rat_name` (str) |
|-------------|------------------|
| 0 | 2G |
| 2 | 3G |
| 7 | LTE CAT1 |
| 8 | LTE-M |
| 9 | NB-IoT |

Five classes total. No missing values observed in any file.

Either `rat` or `rat_name` can serve as the target. `rat` (integer-encoded) is preferred for modeling.

---

## Countries

Four countries appear across the dataset:

| Country | ISO Code | MCC |
|---------|----------|-----|
| Germany | DE | 262 |
| Croatia | HR | 219 |
| Italy | IT | 222 |
| Norway | NO | 242 |

---

## Operators

Four anonymized operators:

| Operator | Appears in |
|----------|-----------|
| Op A | All (largest presence) |
| Op B | All |
| Op C | All |
| Op D | Some files |

---

## Measurement-Specific Columns

### figure4_latency.csv & figure5_packet_loss.csv (Ping)

Both share the same schema — ICMP ping measurements:

| Column | Description |
|--------|-------------|
| `bytes` | ICMP payload size (64) |
| `target_ip` | Target IP (e.g., `8.8.8.8`) |
| `icmp_seq` | ICMP sequence number |
| `ttl` | Time-To-Live |
| `rtt_ms` | Round-trip time in milliseconds |

Note: `figure5_packet_loss.csv` does **not** contain power, current, voltage, or throughput columns. It is primarily useful for packet-loss analysis (examining gaps in `icmp_seq`) and latency statistics.

### figure6_throughput.csv (Throughput)

| Column | Description |
|--------|-------------|
| `size_total` | Total bytes transferred |
| `average_speed` | Throughput (kbit/s) |
| `time_total` | Transfer duration (seconds) |
| `filesize` | Nominal file size label (e.g., `500KB`) |
| `timeout` | Timeout threshold (seconds) |
| `direction` | `Uplink` or `Downlink` |

### figure7_current.csv — Schema Issue

This file is expected (per dataset README) to contain current/voltage measurements during data transfers, with columns `timetamp_ms`, `diff`, `current`, and `voltage`. **In the downloaded data, this file has the same 18-column schema as `figure6_throughput.csv` and the same row count (1,514).** The expected power-related columns are absent.

This issue is **not blocking** for the baseline stage. The throughput schema files (figure6/figure7) are relatively small and can still be used for throughput-based feature extraction. The power-related measurements are available in the figure8 files instead.

### figure8_power_idle.csv & figure8_power_upload.csv (Power)

| Column | Description |
|--------|-------------|
| `timetamp_ms` | Timestamp in milliseconds |
| `diff` | Time delta from previous measurement (seconds) |
| `current` | Current (mA) |
| `voltage` | Voltage (mV) |

`figure8_power_idle.csv` (1.85 GB) contains only the 12 common columns plus these 4 power columns. `figure8_power_upload.csv` (55 MB) contains the 12 common columns, the 6 throughput columns from figure6, plus these 4 power columns.

---

## Columns Excluded from Model Input

The following columns must be excluded from the feature matrix:

| Column | Reason |
|--------|--------|
| `operator_anon` | Leakage risk (operator may correlate with RAT deployment); also anonymized labels limit interpretability |
| `rat` | Target variable |
| `rat_name` | Redundant with `rat` (target) |
| `id` | Session identifier, not a predictive feature |
| `run` | Sub-index within session, not a predictive feature |
| `timestamp` | Temporal metadata; not a direct RAT predictor (may be used for time-window grouping only) |
| `target_ip` | Constant / near-constant IP address |
| `filesize` | String label; can be encoded if needed, but is a test parameter, not a network feature |

Whether to include or exclude `mcc`, `country`, `iso_code`, `location` depends on the task framing. For a **country-agnostic** RAT classifier, these should be excluded. For **transfer learning across countries**, they may be used to split source/target domains but should not be model inputs.

---

## Summary of Data Quality

- No significant missing values in sampled rows (only 2 rows in figure8_power_upload with missing timestamps — negligible)
- All files use consistent column naming
- `rtt_ms` dtype differs between files: `int64` in figure4, `float64` in figure5 — needs alignment during merge or feature extraction
- `timetamp_ms` (note the typo: "timetamp" not "timestamp") is present only in the power files

---

## Recommended Next Step

Build a simple **Random Forest baseline** using a single measurement file (e.g., `figure5_packet_loss.csv`, which has a reasonable 95k rows with 5 RAT classes across 4 countries). Use aggregated features computed over time windows of length *z*, without including `operator_anon` or the target columns. This establishes a minimum performance bar before moving to neural network models and cross-country transfer learning.
