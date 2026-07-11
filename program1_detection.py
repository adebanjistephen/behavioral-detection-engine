"""
CYBR 541 - Program 1
Behavioral Detection Engineering in an Enterprise Environment
Author: ADEBANJI STEPHEN Z2074015

Two behavioral detection models:
  A) Beaconing Detection (Periodic C2 Behavior) using flows.csv
  B) NXDOMAIN Burst Detection (DNS Behavioral Anomaly) using dns.csv
  C) Cross-telemetry Correlation
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# SECTION 1: DATA LOADING & NORMALIZATION
# ─────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV, convert timestamp to datetime, drop unparseable rows,
    and sort chronologically. Handles two common timestamp formats.
    """
    df = pd.read_csv(path)

    # Attempt parsing with inferred format first, then fallback formats
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows where timestamp could not be parsed
    invalid_count = df['timestamp'].isna().sum()
    if invalid_count > 0:
        print(f"  [!] Dropped {invalid_count} rows with unparseable timestamps from {os.path.basename(path)}")
    df = df.dropna(subset=['timestamp'])

    # Sort chronologically (required before delta computation)
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  [+] Loaded {len(df):,} rows from {os.path.basename(path)}")
    return df


# ─────────────────────────────────────────────
# SECTION 2: BEACONING DETECTION (Task A)
# ─────────────────────────────────────────────

def detect_beaconing(flows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect statistically consistent periodic communication suggesting C2 beaconing.

    Algorithm:
      1. Sort by timestamp (already done in load_csv)
      2. Group by src_ip, dst_ip, dst_port
      3. Compute inter-arrival time deltas (Δt) in seconds using shift()
      4. Aggregate: connection_count, mean_delta_seconds, std_delta_seconds
      5. Apply thresholds

    Thresholds (justified below):
      - connection_count   >= 20   : Need sufficient sample for statistical stability
      - std_delta_seconds  <= 30.0 : Low variance = automation / periodic scheduling
      - mean_delta_seconds <= 3600 : Focus on short-cycle beacons (sub-hourly C2)
    """
    print("\n[*] Running Beaconing Detection...")

    results = []

    grouped = flows_df.groupby(['src_ip', 'dst_ip', 'dst_port'])

    for (src_ip, dst_ip, dst_port), group in grouped:
        group = group.sort_values('timestamp')

        # Compute inter-arrival deltas in seconds
        deltas = group['timestamp'].diff().dt.total_seconds()

        # Drop the first NaN (no prior event) and any non-positive deltas
        deltas = deltas.dropna()
        deltas = deltas[deltas > 0]

        connection_count = len(group)

        # Need at least 2 deltas to compute std; set minimum connection_count threshold
        if len(deltas) < 2:
            continue

        mean_delta = deltas.mean()
        std_delta  = deltas.std(ddof=1)   # sample std (ddof=1 per assignment guide)

        results.append({
            'src_ip':             src_ip,
            'dst_ip':             dst_ip,
            'dst_port':           dst_port,
            'connection_count':   connection_count,
            'mean_delta_seconds': round(mean_delta, 3),
            'std_delta_seconds':  round(std_delta, 3),
        })

    all_stats = pd.DataFrame(results)

    if all_stats.empty:
        print("  [!] No flow groups with sufficient data.")
        return pd.DataFrame()

    # ── THRESHOLD APPLICATION ──────────────────────────────────────
    # Threshold 1: Minimum connection count >= 20
    #   Rationale: Low-count groups have statistically unstable std.
    #   A genuine beacon running over hours will accumulate many sessions.
    #   FP risk: Long-running legitimate background services also generate
    #   high counts (e.g., Windows Update, AD health checks).
    #
    # Threshold 2: std_delta_seconds <= 30.0 seconds
    #   Rationale: Automated processes fire at near-constant intervals.
    #   Human-driven traffic is inherently irregular. A C2 implant with a
    #   60-second callback period and minor jitter stays well under 30s std.
    #   FP risk: NTP, heartbeat protocols, monitoring agents.
    #
    # Threshold 3: mean_delta_seconds <= 3600 (1 hour)
    #   Rationale: Sub-hourly callbacks are operationally typical for C2.
    #   Helps exclude weekly cron jobs that also show low std.
    # ──────────────────────────────────────────────────────────────
    MIN_CONNECTIONS   = 20
    MAX_STD_DELTA     = 30.0
    MAX_MEAN_DELTA    = 3600.0

    beacon_candidates = all_stats[
        (all_stats['connection_count']   >= MIN_CONNECTIONS) &
        (all_stats['std_delta_seconds']  <= MAX_STD_DELTA)   &
        (all_stats['mean_delta_seconds'] <= MAX_MEAN_DELTA)
    ].copy()

    beacon_candidates = beacon_candidates.sort_values('std_delta_seconds')

    print(f"  [+] Beacon candidates found: {len(beacon_candidates)}")
    return beacon_candidates


# ─────────────────────────────────────────────
# SECTION 3: NXDOMAIN BURST DETECTION (Task B)
# ─────────────────────────────────────────────

def detect_nxdomain_bursts(dns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect abnormal NXDOMAIN burst behavior suggesting DGA, recon, or C2 lookups.

    Algorithm:
      1. Derive is_nxdomain indicator from 'is_nxdomain' column (already binary)
      2. Floor timestamps into 15-minute windows
      3. Group by src_ip and time_bin
      4. Compute: total_dns_queries, nxdomain_count, nxdomain_ratio, unique_queries
      5. Apply thresholds

    NOTE: Assignment uses field 'response_code'; actual CSV column is 'rcode'.
          Both 'is_nxdomain' flag and 'rcode' == 'NXDOMAIN' are used for robustness.

    Thresholds (justified below):
      - total_dns_queries >= 10  : Filter single-event noise
      - nxdomain_count    >= 5   : Require absolute volume of failures
      - nxdomain_ratio    >= 0.5 : Majority of queries are failing = abnormal
    """
    print("\n[*] Running NXDOMAIN Burst Detection...")

    # Normalize column: use is_nxdomain flag (1 = NXDOMAIN)
    dns_df = dns_df.copy()

    # Defensive: also check rcode column if it exists
    if 'rcode' in dns_df.columns:
        dns_df['is_nxdomain'] = (
            (dns_df['is_nxdomain'].astype(int) == 1) |
            (dns_df['rcode'].str.upper() == 'NXDOMAIN')
        ).astype(int)

    # Create 15-minute time bins
    dns_df['time_bin'] = dns_df['timestamp'].dt.floor('15min')

    # Group and aggregate
    grouped = dns_df.groupby(['src_ip', 'time_bin'])

    agg = grouped.agg(
        total_dns_queries=('query',       'count'),
        nxdomain_count=   ('is_nxdomain', 'sum'),
        unique_queries=   ('query',       'nunique'),
    ).reset_index()

    # Compute ratio (avoid division by zero)
    agg['nxdomain_ratio'] = (agg['nxdomain_count'] / agg['total_dns_queries']).round(3)

    # ── THRESHOLD APPLICATION ──────────────────────────────────────
    # Threshold 1: total_dns_queries >= 10
    #   Rationale: A single failed lookup from a typo would produce ratio=1.0
    #   but is not suspicious. We need volume to distinguish noise from behavior.
    #   FP risk: A host with exactly 10 queries where half fail could be a
    #   misconfigured service.
    #
    # Threshold 2: nxdomain_count >= 5
    #   Rationale: Absolute count prevents small-sample ratio inflation.
    #   DGA-based C2 or recon typically generates dozens of failures per minute.
    #   FP risk: DNS misconfiguration affecting a single known domain.
    #
    # Threshold 3: nxdomain_ratio >= 0.5
    #   Rationale: Legitimate hosts resolve most queries successfully.
    #   Ratio > 0.5 means failures outnumber successes - a strong behavioral signal.
    #   FP risk: A host undergoing a failed domain migration or split-brain DNS.
    # ──────────────────────────────────────────────────────────────
    MIN_TOTAL_QUERIES = 10
    MIN_NXDOMAIN_COUNT = 5
    MIN_NXDOMAIN_RATIO = 0.5

    burst_candidates = agg[
        (agg['total_dns_queries'] >= MIN_TOTAL_QUERIES) &
        (agg['nxdomain_count']    >= MIN_NXDOMAIN_COUNT) &
        (agg['nxdomain_ratio']    >= MIN_NXDOMAIN_RATIO)
    ].copy()

    burst_candidates = burst_candidates.sort_values('nxdomain_count', ascending=False)

    # Reorder columns to match assignment schema exactly
    burst_candidates = burst_candidates[[
        'src_ip', 'time_bin', 'total_dns_queries',
        'nxdomain_count', 'nxdomain_ratio', 'unique_queries'
    ]]

    print(f"  [+] NXDOMAIN burst candidates found: {len(burst_candidates)}")
    return burst_candidates


# ─────────────────────────────────────────────
# SECTION 4: CORRELATION (Task C)
# ─────────────────────────────────────────────

def correlate_beacon_to_dns(beacon_df: pd.DataFrame, dns_df: pd.DataFrame) -> None:
    """
    Correlate beacon destination IPs with DNS answer records.
    Constructs a timeline linking DNS resolution to periodic flow activity.
    """
    print("\n[*] Running Cross-Telemetry Correlation...")

    if beacon_df.empty:
        print("  [!] No beacon candidates to correlate.")
        return

    beacon_ips = set(beacon_df['dst_ip'].unique())

    # Find DNS records where the answer resolved to a beacon destination IP
    correlated = dns_df[dns_df['answer'].isin(beacon_ips)].copy()

    if correlated.empty:
        print("  [!] No DNS resolutions match beacon destination IPs.")
        return

    print(f"\n  [+] DNS records resolving to beacon destination IPs:")
    print(correlated[['timestamp','src_ip','query','answer']].to_string(index=False))

    # Build timeline
    print("\n  [+] Correlation Timeline:")
    for _, row in correlated.drop_duplicates(subset=['src_ip','answer']).iterrows():
        matching_beacons = beacon_df[beacon_df['dst_ip'] == row['answer']]
        for _, b in matching_beacons.iterrows():
            print(f"    HOST {row['src_ip']} resolved '{row['query']}' → {row['answer']}")
            print(f"    ↳ Same host then beaconed to {b['dst_ip']}:{b['dst_port']} "
                  f"({int(b['connection_count'])} connections, "
                  f"μ={b['mean_delta_seconds']}s, σ={b['std_delta_seconds']}s)")


# ─────────────────────────────────────────────
# SECTION 5: MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    FLOWS_PATH = "flows.csv"
    DNS_PATH   = "dns.csv"

    print("=" * 60)
    print("CYBR 541 Program 1 - Behavioral Detection Engine")
    print("=" * 60)

    # ── Step 1: Load & Normalize ──────────────────────────────────
    print("\n[1] Loading and normalizing telemetry data...")
    flows_df = load_csv(FLOWS_PATH)
    dns_df   = load_csv(DNS_PATH)

    # ── Step 2: Beaconing Detection ───────────────────────────────
    beacon_results = detect_beaconing(flows_df)

    print("\n── Beacon Detection Results ──")
    if not beacon_results.empty:
        print(beacon_results.to_string(index=False))
        beacon_results.to_csv("beacon_results.csv", index=False)
        print("  [+] Saved to beacon_results.csv")
    else:
        print("  No beacon candidates identified.")

    # ── Step 3: NXDOMAIN Burst Detection ─────────────────────────
    nxdomain_results = detect_nxdomain_bursts(dns_df)

    print("\n── NXDOMAIN Burst Detection Results ──")
    if not nxdomain_results.empty:
        print(nxdomain_results.to_string(index=False))
        nxdomain_results.to_csv("nxdomain_results.csv", index=False)
        print("  [+] Saved to nxdomain_results.csv")
    else:
        print("  No NXDOMAIN burst candidates identified.")

    # ── Step 4: Correlation ───────────────────────────────────────
    correlate_beacon_to_dns(beacon_results, dns_df)

    print("\n[+] Detection pipeline complete.")


if __name__ == "__main__":
    main()
