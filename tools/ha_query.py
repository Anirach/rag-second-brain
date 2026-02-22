#!/usr/bin/env python3
"""
HA Database Safe Query Wrapper
ALL HA database queries MUST go through this script.
It handles: VPN connect → query → de-identify → output safe text → VPN disconnect

Usage:
  # Aggregate query (no de-ID needed, but still safe)
  python3 tools/ha_query.py "SELECT region, COUNT(*) FROM accreditationHistory GROUP BY region"

  # Detail query (auto de-identified)
  python3 tools/ha_query.py "SELECT hospital_name, hospital_code FROM accreditationHistory LIMIT 10"

  # Save raw (de-identified) output to file
  python3 tools/ha_query.py "SELECT * FROM accreditationHistory LIMIT 5" --output /tmp/safe_output.txt

  # Save mapping for later reverse (report generation)
  python3 tools/ha_query.py "SELECT * FROM ..." --save-mapping

  # Force no de-ID (ONLY for pure aggregates you're sure about)
  python3 tools/ha_query.py "SELECT COUNT(*) FROM ..." --raw-aggregate

Output goes to stdout (de-identified). Safe to read into LLM context.
Mapping saved to /tmp/ha_deid_mapping.json (local only, never sent to API).
"""

import subprocess
import sys
import os
import time
import re
import argparse
import signal

# Import de-identifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ha_deid import DeIdentifier

# HA Database config
VPN_CMD = [
    'sudo', 'openfortivpn', '160.187.249.96:10443',
    '-u', 'Admin-1017-1324_2',
    '-p', '>74Tm£h9RK(D',
    '--trusted-cert', '91153bd1aa210a70d7e19eddf61ef54f4041f5757ae4cd5c17a42dae67097529',
    '--set-dns=0'
]
DB_HOST = '192.168.88.11'
DB_PORT = '30503'
DB_USER = 'anirach'
DB_PASS = 'Ho1(@k5.T&2q@mjG'
DB_NAME = 'datawarehouse'


def is_pure_aggregate(sql):
    """Heuristic: query returns only aggregate/count data, no identifiable fields."""
    sql_upper = sql.upper().strip()
    # Must have aggregate function and no SELECT *
    has_agg = any(fn in sql_upper for fn in ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX('])
    has_star = 'SELECT *' in sql_upper or 'SELECT\n*' in sql_upper
    # Check for known sensitive columns in SELECT
    sensitive_patterns = ['HOSPITAL_NAME', 'HOSPITAL_CODE', 'EMPLOYEE_NAME', 
                         'SURVEYOR_NAME', 'SPEAKER_NAME', 'EMAIL', 'PHONE',
                         'ADDRESS', 'FIRST_NAME', 'LAST_NAME', 'FULL_NAME',
                         'OLD_HOSPITAL_NAME', 'ORGANIZATION_NAME', 'NOTES']
    selects_sensitive = any(p in sql_upper for p in sensitive_patterns)
    return has_agg and not has_star and not selects_sensitive


def wait_for_vpn(timeout=15):
    """Wait for ppp0 interface to come up."""
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(['ip', 'link', 'show', 'ppp0'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return True
        time.sleep(1)
    return False


def run_query(sql, timeout=30):
    """Connect VPN, run query, disconnect. Returns raw output."""
    vpn_proc = None
    try:
        # Start VPN in background
        vpn_proc = subprocess.Popen(
            VPN_CMD,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )
        
        # Wait for VPN
        if not wait_for_vpn(timeout=15):
            raise RuntimeError("VPN failed to connect (ppp0 not up after 15s)")
        
        # Small delay for route stabilization
        time.sleep(1)
        
        # Run query
        env = os.environ.copy()
        env['PGPASSWORD'] = DB_PASS
        result = subprocess.run(
            ['psql', '-h', DB_HOST, '-p', DB_PORT, '-U', DB_USER, '-d', DB_NAME, '-c', sql],
            capture_output=True, text=True, timeout=timeout, env=env
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"psql error: {result.stderr.strip()}")
        
        return result.stdout
        
    finally:
        # Always kill VPN
        if vpn_proc:
            vpn_proc.terminate()
            try:
                vpn_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                vpn_proc.kill()
        # Also kill any lingering openfortivpn
        subprocess.run(['sudo', 'pkill', '-f', 'openfortivpn'], 
                      capture_output=True, timeout=5)


def main():
    parser = argparse.ArgumentParser(description='Safe HA database query wrapper')
    parser.add_argument('sql', help='SQL query to execute')
    parser.add_argument('--output', '-o', help='Save output to file')
    parser.add_argument('--save-mapping', action='store_true', 
                       help='Save de-ID mapping for reverse (report gen)')
    parser.add_argument('--raw-aggregate', action='store_true',
                       help='Skip de-ID (ONLY for verified pure aggregates)')
    parser.add_argument('--timeout', type=int, default=30, help='Query timeout seconds')
    args = parser.parse_args()
    
    # Run query
    print("🔌 Connecting VPN...", file=sys.stderr)
    raw_output = run_query(args.sql, timeout=args.timeout)
    print("✅ Query complete, VPN disconnected", file=sys.stderr)
    
    # De-identify unless explicitly skipped
    if args.raw_aggregate and is_pure_aggregate(args.sql):
        safe_output = raw_output
        print("📊 Pure aggregate — no de-ID needed", file=sys.stderr)
    else:
        deid = DeIdentifier()
        safe_output = deid.deid_psql_output(raw_output)
        stats = deid.stats()
        masked = sum(stats.values())
        if masked > 0:
            print(f"🔒 De-identified: {stats}", file=sys.stderr)
        else:
            print("📊 No identifiable fields detected", file=sys.stderr)
        
        if args.save_mapping:
            path = deid.save_mapping()
            print(f"💾 Mapping saved: {path}", file=sys.stderr)
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(safe_output)
        print(f"📄 Output saved: {args.output}", file=sys.stderr)
    else:
        print(safe_output)


if __name__ == '__main__':
    main()
