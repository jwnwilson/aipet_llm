#!/usr/bin/env python3
"""
Sync Route53 A records for known ingress subdomains to the current inlets tunnel IP.

Usage:
    TUNNEL_IP=1.2.3.4 python3 scripts/sync_route53_dns.py

Environment variables:
    TUNNEL_IP   Public IP of the inlets exit-node (required)
    ZONE_NAME   Hosted zone name, default: jwnwilson.co.uk.
    DRY_RUN     Set to 1 to print changes without applying them
"""
import json
import os
import subprocess
import sys

# Subdomains managed by inlets ingress. Only these A records are touched.
# Matches the records created by infra/terraform/modules/dns/main.tf.
MANAGED_SUBDOMAINS = {
    "llm-api.jwnwilson.co.uk.",
    "temporal.jwnwilson.co.uk.",
}

ZONE_NAME = os.environ.get("ZONE_NAME", "jwnwilson.co.uk.")
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"


def run(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_tunnel_ip() -> str:
    ip = os.environ.get("TUNNEL_IP", "").strip()
    if not ip:
        print("ERROR: TUNNEL_IP environment variable is not set or empty", file=sys.stderr)
        sys.exit(1)
    # Basic IPv4 validation
    parts = ip.split(".")
    if len(parts) != 4 or not all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        print(f"ERROR: TUNNEL_IP '{ip}' is not a valid IPv4 address", file=sys.stderr)
        sys.exit(1)
    return ip


def get_zone_id(zone_name: str) -> str:
    output = run([
        "aws", "route53", "list-hosted-zones",
        "--query", f"HostedZones[?Name=='{zone_name}'].Id",
        "--output", "text",
    ])
    zone_id = output.replace("/hostedzone/", "").strip()
    if not zone_id or zone_id == "None":
        print(f"ERROR: hosted zone '{zone_name}' not found in this AWS account", file=sys.stderr)
        sys.exit(1)
    if " " in zone_id or "\n" in zone_id:
        print(f"ERROR: multiple hosted zones matched '{zone_name}': {zone_id!r}", file=sys.stderr)
        sys.exit(1)
    return zone_id


def get_current_records(zone_id: str) -> dict[str, str]:
    """Return {name: current_ip} for all A records in the zone."""
    output = run([
        "aws", "route53", "list-resource-record-sets",
        "--hosted-zone-id", zone_id,
        "--query", "ResourceRecordSets[?Type=='A'].{Name:Name,Value:ResourceRecords[0].Value}",
        "--output", "json",
    ])
    records = json.loads(output)
    return {r["Name"]: r["Value"] for r in records if r.get("Value")}


def build_change_batch(records: dict[str, str], tunnel_ip: str) -> list[dict]:
    changes = []
    for name in MANAGED_SUBDOMAINS:
        current = records.get(name)
        if current == tunnel_ip:
            print(f"  {name} already → {tunnel_ip} (skip)")
            continue
        if current is None:
            print(f"  {name} not found in zone — will create → {tunnel_ip}")
        else:
            print(f"  {name} {current} → {tunnel_ip}")
        changes.append({
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": name,
                "Type": "A",
                "TTL": 60,
                "ResourceRecords": [{"Value": tunnel_ip}],
            },
        })
    return changes


def main() -> None:
    tunnel_ip = get_tunnel_ip()
    print(f"Tunnel IP: {tunnel_ip}")

    zone_id = get_zone_id(ZONE_NAME)
    print(f"Zone: {ZONE_NAME} ({zone_id})")

    records = get_current_records(zone_id)
    changes = build_change_batch(records, tunnel_ip)

    if not changes:
        print("All managed A records already point to the tunnel IP — no update needed.")
        return

    if DRY_RUN:
        print(f"DRY_RUN: would submit {len(changes)} change(s)")
        return

    change_batch = json.dumps({"Changes": changes})
    result = subprocess.run(
        [
            "aws", "route53", "change-resource-record-sets",
            "--hosted-zone-id", zone_id,
            "--change-batch", change_batch,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Route53 update failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    response = json.loads(result.stdout)
    change_id = response.get("ChangeInfo", {}).get("Id", "unknown")
    print(f"DNS sync submitted (change ID: {change_id})")


if __name__ == "__main__":
    main()
