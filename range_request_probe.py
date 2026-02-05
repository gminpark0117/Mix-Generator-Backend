#!/usr/bin/env python3
"""
Minimal HTTP Range request probe (no third-party deps).

Examples:
  python range_request_probe.py "http://3.35.71.175/storage/mix/...wav" --start 0 --end 1023
  python range_request_probe.py "http://3.35.71.175/storage/mix/...wav" --start 1048576 --end 1052671
"""

from __future__ import annotations

import argparse
import http.client
import sys
from urllib.parse import urlparse


def _fmt_range(start: int, end: int | None) -> str:
    if start < 0:
        raise ValueError("--start must be >= 0")
    if end is not None and end < start:
        raise ValueError("--end must be >= --start")
    return f"bytes={start}-" if end is None else f"bytes={start}-{end}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Send a GET with Range header and print response details.")
    ap.add_argument(
        "url",
        help='Audio URL. Accepts full URL (e.g. "http://127.0.0.1:8000/storage/...") '
        'or host:port/path (e.g. "127.0.0.1:8000/storage/...").',
    )
    ap.add_argument("--start", type=int, default=0, help="Range start byte (default: 0).")
    ap.add_argument("--end", type=int, default=1023, help="Range end byte (default: 1023). Use -1 for open-ended.")
    ap.add_argument(
        "--max-read",
        type=int,
        default=8192,
        help="Safety limit for bytes to read from response body (default: 8192).",
    )
    args = ap.parse_args()

    end: int | None = None if args.end == -1 else args.end
    range_header = _fmt_range(args.start, end)

    raw_url = args.url.strip()
    # Users often paste "127.0.0.1:8000/path" without a scheme; default to http://
    if "://" not in raw_url:
        raw_url = f"http://{raw_url}"

    u = urlparse(raw_url)
    if u.scheme not in ("http", "https"):
        print(f"Unsupported URL scheme: {u.scheme!r}. Try prefixing with http://", file=sys.stderr)
        return 2
    if not u.netloc:
        print("URL must include host, e.g. http://127.0.0.1:8000/path", file=sys.stderr)
        return 2

    path = u.path or "/"
    if u.query:
        path = f"{path}?{u.query}"

    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(u.hostname, u.port, timeout=15)

    headers = {
        "Range": range_header,
        "User-Agent": "range-request-probe/1.0",
        "Accept": "*/*",
        "Connection": "close",
    }
    if u.username or u.password:
        print("URL-embedded credentials are not supported.", file=sys.stderr)
        return 2

    print(f"==> GET {raw_url}")
    print(f"==> Range: {range_header}")

    try:
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        print(f"<== HTTP {resp.status} {resp.reason}")

        # Print high-signal headers first, then everything else.
        hdrs = {k.lower(): v for (k, v) in resp.getheaders()}
        for k in ("accept-ranges", "content-range", "content-length", "content-type", "cache-control", "etag"):
            if k in hdrs:
                print(f"<== {k}: {hdrs[k]}")

        # Read only up to max-read so a server ignoring Range doesn't download a huge file.
        to_read = args.max_read
        if "content-length" in hdrs:
            try:
                to_read = min(to_read, int(hdrs["content-length"]))
            except ValueError:
                pass

        body = resp.read(to_read)
        print(f"<== read_bytes: {len(body)} (limit {args.max_read})")

        if resp.status == 206:
            print("OK: server honored Range (206 Partial Content).")
        elif resp.status == 200:
            print("WARN: server ignored Range (200 OK). Check proxy/CDN config or whether this route supports ranges.")
        elif resp.status == 416:
            print("WARN: 416 Range Not Satisfiable (bad range or unknown length).")
        else:
            print("WARN: unexpected status for a Range request.")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
