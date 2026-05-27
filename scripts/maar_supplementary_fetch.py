#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fetch Maar et al. (2026) supplementary `.zip` from OpenReview.

Maar's paper text documents the contrastive activation steering protocol
only for `Gemma 2 9B` (layer 27, newline token, m=1.5).  Per-model layer
and token-position specs for the other 22 models in their wide sweep
(including `Gemma 2 2B`, `Llama 3.2 1B`, `Llama 3.2 3B`) live in the
supplementary materials attached to the OpenReview submission
`Z10pxu0Q7X`.

This script attempts to download that supplementary `.zip` via the
OpenReview API.  If the API returns an HTML page (login required) or a
403, the script prints a manual-download URL the user can open in a
browser.

## Usage

```bash
python scripts/maar_supplementary_fetch.py \
    --output examples/results/maar_contrastive_steering/maar_supplementary.zip
```

After download, point `examples/maar_contrastive_steering` at the
extracted protocol via `--maar-supplementary <path-to-extracted-dir>`,
or convert the prompts to our JSON schema via the (post-v0.1.12) helper
`scripts/maar_to_candle_mi_prompts.py`.

## Status as of v0.1.12

The OpenReview attachment endpoint requires a logged-in session in most
configurations.  If `--api-token` is not provided, this script will
likely report `403 Forbidden` and print the manual URL.  Manual download
remains the fall-back, recorded in the prompts JSON `source` field.
"""

import argparse
import sys
from pathlib import Path

OPENREVIEW_PAPER_ID = "Z10pxu0Q7X"
OPENREVIEW_FORUM_URL = f"https://openreview.net/forum?id={OPENREVIEW_PAPER_ID}"
OPENREVIEW_ATTACHMENT_URL = (
    f"https://openreview.net/attachment?"
    f"id={OPENREVIEW_PAPER_ID}&name=supplementary_material"
)
OPENREVIEW_API_ATTACHMENT_URL = (
    f"https://api2.openreview.net/attachment?"
    f"id={OPENREVIEW_PAPER_ID}&name=supplementary_material"
)


def fetch(output_path: Path, api_token: str | None) -> int:
    """Attempt to fetch the supplementary .zip.  Returns 0 on success,
    non-zero on failure."""
    try:
        import requests
    except ImportError:
        sys.stderr.write(
            "ERROR: `requests` package is required.  "
            "Install with `python -m pip install requests`.\n"
        )
        return 2

    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    print(f"Attempting download from {OPENREVIEW_API_ATTACHMENT_URL} ...")
    try:
        resp = requests.get(
            OPENREVIEW_API_ATTACHMENT_URL,
            headers=headers,
            timeout=120,
            stream=True,
        )
    except requests.RequestException as exc:
        sys.stderr.write(f"ERROR: request failed: {exc}\n")
        print_manual_fallback()
        return 3

    if resp.status_code != 200:
        sys.stderr.write(
            f"ERROR: HTTP {resp.status_code} from OpenReview API; "
            f"likely requires authentication.\n"
        )
        print_manual_fallback()
        return 4

    content_type = resp.headers.get("Content-Type", "")
    if "zip" not in content_type and "octet-stream" not in content_type:
        sys.stderr.write(
            f"ERROR: unexpected Content-Type '{content_type}'; "
            f"expected a binary zip.  Likely an HTML login redirect.\n"
        )
        print_manual_fallback()
        return 5

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=65_536):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    print(f"Downloaded {total / 1024 / 1024:.2f} MiB to {output_path}")
    return 0


def print_manual_fallback() -> None:
    print(
        "\n=== Manual download fall-back ===\n"
        f"1. Open in a browser:  {OPENREVIEW_FORUM_URL}\n"
        '2. Sign in to OpenReview (free account).\n'
        '3. Scroll to "Supplementary Material" attachment.\n'
        f"4. Save the .zip to your target output path.\n"
        "\n"
        "After download, point the example at the extracted protocol\n"
        "or convert the prompts to our JSON schema (post-v0.1.12).\n",
        file=sys.stderr,
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Fetch Maar et al. (2026) supplementary .zip from OpenReview."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/results/maar_contrastive_steering/maar_supplementary.zip"),
        help="Where to save the downloaded .zip.",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="OpenReview API token (optional; tries unauthenticated first).",
    )
    args = parser.parse_args()
    return fetch(args.output, args.api_token)


if __name__ == "__main__":
    sys.exit(main())
