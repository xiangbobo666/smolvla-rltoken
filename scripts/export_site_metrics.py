#!/usr/bin/env python
"""Convert the lerobot-train SFT log into rl-experiment-site ``metrics.jsonl``.

Stage 1 / Stage 2 / evaluation write the standard site files themselves
(``smolvla_rltoken.site_log``). SFT runs inside ``lerobot-train``, so its curve
is rebuilt from the launcher log instead. Safe to rerun while training: the
file is rewritten atomically and the site reloads it by mtime.

  python scripts/export_site_metrics.py                     # YAML output_dir + launcher log
  python scripts/export_site_metrics.py --log LOG --out DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_rltoken import site_log  # noqa: E402
from smolvla_rltoken.paths import SFT_CONFIG_PATH  # noqa: E402
from smolvla_rltoken.vla.sft import SFTConfig  # noqa: E402

# Same default as scripts/train_sft.sh (SFT_LOG).
DEFAULT_SFT_LOG = ROOT / "outputs" / "sft" / "peg_insertion_run.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=SFT_CONFIG_PATH, help="SFT YAML (log_freq, output_dir)")
    parser.add_argument("--log", type=Path, default=DEFAULT_SFT_LOG, help="lerobot-train stdout log")
    parser.add_argument("--out", type=Path, default=None, help="run directory (default: YAML output_dir)")
    parser.add_argument("--log-freq", type=int, default=None, help="override YAML log_freq")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SFTConfig.from_yaml(args.config)
    out_dir = args.out or Path(cfg.output_dir)
    log_freq = args.log_freq or cfg.log_freq
    if not args.log.is_file():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 1
    rows = site_log.parse_lerobot_train_log(args.log.read_text(errors="replace"), log_freq)
    if not rows:
        print(f"no lerobot step lines in {args.log}", file=sys.stderr)
        return 1
    target = out_dir / site_log.METRICS_FILENAME
    site_log.write_jsonl(target, rows)
    print(f"wrote {len(rows)} rows (last step {int(rows[-1]['step'])}) to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
