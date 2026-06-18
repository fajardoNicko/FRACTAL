"""Quick single-image coastline-extraction preview / tuning tool.

Run this on ONE cropped map image to check that the coastline is detected
cleanly before processing all your segments. It saves an overlay (detected
coastline drawn in red over your map) and prints the fractal dimension.

Examples:
    # Topographic / NAMRIA sheet (blue sea): detect sea by colour
    python tools/preview_coastline.py data/raw/images/seg_01.jpg --method sea_color

    # Clean two-tone land/sea image: brightness threshold
    python tools/preview_coastline.py data/raw/images/seg_01.jpg --method otsu

    # Tune the blue band / drop a leftover frame
    python tools/preview_coastline.py img.jpg --method sea_color \
        --hue-lo 120 --hue-hi 185 --sat-min 25 --crop 0.02

The overlay is written next to the project's output/previews/ folder. Open it
and confirm the red line traces the coastline and nothing else.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fractal.box_counting import box_counting_dimension  # noqa: E402
from src.preprocessing.image_coastline import extract_coastline  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.file_helpers import ensure_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Preview coastline extraction on one image.")
    ap.add_argument("image", help="path to a (cropped) map image")
    ap.add_argument("--method", choices=["otsu", "sea_color"], default="sea_color")
    ap.add_argument("--threshold", default="otsu", help="otsu method: 'otsu' or 0-255")
    ap.add_argument("--crop", type=float, default=0.0, help="crop fraction per side")
    ap.add_argument("--hue-lo", type=float, default=120.0)
    ap.add_argument("--hue-hi", type=float, default=185.0)
    ap.add_argument("--sat-min", type=float, default=25.0)
    ap.add_argument("--val-min", type=float, default=40.0)
    ap.add_argument("--no-keep-largest", action="store_true")
    ap.add_argument("--no-fill-holes", action="store_true")
    args = ap.parse_args()

    config = load_config()
    out_dir = ensure_dir(config.path("previews"))
    stem = os.path.splitext(os.path.basename(args.image))[0]
    overlay = os.path.join(out_dir, f"preview_{stem}.png")

    threshold = args.threshold
    if threshold not in ("otsu",):
        threshold = float(threshold)

    boundary = extract_coastline(
        args.image,
        method=args.method,
        threshold=threshold,
        keep_largest=not args.no_keep_largest,
        fill_holes=not args.no_fill_holes,
        crop_border_frac=args.crop,
        hue_lo=args.hue_lo, hue_hi=args.hue_hi,
        sat_min=args.sat_min, val_min=args.val_min,
        debug_path=overlay,
    )

    result = box_counting_dimension(boundary, config.box_sizes)
    print(f"  image:            {args.image}")
    print(f"  method:           {args.method}")
    print(f"  coastline pixels: {int(boundary.sum())}")
    print(f"  fractal dim D:    {result.dimension:.4f}")
    print(f"  log-log R^2:      {result.r_squared:.4f}"
          + ("  [FLAGGED < 0.97]" if result.flagged else ""))
    print(f"  overlay saved to: {overlay}")
    print("\nOpen the overlay and check the red line traces ONLY the coastline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
