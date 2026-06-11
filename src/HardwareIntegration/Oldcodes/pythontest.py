import argparse
from math import tau
from pathlib import Path

import rerun as rr
from rerun.utilities import build_color_spiral

NUM_POINTS = 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Rerun with a DNA point cloud.")
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Open the native viewer instead of saving an RRD recording.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rerun_test.rrd"),
        help="Recording path used when --spawn is not provided.",
    )
    args = parser.parse_args()

    rr.init("rerun_example_dna_abacus", spawn=args.spawn)
    if not args.spawn:
        rr.save(args.output)

    points1, colors1 = build_color_spiral(NUM_POINTS)
    points2, colors2 = build_color_spiral(NUM_POINTS, angular_offset=tau * 0.5)

    rr.log(
        "dna/structure/left",
        rr.Points3D(points1, colors=colors1, radii=0.08),
    )
    rr.log(
        "dna/structure/right",
        rr.Points3D(points2, colors=colors2, radii=0.08),
    )

    if not args.spawn:
        print(f"Saved Rerun recording to {args.output.resolve()}")


if __name__ == "__main__":
    main()
