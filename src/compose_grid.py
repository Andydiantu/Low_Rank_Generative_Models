import argparse
import os
from typing import List

from PIL import Image


def list_pngs_sorted(input_dir: str) -> List[str]:
    pngs = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.lower().endswith(".png")
    ]
    pngs.sort()
    return pngs


def compose_grid(
    image_paths: List[str],
    rows: int,
    cols: int,
    tile_size: int = 64,
    mode: str = "RGB",
) -> Image.Image:
    grid_width = cols * tile_size
    grid_height = rows * tile_size
    grid_img = Image.new(mode, (grid_width, grid_height))

    for index, img_path in enumerate(image_paths):
        row = index // cols
        col = index % cols
        with Image.open(img_path) as img:
            if mode:
                img = img.convert(mode)
            if img.size != (tile_size, tile_size):
                img = img.resize((tile_size, tile_size), Image.BICUBIC)
            grid_img.paste(img, (col * tile_size, row * tile_size))

    return grid_img


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Compose 16 64x64 PNGs from a folder into a single 4x4 grid image."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/vol/bitbucket/aw624/Low_Rank_Generative_Models/logs/DiT20250906_103859/samples_gen_2",
        help="Folder containing PNG images (default: the provided samples folder)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of rows in the grid (default: 4)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in the grid (default: 4)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="Tile size in pixels for each image (default: 64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: grid_{rows}x{cols}.png in input_dir)",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    all_pngs = list_pngs_sorted(input_dir)
    expected = args.rows * args.cols
    if len(all_pngs) < expected:
        raise SystemExit(
            f"Found {len(all_pngs)} PNGs in {input_dir}, but need at least {expected}."
        )

    selected = all_pngs[:expected]

    grid = compose_grid(
        selected,
        rows=args.rows,
        cols=args.cols,
        tile_size=args.tile_size,
        mode="RGB",
    )

    output_path = (
        args.output
        if args.output
        else os.path.join(input_dir, f"grid_{args.rows}x{args.cols}.png")
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid.save(output_path)
    print(f"Saved grid image to: {output_path}")


if __name__ == "__main__":
    main()


