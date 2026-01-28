# python make_symbolic_links_dataset.py --input /hddevice/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS_newSynthIDs_Arc2Face_sim=[0.5,0.69]_10572ids_DETECTED_FACES_RETINAFACE_scales=[1.0]_nms=0.4/imgs --num-sub-dirs 1000 --output /hddevice/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS_newSynthIDs_Arc2Face_sim=[0.5,0.69]_1000ids_DETECTED_FACES_RETINAFACE_scales=[1.0]_nms=0.4_RANDOM1

import argparse
import os
import random
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create symbolic links for randomly selected sub-directories."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Root dataset directory containing sub-directories"
    )
    parser.add_argument(
        "--num-sub-dirs",
        required=True,
        type=int,
        help="Number of sub-directories to select randomly (-1 or >= N selects all)"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output directory where symbolic links will be created"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    num_sub_dirs = args.num_sub_dirs

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Collect immediate sub-directories
    sub_dirs = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]

    if not sub_dirs:
        print("No sub-directories found in input directory.")
        sys.exit(0)

    N = len(sub_dirs)

    # Selection logic
    if num_sub_dirs == -1 or num_sub_dirs >= N:
        selected = sub_dirs
    elif 0 < num_sub_dirs < N:
        selected = random.sample(sub_dirs, num_sub_dirs)
    else:
        print("Error: --num-sub-dirs must be -1 or a positive integer")
        sys.exit(1)

    # Create symbolic links
    for name in selected:
        src = os.path.join(input_dir, name)
        dst = os.path.join(output_dir, name)

        if os.path.exists(dst):
            print(f"Skipping (already exists): {dst}")
            continue

        os.symlink(src, dst)

    print(f"Created {len(selected)} symbolic links in {output_dir}")


if __name__ == "__main__":
    main()
