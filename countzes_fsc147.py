import json
import os

from countzes_core import (
    build_argument_parser,
    load_class_map,
    load_pipeline_models,
    run_recorded_evaluation,
)


def get_arguments():
    parser = build_argument_parser(
        default_outdir="countzes_fsc147",
        default_fsoc="DSALVANet",
        default_countr_ckpt="./CounTR/output_allnew_dir/FSC147.pth",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="FSC-147 split to evaluate ('test' = 1190 images, 'val' = 1286 images).",
    )
    return parser.parse_args()


def _load_fsc147_samples(args, vis_dir):
    fsc_dir = os.path.join(args.data, "FSC-147")
    json_file_path = os.path.join(fsc_dir, f"{args.split}.json")
    image_class_file_path = os.path.join(fsc_dir, "ImageClasses_FSC147.txt")
    image_folder_path = os.path.join(fsc_dir, "images_384_VarV2")

    if not os.path.isdir(image_folder_path) or not any(
        f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(image_folder_path)
    ):
        raise FileNotFoundError(
            f"FSC-147 image directory is missing or empty: {image_folder_path}. "
            "Please download/extract images_384_VarV2 first."
        )

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    class_map = load_class_map(image_class_file_path)
    samples = []
    for entry in data:
        filename = entry["filename"]
        class_name = class_map.get(filename)
        if class_name is None:
            raise KeyError(f"Class name not found for {filename} in {image_class_file_path}")
        samples.append(
            {
                "filename": filename,
                "image_path": os.path.join(image_folder_path, filename),
                "class_name": class_name,
                "gt_count": len(entry["points"]),
                "vis_save_path": os.path.join(vis_dir, f"refined_{filename}"),
            }
        )
    return samples


def main():
    args = get_arguments()
    print("Args:", args)

    os.makedirs("./outputs", exist_ok=True)
    output_path = os.path.join("./outputs", args.outdir)
    vis_dir = os.path.join(output_path, f"FSC147_{args.split}")
    samples = _load_fsc147_samples(args, vis_dir)
    os.makedirs(vis_dir, exist_ok=True)

    models = load_pipeline_models(args, log_dir=os.path.join(output_path, "logs"))
    run_recorded_evaluation(f"FSC147_{args.split}", output_path, args, samples, models)


if __name__ == "__main__":
    main()
