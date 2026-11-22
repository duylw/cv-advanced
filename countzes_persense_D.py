import json
import os

from countzes_core import (
    build_argument_parser,
    load_pipeline_models,
    run_recorded_evaluation,
)


def get_arguments():
    parser = build_argument_parser(
        default_outdir="countzes_persenseD",
        default_fsoc="countr",
        default_countr_ckpt="./CounTR/output_allnew_dir/FSC147.pth",
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    print("Args:", args)

    os.makedirs("./outputs", exist_ok=True)
    output_path = os.path.join("./outputs", args.outdir)
    models = load_pipeline_models(args, log_dir=os.path.join(output_path, "logs"))

    images_path = os.path.join(args.data, "persenseD", "Images")
    samples = []
    for obj_name in sorted(os.listdir(images_path)):
        test_images_path = os.path.join(images_path, obj_name)
        if ".DS" in obj_name or not os.path.isdir(test_images_path):
            continue
        obj_output_path = os.path.join(output_path, obj_name)
        anno_path = os.path.join(test_images_path, f"points_{obj_name}.json")
        with open(anno_path, "r") as file:
            gt_by_img = {entry["img"]: entry["total_points"] for entry in json.load(file)}
        img_files = sorted(
            f for f in os.listdir(test_images_path) if f.lower().endswith(".jpg")
        )
        for img_file in img_files:
            if img_file not in gt_by_img:
                raise KeyError(f"Ground truth not found for {img_file} in {anno_path}")
            test_idx = os.path.splitext(img_file)[0]
            samples.append(
                {
                    "filename": img_file,
                    "image_path": os.path.join(test_images_path, img_file),
                    "class_name": obj_name,
                    "gt_count": gt_by_img[img_file],
                    "vis_save_path": os.path.join(obj_output_path, f"{test_idx}.png"),
                }
            )

    run_recorded_evaluation("PerSenseD", output_path, args, samples, models)


if __name__ == "__main__":
    main()
