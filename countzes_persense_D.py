import json
import os
import cv2
from tqdm import tqdm

from countzes_core import (
    build_argument_parser,
    load_pipeline_models,
    run_countzes_on_image,
    update_and_log_metrics,
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
    models = load_pipeline_models(args)

    images_path = os.path.join(args.data, "persenseD", "Images")
    os.makedirs("./outputs", exist_ok=True)
    output_path = os.path.join("./outputs", args.outdir)

    count_item = 0
    mae = 0.0
    mse = 0.0

    for obj_name in sorted(os.listdir(images_path)):
        obj_dir = os.path.join(images_path, obj_name)
        if ".DS" not in obj_name and os.path.isdir(obj_dir):
            count_item, mae, mse = countzes(
                args, obj_name, images_path, output_path, models, count_item, mae, mse
            )


def countzes(args, obj_name, images_path, output_path, models, count_item, mae, mse):
    test_images_path = os.path.join(images_path, obj_name)
    obj_output_path = os.path.join(output_path, obj_name)
    os.makedirs(obj_output_path, exist_ok=True)

    anno_path = os.path.join(test_images_path, f"points_{obj_name}.json")
    with open(anno_path, "r") as file:
        gt_by_img = {entry["img"]: entry["total_points"] for entry in json.load(file)}

    print("======> Start Counting")
    img_files = sorted(
        f for f in os.listdir(test_images_path) if f.lower().endswith(".jpg")
    )
    for img_file in tqdm(img_files):
        test_idx = os.path.splitext(img_file)[0]
        test_image_path = os.path.join(test_images_path, img_file)
        if img_file not in gt_by_img:
            raise KeyError(f"Ground truth not found for {img_file} in {anno_path}")
        gt_count = gt_by_img[img_file]

        vis_output, count = run_countzes_on_image(
            image_path=test_image_path,
            filename=img_file,
            class_name=obj_name,
            models=models,
            args=args,
        )

        if args.visualize:
            counter_output_path = os.path.join(obj_output_path, f"{test_idx}.png")
            cv2.imwrite(counter_output_path, vis_output)

        count_item, mae, mse = update_and_log_metrics(
            count_item, mae, mse, count, gt_count
        )

    return count_item, mae, mse


if __name__ == "__main__":
    main()
