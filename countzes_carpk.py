import json
import os
import numpy as np

from countzes_core import (
    build_argument_parser,
    load_pipeline_models,
    run_recorded_evaluation,
)


def get_arguments():
    parser = build_argument_parser(
        default_outdir="countzes_carpk",
        default_fsoc="DSALVANet",
        default_countr_ckpt="./CounTR/output_allnew_dir/CARPK.pth",
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    print("Args:", args)

    os.makedirs("./outputs", exist_ok=True)
    output_path = os.path.join("./outputs", args.outdir)
    models = load_pipeline_models(args, log_dir=os.path.join(output_path, "logs"))
    countzes(args, "CARPK", output_path, models)


def countzes(args, obj_name, output_path, models):
    vis_dir = os.path.join(output_path, obj_name)
    os.makedirs(vis_dir, exist_ok=True)

    carpk_dir = os.path.join(args.data, "CARPK")
    json_file_path = os.path.join(carpk_dir, "test.json")
    preprocessed_json_file_path = os.path.join(carpk_dir, "preprocessed_test.json")
    image_folder_path = os.path.join(carpk_dir, "Images")
    gt_path = os.path.join(carpk_dir, "gt_density_map")

    data_list = []
    with open(json_file_path, "r") as f:
        for line in f:
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")

    with open(preprocessed_json_file_path, "w") as f:
        json.dump(data_list, f, indent=4)
    print(f"Preprocessed JSON data saved to {preprocessed_json_file_path}")

    with open(preprocessed_json_file_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON data is not a list of entries.")

    samples = []
    for entry in data:
        filename = entry["filename"]
        gt_img_path = os.path.join(gt_path, entry["density"])
        gt_density = np.load(gt_img_path)
        samples.append(
            {
                "filename": filename,
                "image_path": os.path.join(image_folder_path, filename),
                "class_name": "cars",
                "gt_count": int(np.sum(gt_density)),
                "vis_save_path": os.path.join(vis_dir, f"refined_{filename}"),
            }
        )

    run_recorded_evaluation("CARPK", output_path, args, samples, models)


if __name__ == "__main__":
    main()
