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
    countzes(args, "FSC147", output_path, models)


def countzes(args, obj_name, output_path, models):
    vis_dir = os.path.join(output_path, obj_name)
    os.makedirs(vis_dir, exist_ok=True)

    fsc_dir = os.path.join(args.data, "FSC-147")
    json_file_path = os.path.join(fsc_dir, "test.json")
    preprocessed_json_file_path = os.path.join(fsc_dir, "preprocessed_test.json")
    image_class_file_path = os.path.join(fsc_dir, "ImageClasses_FSC147.txt")
    image_folder_path = os.path.join(fsc_dir, "images_384_VarV2")

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

    run_recorded_evaluation("FSC147", output_path, args, samples, models)


if __name__ == "__main__":
    main()
