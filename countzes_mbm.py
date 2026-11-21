from countzes_core import evaluate_mask_annotated_dataset


def main():
    evaluate_mask_annotated_dataset(
        dataset_subdir="mbm",
        default_outdir="countzes_mbm",
        default_fsoc="countr",
    )


if __name__ == "__main__":
    main()
