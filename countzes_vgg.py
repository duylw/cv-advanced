from countzes_core import evaluate_mask_annotated_dataset


def main():
    evaluate_mask_annotated_dataset(
        dataset_subdir="vgg",
        default_outdir="countzes_vgg",
        default_fsoc="DSALVANet",
    )


if __name__ == "__main__":
    main()
