from config.oct_config import OctConfig


def generate_optuna_seed(config: OctConfig) -> dict[str, Any]:
    """
    Flattens the base config into a dictionary of Optuna parameters.
    The keys match the string names used in the trial.suggest_* calls.
    """
    seed = {}

    # Convert config to a raw dictionary to easily parse list-of-dicts structures
    # Adjust `model_dump` to `dict()` if you are using older Pydantic versions
    config_dict = config.model_dump()

    # 1. PREPROCESSING
    for step_dict in config_dict.get("preprocessing") or []:
        for name, params in step_dict.items():
            prefix = name  # Matches the default prefix in PreprocessingTuners

            if name == "resizer":
                target_size = params.get("target_size", [256, 120])
                seed[f"{prefix}_target_width"] = target_size[0]
                seed[f"{prefix}_target_height"] = target_size[1]
            elif name == "corner_artifact_remover":
                seed[f"{prefix}_min_threshold"] = params.get("min_threshold")
            elif name == "dwt_denoiser":
                seed[f"{prefix}_wavelet"] = params.get("wavelet")
                seed[f"{prefix}_mode"] = params.get("mode")
                seed[f"{prefix}_wavelet_levels"] = params.get("wavelet_levels")
                seed[f"{prefix}_method"] = params.get("method")
                seed[f"{prefix}_rescale_sigma"] = params.get("rescale_sigma")
            elif name == "rpe_roi_aligner":
                seed[f"{prefix}_roi_above"] = params.get("roi_above")
                seed[f"{prefix}_roi_below"] = params.get("roi_below")
                seed[f"{prefix}_blur_sigma"] = params.get("blur_sigma")
                seed[f"{prefix}_max_jump"] = params.get("max_jump")
                seed[f"{prefix}_poly_degree"] = params.get("poly_degree")
            elif name == "rpe_aligner":
                seed[f"{prefix}_blur_sigma"] = params.get("blur_sigma")
                seed[f"{prefix}_poly_degree"] = params.get("poly_degree")
                seed[f"{prefix}_max_jump"] = params.get("max_jump")

    # 2. FEATURE EXTRACTION
    for step_dict in config_dict.get("feature_extraction") or []:
        for name, params in step_dict.items():
            prefix = name

            if name == "ldch":
                seed[f"{prefix}_sections"] = params.get("sections")
                seed[f"{prefix}_bins"] = params.get("bins")
                seed[f"{prefix}_layout"] = params.get("layout")
                if "pixel_range" in params:
                    seed[f"{prefix}_min_val"] = params["pixel_range"][0]
                    seed[f"{prefix}_max_val"] = params["pixel_range"][1]

            elif name == "hog":
                # Extracted from [16, 16] to 16
                seed[f"{prefix}_cell_size"] = params.get("pixels_per_cell", [8])[0]
                seed[f"{prefix}_cells_per_block"] = params.get("cells_per_block", [1])[
                    0
                ]
                seed[f"{prefix}_orientations"] = params.get("orientations")
                seed[f"{prefix}_transform_sqrt"] = params.get("transform_sqrt")
                seed[f"{prefix}_block_norm"] = params.get("block_norm")

            elif name == "lbp":
                seed[f"{prefix}_n_points"] = params.get("n_points")
                seed[f"{prefix}_radius"] = params.get("radius")
                seed[f"{prefix}_method"] = params.get("method")
                seed[f"{prefix}_norm"] = params.get("norm")
                seed[f"{prefix}_padding_threshold"] = params.get("padding_threshold")

            elif name in ("cdch", "rdch", "rcldch"):
                seed[f"{prefix}_bins"] = params.get("bins")
                if "pixel_range" in params:
                    seed[f"{prefix}_min_val"] = params["pixel_range"][0]
                    seed[f"{prefix}_max_val"] = params["pixel_range"][1]
                if name == "cdch":
                    seed[f"{prefix}_angle_step"] = params.get("angle_step")
                if name == "rdch":
                    seed[f"{prefix}_ring_step"] = params.get("ring_step")
                if name in ("cdch", "rdch") and "center" in params:
                    seed[f"{prefix}_center_x"] = params["center"][0]
                    seed[f"{prefix}_center_y"] = params["center"][1]
                if name == "rcldch":
                    seed[f"{prefix}_sections"] = params.get("sections")

    # 3. DIMENSIONALITY REDUCTION
    for step_dict in config_dict.get("dimensionality_reduction") or []:
        for name, params in step_dict.items():
            prefix = name

            if name == "pca":
                seed[f"{prefix}_solver"] = params.get("svd_solver")
                seed[f"{prefix}_components"] = params.get("n_components")
                seed[f"{prefix}_whiten"] = params.get("whiten")
                seed[f"{prefix}_iterated_power"] = params.get("iterated_power")
                seed[f"{prefix}_power_iteration_normalizer"] = params.get(
                    "power_iteration_normalizer"
                )

    # 4. CLASSIFICATION
    for step_dict in config_dict.get("classification") or []:
        for name, params in step_dict.items():
            prefix = name

            if name == "svm":
                seed[f"{prefix}_kernel"] = params.get("kernel")
                seed[f"{prefix}_c"] = params.get("C")
                seed[f"{prefix}_class_weight"] = params.get("class_weight")
                seed[f"{prefix}_shrinking"] = params.get("shrinking")

                if seed.get(f"{prefix}_kernel") in ("rbf", "poly"):
                    seed[f"{prefix}_gamma"] = params.get("gamma")
                if seed.get(f"{prefix}_kernel") == "poly":
                    seed[f"{prefix}_degree"] = params.get("degree")

            elif name == "random_forest":
                seed[f"{prefix}_bootstrap"] = params.get("bootstrap")
                seed[f"{prefix}_estimators"] = params.get("n_estimators")
                seed[f"{prefix}_max_depth"] = params.get("max_depth")
                seed[f"{prefix}_min_samples_split"] = params.get("min_samples_split")
                seed[f"{prefix}_min_samples_leaf"] = params.get("min_samples_leaf")
                seed[f"{prefix}_max_features"] = params.get("max_features")
                seed[f"{prefix}_criterion"] = params.get("criterion")
                seed[f"{prefix}_class_weight"] = params.get("class_weight")

                if seed.get(f"{prefix}_bootstrap"):
                    seed[f"{prefix}_max_samples"] = params.get("max_samples")

            elif name == "knn":
                seed[f"{prefix}_algorithm"] = params.get("algorithm")
                if seed.get(f"{prefix}_algorithm") not in ("kd_tree", "ball_tree"):
                    seed[f"{prefix}_metric"] = params.get("metric")
                seed[f"{prefix}_neighbors"] = params.get("n_neighbors")
                seed[f"{prefix}_weights"] = params.get("weights")
                seed[f"{prefix}_p"] = params.get("p")
                seed[f"{prefix}_leaf_size"] = params.get("leaf_size")

            elif name == "logistic_regression":
                seed[f"{prefix}_c"] = params.get("C")
                seed[f"{prefix}_class_weight"] = params.get("class_weight")
                seed[f"{prefix}_solver"] = params.get("solver")

    # Clean up any None values to prevent Optuna from crashing
    return {k: v for k, v in seed.items() if v is not None}
