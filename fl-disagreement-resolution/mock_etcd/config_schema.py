"""Configuration validation for federated learning experiments."""


VALID_EXPERIMENT_TYPES = {"cifar10", "mnist", "n_cmapss", "qa", "tabular", "adult"}
VALID_UNLEARNING_STRATEGIES = {"exact_retraining", "federated_exact_retraining", "sisa", "distillation", "mf"}


def validate_config(config):
    """Validate a federated learning configuration dict.

    Args:
        config: Configuration dictionary (parsed JSON).

    Returns:
        list[str]: List of error messages. Empty if config is valid.
    """
    errors = []

    # --- Required top-level sections ---
    for section in ("experiment", "training", "data", "results"):
        if section not in config:
            errors.append(f"Missing required section '{section}'")

    # Early return if critical sections are missing
    if errors:
        return errors

    #--- experiment section ---
    exp = config["experiment"]
    _require_key(errors, exp, "experiment", "type")
    _require_key(errors, exp, "experiment", "fl_rounds")
    _require_key(errors, exp, "experiment", "client_ids")

    if "type" in exp:
        if exp["type"] not in VALID_EXPERIMENT_TYPES:
            errors.append(
                f"'experiment.type' must be one of {sorted(VALID_EXPERIMENT_TYPES)}, "
                f"got '{exp['type']}'"
            )

    if "fl_rounds" in exp:
        if not isinstance(exp["fl_rounds"], int):
            errors.append(
                f"'experiment.fl_rounds' should be int, got {type(exp['fl_rounds']).__name__}"
            )
        elif exp["fl_rounds"] < 1:
            errors.append(f"'experiment.fl_rounds' must be >= 1, got {exp['fl_rounds']}")

    if "client_ids" in exp:
        if not isinstance(exp["client_ids"], list):
            errors.append(
                f"'experiment.client_ids' should be list, got {type(exp['client_ids']).__name__}"
            )
        elif len(exp["client_ids"]) < 1:
            errors.append("'experiment.client_ids' must contain at least 1 client")

    #N-CMAPSS client limit
    if exp.get("type") == "n_cmapss" and isinstance(exp.get("client_ids"), list):
        if len(exp["client_ids"]) > 6:
            errors.append(
                f"N-CMAPSS supports at most 6 clients, got {len(exp['client_ids'])}"
            )

    # --- training section ---
    train = config["training"]
    if "learning_rate" in train:
        if not isinstance(train["learning_rate"], (int, float)):
            errors.append(
                f"'training.learning_rate' should be float, "
                f"got {type(train['learning_rate']).__name__}"
            )

    if "batch_size" in train:
        if not isinstance(train["batch_size"], int) or train["batch_size"] < 1:
            errors.append(f"'training.batch_size' must be a positive int")

    if "local_epochs" in train:
        if not isinstance(train["local_epochs"], int) or train["local_epochs"] < 1:
            errors.append(f"'training.local_epochs' must be a positive int")

    # --- unlearning section (optional) ---
    if "unlearning" in config:
        unl = config["unlearning"]
        if "strategies" in unl and isinstance(unl["strategies"], list):
            for s in unl["strategies"]:
                if s not in VALID_UNLEARNING_STRATEGIES:
                    errors.append(
                        f"Unknown unlearning strategy '{s}', "
                        f"valid: {sorted(VALID_UNLEARNING_STRATEGIES)}"
                    )

    # --- data section ---
    data = config["data"]
    if "train_dir" not in data:
        errors.append("Missing required key 'data.train_dir'")
    if "test_dir" not in data:
        errors.append("Missing required key 'data.test_dir'")

    return errors


def _require_key(errors, section, section_name, key):
    """Check that a required key exists in a config section."""
    if key not in section:
        errors.append(f"Missing required key '{section_name}.{key}'")
