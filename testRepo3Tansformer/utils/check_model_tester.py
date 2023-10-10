
import glob
import os

from get_test_info import get_tester_classes


if __name__ == "__main__":
    failures = []

    pattern = os.path.join("tests", "models", "D3bugM3N0w", "test_modeling_*.py")
    test_files = glob.glob(pattern)
    # TODO: deal with TF/Flax too
    test_files = [
        x for x in test_files if not (x.startswith("test_modeling_tf_") or x.startswith("test_modeling_flax_"))
    ]

    for test_file in test_files:
        tester_classes = get_tester_classes(test_file)
        for tester_class in tester_classes:
            try:
                tester = tester_class(parent=None)
            except Exception:
                continue
            if hasattr(tester, "get_config"):
                config = tester.get_config()
                for k, v in config.to_dict().items():
                    if isinstance(v, int):
                        target = None
                        if k in ["vocab_size"]:
                            target = 100
                        elif k in ["max_position_embeddings"]:
                            target = 128
                        elif k in ["hidden_size", "d_model"]:
                            target = 40
                        elif k == ["num_layers", "num_hidden_layers", "num_encoder_layers", "num_decoder_layers"]:
                            target = 5
                        if target is not None and v > target:
                            failures.append(
                                f"{tester_class.__name__} will produce a `config` of type `{config.__class__.__name__}`"
                                f' with config["{k}"] = {v} which is too large for testing! Set its value to be smaller'
                                f" than {target}."
                            )

    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))
