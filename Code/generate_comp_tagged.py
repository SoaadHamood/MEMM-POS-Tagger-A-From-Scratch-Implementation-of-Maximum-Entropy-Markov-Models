import pickle
from inference import tag_all_test
from preprocessing import Feature2id

# ====== Configuration ======
model_files = {
    "model1": {
        "weights_path": "weights_1.pkl",
        "comp_input": "data/comp1.words",
        "comp_output": "comp_m1_212794762.wtag"
    },
    "model2": {
        "weights_path": "weights_2.pkl",
        "comp_input": "data/comp2.words",
        "comp_output": "comp_m2_212794762.wtag"
    }
}

# ====== Generate competition predictions ======
for model_name, paths in model_files.items():
    print(f"\n[INFO] Loading {model_name} from {paths['weights_path']}")

    with open(paths["weights_path"], "rb") as f:
        (optimal_params, feature2id) = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(f"[INFO] Tagging competition data from {paths['comp_input']}")
    tag_all_test(paths["comp_input"], pre_trained_weights, feature2id, paths["comp_output"])

    print(f"[INFO] Output saved to {paths['comp_output']}")
