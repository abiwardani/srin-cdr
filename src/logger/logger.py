from datetime import datetime

def log_train_results(data_lib, model, trainer, metrics, log_dir="./log"):
    date = datetime.now().strftime("%b_%d_%Y_%H%M")

    source_name = data_lib["source_name"]
    target_name = data_lib["target_name"]
    n_users = data_lib["n_users"]
    n_source_items = data_lib["n_source_items"]
    n_target_items = data_lib["n_target_items"]
    n_source_interactions = len(data_lib["source_interactions"])
    n_target_interactions = len(data_lib["target_interactions"])
    model_type = model.get_model_info().get("model_type")

    log_path = f"{log_dir}/{model_type}_{date}.log"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Data Info:\n")
        f.write(f"Source Name: {source_name}\n")
        f.write(f"Target Name: {target_name}\n")
        f.write(f"Number of Users: {n_users}\n")
        f.write(f"Number of Source Items: {n_source_items}\n")
        f.write(f"Number of Target Items: {n_target_items}\n\n")
        f.write(f"Number of Source Interactions: {n_source_interactions}\n")
        f.write(f"Number of Target Interactions: {n_target_interactions}\n\n")
        f.write("Model Info:\n")
        f.write(str(model) + "\n\n")
        f.write("Trainer Info:\n")
        f.write(str(trainer) + "\n\n")
        f.write("Training Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")