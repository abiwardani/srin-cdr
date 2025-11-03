import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.dataloader import *
from src.model.cmf import CMF, CMFTrainer
from src.model.ncf import NCF, NCFTrainer
from src.logger.logger import log_train_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate CDR models.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["CMF", "NCF"],
        required=True,
        help="Model to use: 'CMF' or 'NCF'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="128,64,32,8",
    )
    parser.add_argument(
        "--data-folder-path",
        type=str,
        default="./dataset",
        help="path to the folder containing dataset folders.",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="amazon-custom-books",
        help="Name of the source dataset.",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default="amazon-custom-movies-tv",
        help="Name of the target dataset.",
    )
    parser.add_argument(
        "--load-embeds",
        type=bool,
        default=False,
        help="Whether to load precomputed item embeddings.",
    )
    args = parser.parse_args()

    if args.model == "NCF":
        args.load_embeds = True
    
    args.layers = [int(x) for x in args.layers.split(",")]

    # Load data
    source_inter_filepath = f"{args.data_folder_path}/{args.source_name}/{args.source_name}.inter"
    target_inter_filepath = f"{args.data_folder_path}/{args.target_name}/{args.target_name}.inter"

    source_file, target_file = get_interaction_data(source_inter_filepath, target_inter_filepath)
    data_lib = build_data_lib(source_file, target_file)

    data_lib["source_name"] = args.source_name
    data_lib["target_name"] = args.target_name

    n_users = data_lib["n_users"]
    n_source_items = data_lib["n_source_items"]
    n_target_items = data_lib["n_target_items"]
    n_items = data_lib["n_items"]

    source_interactions = data_lib["source_interactions"]
    target_interactions = data_lib["target_interactions"]

    train_source_interactions = data_lib["train_source_interactions"]
    valid_source_interactions = data_lib["valid_source_interactions"]
    test_source_interactions = data_lib["test_source_interactions"]

    train_target_interactions = data_lib["train_target_interactions"]
    valid_target_interactions = data_lib["valid_target_interactions"]
    test_target_interactions = data_lib["test_target_interactions"]

    train_source_loader = DataLoader(train_source_interactions, batch_size=args.batch_size, shuffle=True)
    train_target_loader = DataLoader(train_target_interactions, batch_size=args.batch_size, shuffle=True)
    valid_source_loader = DataLoader(valid_source_interactions, batch_size=len(valid_source_interactions), shuffle=False) # no shuffle
    valid_target_loader = DataLoader(valid_target_interactions, batch_size=len(valid_target_interactions), shuffle=False)

    test_source_loader = DataLoader(test_source_interactions, batch_size=args.batch_size, shuffle=False)
    test_target_loader = DataLoader(test_target_interactions, batch_size=args.batch_size, shuffle=False)

    if args.load_embeds:
        source_embed_filepath = f"{args.data_folder_path}/{args.source_name}/{args.source_name}.embed"
        target_embed_filepath = f"{args.data_folder_path}/{args.target_name}/{args.target_name}.embed"

        source_item_embed_dict, target_item_embed_dict = get_item_embedding_data(source_embed_filepath, target_embed_filepath)
        source_item_features, target_item_features = item_embed_dict_to_vectors(
            source_item_embed_dict,
            target_item_embed_dict,
            data_lib["source_item_dict"],
            data_lib["target_item_dict"],
            n_source_items,
            n_target_items,
        )


    # Load model and trainer
    if args.model == "CMF":
        model = CMF(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=64,
        )
        trainer = CMFTrainer(model)
    elif args.model == "NCF":
        model = NCF(
            n_users=n_users,
            source_item_features=source_item_features,
            target_item_features=target_item_features,
            user_embedding_dim=64,
            mlp_layers=args.layers,
            dropout=0.2,
            freeze_item_features=True,
        )
        trainer = NCFTrainer(model)

    trainer.train(
        train_source_loader,
        train_target_loader,
        valid_source_loader,
        valid_target_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-5,
        report_every=1,
    )

    model_loss_type = model.get_model_info().get("loss_type")
    test_target_dataset = Subset(target_interactions, test_target_loader.dataset.indices).dataset.tensors

    target_loss = trainer.evaluate(test_target_loader)
    target_metrics = trainer.calculate_metrics(test_target_dataset[0],   # target users
                                              test_target_dataset[1],   # target items
                                              test_target_dataset[2])   # target ratings

    target_metrics[f"{model_loss_type}"] = target_loss
    
    print(f"\n{args.model} Results:")
    print(f"Target {model_loss_type}: {target_loss:.4f}")
    print("Target Accuracy: {:.4f}, Target Precision: {:.4f}, Target Recall: {:.4f}, Target F1: {:.4f}".format(
        target_metrics["acc"], target_metrics["precision"], target_metrics["recall"], target_metrics["f1"]))

    log_train_results(data_lib, model, trainer, target_metrics)