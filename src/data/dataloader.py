import torch
import numpy as np
from torch.utils.data import TensorDataset

def get_interaction_data(source_inter_path, target_inter_path):
    with open(source_inter_path, "r", encoding="utf-8") as f:
        source_dataset = [line.split("\t") for line in f.read().split("\n")][1:-1]

    with open(target_inter_path, "r", encoding="utf-8") as f:
        target_dataset = [line.split("\t") for line in f.read().split("\n")][1:-1]

    return source_dataset, target_dataset

def build_data_lib(source_dataset, target_dataset, train_test_split=[0.8, 0.2]):
    source_users = list(set([row[0] for row in source_dataset]))
    source_items = list(set([row[1] for row in source_dataset]))

    target_users = list(set([row[0] for row in target_dataset]))
    target_items = list(set([row[1] for row in target_dataset]))

    all_users = list(set(source_users + target_users))
    all_items = list(set(source_items + target_items))

    n_users = len(all_users)
    n_source_items = len(source_items)
    n_target_items = len(target_items)
    n_items = len(all_items)

    user_dict = {user: idx for idx, user in enumerate(all_users)}
    source_item_dict = {item: idx for idx, item in enumerate(source_items)}
    target_item_dict = {item: idx for idx, item in enumerate(target_items)}
    item_dict = {item: idx for idx, item in enumerate(all_items)}

    # raw dataset to torch dataset with integrated user ids but separate item ids

    tensor_source_users = torch.Tensor(np.array([user_dict[row[0]] for row in source_dataset]))
    tensor_source_items = torch.Tensor(np.array([source_item_dict[row[1]] for row in source_dataset]))
    tensor_source_ratings = torch.Tensor(np.array([float(row[2]) for row in source_dataset]))
    tensor_source_clicks = torch.Tensor(np.array([1.0 if float(row[2]) >= 3 else 0.0 for row in source_dataset]))    

    tensor_target_users = torch.Tensor(np.array([user_dict[row[0]] for row in target_dataset]))
    tensor_target_items = torch.Tensor(np.array([target_item_dict[row[1]] for row in target_dataset]))
    tensor_target_ratings = torch.Tensor(np.array([float(row[2]) for row in target_dataset]))
    tensor_target_clicks = torch.Tensor(np.array([1.0 if float(row[2]) >= 3 else 0.0 for row in target_dataset]))

    source_dataset_torch = TensorDataset(tensor_source_users, tensor_source_items, tensor_source_ratings, tensor_source_clicks)
    target_dataset_torch = TensorDataset(tensor_target_users, tensor_target_items, tensor_target_ratings, tensor_target_clicks)

    source_train_dataset, source_test_dataset = torch.utils.data.random_split(source_dataset_torch, train_test_split)
    target_train_dataset, target_test_dataset = torch.utils.data.random_split(target_dataset_torch, train_test_split)

    data_lib = {
        "n_users": n_users,
        "n_source_items": n_source_items,
        "n_target_items": n_target_items,
        "n_items": n_items,

        "user_dict": user_dict,
        "source_item_dict": source_item_dict,
        "target_item_dict": target_item_dict,
        "item_dict": item_dict,

        "source_interactions": source_dataset_torch,
        "target_interactions": target_dataset_torch,

        "train_source_interactions": source_train_dataset,
        "test_source_interactions": source_test_dataset,    
        "train_target_interactions": target_train_dataset,
        "test_target_interactions": target_test_dataset,
    }

    return data_lib

def get_item_embedding_data(source_item_emb_path, target_item_emb_path):
    source_item_embed_dict = {}
    with open(source_item_emb_path, "r", encoding="utf-8") as f:
        for line in f.read().split("\n")[1:-1]:
            parts = line.split("\t")
            item_id = parts[0]
            embed_vector = [float(x) for x in parts[1].replace("[", "").replace("]", "").split(",")]
            source_item_embed_dict[item_id] = embed_vector

    target_item_embed_dict = {}
    with open(target_item_emb_path, "r", encoding="utf-8") as f:
        for line in f.read().split("\n")[1:-1]:
            parts = line.split("\t")
            item_id = parts[0]
            embed_vector = [float(x) for x in parts[1].replace("[", "").replace("]", "").split(",")]
            target_item_embed_dict[item_id] = embed_vector

    return source_item_embed_dict, target_item_embed_dict

def item_embed_dict_to_vectors(source_item_embed_dict, target_item_embed_dict,
                                source_item_dict, target_item_dict,
                                n_source_items, n_target_items):
    reverse_source_item_dict = {source_item_dict[item]: item for item in source_item_dict}
    reverse_target_item_dict = {target_item_dict[item]: item for item in target_item_dict}

    source_item_feature_vectors = np.array([source_item_embed_dict[reverse_source_item_dict[idx]] for idx in range(n_source_items)])
    target_item_feature_vectors = np.array([target_item_embed_dict[reverse_target_item_dict[idx]] for idx in range(n_target_items)])

    return source_item_feature_vectors, target_item_feature_vectors