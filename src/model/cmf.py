import math
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class CMF(nn.Module):
    """
    Collective Matrix Factorization for two domains sharing user embeddings.

    Assumes datasets yield (user_idx, item_idx, rating) with 0-based integer indices.
    """
    def __init__(
        self,
        n_users: int,
        n_items_source: int,
        n_items_target: int,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb_source = nn.Embedding(n_items_source, embedding_dim)
        self.item_emb_target = nn.Embedding(n_items_target, embedding_dim)

        self._reset_parameters()

    def get_model_info(self):
        return {
            "model_type": "CMF",
            "n_users": self.user_emb.num_embeddings,
            "n_items_source": self.item_emb_source.num_embeddings,
            "n_items_target": self.item_emb_target.num_embeddings,
            "embedding_dim": self.user_emb.embedding_dim,
        }

    def _reset_parameters(self):
        nn.init.normal_(self.user_emb.weight, 0, 0.1)
        nn.init.normal_(self.item_emb_source.weight, 0, 0.1)
        nn.init.normal_(self.item_emb_target.weight, 0, 0.1)

    def predict(self, users: torch.LongTensor, items: torch.LongTensor, domain: Literal["source", "target"]):
        """
        Predict ratings for a batch given domain.
        domain: "source" or "target"
        """
        user_embedding = self.user_emb(users)
        if domain == "source":
            item_embedding = self.item_emb_source(items)
        elif domain == "target":
            item_embedding = self.item_emb_target(items)
        else:
            raise ValueError("domain must be 'source' or 'target'")

        dot = torch.mul(user_embedding, item_embedding).sum(dim=-1)
        return dot # shape (batch,)

    def forward(self, users: torch.LongTensor, items: torch.LongTensor, domain: Literal["source", "target"]):
        return self.predict(users, items, domain)
    
    def __str__(self):
        info = self.get_model_info()

        return "\n".join([f"{key}: {value}" for key, value in info.items()])

class CMFTrainer():
    def __init__(self, model: CMF):
        self.model = model

    def train(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        report_every: int = 1,
    ):
        """
        Jointly train on source and target loaders. Each loader yields (user, item, rating).
        """
        model = self.model
        self.epochs = epochs
        self.learning_rate = lr
        self.weight_decay = weight_decay

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        for ep in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            n_batches = 0

            # Train on source domain
            for users, items, ratings in source_loader:
                users = users.long()
                items = items.long()
                ratings = ratings.float()

                preds = model(users, items, domain="source")
                loss = criterion(preds, ratings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            # Train on target domain
            for users, items, ratings in target_loader:
                users = users.long()
                items = items.long()
                ratings = ratings.float()

                preds = model(users, items, domain="target")
                loss = criterion(preds, ratings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if ep % report_every == 0:
                avg_loss = total_loss / max(1, n_batches)
                print(f"Epoch {ep}/{epochs} — avg MSE: {avg_loss:.6f}")

        return model


    def evaluate(self, loader: DataLoader, domain: Literal["source", "target"]):
        """
        Return RMSE on provided loader (domain indicates which item embeddings to use).
        """
        model = self.model
        model.eval()
        se = 0.0
        n = 0
        with torch.no_grad():
            for users, items, ratings in loader:
                users = users.long()
                items = items.long()
                ratings = ratings.float()
                preds = model(users, items, domain=domain)
                se += ((preds - ratings) ** 2).sum().item()
                n += ratings.numel()
        rmse = math.sqrt(se / n) if n > 0 else float("nan")

        return rmse
    
    def calculate_metrics(self, target_users, target_items, ratings):
        y_preds = self.model.predict(target_users.long(), target_items.long(), 'target')
        bin_preds = [1 if e >= 3 else 0 for e in y_preds]
        bin_ratings = [1 if e >= 3 else 0 for e in ratings]

        acc = accuracy_score(bin_ratings, bin_preds)
        conf = precision_recall_fscore_support(bin_ratings, bin_preds, average='binary')

        return {"acc": acc, "precision": conf[0], "recall": conf[1], "f1": conf[2]}

    def get_trainer_info(self):
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
    
    def __str__(self):
        info = self.get_trainer_info()

        return "\n".join([f"{key}: {value}" for key, value in info.items()])