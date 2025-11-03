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
        n_items: int,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        self._reset_parameters()

    def get_model_info(self):
        return {
            "model_type": "CMF",
            "loss_type": "MSE",
            "n_users": self.user_emb.num_embeddings,
            "n_items": self.item_emb.num_embeddings,
            "embedding_dim": self.user_emb.embedding_dim,
        }

    def _reset_parameters(self):
        nn.init.normal_(self.user_emb.weight, 0, 0.1)
        nn.init.normal_(self.item_emb.weight, 0, 0.1)

    def predict(self, users: torch.LongTensor, items: torch.LongTensor):
        """
        Predict ratings for a batch given domain.
        domain: "source" or "target"
        """
        user_embedding = self.user_emb(users)
        item_embedding = self.item_emb(items)

        dot = torch.mul(user_embedding, item_embedding).sum(dim=-1)
        return dot # shape (batch,)

    def forward(self, users: torch.LongTensor, items: torch.LongTensor):
        return self.predict(users, items)
    
    def __str__(self):
        info = self.get_model_info()

        return "\n".join([f"{key}: {value}" for key, value in info.items()])

class CMFTrainer():
    def __init__(self, model: CMF):
        self.model = model

    def train(
        self,
        train_source_loader: DataLoader,
        train_target_loader: DataLoader,
        valid_source_loader: DataLoader,
        valid_target_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-3,
        alpha: float = 0.2,
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

        self.scores = []

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        for ep in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            n_batches = 0

            # Train on source domain
            for users, items, ratings, _ in train_source_loader:
                users = users.long()
                items = items.long()
                ratings = ratings.float()

                preds = model(users, items)
                loss = alpha * criterion(preds, ratings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            # Train on target domain
            for users, items, ratings, _ in train_target_loader:
                users = users.long()
                items = items.long()
                ratings = ratings.float()

                preds = model(users, items)
                loss = (1 - alpha) * criterion(preds, ratings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if ep % report_every == 0:
                avg_loss = total_loss / max(1, n_batches)
                print(f"Epoch {ep}/{epochs} — avg MSE: {avg_loss:.6f}")

            # Validation step
            if valid_source_loader is not None and valid_target_loader is not None:
                for users, items, ratings, _ in valid_source_loader:
                    users = users.long()
                    items = items.long()
                    ratings = ratings.float()

                    source_metrics = self.calculate_metrics(users, items, ratings)
                    break
                
                for users, items, ratings, _ in valid_target_loader:
                    users = users.long()
                    items = items.long()
                    ratings = ratings.float()

                    target_metrics = self.calculate_metrics(users, items, ratings)
                    break
                
                self.scores.append((avg_loss, source_metrics, target_metrics))

        return model


    def evaluate(self, loader: DataLoader):
        """
        Return RMSE on provided loader (domain indicates which item embeddings to use).
        """
        model = self.model
        model.eval()
        se = 0.0
        n = 0
        with torch.no_grad():
            for users, items, ratings, _ in loader:
                users = users.long()
                items = items.long()
                ratings = ratings.float()
                preds = model(users, items)
                se += ((preds - ratings) ** 2).sum().item()
                n += ratings.numel()
        rmse = math.sqrt(se / n) if n > 0 else float("nan")

        return rmse
    
    def calculate_metrics(self, users, items, ratings):
        users = users.long()
        items = items.long()
        ratings = ratings.float()

        y_preds = self.model.predict(users.long(), items.long())
        bin_preds = (y_preds >= 3).long().tolist()
        bin_ratings = (ratings >= 3).long().tolist()

        acc = accuracy_score(bin_ratings, bin_preds)
        conf = precision_recall_fscore_support(bin_ratings, bin_preds, average='binary', zero_division=0.0)

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