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
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self._reset_parameters()

    def get_model_info(self):
        return {
            "model_type": "CMF",
            "loss_type": "BCE",
            "n_users": self.user_emb.num_embeddings,
            "n_items": self.item_emb.num_embeddings,
            "embedding_dim": self.user_emb.embedding_dim,
        }

    def _reset_parameters(self):
        nn.init.normal_(self.user_emb.weight, 0, 0.1)
        nn.init.normal_(self.item_emb.weight, 0, 0.1)

    def predict(self, users: torch.LongTensor, items: torch.LongTensor):
        user_embedding = self.user_emb(users)
        item_embedding = self.item_emb(items)

        return self.sigmoid(torch.mul(user_embedding, item_embedding).sum(dim=-1)) # shape (batch,)

    def forward(self, users: torch.LongTensor, items: torch.LongTensor):
        return self.predict(users, items)
    
    def calculate_loss(self, users: torch.LongTensor, items: torch.LongTensor, clicks: torch.FloatTensor):
        preds = self.forward(users, items)
        loss = self.loss(preds, clicks)

        return loss

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
        self.alpha = alpha
        self.weight_decay = weight_decay

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for ep in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            n_batches = 0

            # Train on source domain
            for users, items, _, clicks in source_loader:
                users = users.long()
                items = items.long()
                clicks = clicks.float()

                loss = alpha * model.calculate_loss(users, items, clicks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            # Train on target domain
            for users, items, _, clicks in target_loader:
                users = users.long()
                items = items.long()
                clicks = clicks.float()

                loss = (1 - alpha) * model.calculate_loss(users, items, clicks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if ep % report_every == 0:
                avg_bce = total_loss / max(1, n_batches)
                print(f"Epoch {ep}/{epochs} — BCE: {total_loss:.6f} | avg BCE: {avg_bce:.6f}")

        return model


    def evaluate(self, loader: DataLoader):
        """
        Return BCE Loss on provided loader.
        """
        model = self.model
        model.eval()
        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for users, items, _, clicks in loader:
                users = users.long()
                items = items.long()
                clicks = clicks.float()

                loss = model.calculate_loss(users, items, clicks)

                total_loss += loss.item()
                n += 1
                
        bce = math.sqrt(total_loss / n) if n > 0 else float("nan")

        return bce
    
    def calculate_metrics(self, eval_dataset):
        target_users = eval_dataset[0]
        target_items = eval_dataset[1]
        clicks = eval_dataset[3].numpy()

        preds = self.model.predict(target_users.long(), target_items.long())
        bin_preds = preds.detach().numpy() >= 0.5

        acc = accuracy_score(clicks, bin_preds)
        conf = precision_recall_fscore_support(clicks, bin_preds, average='binary')

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