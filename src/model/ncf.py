import math
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Neural collaborative filtering (NCF) for cross-domain recommendation
# Uses precomputed item feature vectors for source / target domains

class NCF(nn.Module):
    """
    Neural Collaborative Filtering that concatenates a learned user embedding with provided
    item feature vectors (from source/target) and passes them through an MLP to predict rating.
    """
    def __init__(
        self,
        n_users: int,
        source_item_features,  # numpy array or torch.Tensor (n_source_items, feat_dim)
        target_item_features,  # numpy array or torch.Tensor (n_target_items, feat_dim)
        user_embedding_dim: int = 64,
        mlp_layers: list = (128, 64, 32),
        dropout: float = 0.2,
        freeze_item_features: bool = True,
    ):
        super().__init__()
        # user embedding
        self.user_embedding = nn.Embedding(n_users, user_embedding_dim)

        # item features: convert to tensors and make Embedding-like lookup via from_pretrained
        source_feat = torch.as_tensor(source_item_features, dtype=torch.float32)
        target_feat = torch.as_tensor(target_item_features, dtype=torch.float32)
        self.source_feat_dim = source_feat.shape[1]
        self.target_feat_dim = target_feat.shape[1]

        self.source_item_feature_embedding = nn.Embedding.from_pretrained(source_feat, freeze=freeze_item_features)
        self.target_item_feature_embedding = nn.Embedding.from_pretrained(target_feat, freeze=freeze_item_features)

        # MLP that takes concat(user_embedding, item_feat) -> rating scalar
        input_dim_src = user_embedding_dim + self.source_feat_dim
        input_dim_tgt = user_embedding_dim + self.target_feat_dim

        def build_mlp(input_dim):
            layers = []
            prev_dim = input_dim

            for hidden_dim in mlp_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))
            return nn.Sequential(*layers)

        self.source_mlp = build_mlp(input_dim_src)
        # reuse architecture for target but with target input dim
        self.target_mlp = build_mlp(input_dim_tgt)

        self.loss = nn.MSELoss()

        self._reset_parameters()
    
    def get_model_info(self):
        return {
            "model_type": "NCF",
            "loss_type": "MSE",
            "n_users": self.user_embedding.num_embeddings,
            "user_embedding_dim": self.user_embedding.embedding_dim,
            "source_feat_dim": self.source_feat_dim,
            "target_feat_dim": self.target_feat_dim,
            "source_mlp_layers": [layer.out_features for layer in self.source_mlp if isinstance(layer, nn.Linear)],
            "target_mlp_layers": [layer.out_features for layer in self.target_mlp if isinstance(layer, nn.Linear)],
        }

    def _reset_parameters(self):
        nn.init.normal_(self.user_embedding.weight, 0, 0.1)

        # pretrained item features already set; MLP init
        for m in self.source_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.target_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, users: torch.LongTensor, items: torch.LongTensor, domain: str):
        """
        users: (batch,)
        items: (batch,) -- item indices local to domain (0..n_items_domain-1)
        domain: "source" or "target"
        returns: (batch,) predicted scalar ratings
        """
        u = self.user_embedding(users)  # (batch, user_embedding_dim)

        if domain == "source":
            i_feat = self.source_item_feature_embedding(items)  # (batch, source_feat_dim)
            x = torch.cat([u, i_feat], dim=-1)
            out = self.source_mlp(x).squeeze(-1)

        elif domain == "target":
            i_feat = self.target_item_feature_embedding(items)  # (batch, target_feat_dim)
            x = torch.cat([u, i_feat], dim=-1)
            out = self.target_mlp(x).squeeze(-1)

        else:
            raise ValueError("domain must be 'source' or 'target'")
        
        return out
    
    def calculate_loss(self, users: torch.LongTensor, items: torch.LongTensor, ratings: torch.FloatTensor, domain: str):
        preds = self.forward(users, items, domain)
        loss = self.loss(preds, ratings)

        return loss
    
    def __str__(self):
        info = self.get_model_info()

        return "\n".join([f"{key}: {value}" for key, value in info.items()])

class NCFTrainer():
    def __init__(self, model: NCF):
        self.model = model

    def train(
            self,
            source_loader,
            target_loader,
            epochs: int = 10,
            lr: float = 1e-3,
            alpha: float = None,
            weight_decay: float = 0.0,
            report_every: int = 1,
        ):
        model = self.model
        self.epochs = epochs
        self.learning_rate = lr
        self.weight_decay = weight_decay

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        for ep in range(1, epochs + 1):
            model.train()
            total_source_loss = 0.0
            total_target_loss = 0.0
            n_source_batches = 0
            n_target_batches = 0

            for users, items, ratings, _ in source_loader:
                users = users.long()
                items = items.long()
                ratings = ratings.float()

                loss = model.calculate_loss(users, items, ratings, domain="source")
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_source_loss += loss.item()
                n_source_batches += 1

            for users, items, ratings, _ in target_loader:
                users = users.long()
                items = items.long()
                ratings = ratings.float()

                loss = model.calculate_loss(users, items, ratings, domain="target")
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_target_loss += loss.item()
                n_target_batches += 1

            if ep % report_every == 0:
                print(f"Epoch {ep}/{epochs} source MSE: {total_source_loss / max(1, n_source_batches):.6f} | target MSE: {total_target_loss / max(1, n_target_batches):.6f}")

        return model

    def evaluate(self, loader, domain="target"):
        model = self.model
        model.eval()
        mse = 0.0
        n = 0
        with torch.no_grad():
            for users, items, ratings, _ in loader:
                users = users.long()
                items = items.long()
                ratings = ratings.float()
                mse += model.calculate_loss(users, items, ratings, domain=domain).item() * ratings.numel()
                n += ratings.numel()
        return mse / max(1, n)
    
    def calculate_metrics(self, eval_dataset):
        target_users = eval_dataset[0]
        target_items = eval_dataset[1]
        ratings = eval_dataset[2].numpy()

        preds = self.model.forward(target_users.long(), target_items.long(), 'target')
        bin_preds = preds.detach().numpy() >= 3
        bin_ratings = ratings >= 3

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