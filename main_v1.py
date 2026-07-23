"""
main_v1.py — STAIR + Stepwise Graph Contrastive Learning (STAIR-GCL, Ý tưởng 1)
==============================================================================
Đặc điểm cải tiến:
  1. Tạo 2 views đồ thị phụ trợ bằng Edge Dropout trên ma trận kề tương tác (Adj).
  2. Hàm mất mát đối chiếu InfoNCE (Collaborative Contrastive Loss) chỉ áp dụng cho
     32 chiều đầu tiên (Collaborative Subspace), nơi diễn ra sự lan truyền hành vi chính.
  3. Không áp dụng Contrastive Loss cho 32 chiều sau (Multimodal Subspace) để bảo tồn
     trọn vẹn đặc trưng đa phương thức gốc từ FSC mà không làm oversmoothing.
  4. Trọng số tổng hợp: Loss = Rec_Loss (BPR) + cl_weight * (CL_user + CL_item).
"""

from typing import Dict, Tuple, Optional
import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import freerec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3, help="the number of layers for FSC/BSC")

cfg.add_argument("--mfiles", type=str, default="textual_modality.pkl,visual_modality.pkl", help="the files saving modality")
cfg.add_argument("--num-neighbors", type=str, default='5-1', help="for kNN graph")
cfg.add_argument("--gamma", type=float, default=0.2)

# Tham số mới cho STAIR-GCL (Ý tưởng 1)
cfg.add_argument("--cl-weight", type=float, default=1e-3, help="contrastive loss weight")
cfg.add_argument("--edge-drop", type=float, default=0.2, help="edge dropout rate for graph views")
cfg.add_argument("--cl-temp", type=float, default=0.2, help="temperature for InfoNCE loss")

cfg.set_defaults(
    description="STAIR-v1 (Stepwise Graph Contrastive Learning)",
    root="../../data",
    dataset='Amazon2014Baby_550_MMRec',
    epochs=500,
    batch_size=1024,
    optimizer='adamwsevo',
    lr=1e-3,
    weight_decay=0.1,
    seed=1,
    monitors=["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
    which4best="NDCG@20",
)
cfg.compile()

cfg.mfiles = cfg.mfiles.split(',')
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-'))) 

# beta3 here is the 1 - beta_j for BSC
cfg.beta3 = (0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)).to(cfg.device)


class STAIR(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet
    ) -> None:
        super().__init__(dataset)

        self.num_layers = cfg.num_layers

        self.User.add_module(
            "embeddings", nn.Embedding(
                self.User.count, cfg.embedding_dim
            )
        )

        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count, cfg.embedding_dim
            )
        )

        self.register_buffer(
            "Adj",
            self.dataset.train().to_normalized_adj(
                normalization='sym'
            )
        )

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def marked_params(self):
        params = [
            {
                'params': self.User.parameters(),
                'smoother': None
            },
            {
                'params': self.Item.parameters(), 
                'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann')
            },
        ]
        return params

    def whitening(self, feats: torch.Tensor):
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        features = F.normalize(features, dim=-1) # (N, D)
        sim = features @ features.t() # (N, N)
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(
            sim, k, symmetric=False
        )
        return edge_index

    def prepare(self, path: str):
        from freerec.utils import import_pickle

        mfeats = [
            import_pickle(
                os.path.join(path, mfile)
            )
            for mfile in cfg.mfiles
        ]

        edge_index = torch.cat(
            [self.get_knn_graph(feats, k) for feats, k in zip(mfeats, cfg.num_neighbors)],
            dim=1
        )
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(
            edge_index, edge_weight, reduce='sum'
        )
        edge_index, edge_weight = freerec.graph.to_undirected(
            edge_index, edge_weight, reduce='max'
        )
        edge_index, edge_weight = freerec.graph.to_normalized(
            edge_index, edge_weight,
            normalization='sym'
        )
        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight,
            size=(self.Item.count, self.Item.count)
        )
        self.register_buffer(
            'mAdj',
            mAdj.to_sparse_csr()
        )

        # MI
        mfeats = [self.whitening(mfeat) * k for mfeat, k in zip(mfeats, cfg.num_neighbors)]
        mfeats = sum(mfeats).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats)

        edge_index = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, normalization='left')
        R = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.User.count, self.Item.count)
        ).to_sparse_csr()

        self.User.embeddings.weight.data.copy_(R @ mfeats)

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_pairs_source(
        ).gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def graph_dropout(self, adj: torch.Tensor, drop_rate: float) -> torch.Tensor:
        """Randomly drop edges from sparse adjacency matrix during training."""
        if drop_rate <= 0.0 or not self.training:
            return adj
        
        # Format check for sparse matrix
        if adj.is_sparse or adj.is_sparse_csr:
            adj_coo = adj.to_sparse_coo().coalesce()
            indices = adj_coo.indices()
            values = adj_coo.values()
            num_edges = values.size(0)
            
            mask = torch.rand(num_edges, device=adj.device) >= drop_rate
            new_indices = indices[:, mask]
            new_values = values[mask] / (1.0 - drop_rate)
            
            return torch.sparse_coo_tensor(
                new_indices, new_values, adj.shape
            ).coalesce().to_sparse_csr()
        return adj

    def calc_contrastive_loss(self, view1: torch.Tensor, view2: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
        """
        InfoNCE Loss applied ONLY to Collaborative Subspace (first 32 dimensions).
        Multimodal Subspace (last 32 dimensions) is untouched to preserve modal features.
        """
        collab_dim = cfg.embedding_dim // 2 # 32 channels
        v1 = F.normalize(view1[:, :collab_dim], dim=-1)
        v2 = F.normalize(view2[:, :collab_dim], dim=-1)

        pos_score = torch.exp(torch.sum(v1 * v2, dim=-1) / temp)
        all_score = torch.matmul(v1, v2.t())
        all_score = torch.exp(all_score / temp).sum(dim=-1)

        return -torch.log(pos_score / (all_score + 1e-8)).mean()

    def encode(self, adj: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if adj is None:
            adj = self.Adj

        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        ) # (N, D)

        features = allEmbds
        smoothed = allEmbds
        
        # FSC
        beta = 1 - cfg.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        userEmbds = userEmbds[users] # (B, 1, D)
        iposEmbds = itemEmbds[positives] # (B, 1, D)
        inegEmbds = itemEmbds[negatives] # (B, K, D)

        rec_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userEmbds, iposEmbds),
            torch.einsum("BKD,BKD->BK", userEmbds, inegEmbds)
        )

        # Stepwise Graph Contrastive Learning (STAIR-GCL)
        if self.training and cfg.cl_weight > 0:
            adj_v1 = self.graph_dropout(self.Adj, cfg.edge_drop)
            adj_v2 = self.graph_dropout(self.Adj, cfg.edge_drop)

            u_v1, i_v1 = self.encode(adj_v1)
            u_v2, i_v2 = self.encode(adj_v2)

            u_idx = users.view(-1)
            i_idx = positives.view(-1)

            cl_loss_u = self.calc_contrastive_loss(u_v1[u_idx], u_v2[u_idx], cfg.cl_temp)
            cl_loss_i = self.calc_contrastive_loss(i_v1[i_idx], i_v2[i_idx], cfg.cl_temp)

            cl_loss = cfg.cl_weight * (cl_loss_u + cl_loss_i)
            return rec_loss + cl_loss

        return rec_loss

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum("BKD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]] # (B, 101, D)
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)


class CoachForSTAIR(freerec.launcher.Coach):

    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.marked_params(), lr=self.cfg.lr, 
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamsevo':
            self.optimizer = AdamSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamwsevo':
            self.optimizer = AdamWSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unexpected optimizer {self.cfg.optimizer} ..."
            )

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(
                loss.item(), 
                n=len(data[self.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )


def main():

    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = STAIR(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()
