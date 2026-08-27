"""
main_stair_lia_v3.py -- STAIR-LIA v3 (Local Interest Aligned)
=============================================================
Kien truc: SVD/ZCA Whitening + Offline Bidirectional ROI Attention (CLID-style)
           + Lightweight Fusion + Denoised kNN Graph on E_fused
           + BPR Loss (+ optional ROI InfoNCE)

Chien luoc (Plan.md):
  Phase 1: ZCA Whitening (preprocess_stair_lia.py da chay offline)
  Phase 2: Offline ROI Extraction (preprocess_stair_lia.py da chay offline)
  Phase 3: Load E_fused tu disk, xay dung Denoised kNN Graph tren E_fused,
           Khoi tao MI cho STAIR, train voi BPR + optional roi_cl loss.
"""
from typing import Dict, Tuple
import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import freerec

from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

freerec.declare(version='1.0.1')

# ============================================================================
# Config
# ============================================================================
cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim",  type=int,   default=64)
cfg.add_argument("--num-layers",     type=int,   default=3)
cfg.add_argument("--mfiles",         type=str,   default="textual_modality.pkl,visual_modality.pkl")
cfg.add_argument("--num-neighbors",  type=str,   default='5-1')
cfg.add_argument("--gamma",          type=float, default=0.2)

# LIA-specific
cfg.add_argument("--lia-precomputed-dir", type=str, default="preprocessed_lia",
                 help="Thu muc chua E_fused_t.pt / E_fused_v.pt (ket qua preprocess_stair_lia.py)")
cfg.add_argument("--lia-alpha",      type=float, default=0.5,
                 help="Weight for text vs visual in modal fusion: E_modal = alpha*t + (1-alpha)*v")
cfg.add_argument("--lia-roi-cl",     type=float, default=0.0,
                 help="Weight cho ROI Contrastive loss (0.0 = disable, 0.01 = enable nhe)")
cfg.add_argument("--lia-temperature",type=float, default=0.07,
                 help="Temperature cho InfoNCE ROI contrastive loss")

cfg.set_defaults(
    description="STAIR-LIA-v3",
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
cfg.mfiles        = cfg.mfiles.split(',')
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-')))

# BSC Smoother spectral decay (dong nhat voi STAIR goc)
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ============================================================================
# ROI Contrastive Loss (InfoNCE) -- optional
# ============================================================================
def roi_contrastive_loss(
    roi_t: torch.Tensor,
    roi_v: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE Cross-Modal Contrastive Loss tren batch.
    Matching CLID.py multi_loss():
        logits = roi_v @ roi_t^T / temperature
        loss   = CE(logits, diag) + CE(logits^T, diag)
    Args:
        roi_t: (B, D) - normalized text ROI embeddings
        roi_v: (B, D) - normalized visual ROI embeddings
    Returns:
        scalar loss
    """
    roi_t = F.normalize(roi_t, p=2, dim=-1)
    roi_v = F.normalize(roi_v, p=2, dim=-1)
    logits = roi_v @ roi_t.t() / temperature       # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_v2t = F.cross_entropy(logits,   labels)
    loss_t2v = F.cross_entropy(logits.t(), labels)
    return (loss_v2t + loss_t2v) / 2.0


# ============================================================================
# Model: STAIR-LIA v3
# ============================================================================
class STAIR_LIA_v3(freerec.models.GenRecArch):
    """
    STAIR-LIA v3 (Local Interest Aligned).

    Khac voi v2a:
      1. KHONG co ResidualProjector trainable -> KHONG co gradient xung dot voi BPR.
      2. Dung E_fused = load tu disk (da qua ZCA + Offline ROI Attention).
      3. Denoised kNN Graph duoc tinh tren E_fused (sach nhieu hon raw features).
      4. E_modal = alpha * E_fused_t + (1-alpha) * E_fused_v  (Lightweight Fusion).
      5. Optional: ROI Contrastive loss voi weight rat nho (0.01).
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)
        self.num_layers = cfg.num_layers

        self.User.add_module('embeddings', nn.Embedding(self.User.count, cfg.embedding_dim))
        self.Item.add_module('embeddings', nn.Embedding(self.Item.count, cfg.embedding_dim))

        self.register_buffer(
            'Adj',
            self.dataset.train().to_normalized_adj(normalization='sym')
        )

        # alpha: learnable scalar de balance text/visual (init = cfg.lia_alpha)
        self.alpha = nn.Parameter(torch.tensor(cfg.lia_alpha))

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)

    def marked_params(self):
        return [
            {'params': self.User.parameters(), 'smoother': None},
            {
                'params': self.Item.parameters(),
                'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann'),
            },
            {'params': [self.alpha], 'smoother': None, 'weight_decay': 0.0},
        ]

    # ----------------------------------------------------------------
    # kNN Graph helpers
    # ----------------------------------------------------------------
    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        """Tinh kNN graph tu cosine similarity."""
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        return edge_index

    def build_denoised_knn_graph(self, E_fused: torch.Tensor, num_neighbors_list) -> None:
        """
        Xay dung Denoised kNN Graph tren E_fused (vector sach nhieu).
        Thay the toan bo kNN graph goc cua STAIR (duoc xay tren raw 4096-D/384-D).
        """
        print("[LIA] Building Denoised kNN Graph on E_fused...")
        # Dung cung E_fused cho ca text va visual (da fuse roi)
        # k = sum(num_neighbors_list) = 6 (5+1)
        k_total = sum(num_neighbors_list)
        edge_index = self.get_knn_graph(E_fused, k=k_total)
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce='sum')
        edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce='max')
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, edge_weight, normalization='sym')
        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.Item.count, self.Item.count)
        )
        self.register_buffer('mAdj', mAdj.to_sparse_csr())
        print(f"[LIA] Denoised kNN Graph built: {edge_index.shape[1]} edges, k={k_total}")

    # ----------------------------------------------------------------
    # Prepare: load precomputed LIA features
    # ----------------------------------------------------------------
    def whitening(self, feats: torch.Tensor) -> torch.Tensor:
        """SVD Whitening fallback (dung khi khong co precomputed features)."""
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def prepare(self, path: str):
        """
        Load E_fused tu disk (precomputed boi preprocess_stair_lia.py),
        xay dung Denoised kNN Graph, khoi tao MI.

        Neu khong tim thay E_fused (chua chay preprocessing), fallback ve SVD whitening.
        """
        from freerec.utils import import_pickle

        # --- Thu load precomputed LIA features ---
        lia_dir = os.path.join(path, cfg.lia_precomputed_dir, cfg.dataset)
        fused_t_path = os.path.join(lia_dir, "E_fused_t.pt")
        fused_v_path = os.path.join(lia_dir, "E_fused_v.pt")

        if os.path.exists(fused_t_path) and os.path.exists(fused_v_path):
            print(f"[LIA] Loading precomputed E_fused from: {lia_dir}")
            E_fused_t = torch.load(fused_t_path, map_location='cpu').float()  # (N, 64)
            E_fused_v = torch.load(fused_v_path, map_location='cpu').float()  # (N, 64)
            # Fallback: co the load roi (roi da fuse vao E_fused roi)
            roi_t_path = os.path.join(lia_dir, "E_roi_t.pt")
            roi_v_path = os.path.join(lia_dir, "E_roi_v.pt")
            if os.path.exists(roi_t_path):
                E_roi_t = torch.load(roi_t_path, map_location='cpu').float()
                E_roi_v = torch.load(roi_v_path, map_location='cpu').float()
                self.register_buffer('E_roi_t', E_roi_t.to(cfg.device))
                self.register_buffer('E_roi_v', E_roi_v.to(cfg.device))
                print(f"[LIA] E_roi loaded: t={tuple(E_roi_t.shape)}, v={tuple(E_roi_v.shape)}")
            else:
                self.E_roi_t = None
                self.E_roi_v = None
            use_lia = True
        else:
            # Fallback: dung SVD Whitening nhu STAIR goc
            print(f"[LIA] WARNING: Precomputed features not found at {lia_dir}")
            print("[LIA] Fallback to SVD Whitening (original STAIR)...")
            mfeats_raw = [
                import_pickle(os.path.join(path, mfile)) for mfile in cfg.mfiles
            ]
            text_feat   = mfeats_raw[0].float()
            visual_feat = mfeats_raw[1].float()
            E_fused_t = self.whitening(text_feat)
            E_fused_v = self.whitening(visual_feat)
            self.E_roi_t = None
            self.E_roi_v = None
            use_lia = False

        print(f"[LIA] E_fused_t: {tuple(E_fused_t.shape)}, mean_norm={E_fused_t.norm(dim=-1).mean():.3f}")
        print(f"[LIA] E_fused_v: {tuple(E_fused_v.shape)}, mean_norm={E_fused_v.norm(dim=-1).mean():.3f}")

        # --- Xay dung Denoised kNN Graph tren E_fused ---
        # E_modal_for_graph: weighted combination cho graph construction
        E_for_graph = 0.5 * E_fused_t + 0.5 * E_fused_v
        self.build_denoised_knn_graph(E_for_graph, cfg.num_neighbors)

        # --- Luu E_fused lam frozen buffer ---
        self.register_buffer('E_fused_t', E_fused_t.to(cfg.device))
        self.register_buffer('E_fused_v', E_fused_v.to(cfg.device))

        # --- Khoi tao User theo R @ E_modal ---
        alpha_init = float(cfg.lia_alpha)
        E_modal_init = alpha_init * E_fused_t + (1 - alpha_init) * E_fused_v

        edge_index_ui = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index_ui, edge_weight_ui = freerec.graph.to_normalized(
            edge_index_ui, normalization='left'
        )
        R = torch.sparse_coo_tensor(
            edge_index_ui, edge_weight_ui,
            size=(self.User.count, self.Item.count)
        ).to_sparse_csr().to(cfg.device)
        self.User.embeddings.weight.data.copy_(R @ E_modal_init.to(cfg.device))

        # Item ID embeddings = 0 (e_item = e_ID + e_modal, e_modal = E_fused)
        self.Item.embeddings.weight.data.zero_()
        print(f"[LIA] Initialized: alpha={alpha_init:.2f}, use_lia={use_lia}")
        print("[LIA] Ready for training!")

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------
    def get_modal_item_embeddings(self) -> torch.Tensor:
        """
        Lightweight Fusion:
            E_modal = alpha * E_fused_t + (1-alpha) * E_fused_v

        alpha la nn.Parameter (scalar, init=0.5), duoc hoc nhe trong qua trinh train.
        E_fused_t, E_fused_v la frozen buffers (khong thay doi trong train).
        """
        alpha = torch.sigmoid(self.alpha)          # (0, 1) bounded
        e_modal = alpha * self.E_fused_t + (1 - alpha) * self.E_fused_v  # (N, 64)
        return e_modal

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FSC/BSC smoothing (dong nhat voi STAIR goc):
            item_total = Item.embeddings + E_modal
        """
        modal_items = self.get_modal_item_embeddings()
        total_items = self.Item.embeddings.weight + modal_items

        allEmbds = torch.cat((self.User.embeddings.weight, total_items), dim=0)
        features = allEmbds
        smoothed = allEmbds
        beta     = 1 - cfg.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(avgEmbds, (self.User.count, self.Item.count))
        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        """BPR + optional ROI Contrastive loss."""
        userEmbds, itemEmbds = self.encode()
        users     = data[self.User]
        positives = data[self.Item]
        negatives = data[self.INeg]

        bpr_loss = self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[positives]),
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[negatives]),
        )

        # Optional ROI Contrastive Loss (Plan.md: "only CFD or InfoNCE with 0.01 weight")
        if cfg.lia_roi_cl > 0.0 and self.E_roi_t is not None:
            pos_items_flat = positives.view(-1)
            roi_t_batch = self.E_roi_t[pos_items_flat]
            roi_v_batch = self.E_roi_v[pos_items_flat]
            cl_loss = roi_contrastive_loss(roi_t_batch, roi_v_batch, cfg.lia_temperature)
            return bpr_loss + cfg.lia_roi_cl * cl_loss
        return bpr_loss

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum('BKD,ND->BN', userEmbds, itemEmbds)

    def recommend_from_pool(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum('BKD,BKD->BK', userEmbds, itemEmbds)


# ============================================================================
# Coach
# ============================================================================
class CoachForSTAIR_LIA_v3(freerec.launcher.Coach):
    """Coach cho STAIR-LIA v3: theo doi alpha va loss."""

    def set_optimizer(self):
        self.optimizer = AdamWSEvo(
            self.model.marked_params(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )
        print(f"[Optimizer] AdamWSEvo | lr={self.cfg.lr} | wd={self.cfg.weight_decay}")

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.monitor(
                loss.item(), n=len(data[self.User]),
                reduction='mean', mode='train', pool=['LOSS'],
            )
        alpha_val = torch.sigmoid(self.model.alpha).item()
        print(f"  [alpha @epoch {epoch:3d}]: {alpha_val:.4f} (text weight)", flush=True)


# ============================================================================
# Entry Point
# ============================================================================
def main():
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = STAIR_LIA_v3(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_LIA_v3(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == '__main__':
    main()
