"""V8.1 NLGCL query override with unchanged keys and denominators."""

import numbers

import torch

from .stair5_v4_objectives import NLGCL_Module


class NLGCLConsensus_Module(NLGCL_Module):
    """Replace only first-hop item queries; preserve V8 InfoNCE orientation.

    item_query must already be expanded to sampled batch order, including
    duplicate item IDs. No ranking/FSC tensor or caller-owned list is mutated.
    """

    def __init__(self, n_users, n_items, G=1, tau=0.2, alpha=0.5, chunk_size=None):
        if G != 1 or isinstance(G, bool) or not isinstance(G, numbers.Integral):
            raise ValueError("The V8.1 query override requires G=1.")
        super().__init__(n_users, n_items, G, tau, alpha, chunk_size)

    def forward(self, layer_embeds, users, pos_items, item_query=None):
        if item_query is None:
            return super().forward(layer_embeds, users, pos_items)
        users, pos_items = users.view(-1), pos_items.view(-1)
        if users.numel() == 0 or users.numel() != pos_items.numel():
            raise ValueError("NLGCL requires equally sized, nonempty sampled pairs.")
        if len(layer_embeds) < 2:
            raise ValueError("A consensus query requires FSC layers zero and one.")
        h0, h1 = layer_embeds[:2]
        if h0.ndim != 2 or h0.shape != h1.shape or h0.shape[0] != self.n_users + self.n_items:
            raise ValueError("FSC layers must have matching [N_users+N_items, d] shapes.")
        if item_query.ndim != 2 or item_query.shape != (users.numel(), h0.shape[1]):
            raise ValueError("item_query must have shape [sampled_batch_size, d].")
        if item_query.device != h0.device or item_query.dtype != h0.dtype:
            raise ValueError("item_query must match the FSC device and dtype.")
        user0, item0 = torch.split(h0, [self.n_users, self.n_items])
        user1, _ = torch.split(h1, [self.n_users, self.n_items])
        # Bind one key tensor and pass that exact object twice, enabling the
        # baseline helper's normalized-key reuse, including checkpoint chunks.
        user_keys, item_keys = user0[users], item0[pos_items]
        user_loss = self.info_nce_in_batch(item_query, user_keys, user_keys)
        item_loss = self.info_nce_in_batch(user1[users], item_keys, item_keys)
        return self.alpha * user_loss + (1.0 - self.alpha) * item_loss
