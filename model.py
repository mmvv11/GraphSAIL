import layers
import torch
import torch.nn as nn

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(
            full_graph, ntype, hidden_dims
        )

        self.user_proj = layers.LinearProjector(
            full_graph, 'user', hidden_dims
        )

        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.user_sage = layers.SAGENet(hidden_dims, n_layers)

        self.scorer = layers.ItemToItemScorer(full_graph, ntype)  # 현재는 item-item graph
        self.user_scorer = layers.UserToUserScorer(full_graph, 'user')

    def forward(self, pos_graph, neg_graph, blocks):
        # 여기서 주어지는 pos, neg graph는 item-item graph임
        # ItemToItemBatchSampler 여기서 item-item graph가 만들어지는군
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def user_forward(self, pos_graph, neg_graph, blocks):
        h_user = self.get_user_repr(blocks)
        pos_score = self.user_scorer(pos_graph, h_user)
        neg_score = self.user_scorer(neg_graph, h_user)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):  # 아이템 임베딩을 구하는 부분
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)

    def get_user_repr(self, blocks):  # 유저 임베딩을 구하는 부분
        h_user = self.user_proj(blocks[0].srcdata)
        h_user_dst = self.user_proj(blocks[-1].dstdata)
        return h_user_dst + self.user_sage(blocks, h_user)