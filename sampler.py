import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


# from torchtext.data.functional import numericalize_tokens_from_iterator


def padding(array, yy, val):
    """
    :param array: torch tensor array
    :param yy: desired width
    :param val: padded value
    :return: padded array
    """
    w = array.shape[0]
    b = 0
    bb = yy - b - w

    return torch.nn.functional.pad(
        array, pad=(b, bb), mode="constant", value=val
    )


def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block


class ItemToItemBatchSampler(IterableDataset):
    """
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        train_graph, user_ntype, item_ntype, args.batch_size
    )

    랜덤으로 뽑은 heads(node ID) 를 기준으로 랜덤워크를 하여 tails을 찾아내고.. 랜덤으로 뽑은 neg_tails를 만든다.
    그리고 heads, tails, neg_tails를 return 한다.
    """

    def __init__(self, train_g, prev_g, user_type, item_type, batch_size):
        self.train_g = train_g
        self.prev_g = prev_g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(train_g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(train_g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            # 출발 아이템 노드.. 0부터 item 노드 갯수까지, batch_size(32) 갯수만큼 할당
            heads = torch.randint(
                0, self.train_g.num_nodes(self.item_type), (self.batch_size,)
            )
            # 랜덤 워크를 통해 얻은 출발 아이템 노드와 아이템-유저-아이템으로 이어진 노드
            tails = dgl.sampling.random_walk(
                self.train_g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype],
            )[0][:, 2]
            # 랜덤  노드. 그래프가 충분히 크면 랜덤으로 뽑으면 그냥 안 연결되어있다고 가정하는듯
            neg_tails = torch.randint(
                0, self.train_g.num_nodes(self.item_type), (self.batch_size,)
            )

            # prev_g 존재하는 heads
            heads_common = torch.randint(
                0, self.prev_g.num_nodes(self.item_type), (self.batch_size,)
            )
            tails_prev = dgl.sampling.random_walk(
                self.prev_g,
                heads_common,
                metapath=[self.item_to_user_etype, self.user_to_item_etype],
            )[0][:, 2]
            neg_tails_prev = torch.randint(
                0, self.prev_g.num_nodes(self.item_type), (len(heads_common),)
            )

            mask = tails != -1
            mask_prev = tails_prev != -1
            # mask_cur = tails_cur != -1
            yield heads[mask], tails[mask], neg_tails[mask], heads_common[mask_prev], tails_prev[mask_prev], \
                neg_tails_prev[mask_prev]


class UserToUserBatchSampler(IterableDataset):
    """
    # user batch sampler
    user_batch_sampler = sampler_module.UserToUserBatchSampler(
        train_graph, user_ntype, item_ntype, args.batch_size
    )
    """

    def __init__(self, train_g, prev_g, user_type, item_type, batch_size):
        self.train_g = train_g
        self.prev_g = prev_g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(train_g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(train_g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            heads = torch.randint(
                0, self.train_g.num_nodes(self.user_type), (self.batch_size,)
            )
            tails = dgl.sampling.random_walk(
                self.train_g,
                heads,
                metapath=[self.user_to_item_etype, self.item_to_user_etype],
            )[0][:, 2]
            neg_tails = torch.randint(
                0, self.train_g.num_nodes(self.user_type), (self.batch_size,)
            )

            # train_g/prev_g 모두에 존재하는 heads
            heads_common = torch.randint(
                0, self.prev_g.num_nodes(self.user_type), (self.batch_size,)
            )
            tails_prev = dgl.sampling.random_walk(
                self.prev_g,
                heads_common,
                metapath=[self.user_to_item_etype, self.item_to_user_etype],
            )[0][:, 2]
            neg_tails_prev = torch.randint(
                0, self.prev_g.num_nodes(self.user_type), (len(heads_common),)
            )

            mask = tails != -1
            mask_prev = tails_prev != -1
            # mask_cur = tails_cur != -1
            yield heads[mask], tails[mask], neg_tails[mask], heads_common[mask_prev], tails_prev[mask_prev], \
                neg_tails_prev[mask_prev]

            # # 출발 아이템 노드. g안에 존재하는 랜덤한 node ID를 batch_size만큼 뽑아낸다.
            # indices = torch.randperm(self.g.num_nodes(self.user_type))[:self.batch_size]
            # heads = self.g.nodes[self.user_type].data[dgl.NID][indices]
            #
            # tails = dgl.sampling.random_walk(
            #     self.g,
            #     heads,
            #     metapath=[self.user_to_item_etype, self.item_to_user_etype],
            # )[0][:, 2]
            #
            # indices_neg_tails = torch.randperm(self.g.num_nodes(self.user_type))[:self.batch_size]
            # neg_tails = self.g.nodes[self.user_type].data[dgl.NID][indices_neg_tails]


class NeighborSampler(object):
    """
        neighbor_sampler = sampler_module.NeighborSampler(
        train_graph,
        user_ntype,
        item_ntype,
        args.random_walk_length,
        args.random_walk_restart_prob,
        args.num_random_walks,
        args.num_neighbors,
        args.num_layers,
    )
    """

    def __init__(
            self,
            train_g,
            total_g,
            prev_g,
            user_type,
            item_type,
            random_walk_length,
            random_walk_restart_prob,
            num_random_walks,
            num_neighbors,
            num_layers,
    ):
        self.train_g = train_g
        self.total_g = total_g
        self.prev_g = prev_g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(train_g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(train_g.metagraph()[item_type][user_type])[0]
        # https://docs.dgl.ai/en/0.9.x/generated/dgl.sampling.PinSAGESampler.html
        self.samplers = [
            dgl.sampling.PinSAGESampler(
                train_g,
                item_type,
                user_type,
                random_walk_length,
                random_walk_restart_prob,
                num_random_walks,
                num_neighbors,
            )
            for _ in range(num_layers)
        ]
        self.total_samplers = [
            dgl.sampling.PinSAGESampler(
                total_g,
                item_type,
                user_type,
                random_walk_length,
                random_walk_restart_prob,
                num_random_walks,
                num_neighbors,
            )
            for _ in range(num_layers)
        ]
        self.prev_samplers = [
            dgl.sampling.PinSAGESampler(
                prev_g,
                item_type,
                user_type,
                random_walk_length,
                random_walk_restart_prob,
                num_random_walks,
                num_neighbors,
            )
            for _ in range(num_layers)
        ]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None, is_prev=False):
        """
        collator_test에서는 아래와 같이 호출하고
        blocks = self.sampler.sample_blocks(batch)

        sample_from_item_pairs가 호출되는 collator_train에서는 아래와 같이 호출된다.
        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        """
        blocks = []
        # 각 sampler는 dgl.sampling.PinSAGESampler 객체임

        # collator_train에서 호출됀다면 heads, tails.. 등이 존재함
        if heads is not None:
            samplers = self.prev_samplers if is_prev else self.samplers

            for sampler in samplers:
                frontier = sampler(seeds)

                # heads가 전달된 상황 => collator_test
                if heads is not None:
                    eids = frontier.edge_ids(
                        torch.cat([heads, heads]),
                        torch.cat([tails, neg_tails]),
                        return_uv=True,
                    )[2]
                    if len(eids) > 0:
                        old_frontier = frontier
                        frontier = dgl.remove_edges(old_frontier, eids)

                block = compact_and_copy(frontier, seeds)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)
            return blocks
        # collator_test에서 호출된다면 heads.. 등이 없음
        else:
            for sampler in self.total_samplers:
                frontier = sampler(seeds)
                block = compact_and_copy(frontier, seeds)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)
            return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails, is_prev=False):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        g = self.prev_g if is_prev else self.train_g

        pos_graph = dgl.graph(
            (heads, tails), num_nodes=g.num_nodes(self.item_type)
        )
        neg_graph = dgl.graph(
            (heads, neg_tails), num_nodes=g.num_nodes(self.item_type)
        )
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        # pos_graph = dgl.compact_graphs(pos_graph)
        # neg_graph = dgl.compact_graphs(neg_graph)

        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails, is_prev)
        return pos_graph, neg_graph, blocks


class UserNeighborSampler(object):
    def __init__(
            self,
            train_g,
            total_g,
            prev_g,
            user_type,
            item_type,
            random_walk_length,
            random_walk_restart_prob,
            num_random_walks,
            num_neighbors,
            num_layers,
    ):
        self.train_g = train_g
        self.total_g = total_g
        self.prev_g = prev_g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(train_g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(train_g.metagraph()[item_type][user_type])[0]
        # https://docs.dgl.ai/en/0.9.x/generated/dgl.sampling.PinSAGESampler.html
        self.samplers = [
            dgl.sampling.PinSAGESampler(
                train_g,
                user_type,
                item_type,
                random_walk_length,
                random_walk_restart_prob,
                num_random_walks,
                num_neighbors,
            )
            for _ in range(num_layers)
        ]
        self.total_samplers = [
            dgl.sampling.PinSAGESampler(
                total_g,
                user_type,
                item_type,
                random_walk_length,
                random_walk_restart_prob,
                num_random_walks,
                num_neighbors,
            )
            for _ in range(num_layers)
        ]
        self.prev_samplers = [
            dgl.sampling.PinSAGESampler(
                prev_g,
                user_type,
                item_type,
                random_walk_length,
                random_walk_restart_prob,
                num_random_walks,
                num_neighbors,
            )
            for _ in range(num_layers)
        ]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None, is_prev=False):
        blocks = []
        # 각 sampler는 dgl.sampling.PinSAGESampler 객체임

        # collator_train에서 호출됀다면 heads, tails.. 등이 존재함
        if heads is not None:
            samplers = self.prev_samplers if is_prev else self.samplers

            for sampler in samplers:
                frontier = sampler(seeds)

                # heads가 전달된 상황 => collator_test
                if heads is not None:
                    eids = frontier.edge_ids(
                        torch.cat([heads, heads]),
                        torch.cat([tails, neg_tails]),
                        return_uv=True,
                    )[2]
                    if len(eids) > 0:
                        old_frontier = frontier
                        frontier = dgl.remove_edges(old_frontier, eids)

                block = compact_and_copy(frontier, seeds)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)
            return blocks
        # collator_test에서 호출된다면 heads.. 등이 없음
        else:
            for sampler in self.total_samplers:
                frontier = sampler(seeds)
                block = compact_and_copy(frontier, seeds)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)
            return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails, is_prev=False):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        g = self.prev_g if is_prev else self.train_g

        pos_graph = dgl.graph(
            (heads, tails), num_nodes=g.num_nodes(self.user_type)
        )
        neg_graph = dgl.graph(
            (heads, neg_tails), num_nodes=g.num_nodes(self.user_type)
        )
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails, is_prev)
        return pos_graph, neg_graph, blocks


def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """
    Copies data to the given block from the corresponding nodes in the original graph.
    """
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]


def assign_features_to_blocks(blocks, g, ntype):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)


class PinSAGECollator(object):
    """
    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, train_graph, item_ntype
    )
    user_collator = sampler_module.PinSAGECollator(
        user_neighbor_sampler, train_graph, user_ntype
    )
    """

    def __init__(self, sampler, train_g, total_g, prev_g, ntype):
        self.sampler = sampler
        self.ntype = ntype
        self.train_g = train_g
        self.total_g = total_g
        self.prev_g = prev_g

    def collate_train(self, batches):
        heads, tails, neg_tails, heads_prev, tails_prev, neg_tails_prev = batches[0]

        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(
            heads, tails, neg_tails
        )
        pos_graph_prev, _, blocks_prev = self.sampler.sample_from_item_pairs(
            heads_prev, tails_prev, neg_tails_prev, is_prev=True
        )
        # pos_graph_cur, _, blocks_cur = self.sampler.sample_from_item_pairs(
        #     heads_cur, tails_cur, neg_tails_cur
        # )
        assign_features_to_blocks(blocks, self.train_g, self.ntype)
        assign_features_to_blocks(blocks_prev, self.prev_g, self.ntype)
        # assign_features_to_blocks(blocks_cur, self.train_g, self.ntype)

        # print(heads, heads_prev, heads_cur)
        # print(tails, tails_prev, tails_cur)
        # print(neg_tails, neg_tails_prev, neg_tails_cur)
        # print(heads.shape, heads_prev.shape, heads_cur.shape)
        # print(tails.shape, tails_prev.shape, tails_cur.shape)
        # print(neg_tails.shape, neg_tails_prev.shape, neg_tails_cur.shape)

        return pos_graph, neg_graph, blocks, pos_graph_prev, blocks_prev

    def collate_test(self, samples):
        # batch에는 total_graph의 node index가 batch_size만큼 들어올거고..
        batch = torch.LongTensor(samples)
        # sample_blocks는 batch를 받아서 block으로 만든다.
        # block 만드는 과정을 뜯어야겠다.
        blocks = self.sampler.sample_blocks(batch)
        assign_features_to_blocks(blocks, self.total_g, self.ntype)
        return blocks
