import argparse
import os
import pickle
import pandas as pd
import numpy as np
import dgl
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
import dgl.function as fn
from sklearn.cluster import KMeans
import logging

import evaluation
import layers
import sampler as sampler_module
from model import PinSAGEModel


# data
def build_subgraph(g, indices, utype, itype, etype, etype_rev):
    """
        # Build the graph with training interactions only.
        train_g = build_train_graph(
            g, train_indices, "user", "movie", "watched", "watched-by"
        )
        assert train_g.out_degrees(etype="watched").min() > 0
    """
    subgraph = g.edge_subgraph(
        {etype: indices, etype_rev: indices}
    )

    # 기존 id 보존
    subgraph.nodes[utype].data[dgl.NID] = subgraph.nodes[utype].data[f"{utype}_id"]
    subgraph.nodes[itype].data[dgl.NID] = subgraph.nodes[itype].data[f"{itype}_id"]

    return subgraph


def get_dataloader(train_graph, prev_graph, total_graph, dataset, args):
    """
    return train dataloader, test dataloader

    note,
    train dataloader consist of train graph
    test dataloader consist of total graph
    """

    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]

    ## Sampler
    # it returns heads, tails, neg_tails
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        train_graph, prev_graph, user_ntype, item_ntype, args.batch_size
    )
    # user batch sampler
    user_batch_sampler = sampler_module.UserToUserBatchSampler(
        train_graph, prev_graph, user_ntype, item_ntype, args.batch_size
    )

    ## neighbor_sampler
    # it returns pos_graph, neg_graph, blocks when train
    # it returns blocks when test
    neighbor_sampler = sampler_module.NeighborSampler(
        train_graph,
        total_graph,
        prev_graph,
        user_ntype,
        item_ntype,
        args.random_walk_length,
        args.random_walk_restart_prob,
        args.num_random_walks,
        args.num_neighbors,
        args.num_layers,
    )
    # user neighbor sampler
    user_neighbor_sampler = sampler_module.UserNeighborSampler(
        train_graph,
        total_graph,
        prev_graph,
        user_ntype,
        item_ntype,
        args.random_walk_length,
        args.random_walk_restart_prob,
        args.num_random_walks,
        args.num_neighbors,
        args.num_layers,
    )

    # 아이템-아이템 graph를 실질적으로 처리해서 넘기게 도와주는 컴포넌트
    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, train_graph, total_graph, prev_graph, item_ntype
    )
    user_collator = sampler_module.PinSAGECollator(
        user_neighbor_sampler, train_graph, total_graph, prev_graph, user_ntype
    )

    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers,
    )
    user_dataloader = DataLoader(
        user_batch_sampler,
        collate_fn=user_collator.collate_train,
        num_workers=args.num_workers,
    )

    dataloader_test = DataLoader(
        torch.arange(total_graph.num_nodes(item_ntype)),
        #         total_graph.nodes[item_ntype].data[dgl.NID],
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )
    user_dataloader_test = DataLoader(
        torch.arange(total_graph.num_nodes(user_ntype)),
        #         total_graph.nodes[user_ntype].data[dgl.NID],
        batch_size=args.batch_size,
        collate_fn=user_collator.collate_test,
        num_workers=args.num_workers,
    )

    dataloader_it = iter(dataloader)
    user_dataloader_it = iter(user_dataloader)

    return dataloader_it, user_dataloader_it, dataloader_test, user_dataloader_test


def get_valtest(stage):
    labels = pd.read_csv(f"./dataset/increase/ml_1m_inc{stage + 1}.csv")
    labels_series = labels.groupby("user_id")['movie_id'].agg(list)
    labels_series = labels_series.sample(frac=1, random_state=30)
    pivot = labels_series.shape[0] // 2
    val_data = labels_series[:pivot]
    test_data = labels_series[pivot:]
    return val_data, test_data


# calc
def get_node_emb(model, dataloader_test, user_dataloader_test, args):
    # get previous h_user, h_item
    model.eval()
    with torch.no_grad():
        h_item_batches = []
        h_user_batches = []

        for blocks in dataloader_test:
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(args.device)
            h_item_batches.append(model.get_repr(blocks))

        h_item = torch.cat(h_item_batches, 0)

        if user_dataloader_test is None:
            return h_item

        for blocks in user_dataloader_test:
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(args.device)
            h_user_batches.append(model.get_user_repr(blocks))

        h_user = torch.cat(h_user_batches, 0)
    return h_item, h_user


def get_anchors(emb, args):
    # get anchors by k-means clustering
    kmeans = KMeans(n_clusters=args.n_cluster, n_init='auto')
    kmeans.fit(emb.cpu())
    anchors = kmeans.cluster_centers_
    anchors = torch.from_numpy(anchors).to(args.device).to(torch.float32)

    return torch.transpose(anchors, 0, 1)


def get_local_loss(h_item_batch_prev, h_user_batch_prev, h_item_batch_cur, h_user_batch_cur, pos_graph_prev,
                   user_pos_graph_prev):
    # pos_graph에 임베딩을 h_type feature로 할당
    pos_graph_prev.ndata['h_item_prev'] = h_item_batch_prev
    pos_graph_prev.ndata['h_item_cur'] = h_item_batch_cur
    user_pos_graph_prev.ndata['h_user_prev'] = h_user_batch_prev
    user_pos_graph_prev.ndata['h_user_cur'] = h_user_batch_cur

    # 인접 이웃의 평균 임베딩을 n_type feature로 할당
    pos_graph_prev.update_all(fn.copy_u('h_item_prev', "m"), fn.mean("m", "n_item_prev"))
    pos_graph_prev.update_all(fn.copy_u('h_item_cur', "m"), fn.mean("m", "n_item_cur"))
    user_pos_graph_prev.update_all(fn.copy_u('h_user_prev', "m"), fn.mean("m", "n_user_prev"))
    user_pos_graph_prev.update_all(fn.copy_u('h_user_cur', "m"), fn.mean("m", "n_user_cur"))

    n_item_prev = pos_graph_prev.ndata["n_item_prev"]
    n_item_cur = pos_graph_prev.ndata["n_item_cur"]
    n_user_prev = user_pos_graph_prev.ndata["n_user_prev"]
    n_user_cur = user_pos_graph_prev.ndata["n_user_cur"]

    num_user = user_pos_graph_prev.num_nodes()
    num_item = pos_graph_prev.num_nodes()

    # local loss = ((h_item_prev * n_item_prev - h_item_cur * n_item_cur)^2)/|I| + ((h_user_prev * n_user_prev - h_user_cur * n_user_cur)^2)/|U|
    local_item_loss = torch.sum(h_item_batch_prev * n_item_prev - h_item_batch_cur * n_item_cur, dim=1)
    local_item_loss = torch.sum(torch.pow(local_item_loss, 2)) / num_item
    local_user_loss = torch.sum(h_user_batch_prev * n_user_prev - h_user_batch_cur * n_user_cur, dim=1)
    local_user_loss = torch.sum(torch.pow(local_user_loss, 2)) / num_user
    local_loss = local_item_loss + local_user_loss
    return local_loss


def get_global_loss(h_item_batch_prev, h_user_batch_prev, h_item_batch_cur, h_user_batch_cur, a_item_prev, a_user_prev,
                    a_item_cur, a_user_cur, pos_graph_prev,
                    user_pos_graph_prev, args):
    # exp^(sim/t) 계산
    sim_h_item_a_item_cur = torch.exp(torch.div(torch.matmul(h_item_batch_cur, a_item_cur), args.global_structure_t))
    sim_h_item_a_user_cur = torch.exp(torch.div(torch.matmul(h_item_batch_cur, a_user_cur), args.global_structure_t))
    sim_h_user_a_item_cur = torch.exp(torch.div(torch.matmul(h_user_batch_cur, a_item_cur), args.global_structure_t))
    sim_h_user_a_user_cur = torch.exp(torch.div(torch.matmul(h_user_batch_cur, a_user_cur), args.global_structure_t))
    sim_h_item_a_item_prev = torch.exp(torch.div(torch.matmul(h_item_batch_prev, a_item_prev), args.global_structure_t))
    sim_h_item_a_user_prev = torch.exp(torch.div(torch.matmul(h_item_batch_prev, a_user_prev), args.global_structure_t))
    sim_h_user_a_item_prev = torch.exp(torch.div(torch.matmul(h_user_batch_prev, a_item_prev), args.global_structure_t))
    sim_h_user_a_user_prev = torch.exp(torch.div(torch.matmul(h_user_batch_prev, a_user_prev), args.global_structure_t))

    # GS 계산
    gs_h_item_a_item_cur = torch.div(sim_h_item_a_item_cur, torch.sum(sim_h_item_a_item_cur, dim=1).view(-1, 1))
    gs_h_item_a_user_cur = torch.div(sim_h_item_a_user_cur, torch.sum(sim_h_item_a_user_cur, dim=1).view(-1, 1))
    gs_h_user_a_item_cur = torch.div(sim_h_user_a_item_cur, torch.sum(sim_h_user_a_item_cur, dim=1).view(-1, 1))
    gs_h_user_a_user_cur = torch.div(sim_h_user_a_user_cur, torch.sum(sim_h_user_a_user_cur, dim=1).view(-1, 1))
    gs_h_item_a_item_prev = torch.div(sim_h_item_a_item_prev, torch.sum(sim_h_item_a_item_prev, dim=1).view(-1, 1))
    gs_h_item_a_user_prev = torch.div(sim_h_item_a_user_prev, torch.sum(sim_h_item_a_user_prev, dim=1).view(-1, 1))
    gs_h_user_a_item_prev = torch.div(sim_h_user_a_item_prev, torch.sum(sim_h_user_a_item_prev, dim=1).view(-1, 1))
    gs_h_user_a_user_prev = torch.div(sim_h_user_a_user_prev, torch.sum(sim_h_user_a_user_prev, dim=1).view(-1, 1))

    # SA 계산
    SA = nn.KLDivLoss(reduction='batchmean')

    sa_uu = SA(gs_h_user_a_user_cur.log(), gs_h_user_a_user_prev)
    sa_ui = SA(gs_h_user_a_item_cur.log(), gs_h_user_a_item_prev)
    sa_iu = SA(gs_h_item_a_user_cur.log(), gs_h_item_a_user_prev)
    sa_ii = SA(gs_h_item_a_item_cur.log(), gs_h_item_a_item_prev)

    num_user = user_pos_graph_prev.num_nodes()
    num_item = pos_graph_prev.num_nodes()

    global_loss = (sa_uu + sa_ui) / num_user + (sa_iu + sa_ii) / num_item
    return global_loss


# train
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_recall):
        if self.best_score is None:
            self.best_score = val_recall
        elif val_recall < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_recall
            self.counter = 0


def base_train(stage, args):
    with open(os.path.join(args.dataset_path, args.scenario, f"dataset{stage}.pkl"), "rb") as f:
        dataset = pickle.load(f)

    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]
    user_to_item_etype = dataset["user-to-item-type"]
    item_to_user_etpye = dataset["item-to-user-type"]
    timestamp = dataset["timestamp-edge-column"]

    # set data
    train_g = dataset['total_g']
    prev_g = dataset['total_g']
    total_g = dataset['total_g']
    dataloader_it, user_dataloader_it, dataloader_test, user_dataloader_test = get_dataloader(train_graph=train_g,
                                                                                              prev_graph=prev_g,
                                                                                              total_graph=total_g,
                                                                                              dataset=dataset,
                                                                                              args=args)
    val_data = dataset['val_data']
    test_data = dataset['test_data']

    # set model
    init = nn.init.xavier_uniform_
    model = PinSAGEModel(
        total_g, item_ntype, args.hidden_dims, args.num_layers
    ).to(args.device)
    for param in model.parameters():
        if len(param.shape) > 1:
            init(param)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_base)

    early_stop = EarlyStopping(patience=args.patience_base)

    for epoch_id in range(args.num_epochs_base):
        """
        train
        """
        model.train()
        for batch_id in tqdm.trange(args.batches_per_epoch_base):
            # get batch
            pos_graph, neg_graph, blocks, _, _ = next(dataloader_it)
            user_pos_graph, user_neg_graph, user_blocks, _, _ = next(user_dataloader_it)
            # and copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(args.device)
            for i in range(len(user_blocks)):
                user_blocks[i] = user_blocks[i].to(args.device)
            pos_graph = pos_graph.to(args.device)
            neg_graph = neg_graph.to(args.device)
            user_pos_graph = user_pos_graph.to(args.device)
            user_neg_graph = user_neg_graph.to(args.device)

            # calc loss
            loss = model(pos_graph, neg_graph, blocks).mean()
            user_loss = model.user_forward(user_pos_graph, user_neg_graph, user_blocks).mean()
            loss += user_loss

            # update weight
            opt.zero_grad()
            loss.backward()
            opt.step()
        """
        test
        """
        h_item = get_node_emb(model, dataloader_test, None, args)
        # recall, hr, ndcg5, f1 = evaluation.evaluate_nn(total_g, h_item, args.k, val_data, args.batch_size)
        recall, hr = evaluation.evaluate_nn(total_g, h_item, args.k, val_data, args.batch_size)
        # print(f"\nrecall at stage {stage} epoch {epoch_id} is {recall}")
        logger.info(
            f"stage {stage} epoch {epoch_id}, recall: {recall}, hr: {hr}, loss: {loss}")
        # logger.info(
        #     f"stage {stage} epoch {epoch_id}, recall: {recall}, hr: {hr}, ndcg5: {ndcg5}, f1: {f1}, loss: {loss}")
        early_stop(recall)
        if early_stop.early_stop:
            logger.info(f"early stopped at stage {stage} epoch {epoch_id}")
            break

    # recall, hr, ndcg5, f1 = evaluation.evaluate_nn(total_g, h_item, args.k, test_data, args.batch_size)
    recall, hr = evaluation.evaluate_nn(total_g, h_item, args.k, test_data, args.batch_size)
    # print(f"\ntraining finished at stage {stage}, recall20 on test data is {recall}")
    logger.info(f"training finished at stage {stage}, recall: {recall}, hr: {hr}\n\n")
    # logger.info(f"training finished at stage {stage}, recall: {recall}, hr: {hr}, ndcg5: {ndcg5}, f1: {f1}\n\n")

    return model, dataloader_test, user_dataloader_test


def inc_train(stage, args, model_prev, dataloader_test_prev, user_dataloader_test_prev):
    with open(os.path.join(args.dataset_path, args.scenario, f"dataset{stage}.pkl"), "rb") as f:
        dataset = pickle.load(f)

    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]
    user_to_item_etype = dataset["user-to-item-type"]
    item_to_user_etpye = dataset["item-to-user-type"]
    timestamp = dataset["timestamp-edge-column"]

    # set data
    train_g = dataset['train_g']
    prev_g = dataset['prev_g']
    total_g = dataset['total_g']
    dataloader_it, user_dataloader_it, dataloader_test, user_dataloader_test = get_dataloader(train_graph=train_g,
                                                                                              prev_graph=prev_g,
                                                                                              total_graph=total_g,
                                                                                              dataset=dataset,
                                                                                              args=args)
    val_data = dataset['val_data']
    test_data = dataset['test_data']

    # set model
    init = nn.init.xavier_uniform_
    model = PinSAGEModel(
        total_g, item_ntype, args.hidden_dims, args.num_layers
    ).to(args.device)
    for param in model.parameters():
        if len(param.shape) > 1:
            init(param)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_inc)

    early_stop = EarlyStopping(patience=args.patience_inc)

    # pre-work in inc training
    # calc user/item embedding anchor in (t-1)
    # t-1 시점 유저/아이템 anchor 사전 계산
    h_item_prev, h_user_prev = get_node_emb(model_prev, dataloader_test_prev, user_dataloader_test_prev, args)
    a_item_prev = get_anchors(h_item_prev, args)
    a_user_prev = get_anchors(h_user_prev, args)

    for epoch_id in range(args.num_epochs_inc):
        """
        t 시점 유저/아이템 anchor 계산
        """
        h_item_cur, h_user_cur = get_node_emb(model, dataloader_test, user_dataloader_test, args)
        a_item_cur = get_anchors(h_item_cur, args)
        a_user_cur = get_anchors(h_user_cur, args)

        """
        train
        """
        model.train()
        for batch_id in tqdm.trange(args.batches_per_epoch_inc):
            # get batch
            pos_graph, neg_graph, blocks, pos_graph_prev, blocks_prev = next(dataloader_it)
            user_pos_graph, user_neg_graph, user_blocks, user_pos_graph_prev, user_blocks_prev = next(
                user_dataloader_it)
            # and copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(args.device)
            for i in range(len(user_blocks)):
                user_blocks[i] = user_blocks[i].to(args.device)
            for i in range(len(blocks_prev)):
                blocks_prev[i] = blocks_prev[i].to(args.device)
            for i in range(len(user_blocks_prev)):
                user_blocks_prev[i] = user_blocks_prev[i].to(args.device)
            pos_graph = pos_graph.to(args.device)
            neg_graph = neg_graph.to(args.device)
            user_pos_graph = user_pos_graph.to(args.device)
            user_neg_graph = user_neg_graph.to(args.device)
            pos_graph_prev = pos_graph_prev.to(args.device)
            user_pos_graph_prev = user_pos_graph_prev.to(args.device)

            # calculate common data used for local, global loss
            # 배치 임베딩을 계산
            model_prev.eval()
            # 이전 모델로 이전 block에 대한 emb 계산
            with torch.no_grad():
                h_item_batch_prev = model_prev.get_repr(blocks_prev)
                h_user_batch_prev = model_prev.get_user_repr(user_blocks_prev)
            # 현재 모델로 이전 block에 대한 emb 계산
            h_item_batch_cur = model.get_repr(blocks_prev)
            h_user_batch_cur = model.get_user_repr(user_blocks_prev)

            local_loss = get_local_loss(h_item_batch_prev, h_user_batch_prev, h_item_batch_cur, h_user_batch_cur,
                                        pos_graph_prev,
                                        user_pos_graph_prev)
            global_loss = get_global_loss(h_item_batch_prev, h_user_batch_prev, h_item_batch_cur, h_user_batch_cur,
                                          a_item_prev, a_user_prev,
                                          a_item_cur, a_user_cur, pos_graph_prev,
                                          user_pos_graph_prev, args)
            loss = model(pos_graph, neg_graph, blocks).mean()
            user_loss = model.user_forward(user_pos_graph, user_neg_graph, user_blocks).mean()
            # loss += user_loss
            loss = loss + user_loss + args.local_lam * local_loss + args.global_lam * global_loss

            # update weight
            opt.zero_grad()
            loss.backward()
            opt.step()
        """
        test
        """
        h_item = get_node_emb(model, dataloader_test, None, args)
        # recall, hr, ndcg5, f1 = evaluation.evaluate_nn(total_g, h_item, args.k, val_data, args.batch_size)
        recall, hr= evaluation.evaluate_nn(total_g, h_item, args.k, val_data, args.batch_size)
        # print(f"\nrecall at stage {stage} epoch {epoch_id} is {recall}")
        logger.info(
            f"stage {stage} epoch {epoch_id}, recall: {recall}, hr: {hr} loss: {loss}")
        # logger.info(
        #     f"stage {stage} epoch {epoch_id}, recall: {recall}, hr: {hr}, ndcg5: {ndcg5}, f1: {f1}, loss: {loss}")
        early_stop(recall)
        if early_stop.early_stop:
            logger.info(f"early stopped at stage {stage} epoch {epoch_id}")
            break

    # recall, hr, ndcg5, f1 = evaluation.evaluate_nn(total_g, h_item, args.k, test_data, args.batch_size)
    recall, hr = evaluation.evaluate_nn(total_g, h_item, args.k, test_data, args.batch_size)
    # print(f"\ntraining finished at stage {stage}, recall20 on test data is {recall}")
    # logger.info(f"training finished at stage {stage}, recall: {recall}, hr: {hr}, ndcg5: {ndcg5}, f1: {f1}\n\n")
    logger.info(f"training finished at stage {stage}, recall: {recall}, hr: {hr}\n\n")
    return model, dataloader_test, user_dataloader_test


def train(args):
    """
    this function controls entire training sequence
    """
    if args.is_full == "full":
        for stage in range(6):
            logger.info(f"{args.scenario} scenario training start at stage {stage}")
            base_train(stage, args)
    else:
        for stage in range(6):
            # print(f"\ntraining start at stage {stage}\n")
            logger.info(f"{args.scenario} scenario training start at stage {stage}")
            if stage == 0:
                ##### 0-th training
                model_prev, dataloader_test_prev, user_dataloader_test_prev = base_train(stage, args)
            else:
                ##### 1, 2, 3, 4-th training
                model_prev, dataloader_test_prev, user_dataloader_test_prev = inc_train(stage, args, model_prev,
                                                                                        dataloader_test_prev,
                                                                                        user_dataloader_test_prev)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--logger", type=str)
    parser.add_argument("--scenario", type=str, default="increase")
    parser.add_argument("--is-full", type=str)
    parser.add_argument("--dataset_path", type=str, default="./dataset")
    parser.add_argument("--random-walk-length", type=int, default=2)
    parser.add_argument("--random-walk-restart-prob", type=float, default=0.5)
    parser.add_argument("--num-random-walks", type=int, default=10)
    parser.add_argument("--num-neighbors", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dims", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda:1"
    )
    parser.add_argument("--num-epochs-base", type=int, default=50)  # 50
    parser.add_argument("--num-epochs-inc", type=int, default=15)  # 15
    parser.add_argument("--batches-per-epoch-base", type=int, default=1000)
    parser.add_argument("--batches-per-epoch-inc", type=int, default=1000)
    parser.add_argument("--patience-base", type=int, default=10)
    parser.add_argument("--patience-inc", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-base", type=float, default=1e-3)
    parser.add_argument("--lr-inc", type=float, default=5e-4)
    parser.add_argument("-k", type=int, default=20)
    parser.add_argument("--n_cluster", type=int, default=5)
    parser.add_argument("--global-structure-t", type=float, default=0.1)
    parser.add_argument("--local-lam", type=float, default=1)
    parser.add_argument("--global-lam", type=float, default=1)

    args = parser.parse_args()

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(f"./log/{args.logger}.txt")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Load dataset
    train(args)
