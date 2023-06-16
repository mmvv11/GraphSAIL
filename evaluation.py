import argparse
import pickle
import dgl
import numpy as np
import torch
from sklearn.metrics import f1_score


def prec(recommendations, ground_truth):
    """
    return prec(recommendations, val_matrix)

    :param recommendations: 이건. 유저의 최근 아이템과 유사한 top-k 아이템
    :param ground_truth: 이건. 실제 유저가 선택한 아이템. val_matrix 가 될 수도 있고. test_matrix가 될수도.
    :return: hit rate
    """
    n_users, n_items = ground_truth.shape
    K = recommendations.shape[1]
    user_idx = np.repeat(np.arange(n_users), K)  # user index를 k번 반복한 array.. ex 0,0,1,1,2,2 ...
    item_idx = recommendations.flatten()  # n_users x k 형태의 item index를 가진 recommendations를 1차원 배열로
    relevance = ground_truth[user_idx, item_idx].reshape((n_users, K))  # ground_truth에 해당 user, item index 부분만
    hit = relevance.any(axis=1).mean()  # hit rate 계산.
    return hit


def ndcg_at_k(rec, val, k):
    """
    :param rec: 각 유저에 대한 top-k 추천 (pandas df)
    :param val: 실제 각 유저의 interaction 정보 (pandas df)
    :param k: 추천 리스트에서 고려할 상위 k개의 아이템
    :return: ndcg@k
    """
    ndcg_sum = 0
    num_user = 0

    for ui in val.index:
        if ui not in rec:
            continue

        num_user += 1
        actual_list = list(val[ui])
        predicted_list = list(rec[ui][:k])

        ndcg = ndcg_score(actual_list, predicted_list, k)
        ndcg_sum += ndcg

    mean_ndcg = ndcg_sum / num_user if num_user > 0 else 0

    return mean_ndcg


def ndcg_score(actual_list, predicted_list, k):
    """
    :param actual_list: 실제 상호작용 정보 (리스트)
    :param predicted_list: 추천 결과 (리스트)
    :param k: 추천 리스트에서 고려할 상위 k개의 아이템
    :return: ndcg score
    """
    dcg = 0
    idcg = ideal_dcg_score(actual_list, k)

    for i in range(min(k, len(predicted_list))):
        item = predicted_list[i]
        if item in actual_list:
            item_rank = actual_list.index(item) + 1
            dcg += 1 / np.log2(i + 2) if i < k else 0

    ndcg = dcg / idcg if idcg > 0 else 0

    return ndcg


def ideal_dcg_score(actual_list, k):
    """
    :param actual_list: 실제 상호작용 정보 (리스트)
    :param k: 추천 리스트에서 고려할 상위 k개의 아이템
    :return: ideal dcg score
    """
    idcg = 0

    for i in range(min(k, len(actual_list))):
        idcg += 1 / np.log2(i + 2) if i < k else 0

    return idcg


def f1(rec, val):
    """
    :param rec: 각 유저에 대한 top-k 추천 (pandas df)
    :param val: 실제 각 유저의 interaction 정보 (pandas df)
    :return: f1 score
    """
    y_true = []
    y_pred = []

    for ui in val.index:
        if ui not in rec:
            continue

        actual_set = set(val[ui])
        predicted_set = set(rec[ui])

        y_true.append(actual_set)
        y_pred.append(predicted_set)

    f1 = f1_score(y_true, y_pred, average='micro')

    return f1


def hit_rate(rec, val):
    """
    :param rec: 각 유저에 대한 top-k 추천 (pandas df)
    :param val: 실제 각 유저의 interaction 정보 (pandas df)
    :return: hit_rate
    """
    num_user = 0
    hit_count = 0

    for ui in val.index:
        if ui not in rec:
            continue

        num_user += 1
        actual_set = set(val[ui])
        predicted_set = set(rec[ui])

        if len(actual_set.intersection(predicted_set)) > 0:
            hit_count += 1

    hit_rate = hit_count / num_user if num_user > 0 else 0

    return hit_rate
    # return "{:.5f}".format(hit_rate)


def recall(rec, val):
    """

    :param recommendations: 각 유저에 대한 top-k 추천 (pandas df)
    :param truth: 실제 각 유저의 interaction 정보 (pandas series)
    :return: recall
    """
    num_user = 0
    recall_sum = 0

    for ui in val.index:
        if ui not in rec:
            continue

        num_user += 1
        actual_set = set(val[ui])
        predicted_set = set(rec[ui])
        num_actual = len(actual_set)
        num_correct = len(actual_set.intersection(predicted_set))

        recall = num_correct / num_actual if num_actual > 0 else 0
        recall_sum += recall
    mean_recall = recall_sum / num_user

    return mean_recall
    # return "{:.5f}".format(mean_recall)


class LatestNNRecommender(object):
    """
    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size
    )

    recommendations = rec_engine.recommend(g, k, None, h_item).cpu().numpy()
    """

    def __init__(
            self, user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size
    ):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, K, h_user, h_item):
        """
        Return a (n_user, K) matrix of recommended items for each user

        각 유저의 마지막 interaction을 바탕으로 top-k를 추출해냄.
        """
        # user to item subgraph 만들고
        graph_slice = full_graph.edge_type_subgraph([self.user_to_item_etype])
        # 총 유저 수
        n_users = full_graph.num_nodes(self.user_ntype)
        # 마지막 interaction
        latest_interactions = dgl.sampling.select_topk(
            graph_slice, 1, self.timestamp, edge_dir="out"
        )
        original_user_id = latest_interactions.nodes['user'].data['user_id'].cpu().numpy()
        original_movie_id = latest_interactions.nodes['movie'].data['movie_id'].cpu().numpy()
        movie_id_dict = {k: v for k, v in enumerate(original_movie_id)}
        user_id_dict = {k: v for k, v in enumerate(original_user_id)}

        # 유저, 가장 최근 아이템
        user, latest_items = latest_interactions.all_edges(
            form="uv", order="srcdst"
        )
        # each user should have at least one "latest" interaction
        assert torch.equal(user, torch.arange(n_users))

        recommended_batches = []
        user_batches = torch.arange(n_users).split(self.batch_size)
        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch].to(
                device=h_item.device
            )

            # 최근 아이템과 다른 아이템간의 행렬곱으로 유사성 계산
            dist = h_item[latest_item_batch] @ h_item.t()
            # exclude items that are already interacted
            for i, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(
                    u, etype=self.user_to_item_etype
                )
                dist[i, interacted_items] = -np.inf
            # 가장 유사한 top-k
            recommended_batches.append(dist.topk(K, 1)[1])

        recommendations = torch.cat(recommended_batches, 0)
        recommendations = recommendations.cpu().apply_(lambda x: movie_id_dict[x])
        recommendations = {k: v for k, v in zip(original_user_id, recommendations.tolist())}
        return recommendations


def evaluate_nn(full_graph, h_item, k, truth, batch_size):
    """
    evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size)

    """
    user_ntype = "user"
    item_ntype = "movie"
    user_to_item_etype = "watched"
    timestamp = "timestamp"

    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size
    )

    # train data 기준.. 각 유저별 마지막 아이템을 기준으로 가장 유사한 topk item list
    recommendations = rec_engine.recommend(full_graph, k, None, h_item)

    return recall(recommendations, truth), hit_rate(recommendations, truth)
    # return recall(recommendations, truth), hit_rate(recommendations, truth), ndcg_at_k(recommendations, truth, 5), f1(recommendations, truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("item_embedding_path", type=str)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset_path, "rb") as f:
        dataset = pickle.load(f)
    with open(args.item_embedding_path, "rb") as f:
        emb = torch.FloatTensor(pickle.load(f))
    print(evaluate_nn(dataset, emb, args.k, args.batch_size))
