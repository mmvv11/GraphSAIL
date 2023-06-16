import argparse
import os
import pickle
import re
import numpy as np
import pandas as pd
import torch
from builder import PandasGraphBuilder
# from data_utils import *
import dgl

pd.set_option('mode.chained_assignment',  None)


def sort_by_time(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df.sort_values(by=['timestamp'])


def df2graph(users, movies, ratings):
    # Filter the users and items that never appear in the rating table.
    distinct_users_in_ratings = ratings["user_id"].unique()  # 평점을 남긴 유저
    distinct_movies_in_ratings = ratings["movie_id"].unique()  # 평점이 남겨진 영화
    users = users[users["user_id"].isin(distinct_users_in_ratings)]  # 평점이 있는 유저만
    movies = movies[movies["movie_id"].isin(distinct_movies_in_ratings)]  # 평점이 있는 영화만
    users = users.astype("category")
    movies = movies.astype({"movie_id": "category"})

    # Group the movie features into genres (a vector), year (a category), title (a string)
    genre_columns = movies.columns.drop(["movie_id", "title", "year"])
    movies[genre_columns] = movies[genre_columns].fillna(False).astype("bool")
    movies_categorical = movies.drop("title", axis=1)

    # Build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, "user_id", "user")
    graph_builder.add_entities(movies_categorical, "movie_id", "movie")
    graph_builder.add_binary_relations(
        ratings, "user_id", "movie_id", "watched"
    )
    graph_builder.add_binary_relations(
        ratings, "movie_id", "user_id", "watched-by"
    )

    g = graph_builder.build()

    # Assign features.
    # Note that variable-sized features such as texts or images are handled elsewhere.
    g.nodes['user'].data['user_id'] = torch.LongTensor(
        users["user_id"].values
    )
    g.nodes['user'].data[dgl.NID] = g.nodes['user'].data['user_id']
    g.nodes["user"].data["gender"] = torch.LongTensor(
        users["gender"].cat.codes.values
    )
    g.nodes["user"].data["age"] = torch.LongTensor(
        users["age"].cat.codes.values
    )
    g.nodes["user"].data["occupation"] = torch.LongTensor(
        users["occupation"].cat.codes.values
    )
    g.nodes["user"].data["zip"] = torch.LongTensor(
        users["zip"].cat.codes.values
    )

    g.nodes['movie'].data['movie_id'] = torch.LongTensor(
        movies["movie_id"].values
    )
    g.nodes['movie'].data[dgl.NID] = g.nodes['movie'].data['movie_id']
    g.nodes["movie"].data["year"] = torch.LongTensor(
        movies["year"].cat.codes.values
    )
    g.nodes["movie"].data["genre"] = torch.FloatTensor(
        movies[genre_columns].values
    )

    g.edges["watched"].data["rating"] = torch.LongTensor(
        ratings["rating"].values
    )
    g.edges["watched"].data["timestamp"] = torch.LongTensor(
        ratings["timestamp"].values
    )
    g.edges["watched-by"].data["rating"] = torch.LongTensor(
        ratings["rating"].values
    )
    g.edges["watched-by"].data["timestamp"] = torch.LongTensor(
        ratings["timestamp"].values
    )

    dataset = {
        "user-type": "user",
        "item-type": "movie",
        "user-to-item-type": "watched",
        "item-to-user-type": "watched-by",
        "timestamp-edge-column": "timestamp",
    }

    return g


def get_IncIndices(df):
    """
    각 inc block에 들어가야할 interaction row에 mask를 생성한다.
    """
    BASE_DATA_RATIO = 6
    INC_STEP = 5

    # base block 설정
    pivot = len(df) * BASE_DATA_RATIO // 10
    df['mask_0'] = False
    df.iloc[:pivot, df.columns.get_loc("mask_0")] = True

    # inc block 설정
    len_per_block = df[pivot:].shape[0] // INC_STEP
    start = pivot
    for i in range(INC_STEP):
        df[f"mask_{i + 1}"] = False
        if i != INC_STEP - 1:  # 1, 2, 3, 4 block이라면
            df.iloc[start:start + len_per_block, df.columns.get_loc(f"mask_{i + 1}")] = True
        else:  # 마지막 block이라면
            df.iloc[start:, df.columns.get_loc(f"mask_{i + 1}")] = True
        start += len_per_block

    return [
        df[f"mask_{i}"].to_numpy().nonzero()[0] for i in range(INC_STEP + 1)
    ]


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

def split_trainvaltest(df):
    """
    df의 80%는 train,
    10% val
    10% test
    """
    train_pivot = df.shape[0] * 8 // 10
    train_df = df[:train_pivot]
    labels = df[train_pivot:]
    labels_series = labels.groupby("user_id")['movie_id'].agg(list)
    labels_series = labels_series.sample(frac=1, random_state=30)
    pivot = labels_series.shape[0] // 2
    val_data = labels_series[:pivot]
    test_data = labels_series[pivot:]
    return train_df, val_data, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="./dataset/ml-1m")
    args = parser.parse_args()
    directory = args.directory
    out_directory = "/dataset"

    # Load data
    users = []
    with open(os.path.join(directory, "users.dat"), encoding="latin1") as f:
        for l in f:
            id_, gender, age, occupation, zip_ = l.strip().split("::")
            users.append(
                {
                    "user_id": int(id_),
                    "gender": gender,
                    "age": age,
                    "occupation": occupation,
                    "zip": zip_,
                }
            )
    # users = pd.DataFrame(users).astype("category")
    users = pd.DataFrame(users)

    movies = []
    with open(os.path.join(directory, "movies.dat"), encoding="latin1") as f:
        for l in f:
            id_, title, genres = l.strip().split("::")
            genres_set = set(genres.split("|"))

            # extract year
            assert re.match(r".*\([0-9]{4}\)$", title)
            year = title[-5:-1]
            title = title[:-6].strip()

            data = {"movie_id": int(id_), "title": title, "year": year}
            for g in genres_set:
                data[g] = True
            movies.append(data)
    movies = pd.DataFrame(movies).astype({"year": "category"})

    # ratings
    ratings = []
    with open(os.path.join(directory, "ratings.dat"), encoding="latin1") as f:
        for l in f:
            user_id, movie_id, rating, timestamp = [
                int(_) for _ in l.split("::")
            ]
            ratings.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
    ratings = pd.DataFrame(ratings)

    # filter out less than 10 records
    print("before filter out")
    print(ratings.shape)

    while True:
        user_counts = ratings['user_id'].value_counts()
        movie_counts = ratings['movie_id'].value_counts()
        if (user_counts >= 10).all() and (movie_counts >= 10).all():
            break

        filtered_index = ratings[~ratings['user_id'].isin(user_counts[user_counts < 10].index) &
                            ~ratings['movie_id'].isin(movie_counts[movie_counts < 10].index)].index
        ratings = ratings.loc[filtered_index]

    print("after filter out")
    print(ratings.shape)

    # sort by timestamp
    ratings = sort_by_time(ratings)
    ratings['timestamp'] = ratings['timestamp'].dt.year





    # for i in range(6):
    #     # 단계별로 그래프 설정해주고..
    #     if i == 0:
    #         train_g = df2graph(users, movies, train_dfs[i])
    #         prev_g = train_g
    #         total_g = train_g
    #     else:
    #         train_g = df2graph(users, movies, train_dfs[i])
    #         prev_g = df2graph(users, movies, pd.concat(train_dfs[:i], axis=0))
    #         total_g = df2graph(users, movies,pd.concat(train_dfs[:i+1], axis=0))
    #
    #     # set dataset
    #     dataset = {
    #         "train_g" : train_g,
    #         "prev_g" : prev_g,
    #         "total_g" : total_g,
    #         "val_data" : val_dataset[i],
    #         "test_data" : test_dataset[i],
    #         "user-type": "user",
    #         "item-type": "movie",
    #         "user-to-item-type": "watched",
    #         "item-to-user-type": "watched-by",
    #         "timestamp-edge-column": "timestamp",
    #     }
    #
    #     # save data
    #     with open(os.path.join(out_directory, f"dataset{i}.pkl"), "wb") as f:
    #         pickle.dump(dataset, f)
