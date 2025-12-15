import random
import shutil
import numpy as np
import json
import pandas as pd
from cyipopt import minimize_ipopt

from models.types import *
from train.predictors import *
from train.utils import *
from opt.utils import *

import opt.nlp as nlp

import opt.opt_perx as opt_perx

SERVER_RANK_L_FRAC = 2 # 1 over this
SCENARIO = "s1"

NUM_SERVERS = [2, 4, 6, 8, 10]
TIME_LIMITS = [2, 4, 6, 15, 35]



SEED = [19, 21, 33, 31, 56, 5, 6, 93, 48, 76, 54, 11, 23, 94, 12]
START_IX = 0
NUM_RUNS = 5
BENCH_NUM_ITERS = 3
C_S = 32
CLEAR = False

APPLICATION_CATALOG = ApplicationCatalog.E

match APPLICATION_CATALOG:
    case ApplicationCatalog.V1:
        DS_FOLDER = (
            f"{HOME_PATH}/data_train_p/data_train_uniform_v1_test"
        )
    case ApplicationCatalog.V22:
        DS_FOLDER = f"{HOME_PATH}/data_train_p/data_train_uniform_v22_test"

service_predictors = {}

for server in range(2):
    if server == 0:
        WEIGHTS_FOLDER = f"{HOME_PATH}/server_1/results/models_mp75p95/neuro_e1e1oe2"
    elif server == 1:
        WEIGHTS_FOLDER = f"{HOME_PATH}/server_2/results/models_mp75p95/neuro"

    service_predictor = ServicePredictorNNSetReprHydra(
        use_base_param=True,
        embedding_type=EmbeddingTypes.USE_EMBEDDING_LAYER,
        application_catalog=APPLICATION_CATALOG,
    )
    service_predictor.load_weights(path=WEIGHTS_FOLDER, expand_embedd=False)

    service_predictors[server] = service_predictor


for ix, num_servers in enumerate(NUM_SERVERS):
    # num_runs = NUM_RUNS if num_servers <= 5 else 3
    num_runs = NUM_RUNS

    NUM_REQS = get_num_reqs(SCENARIO, num_servers)
    BENCH_TIMELIMIT = TIME_LIMITS[ix]

    RUN_DESC = {
        "params": {
            "SEED": SEED,
            "num_runs": num_runs,
            "num_servers": num_servers,
            "num_reqs": NUM_REQS,
            "C_S": C_S,
            "EPS": nlp.EPS,
            "ALGO_ITERS": nlp.ALGO_ITERS,
            "ROUND_TIERS": nlp.ROUND_ITERS,
            "BENCH_TIMELIMIT": BENCH_TIMELIMIT,
            "BENCH_NUM_ITERS": BENCH_NUM_ITERS,
        },
        "mer_info": {},
    }

    SAVE_RES = True
    SAVE_RES_PATH = f"{HOME_PATH}/results/test_{SCENARIO}_{num_servers}"

    if CLEAR:
        shutil.rmtree(SAVE_RES_PATH, ignore_errors=True)

    if not os.path.exists(SAVE_RES_PATH):
        os.mkdir(SAVE_RES_PATH)

    json.dump(
        RUN_DESC,
        open(os.path.join(SAVE_RES_PATH, "RUN_DESC.json"), "w"),
        indent=4,
    )

    try:
        results = pickle.load(open(os.path.join(SAVE_RES_PATH, "opt_results.pkl"), "rb"))
    except FileNotFoundError as e:
        results: List[Dict[str, OptResult]] = []

    # low to high index (or high to low ranks)
    SERVER_RANKS_L2H_ix = []

    num_h = max(1, num_servers // SERVER_RANK_L_FRAC)
    for _ in range(num_h):
        SERVER_RANKS_L2H_ix.append(1)

    for i in range(max(0, num_servers - num_h)):
        SERVER_RANKS_L2H_ix.append(0)

    # high to low index
    SERVER_RANKS_H2L_ix = list(reversed(SERVER_RANKS_L2H_ix))

    for run_ix in range(START_IX, num_runs):

        print("*" * 10, f"{num_servers=}, {run_ix=}")
        result = {}

        nlp.C_S = C_S
        nlp.SERVER_RANKS = SERVER_RANKS_L2H_ix
        opt_perx.C_S = C_S
        opt_perx.SCENARIO = SCENARIO
        opt_perx.SERVER_RANKS = SERVER_RANKS_L2H_ix


        requests_df = GEN_REQS_FUNCS.get(SCENARIO)(
            num=NUM_REQS,
            application_catalog=APPLICATION_CATALOG,
            seed=SEED[run_ix],
        )

        ### >>> benchmark
        opt_results_b = opt_perx.solve_server_instance(
            requests_df=requests_df,
            num_servers=num_servers,
            time_limit=BENCH_TIMELIMIT,
            num_iters=BENCH_NUM_ITERS,
        )

        result |= opt_results_b

        print("*" * 5, "benchmark")
        print(opt_results_b)

        ### <<< benchmark

        ### >>> neuro(1)
        opt_results = nlp.algo_solve_multi_server_method1(num_servers, requests_df, service_predictors)
        sol = opt_results["rnd"].solution
        sol.loc[sol.sum(axis=1) < 1] = 0

        opt_results["rnd"]._replace(solution=sol)
        rev1 = requests_df[opt_results["rnd"].solution.sum(axis=1) >= 1]["payment"].sum()

        opt_results["rnd"]._replace(obj=rev1)

        result["neuro (l)"] = opt_results["rnd"]

        ###

        # nlp.C_S = C_S
        # nlp.SERVER_RANKS = SERVER_RANKS_H
        # opt_perx.C_S = C_S
        # opt_perx.SCENARIO = SCENARIO
        # opt_perx.SERVER_RANKS = SERVER_RANKS_H

        # opt_results = nlp.algo_solve_multi_server_method1(num_servers, requests_df, service_predictors)
        # sol = opt_results["rnd"].solution
        # sol.loc[sol.sum(axis=1) < 1] = 0

        # opt_results["rnd"]._replace(solution=sol)
        # rev1 = requests_df[opt_results["rnd"].solution.sum(axis=1) >= 1]["payment"].sum()

        # opt_results["rnd"]._replace(obj=rev1)

        # result["neuro (h)"] = opt_results["rnd"]
        ### <<< neuro(1)

        results.append(result)

        if SAVE_RES:
            requests_df.to_csv(os.path.join(SAVE_RES_PATH, f"requests_r-{run_ix}.csv"))

            pickle.dump(results, open(os.path.join(SAVE_RES_PATH, "opt_results.pkl"), "wb"))