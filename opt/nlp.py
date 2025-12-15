import time
import random
from itertools import product
from typing import Iterable
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import coo_array
import json
import pandas as pd
from cyipopt import minimize_ipopt

from models.types import *
from train.predictors import *
from train.utils import *
from opt.utils import *

C_S = 32
EPS = 0.025
### abs penalty
# PENALTY = 2
### weighed by payments
PENALTY = 0.5
# PENALTY = 0.75
ROUND_ITERS = 15
ALGO_ITERS = 1

def cumul_z(z_row, s_ix):
    if isinstance(z_row, pd.Series):
        z = np.array([z_row[f"z_{s}"] for s in range(s_ix)])
    else:
        z = z_row[:s_ix]

    cumsum = np.cumsum(z)
    shifted_cumsum = np.concatenate(([0], cumsum[:-1]))
    remaining = 1 - shifted_cumsum
    return np.sum(z * remaining)


def flatten(mat: np.array, *numel:int) -> np.array:
    return mat.reshape(*numel, order="C")

def matify(arr: np.array, r:int, c:int) -> np.array:
    return arr.reshape(r, c, order="C")

def update_df(requests_df: pd.DataFrame, solution, service_predictor, update_preds=True, trim=False) -> pd.DataFrame:
    df = requests_df.copy(deep=True)

    df["num_instances"] *= solution
    # df["num_instances"] = df["num_instances"].round()

    if update_preds:
        dl = get_predictor_inputs(solution, df, trim=trim)

        update_preds = []
        for (_, row), (x, _) in zip(df.iterrows(), dl):
            ts, _ = service_predictor(x)
            ts = ts.squeeze()
            ts = ts[row["sla_type"]]
            update_preds.append(ts.item())

        df["pred_st"] = update_preds * df["exec_scaling"]

    return df


def compute_revenue_method2(requests_df: pd.DataFrame, solution: np.ndarray) -> float:
    rev = 0.0
    shape = solution.shape
    if len(shape) == 1:
        n_servers = 1
        solution = solution.reshape(-1, 1)
    else:
        n_servers = solution.shape[1]
        # solution = np.apply_along_axis(cumul_z, axis=1, arr=solution, s_ix=n_servers)
        # solution = solution.apply(cumul_z, axis=1, args=(n_servers,))
    for s_ix in range(n_servers):
        sol = solution[:, s_ix]
        rev += (requests_df["payment"] * sol).sum()

    return rev


def compute_revenue_method1(requests_df: pd.DataFrame, solution: np.ndarray) -> float:
    rev = 0.0
    shape = solution.shape
    if len(shape) == 1:
        n_servers = 1
        solution = solution.reshape(-1, 1)
    else:
        n_servers = solution.shape[1]
    for s_ix in range(n_servers):
        sol = solution[:, s_ix]
        rev += (requests_df["payment"] * sol).sum()

    return rev


def randomized_rounding(solution) -> np.array:
    rounded_solution = (
        (np.random.rand(solution.shape[0]) < solution)
        .astype(int)
        .reshape(
            -1,
        )
    )
    # rounded_solution = solution.copy()
    # rounded_solution[np.random.rand(solution.shape[0]) < (1 - solution)] = 0

    return rounded_solution

def round_best_ratio_greedy(
    df: pd.DataFrame, opt_result_rel: OptResult, service_predictor, constr_solution=[], n_iters: int = 100
) -> np.ndarray:
    print(len(df))
    best_ratio = 0
    best_rnd_result: OptResult = opt_result_rel

    relaxed_solution = opt_result_rel.solution
    # relaxed_solution[relaxed_solution < df["g_tau"]] = -1
    sort_ix = np.argsort(opt_result_rel.solution)

    solution_rnd = np.zeros_like(relaxed_solution)
    for ix in sort_ix[::-1]:
        # if relaxed_solution[ix] == -1:
        #     continue

        solution_rnd[ix] = 1

        df_rnd = update_df(df, solution_rnd, service_predictor, update_preds=True)

        rev = compute_revenue_method1(df_rnd, solution_rnd)
        cap = verify_sol_constr_capacity(df, solution_rnd)
        n_v, max_v = verify_sol_constr_latency(df_rnd, solution_rnd)

        if n_v > 0 or cap > C_S + 1e-1:
            solution_rnd[ix] = 0
        else:
            print("rounded", relaxed_solution[ix], n_v, cap)
            # if rev > best_ratio:
            best_ratio = rev
            best_rnd_result = best_rnd_result._replace(solution=solution_rnd)
            best_rnd_result = best_rnd_result._replace(obj=rev)

    return best_rnd_result


def round_up_frac(x, a, cs=1.0) -> float:
    if cs != 1.0:
        max_val = np.floor(cs / a) * a
    else:
        max_val = 1.0

    rnd = min(np.ceil(x / a) * a, max_val)
    return rnd


# def round_up_frac(x, a, cs=1.0) -> float:
#     max_val = cs
#     return min(np.ceil(x / a) * a, max_val)

def round_down_frac(x, a) -> float:
    return max(np.floor(x / a) * a, 0.0)

def round_down_method1(requests_df:pd.DataFrame, solution: pd.DataFrame, service_predictor) -> pd.DataFrame:
    solution = solution.copy(deep=True)

    zr = solution.sum(axis=1)
    ids = solution.loc[(zr > 0) & (zr < 1 - EPS)].index

    print("rounding down", len(ids), "lost rev.", (requests_df.loc[ids, "payment"] * (solution.loc[ids].sum(axis=1))).sum())

    print("* revenue before rounding down", compute_revenue_method1(requests_df, solution.to_numpy()))

    solution.loc[ids] = 0

    print("* revenue after rounding down", compute_revenue_method1(requests_df, solution.to_numpy()))

    return solution


def round_down_method2(requests_df:pd.DataFrame, solution: pd.DataFrame, service_predictor) -> pd.DataFrame:
    # zr = solution.apply(cumul_z, axis=1, args=(solution.shape[1],))
    zr = solution.sum(axis=1)
    ids = solution.loc[(zr > 0) & (zr < 1 - EPS)].index

    print("num. partially served reqs", len(ids))

    solution.loc[ids] = 0

    return solution


def merge_final_method(requests_df:pd.DataFrame, solution: pd.DataFrame, service_predictor) -> pd.DataFrame:

    solution = solution.copy(deep=True)

    print("* revenue before merge", compute_revenue_method1(requests_df, solution.to_numpy()))
    num_servers = solution.shape[1]

    zr = solution.sum(axis=1)

    ids = solution.loc[(zr > 0) & (zr < 1 - EPS)].index
    print(ids)

    part_requests = requests_df.loc[ids]

    tenant_reqs_group = part_requests.groupby(by="application")

    for app, tenant_group in tenant_reqs_group:
        avail_tenant_ids = tenant_group.index.tolist()

        avail_tenant_group = tenant_group.loc[avail_tenant_ids]

        for i, id_a in enumerate(avail_tenant_ids):
            for j, id_b in enumerate(avail_tenant_ids):
                tenant_a = tenant_group.loc[id_a]
                tenant_b = tenant_group.loc[id_b]

                z_a_sol = solution.loc[id_a]
                z_b_sol = solution.loc[id_b]

                z_a = z_a_sol.sum()
                z_b = z_b_sol.sum()

                if id_a == id_b or z_a >= 1 or z_b <= 0 or z_b >=1:
                    continue

                N_a = tenant_a["num_instances"]
                N_b = tenant_b["num_instances"]

                # merge b to a (borrow from b)
                if tenant_b["arrival_rate"] >= tenant_a["arrival_rate"] and tenant_b["delay"] <= tenant_a["delay"]:
                    print("found a match", id_a, id_b, "a:", solution.loc[id_a].sum())
                    for s_ix in range(num_servers):
                        z_a_sol = solution.loc[id_a]
                        z_b_sol = solution.loc[id_b]

                        z_a = z_a_sol.sum()
                        z_b = z_b_sol.sum()

                        if z_a >= 1.0 or z_b <= 0:
                            break

                        if z_b_sol[f"z_{s_ix}"] <= 0:
                            continue

                        needed_a = N_a * (1 - z_a)
                        used_b = N_b * z_b_sol[f"z_{s_ix}"]

                        borrow_a = min(needed_a, used_b)

                        old_a = z_a_sol.loc[f"z_{s_ix}"]
                        old_b = z_b_sol.loc[f"z_{s_ix}"]
                        z_a_sol.loc[f"z_{s_ix}"] += borrow_a / N_a
                        z_b_sol.loc[f"z_{s_ix}"] -= borrow_a / N_b

                        df = update_df(requests_df, solution.loc[:, f"z_{s_ix}"], service_predictor)
                        nv, mv, vdf = verify_sol_constr_latency(df, solution.loc[:, f"z_{s_ix}"], ret_df=True)

                        if nv != 0:
                            z_a_sol.loc[f"z_{s_ix}"] = old_a
                            z_b_sol.loc[f"z_{s_ix}"] = old_b

                        else:
                            print("* verify constr", nv, mv)
                            print("updated tenant a", solution.loc[id_a].sum())

    # lost_rev = nlp.calc_lost_rev_lat_v(requests_df, solution, service_predictor)

    # print(lost_rev)

    for s_ix in range(num_servers):
        sol = solution[f"z_{s_ix}"]
        df = update_df(requests_df, sol, service_predictor)
        nv, mv, vdf = verify_sol_constr_latency(df, sol, ret_df=True)
        print(vdf.index)

    print("* revenue after merge", compute_revenue_method1(requests_df, solution.to_numpy()))

    return solution


def merge_final_method_single(
    requests_df: pd.DataFrame, solution: pd.Series, service_predictor, constr_solution: List=[]
) -> pd.DataFrame:

    # allocated
    if len(constr_solution) == 0:
        constr_solution = np.zeros(shape=(len(requests_df,)))
    else:
        constr_solution = 1 - np.array(constr_solution)

    requests_df = requests_df.copy(deep=True)
    requests_df["constr_sol"] = constr_solution

    solution = solution.copy(deep=True)

    print("** revenue before merge", compute_revenue_method1(requests_df, solution.to_numpy()))

    zr = solution.sum(axis=1)

    ids = solution.loc[(zr > 0) & (zr < 1 - EPS)].index
    print(ids)

    part_requests = requests_df.loc[ids]

    tenant_reqs_group = part_requests.groupby(by="application")

    for app, tenant_group in tenant_reqs_group:
        avail_tenant_ids = tenant_group.index.tolist()

        avail_tenant_group = tenant_group.loc[avail_tenant_ids]

        print(app, avail_tenant_ids)
        for i, id_a in enumerate(avail_tenant_ids):
            for j, id_b in enumerate(avail_tenant_ids):
                tenant_a = tenant_group.loc[id_a]
                tenant_b = tenant_group.loc[id_b]

                z_a_sol = solution.loc[id_a, 0]
                z_b_sol = solution.loc[id_b, 0]

                if (
                    id_a == id_b
                    or z_a_sol + tenant_a["constr_sol"] >= 1
                    or z_b_sol <= 0
                    or tenant_a["sla_type"] > tenant_b["sla_type"]
                    # or z_a_sol >= 1
                ):
                    continue

                N_a = tenant_a["num_instances"]
                N_b = tenant_b["num_instances"]

                # merge b to a (borrow from b)
                if (
                    tenant_b["arrival_rate"] >= tenant_a["arrival_rate"]
                    and tenant_b["delay"] <= tenant_a["delay"]
                ):
                    # print("found a match", id_a, id_b, "a:", solution.loc[id_a].sum())
                    z_a_sol = solution.loc[id_a, 0]
                    z_b_sol = solution.loc[id_b, 0]

                    if z_a_sol + tenant_a["constr_sol"] >= 1.0 or z_b_sol <= 0:
                        break

                    if z_b_sol <= 0:
                        continue

                    needed_a = N_a * (1 - (z_a_sol + tenant_a["constr_sol"]))
                    used_b = N_b * z_b_sol

                    borrow_a = min(needed_a, used_b)

                    old_a = z_a_sol
                    old_b = z_b_sol
                    solution.loc[id_a, 0] += borrow_a / N_a
                    solution.loc[id_b, 0] -= borrow_a / N_b

                    # df = update_df(
                    #     requests_df, solution[0].to_numpy(), service_predictor
                    # )
                    # nv, mv, vdf = verify_sol_constr_latency(
                    #     df, solution[0], ret_df=True
                    # )

                    # if nv != 0:
                    #     print("* verify constr", (id_a, id_b), nv, mv)

                    #     solution.loc[id_a, 0] = old_a
                    #     solution.loc[id_b, 0] = old_b

                    # else:
                    #     print("* verify constr", (id_a, id_b), nv, mv)
                    #     print("+++ updated tenant a", app, solution.loc[id_a, 0])

    # lost_rev = nlp.calc_lost_rev_lat_v(requests_df, solution, service_predictor)

    # print(lost_rev)

    # for s_ix in range(num_servers):
    #     sol = solution[f"z_{s_ix}"]
    #     df = update_df(requests_df, sol, service_predictor)
    #     nv, mv, vdf = verify_sol_constr_latency(df, sol, ret_df=True)
    #     print(vdf.index)

    print("** revenue after merge", compute_revenue_method1(requests_df, solution.to_numpy()))

    return solution


def round_final_method12(requests_df:pd.DataFrame, solution: pd.DataFrame, service_predictor) -> pd.DataFrame:
    solution = solution.copy(deep=True)
    zr = solution.sum(axis=1)

    ids = solution.loc[(zr > 0) & (zr < 1 - EPS)].index
    print("round final", len(ids), "lost rev.", (requests_df.loc[ids, "payment"] * (solution.loc[ids].sum(axis=1))).sum())

    print("* revenue before final", compute_revenue_method1(requests_df, solution.to_numpy()))

    zr_part = zr.loc[ids].to_numpy()
    sort_ix = np.argsort(zr_part)
    sort_ix = sort_ix[::-1]

    # for id in ids:
    for ix in sort_ix:
        id = ids[ix]
        z = solution.loc[id]

        sort_ix = np.argsort(z)
        sort_ix = sort_ix[::-1]

        for s_ix in sort_ix:
            z_rs = z[f"z_{s_ix}"]

            if z_rs == 0:
                continue

            z_s = solution.loc[:, f"z_{s_ix}"].to_numpy(copy=True).squeeze()
            z_s[ix] = 1

            df = update_df(requests_df, z_s, service_predictor)

            cap = verify_sol_constr_capacity(requests_df, z_s)
            num_v, max_v = verify_sol_constr_latency(df, z_s)

            if cap <= C_S and num_v == 0:
                solution.loc[id] = 0
                solution.loc[id, f"z_{s_ix}"] = 1

                print("rounded", id, s_ix, requests_df.loc[id, "application"], requests_df.loc[id, "sla_type"], z_rs, z.sum())

            else:
                print("failed", id, s_ix, requests_df.loc[id, "application"], requests_df.loc[id, "sla_type"], z_rs, z.sum())
                solution.loc[id, f"z_{s_ix}"] = 0

    print("* revenue after final", compute_revenue_method1(requests_df, solution.to_numpy()))

    return solution


def round_final_method12_iter(requests_df:pd.DataFrame, solution: pd.DataFrame, service_predictor) -> pd.DataFrame:
    solution = solution.copy(deep=True)

    zr = solution.sum(axis=1)

    ids = solution.loc[(zr > 0) & (zr < 1 - EPS)].index
    solution.loc[zr < 0.5] = 0
    print("round final", len(ids), "lost rev.", (requests_df.loc[ids, "payment"] * (solution.loc[ids].sum(axis=1))).sum())

    print("* revenue before final", compute_revenue_method1(requests_df, solution.to_numpy()))

    for id in ids:
        z = solution.loc[id]

        sort_ix = np.argsort(z)
        sort_ix = sort_ix[::-1]

        
        while True:
            flag = False
            for s_ix in sort_ix:
                z_rs = z[f"z_{s_ix}"]

                if z_rs == 0:
                    continue

                z_s = solution.loc[:, f"z_{s_ix}"].to_numpy(copy=True).squeeze()
                # z_s[id - 1] = 1

                z_s_original = z_s[id - 1]
                frac = requests_df.loc[id, "g_tau"]
                if z_s_original % frac == 0:
                    z_s[id - 1] = z_s_original + frac

                else:
                    z_s[id - 1] = round_up_frac(z_s_original, frac)

                df = update_df(requests_df, z_s, service_predictor)

                cap = verify_sol_constr_capacity(requests_df, z_s)
                num_v, max_v = verify_sol_constr_latency(df, z_s)

                if cap <= C_S and num_v == 0:
                    flag = True
                    solution.loc[id, f"z_{s_ix}"] = z_s[id - 1]
                    # solution.loc[id] = 0
                    # solution.loc[id, f"z_{s_ix}"] = 1

                    print("rounded", id, s_ix, requests_df.loc[id, "application"], z_s_original, z_s[id - 1])

                else:
                    solution.loc[id, f"z_{s_ix}"] = z_s_original
                    print("failed", id, s_ix, requests_df.loc[id, "application"], requests_df.loc[id, "num_instances"], z_rs, z.sum(), requests_df.loc[id, "delay"], requests_df.loc[id, "arrival_rate"])
                    # solution.loc[id, f"z_{s_ix}"] = 0
                
            if solution.loc[id].sum() >= 1.0 or not flag:
                break
        
        if solution.loc[id].sum() < 1.0:
            print("resetting", id)
            solution.loc[id, :] = 0

    print("* revenue after final", compute_revenue_method1(requests_df, solution.to_numpy()))

    return solution


def round_best_ratio_greedy_new(
    df: pd.DataFrame,
    opt_result_rel: OptResult,
    service_predictor,
    constr_solution=[],
    n_iters: int = 100,
) -> np.ndarray:
    if len(constr_solution) == 0:
        constr_solution = np.ones(shape=(len(df),))
    else:
        constr_solution = np.array(constr_solution)

    best_ratio = opt_result_rel.obj_raw
    best_rnd_result: OptResult = opt_result_rel

    relaxed_solution = opt_result_rel.solution.squeeze()
    # relaxed_solution[relaxed_solution < df["g_tau"]] = 0
    # relaxed_solution[relaxed_solution > 1 - EPS] = 1

    rev = compute_revenue_method1(df, relaxed_solution)
    best_rnd_result = OptResult(
        solution=relaxed_solution,
        obj_raw=rev,
    )

    solution_rnd = relaxed_solution.copy()

    metric = solution_rnd * df["payment"]
    sort_ix = np.argsort(metric)
    sort_ix = sort_ix[::-1]

    # for ix in range(len(relaxed_solution)):
    for ix in sort_ix:
        g_tau = df.iloc[ix]["g_tau"]

        # solution_rnd[ix] = 1.0

        # df_rnd = update_df(df, solution_rnd, service_predictor, update_preds=True)

        # rev = compute_revenue_method1(df_rnd, solution_rnd)
        # cap = verify_sol_constr_capacity(df, solution_rnd)
        # n_v, max_v = verify_sol_constr_latency(df_rnd, solution_rnd, strict=True)

        # if n_v > 0 or cap > C_S + 1e-1:
        solution_rnd[ix] = round_up_frac(relaxed_solution[ix], g_tau, constr_solution[ix])

        if np.abs(solution_rnd[ix] - relaxed_solution[ix]) <= EPS:
            print("EPS:", df.iloc[ix]["application"], "rel", relaxed_solution[ix], "rnd", solution_rnd[ix], "g_tau", g_tau)
            continue


        df_rnd = update_df(df, solution_rnd, service_predictor, update_preds=True)

        rev = compute_revenue_method1(df_rnd, solution_rnd)
        cap = verify_sol_constr_capacity(df, solution_rnd)
        n_v, max_v = verify_sol_constr_latency(df_rnd, solution_rnd, strict=True)

        if n_v > 0 or cap > C_S + 1e-1:

            # solution_rnd[ix] = relaxed_solution[ix]
            solution_rnd[ix] = round_down_frac(relaxed_solution[ix], g_tau)
            print(df.iloc[ix]["application"], "rel", relaxed_solution[ix], "rnd", solution_rnd[ix], "g_tau", g_tau)
            # print(
            #     "- rounded down frac",
            #     df.iloc[ix].name,
            #     df.iloc[ix]["application"],
            #     relaxed_solution[ix],
            #     solution_rnd[ix],
            #     g_tau,
            # )

            rev = compute_revenue_method1(df_rnd, solution_rnd)
            best_rnd_result = best_rnd_result._replace(obj=rev)
            best_rnd_result = best_rnd_result._replace(solution=solution_rnd)

        else:
            # print(
            #     "+ rounded up frac",
            #     df.iloc[ix].name, df.iloc[ix]["application"],
            #     relaxed_solution[ix],
            #     solution_rnd[ix],
            #     g_tau,
            # )

            # print("rounded", n_v, cap, rev)
            best_ratio = rev
            best_rnd_result = best_rnd_result._replace(solution=solution_rnd)
            best_rnd_result = best_rnd_result._replace(obj=rev)
        # else:
        #     best_ratio = rev
        #     best_rnd_result = best_rnd_result._replace(solution=solution_rnd)
        #     best_rnd_result = best_rnd_result._replace(obj=rev)

    return best_rnd_result


def round_best_ratio(
    df: pd.DataFrame, opt_result_rel: OptResult, service_predictor, constr_solution=[], n_iters: int = 100
) -> OptResult:
    if len(constr_solution) == 0:
        constr_solution = np.zeros_like(opt_result_rel.solution)

    best_ratio = 0
    best_rnd_result: OptResult = None
    for _ in range(n_iters):
        solution_rnd = randomized_rounding(opt_result_rel.solution)
        # solution_rnd = np.round(opt_result_rel.solution)

        if constr_solution is not None:
            solution_rnd[constr_solution == 1] = 1

        df_rnd = update_df(df, solution_rnd, service_predictor, update_preds=True)

        rev = compute_revenue_method1(df_rnd, solution_rnd)
        cap = verify_sol_constr_capacity(df_rnd, solution_rnd)
        n_v, max_v = verify_sol_constr_latency(df_rnd, solution_rnd)

        # ix = df_rnd["pred_st"] > df_rnd["delay"]

        # df_rel = update_df(df, opt_result_rel.solution, service_predictor, update_preds=True)
        # df_rel["temp"] = opt_result_rel.solution * df_rel["pred_st"]
        # df_rel["solution"] = opt_result_rel.solution

        # print(
        #     df_rel.loc[(ix == True) & (solution_rnd == 1)][
        #         ["solution", "num_instances", "temp", "pred_st", "delay"]
        #     ]
        # )
        # print(
        #     df_rnd.loc[(ix == True) & (solution_rnd == 1)][["num_instances", "pred_st", "delay"]]
        # )

        # print("** before", rev, n_v, cap)

        if n_v > 0 or cap > C_S:
            # start eliminating
            while n_v > 0 or cap > C_S + 1e-1:
                # temp_df = df_rnd.loc[solution_rnd]

                # min_value = temp_df["payment"].min()
                # min_rows = temp_df[temp_df["payment"] == min_value]

                temp_df = df.loc[(solution_rnd == 1)]
                # temp_df = df.loc[(solution_rnd == 1) & (constr_solution == 0)]
                weights = 1 - opt_result_rel.solution[solution_rnd == 1]
                weights[weights < 0] = 0

                row = temp_df.sample(n=1, weights=weights)
                solution_rnd[row.index[0] - 1] = 0

                df_rnd = update_df(df, solution_rnd, service_predictor)

                rev = compute_revenue_method1(df_rnd, solution_rnd)
                cap = verify_sol_constr_capacity(df_rnd, solution_rnd)
                n_v, max_v = verify_sol_constr_latency(df_rnd, solution_rnd)

                # print(" * eliminating", n_v, cap)

        # print("** after", rev, n_v, cap)

        if n_v == 0 and cap <= C_S:
            ratio = rev
            # ratio = rev / np.count_nonzero(solution_rnd)
            # ratio = rev / cap
            # print(rev, cap, n_v)
            if ratio > best_ratio:
                best_ratio = ratio
                best_rnd_result = opt_result_rel._replace(obj=rev)
                best_rnd_result = best_rnd_result._replace(solution=solution_rnd)

    return best_rnd_result

def verify_sol_constr_capacity(requests_df: pd.DataFrame, solution) -> float:
    return (solution * requests_df["num_instances"] * requests_df["app_mem"]).sum()

# def verify_sol_constr_latency(requests_df: pd.DataFrame, solution) -> int:
#     sat = ((requests_df.loc[solution]["pred_st"] <= requests_df.loc[solution]["delay"]) == True)

#     n_violations = np.count_nonzero(1 - sat)

#     return n_violations


def verify_sol_constr_latency(df: pd.DataFrame, solution, strict=False, ret_df=False, trim=False) -> int:

    df["solution"] = solution

    if strict:
        sat_ix = df["pred_st"] <= df["delay"]
    else:
        sat_ix = solution * df["pred_st"] <= df["delay"]

    max_v = np.max(
        np.abs(-df.loc[sat_ix == False]["delay"] - solution[sat_ix == False] * df.loc[sat_ix == False]["pred_st"])
    )

    n_violations = np.count_nonzero(sat_ix == False)

    if ret_df:
        violations_df = df.loc[sat_ix == False]

        return n_violations, max_v, violations_df
    else:
        return n_violations, max_v

def calc_lost_rev_lat_v(request_df, solution_df: pd.DataFrame, service_predictor):
    n_servers = solution_df.shape[1]

    lost_rev = 0.0
    for s_ix in range(n_servers):
        z_r_s = solution_df[f"z_{s_ix}"]

        df = update_df(request_df, z_r_s, service_predictor)

        nv, mv, vdf = verify_sol_constr_latency(df, z_r_s, ret_df=True)

        lost_rev += (z_r_s * vdf["payment"]).sum()
    
    return lost_rev


def solve_server_instance(requests_df: pd.DataFrame, service_predictor: CustomDatasetSetReprHydra, constr_solution: List=[], num_servers: int=1, server_ix: int = 0, freeze_solution: np.array = None) -> OptResult:
    if len(constr_solution) > 0:
        constr_solution = np.array(constr_solution).reshape(-1,)
    else:
        constr_solution = np.zeros(shape=(len(requests_df,)))
    
    if freeze_solution is not None:
        freeze_solution = np.array(freeze_solution).reshape(-1,)
    else:
        freeze_solution = np.zeros(shape=(len(requests_df,)))

    NUM_REQS = len(requests_df)

    # tenant_ids = np.array([i for i in range(1, NUM_REQS+1)])
    tenant_ids = requests_df.index

    r_apps = requests_df.loc[tenant_ids, "application"].to_numpy().reshape(-1,)
    r_exec_scaling = (
        requests_df.loc[tenant_ids, "exec_scaling"]
        .to_numpy()
        .reshape(
            -1,
        )
    )
    r_payments = (
        requests_df.loc[tenant_ids, "payment"]
        .to_numpy()
        .reshape(
            -1,
        )
    )
    r_app_mems = requests_df.loc[tenant_ids, "app_mem"].to_numpy().reshape(-1,)
    r_num_instances = requests_df.loc[tenant_ids, "num_instances"].to_numpy().reshape(-1,)
    r_arrival_rate_per = requests_df.loc[tenant_ids, "arrival_rate_per"].to_numpy().reshape(-1,)
    r_delays = requests_df.loc[tenant_ids, "delay"].to_numpy().reshape(-1,)
    r_g_tau = requests_df.loc[tenant_ids, "g_tau"].to_numpy().reshape(-1,)
    r_sla_type = requests_df.loc[tenant_ids, "sla_type"].to_numpy().reshape(-1,)

    def objective(z: np.ndarray):
        ### vanilla
        return -np.dot(r_payments, z * (1 + PENALTY * constr_solution))

        ### original - abs penalty
        # return -np.dot(r_payments, z) + PENALTY * np.sum(z * (1 - z))

        ### abs penalty, cumulative z
        # return -np.dot(r_payments, z) + PENALTY * np.sum(
        #     (z + constr_solution) * (1 - (z + constr_solution)) * r_payments
        # )

        ### original - penalty weighed by payments
        # return -np.dot(r_payments, z) + PENALTY * np.dot(z * (1 - z), r_payments)

        ### cumulative penalty
        # return -np.dot(r_payments, z) + PENALTY * np.dot(
        #     (1) * (z + constr_solution) * (1 - (z + constr_solution)), np.ones_like(z)
        # )

        ### even dist
        # return -np.dot(r_payments, z) + PENALTY * np.dot((1 / num_servers - z) ** 2, r_payments)
        # return -np.dot(r_payments, z) + PENALTY * np.dot(z * (1 / num_servers - z) ** 2, r_payments)

    def grad_objective(z: np.ndarray):
        ### vanilla
        return -r_payments * (1 + PENALTY * constr_solution)

        ### original - abs penalty
        # return -r_payments + PENALTY * (1 - 2 * z)

        ### abs penalty, cumulative z
        # return -r_payments + PENALTY * (1 - 2 * (z + constr_solution)) * r_payments

        ### original - penalty weighed by payments
        # return -r_payments + PENALTY * (1 - 2 * z) * r_payments

        ### cumulative penalty
        # return -r_payments + PENALTY * (1) * (1 - 2 * (z + constr_solution)) * np.ones_like(z)

        ### even dist
        # return -r_payments + PENALTY * -2 * (1 / num_servers - z) * r_payments
        # return -r_payments + PENALTY * ((1 - z) ** 2 -2 * (1 / num_servers - 2 * z)) * r_payments

    def constr_capacity(z: np.ndarray):
        used_capacity = np.dot(r_app_mems, (z * r_num_instances))
        return np.array([-used_capacity + C_S])

    def jac_constr_capacity(z: np.ndarray):
        jac = -1 * (r_app_mems * r_num_instances)
        return jac

    def constr_delay(z: np.ndarray):
        dl = get_predictor_inputs(z, requests_df)

        preds = []
        with th.no_grad():
            for i, (x, _) in enumerate(dl):
                ts, _ = service_predictor(x)
                ts = ts.squeeze()
                preds.append(ts[r_sla_type[i]].item())

        preds = np.array(preds)
        preds *= r_exec_scaling

        # print("delay eval", np.count_nonzero(z * preds > r_delays))
        constr = -preds + r_delays
        constr = constr.reshape(NUM_REQS,)

        # if server_ix == 0:
        #     print("***", r_apps[15], z[15], constr[15])
        #     print("***", r_apps[20], z[20], constr[20])
        return constr

    def jac_constr_delay(z: np.ndarray):
        dl = get_predictor_inputs(z, requests_df)

        # preds = []
        # grads = []
        jacs = np.zeros(shape=(NUM_REQS, NUM_REQS))
        for i, (x, _) in enumerate(dl):
            # length = x["lengths"]
            # x["x_features"] = x["x_features"][:, :length]

            for k, v in x.items():
                if k != "lengths":
                    v.requires_grad_(True)

            ts, _ = service_predictor(x)
            ts = ts.squeeze()
            ts = ts[r_sla_type[i]]
            # preds.append(ts[r_sla_type[i]].item())
            length = x["lengths"]
            # grad_f = th.autograd.grad(ts, x["x_features"], retain_graph=True)
            # grad_c = th.autograd.grad(ts, x["features"], retain_graph=True)

            grad_f, grad_c = th.autograd.grad(ts, [x["x_features"], x["features"]], retain_graph=True, allow_unused=True)

            # app_index -> grad
            grad_c_map = {}
            for ix in range(length):

                grad_c_map[x["features"][0][0][ix].item()] = grad_c[0][1][ix]

            jacs[i][i] = -grad_f[0][1].item() * r_num_instances[i] / 16 * r_exec_scaling[i]

            for j in range(NUM_REQS):
                if j == i:
                    continue

                r_app_j = r_apps[j]
                r_app_j_ix = APPLICATIONS_E.index(r_app_j)

                if r_app_j_ix not in grad_c_map:
                    jacs[i][j] = 0
                else:
                    jacs[i][j] = -grad_c_map[r_app_j_ix] * r_exec_scaling[i] * r_num_instances[j] / 16

        # preds = np.array(preds)
        # preds *= r_exec_scaling
        # jacs = np.array(jacs)
        # jacs *= r_exec_scaling

        # rows = []
        # cols = []
        # data = []
        # for ix in range(NUM_REQS):
        #     rows.append(ix)
        #     cols.append(ix)
        #     data.append(jac[ix])

        # rows = np.array(rows)
        # cols = np.array(cols)
        # data = np.array(data)
        # return coo_array((data, (rows, cols)))

        # jacs = jacs * r_num_instances.reshape(-1, 1).repeat(NUM_REQS, axis=1).T
        # jacs /= 16

        # for ix in range(NUM_REQS):
        #     jacs[ix][ix] = -1 * (jacs[ix][ix])

        # jac = -1 * (preds + z * jacs * r_num_instances.repeat(repeats=NUM_REQS, axis=0))

        return jacs

    ### old impl (incorrect: off diag are missing)
    # def jac_constr_delay(z: np.ndarray):
    #     dl = get_predictor_inputs(z, requests_df)

    #     preds = []
    #     grads = []
    #     for x, _ in dl:
    #         for k, v in x.items():
    #             if k != "lengths":
    #                 v.requires_grad_(True)
    #         ts, _ = service_predictor(x)
    #         preds.append(ts.item())
    #         grad = th.autograd.grad(ts, x["x_features"], retain_graph=True)
    #         grads.append(grad[0][0][1])

    #     preds = np.array(preds)
    #     preds *= r_apps_base_exec
    #     grads = np.array(grads)
    #     grads *= r_apps_base_exec

    #     jac = -1 * (preds + z * grads * r_num_instances)

    #     return jac.reshape(1, -1)

    # z0 = np.zeros(shape=(NUM_REQS,))
    z0 = []
    bounds = []
    for ix in range(NUM_REQS):
        if freeze_solution[ix] != 0:
            bounds.append((freeze_solution[ix] - 1e-2, freeze_solution[ix]))
            z0.append(freeze_solution[ix] - 1e-2 / 2)
        else:
            if constr_solution[ix] != 0:
                # bounds.append((1-EPS, 1))
                # z0.append(1 - EPS)
                bounds.append((0, constr_solution[ix]))
                z0.append(0)
            else:
                bounds.append((0, 1))
                z0.append(0)
    # bounds = [(0, 1) for _ in range(NUM_REQS)]

    z0 = np.array(z0).reshape(-1,)
    constraints = [
        {"type": "ineq", "fun": constr_capacity, "jac": jac_constr_capacity},
        {"type": "ineq", "fun": constr_delay, "jac": jac_constr_delay},
    ]

    start = time.perf_counter()
    result = minimize_ipopt(
        fun=objective,
        x0=z0,
        jac=grad_objective,
        constraints=constraints,
        options={
            "max_iter": 3000,
            # "alpha_min_frac": 0.001,
            # "mu_strategy": "adaptive",
            # "derivative_test": "second-order",
            # "print_level": 12,
            # "jacobian_approximation": "finite-difference-values",
            # "gradient_approximation": "finite-difference-values",
            # "max_soc": 10,
            # "tol": 1e-3,
            # "dual_inf_tol": 1e-6,
            # "constr_viol_tol": 1e-3,  # tolerance for constraint violations
            # "tol": 1e-9,  # tolerance for convergence
            # "dual_inf_tol": 1e-9,  # tolerance for dual infeasibility
        },
        bounds=bounds,
    )
    end = time.perf_counter()

    opt_result = OptResult(
        status=result.status,
        message=result.message,
        obj_raw=result.fun,
        obj=result.fun,
        exec_time=end - start,
        solution=result.x,
        num_iters=result.nit,
    )

    return opt_result


def solve_server_instance_recl(
    requests_df: pd.DataFrame,
    service_predictor: CustomDatasetSetReprHydra,
    constr_solution: List = [],
) -> OptResult:
    if len(constr_solution) > 0:
        constr_solution = np.array(constr_solution).reshape(
            -1,
        )
    else:
        constr_solution = np.zeros(
            shape=(
                len(
                    requests_df,
                )
            )
        )

    NUM_REQS = len(requests_df)

    # tenant_ids = np.array([i for i in range(1, NUM_REQS+1)])
    tenant_ids = requests_df.index

    r_apps = (
        requests_df.loc[tenant_ids, "application"]
        .to_numpy()
        .reshape(
            -1,
        )
    )
    r_apps_base_exec = (
        requests_df.loc[tenant_ids, "app_base_exec"]
        .to_numpy()
        .reshape(
            -1,
        )
    )
    r_payments = (
        requests_df.loc[tenant_ids, "payment"]
        .to_numpy()
        .reshape(
            -1,
        )
    )
    r_app_mems = (
        requests_df.loc[tenant_ids, "app_mem"]
        .to_numpy()
        .reshape(
            -1,
        )
    )
    r_num_instances = (
        requests_df.loc[tenant_ids, "num_instances"]
        .to_numpy()
        .reshape(
            -1,
        )
    )
    r_arrival_rate_per = (
        requests_df.loc[tenant_ids, "arrival_rate_per"]
        .to_numpy()
        .reshape(
            -1,
        )
    )
    r_delays = (
        requests_df.loc[tenant_ids, "delay"]
        .to_numpy()
        .reshape(
            -1,
        )
    )
    r_g_tau = (
        requests_df.loc[tenant_ids, "g_tau"]
        .to_numpy()
        .reshape(
            -1,
        )
    )

    def objective(z: np.ndarray):
        return -np.dot(r_payments, z) + PENALTY * np.sum(z * (1 - z))
        # return -np.dot(r_payments, z) + PENALTY * np.sum(z ** 2) + np.sum((z - 1) ** 2)
        # return -np.dot(r_payments, g_z * (1 - constr_solution))
        # return -np.dot(r_payments, z * (1 - constr_solution)) + np.sum(z * (1 - z))

    def grad_objective(z: np.ndarray):
        return -r_payments + PENALTY * (1 - 2 * z)
        # return -r_payments + PENALTY * (4 * z - 2)
        # return -r_payments * -SIG_A * SIG_B * g_z * (1-g_z) * (1 - constr_solution)
        # return -r_payments * (1 - constr_solution) + 1 - 2 * z

    def constr_capacity(z: np.ndarray):
        used_capacity = np.dot(r_app_mems, (z * r_num_instances))
        return np.array([-used_capacity + C_S])

    def jac_constr_capacity(z: np.ndarray):
        jac = -1 * (r_app_mems * r_num_instances)
        return jac

    def constr_delay(z: np.ndarray):
        dl = get_predictor_inputs(z, requests_df)

        preds = []
        with th.no_grad():
            for x, _ in dl:
                ts, _ = service_predictor(x)
                preds.append(ts.item())

        preds = np.array(preds)
        preds *= r_apps_base_exec

        # print("delay eval", np.count_nonzero(z * preds > r_delays))
        constr = -preds + r_delays
        constr = constr.reshape(
            NUM_REQS,
        )
        return constr

    def jac_constr_delay(z: np.ndarray):
        dl = get_predictor_inputs(z, requests_df)

        preds = []
        # grads = []
        jacs = np.zeros(shape=(NUM_REQS, NUM_REQS))
        for i, (x, _) in enumerate(dl):
            for k, v in x.items():
                if k != "lengths":
                    v.requires_grad_(True)

            ts, _ = service_predictor(x)
            preds.append(ts.item())
            length = x["lengths"]
            # grad_f = th.autograd.grad(ts, x["x_features"], retain_graph=True)
            # grad_c = th.autograd.grad(ts, x["features"], retain_graph=True)

            grad_f, grad_c = th.autograd.grad(
                ts, [x["x_features"], x["features"]], retain_graph=True
            )

            grad_c_map = {}
            for ix in range(length):
                grad_c_map[x["features"][0][0][ix].item()] = grad_c[0][1][ix]

            jacs[i][i] = grad_f[0][1].item()

            for j in range(NUM_REQS):
                if j == i:
                    continue

                r_app_j = r_apps[j]
                r_app_j_ix = APPLICATIONS_V22.index(r_app_j)

                jacs[i][j] = z[i] * grad_c_map[r_app_j_ix]

        preds = np.array(preds)
        preds *= r_apps_base_exec
        jacs = np.array(jacs)
        jacs *= r_apps_base_exec

        # rows = []
        # cols = []
        # data = []
        # for ix in range(NUM_REQS):
        #     rows.append(ix)
        #     cols.append(ix)
        #     data.append(jac[ix])

        # rows = np.array(rows)
        # cols = np.array(cols)
        # data = np.array(data)
        # return coo_array((data, (rows, cols)))

        jacs = jacs * r_num_instances.reshape(-1, 1).repeat(NUM_REQS, axis=1).T
        jacs /= 16

        for ix in range(NUM_REQS):
            jacs[ix][ix] = -1 * (jacs[ix][ix])

        # jac = -1 * (preds + z * jacs * r_num_instances.repeat(repeats=NUM_REQS, axis=0))

        return jacs

    ### old impl (incorrect: off diag are missing)
    # def jac_constr_delay(z: np.ndarray):
    #     dl = get_predictor_inputs(z, requests_df)

    #     preds = []
    #     grads = []
    #     for x, _ in dl:
    #         for k, v in x.items():
    #             if k != "lengths":
    #                 v.requires_grad_(True)
    #         ts, _ = service_predictor(x)
    #         preds.append(ts.item())
    #         grad = th.autograd.grad(ts, x["x_features"], retain_graph=True)
    #         grads.append(grad[0][0][1])

    #     preds = np.array(preds)
    #     preds *= r_apps_base_exec
    #     grads = np.array(grads)
    #     grads *= r_apps_base_exec

    #     jac = -1 * (preds + z * grads * r_num_instances)

    #     return jac.reshape(1, -1)

    # z0 = np.zeros(shape=(NUM_REQS,))
    z0 = []
    bounds = []
    for ix in range(NUM_REQS):
        if constr_solution[ix] != 0:
            # bounds.append((1-EPS, 1))
            # z0.append(1 - EPS)
            bounds.append((r_g_tau[ix], constr_solution[ix]))
            z0.append(0)
        else:
            bounds.append((r_g_tau[ix], 1))
            z0.append(0)
    # bounds = [(0, 1) for _ in range(NUM_REQS)]

    z0 = np.array(z0).reshape(
        -1,
    )
    constraints = [
        {"type": "ineq", "fun": constr_capacity, "jac": jac_constr_capacity},
        {"type": "ineq", "fun": constr_delay, "jac": jac_constr_delay},
    ]

    start = time.perf_counter()
    result = minimize_ipopt(
        fun=objective,
        x0=z0,
        jac=grad_objective,
        constraints=constraints,
        options={
            "max_iter": 250,
            # "alpha_min_frac": 0.001,
            # "mu_strategy": "adaptive",
            # "derivative_test": "second-order",
            # "print_level": 12,
            # "jacobian_approximation": "finite-difference-values",
            # "gradient_approximation": "finite-difference-values",
            # "max_soc": 10,
            # "tol": 1e-6,
            # "dual_inf_tol": 1e-6,
            # "constr_viol_tol": 1e-3,
            # "tol": 1e-9,  # tolerance for convergence
            # "dual_inf_tol": 1e-9,  # tolerance for dual infeasibility
        },
        bounds=bounds,
    )
    end = time.perf_counter()

    opt_result = OptResult(
        status=result.status,
        message=result.message,
        obj_raw=result.fun,
        exec_time=end - start,
        solution=result.x,
        num_iters=result.nit,
    )

    return opt_result


def algo_solve_server(requests_df: pd.DataFrame, service_predictor) -> OptResult:
    constr_solution = []

    best_obj = 0.0
    best_result: OptResult = None

    time1 = time.time()
    for itr in range(ALGO_ITERS):

        opt_result_rel = solve_server_instance(
            requests_df=requests_df,
            service_predictor=service_predictor,
            constr_solution=constr_solution,
        )

        sol_rel = opt_result_rel.solution

        df_rel = update_df(requests_df, sol_rel, service_predictor, update_preds=True)

        print("****** rel")
        print("used capacity", verify_sol_constr_capacity(requests_df, sol_rel))
        print(
            "rev",
            compute_revenue_method1(df_rel, sol_rel),
            "n_v",
            verify_sol_constr_latency(df_rel, sol_rel),
        )

        print(opt_result_rel)

        print("****** rnd")

        for _ in range(3):
            # best_rnd_result = round_best_ratio(
            #     requests_df,
            #     opt_result_rel,
            #     service_predictor,
            #     constr_solution,
            #     n_iters=ROUND_ITERS,
            # )
            best_rnd_result = round_best_ratio_greedy(
                requests_df,
                opt_result_rel,
                service_predictor,
                constr_solution,
                n_iters=ROUND_ITERS,
            )
            if best_rnd_result is not None:
                break

        if best_rnd_result is None:
            print("breaking prematurely")
            break

        print("used capacity", verify_sol_constr_capacity(requests_df, best_rnd_result.solution))
        print(best_rnd_result)

        df = update_df(
            requests_df, best_rnd_result.solution, service_predictor, update_preds=False
        )

        if best_rnd_result.obj > best_obj:
            best_obj = best_rnd_result.obj
            best_result = best_rnd_result

        constr_solution = best_rnd_result.solution

    time2 = time.time()

    best_result = best_result._replace(exec_time=time2 - time1)
    print("****** final")
    df = update_df(df, best_result.solution, service_predictor, update_preds=True)

    print("used capacity", verify_sol_constr_capacity(requests_df, best_rnd_result.solution))
    print("violations", verify_sol_constr_latency(df, best_result.solution))
    print(best_result)

    opt_result_rel = opt_result_rel._replace(obj=-opt_result_rel.obj_raw)
    opt_results: Dict[str, OptResult] = {
        "rel": opt_result_rel,
        "rnd": best_result,
    }

    return opt_results

def algo_solve_multi_server_method1(num_servers: int, requests_df: pd.DataFrame, service_predictor: CustomDatasetSetReprHydra) -> OptResult:
    z_rnd = pd.DataFrame(
        index=requests_df.index,
        data={f"z_{s}": np.zeros(shape=(len(requests_df))) for s in range(num_servers)},
    )
    z_rel = pd.DataFrame(
        index=requests_df.index,
        data={f"z_{s}": np.zeros(shape=(len(requests_df))) for s in range(num_servers)},
    )

    rev_rnd = 0
    rev_rel = 0
    obj_rel = 0
    requests_df_s = requests_df.copy(deep=True)
    coupling_constrs = []

    t1 = time.time()
    for s_ix in range(num_servers):
        ids = requests_df_s.index
        opt_result_s = solve_server_instance(requests_df_s, service_predictor, coupling_constrs, server_ix=s_ix)
        obj_rel += opt_result_s.obj_raw
        z_rel.loc[ids, f"z_{s_ix}"] = opt_result_s.solution
        rev_rel += compute_revenue_method1(requests_df_s, opt_result_s.solution)
        # opt_result_s = round_best_ratio_greedy(requests_df_s, opt_result_s, service_predictor)

        print("iter: ", s_ix, opt_result_s)

        bar_z_s = opt_result_s.solution
        # bar_z_s[bar_z_s < requests_df_s["g_tau"]] = 0
        # bar_z_s[bar_z_s > 1 - EPS] = 1

        solution_df = merge_final_method_single(
            requests_df_s,
            pd.DataFrame(data=bar_z_s, index=ids),
            service_predictor,
            constr_solution=coupling_constrs,
        )
        bar_z_s = solution_df[0].to_numpy()

        opt_result_s_rnd = round_best_ratio_greedy_new(
            requests_df_s,
            opt_result_s._replace(solution=bar_z_s),
            service_predictor,
            constr_solution=coupling_constrs,
        )
        print("* lost frac rounding", compute_revenue_method1(requests_df_s, opt_result_s.solution), compute_revenue_method1(requests_df_s, opt_result_s_rnd.solution))
        bar_z_s = opt_result_s_rnd.solution
        rev_rnd += compute_revenue_method1(requests_df_s, bar_z_s)

        z_rnd.loc[ids, f"z_{s_ix}"] = bar_z_s

        zr = z_rnd.loc[:, [f"z_{s_jx}" for s_jx in range(s_ix + 1)]].sum(axis=1)
        new_request_df = requests_df.loc[zr < 1 - EPS, :]
        new_ids = new_request_df.index

        coupling_constrs = 1 - zr
        coupling_constrs = coupling_constrs.loc[new_ids]
        requests_df_s = new_request_df

    # zr = z_rnd.sum(axis=1)
    # ix = zr > 0 and zr < 1 - EPS
    # requests_df_recl = requests_df.loc[ix == True, :]
    # ids = requests_df_recl.index

    # for s_ix in range(num_servers):
    #     z_rel_s = z_rel.loc[:, ]
    #     z_rel_s.loc[ids] = 0

    t2 = time.time()
    opt_result_rnd: OptResult = OptResult(
        status=opt_result_s.status,
        message=opt_result_s.message,
        exec_time=t2 - t1,
        obj_raw=rev_rnd,
        obj=rev_rnd,
        solution=z_rnd,
        num_iters=num_servers,
    )

    opt_result_rel: OptResult = OptResult(
        status=opt_result_s.status,
        message=opt_result_s.message,
        exec_time=t2 - t1,
        obj_raw=-obj_rel,
        obj=rev_rel,
        solution=z_rel,
        num_iters=num_servers,
    )

    opt_results = {
        "rel": opt_result_rel,
        "rnd": opt_result_rnd,
    }

    return opt_results


def algo_solve_multi_server_method2(
    num_servers: int, requests_df: pd.DataFrame, service_predictor: CustomDatasetSetReprHydra
) -> OptResult:
    z_rnd = pd.DataFrame(
        index=requests_df.index,
        data={f"z_{s}": np.zeros(shape=(len(requests_df))) for s in range(num_servers)},
    )
    x_rnd = pd.DataFrame(
        index=requests_df.index,
        data={f"z_{s}": np.zeros(shape=(len(requests_df))) for s in range(num_servers)},
    )
    z_rel = pd.DataFrame(
        index=requests_df.index,
        data={f"z_{s}": np.zeros(shape=(len(requests_df))) for s in range(num_servers)},
    )
    x_rel = pd.DataFrame(
        index=requests_df.index,
        data={f"z_{s}": np.zeros(shape=(len(requests_df))) for s in range(num_servers)},
    )

    rev_rnd = 0
    rev_rel = 0
    obj_rel = 0
    requests_df_s = requests_df.copy(deep=True)
    coupling_constrs = []

    t1 = time.time()
    for s_ix in range(num_servers):
        ids = requests_df_s.index

        opt_result_s = solve_server_instance(requests_df_s, service_predictor, coupling_constrs, num_servers=s_ix)

        obj_rel += opt_result_s.obj_raw
        z_rel.loc[ids, f"z_{s_ix}"] = opt_result_s.solution

        # compute rel revenue
        num_instances_s = opt_result_s.solution * requests_df_s["num_instances"]
        x_rel.loc[ids, f"z_{s_ix}"] = num_instances_s
        z_s = num_instances_s / requests_df.loc[ids, "num_instances"].to_numpy().reshape(-1,)
        z_s = z_s.to_numpy()
        rev_rel += compute_revenue_method2(requests_df_s, z_s)

        print("iter: ", s_ix, opt_result_s)

        # compute rnd solution
        bar_z_s = opt_result_s.solution
        opt_result_s_rnd = round_best_ratio_greedy_new(
            requests_df_s,
            opt_result_s._replace(solution=bar_z_s),
            service_predictor,
        )
        bar_z_s = opt_result_s_rnd.solution
        z_rnd.loc[ids, f"z_{s_ix}"] = bar_z_s

        num_instances_s = bar_z_s * requests_df_s["num_instances"]
        x_rnd.loc[ids, f"z_{s_ix}"] = num_instances_s

        bar_z_s = num_instances_s / requests_df.loc[ids, "num_instances"].to_numpy().reshape(-1,)
        bar_z_s = bar_z_s.to_numpy()
        rev_rnd += compute_revenue_method2(requests_df_s, bar_z_s)

        # filter requests for the next iter

        requests_df_s["num_instances"] -= num_instances_s
        requests_df_s[requests_df_s["num_instances"] < 0] = 0

        zr = x_rnd.loc[ids, [f"z_{s_jx}" for s_jx in range(s_ix + 1)]].sum(axis=1) / requests_df.loc[ids, "num_instances"]
        requests_df_s = requests_df_s.loc[zr < 1, :]

        requests_df_s["g_tau"] = 1 / requests_df_s["num_instances"]

    # zr = z_rnd.sum(axis=1)
    # ix = zr > 0 and zr < 1 - EPS
    # requests_df_recl = requests_df.loc[ix == True, :]
    # ids = requests_df_recl.index

    # for s_ix in range(num_servers):
    #     z_rel_s = z_rel.loc[:, ]
    #     z_rel_s.loc[ids] = 0

    t2 = time.time()
    opt_result_rnd: OptResult = OptResult(
        status=opt_result_s.status,
        message=opt_result_s.message,
        exec_time=t2 - t1,
        obj_raw=rev_rnd,
        obj=rev_rnd,
        solution=z_rnd,
        solution_x=x_rnd,
        num_iters=num_servers,
    )

    opt_result_rel: OptResult = OptResult(
        status=opt_result_s.status,
        message=opt_result_s.message,
        exec_time=t2 - t1,
        obj_raw=obj_rel,
        obj=rev_rel,
        solution=z_rel,
        num_iters=num_servers,
    )

    opt_results = {
        "rel": opt_result_rel,
        "rnd": opt_result_rnd,
    }

    return opt_results


def solve_multi_server_instance(num_servers: int, requests_df: pd.DataFrame, service_predictor: CustomDatasetSetReprHydra, constr_solution: List=[]) -> OptResult:
    if len(constr_solution) > 0:
        constr_solution = np.array(constr_solution).reshape(-1, num_servers)
    else:
        constr_solution = np.zeros(shape=(len(requests_df), num_servers))

    NUM_REQS = len(requests_df)

    # tenant_ids = np.array([i for i in range(1, NUM_REQS+1)])
    tenant_ids = requests_df.index

    r_apps = requests_df.loc[tenant_ids, "application"].to_numpy().reshape(-1, 1)
    r_apps_base_exec = (
        requests_df.loc[tenant_ids, "app_base_exec"]
        .to_numpy()
        .reshape(
            -1, 1 
        )
    )
    r_payments = (
        requests_df.loc[tenant_ids, "payment"]
        .to_numpy()
        .reshape(
            -1, 1
        )
    )
    r_app_mems = requests_df.loc[tenant_ids, "app_mem"].to_numpy().reshape(-1, 1)
    r_num_instances = requests_df.loc[tenant_ids, "num_instances"].to_numpy().reshape(-1, 1)
    r_arrival_rate_per = requests_df.loc[tenant_ids, "arrival_rate_per"].to_numpy().reshape(-1, 1)
    r_delays = requests_df.loc[tenant_ids, "delay"].to_numpy().reshape(-1, 1)

    def objective(z: np.ndarray):
        z_mat = matify(z, NUM_REQS, num_servers)
        zr = np.sum(z_mat, axis=1)
        obj = -np.dot(
            r_payments.reshape(
                -1,
            ),
            zr,
        ) + PENALTY * np.sum(1 - zr)
        return obj

    def grad_objective(z: np.ndarray):
        grad = np.repeat(-r_payments + PENALTY * -1, repeats=num_servers, axis=1)
        grad = flatten(grad, NUM_REQS * num_servers)
        return grad

    def constr_capacity(z: np.ndarray):
        z_mat = matify(z, NUM_REQS, num_servers)
        used_capacity = (r_app_mems * r_num_instances).T @ z_mat
        constr = np.array([-used_capacity + C_S]).reshape(num_servers, 1)
        # print("cap:constr", constr.shape)
        return constr

    # def jac_constr_capacity(z: np.ndarray):
    #     jac = -1 * (r_app_mems * r_num_instances)
    #     jac = jac.reshape(-1, 1)
    #     jac = jac.repeat(repeats=num_servers, axis=1)
    #     jac_flat = flatten(jac, NUM_REQS * num_servers)
    #     return jac_flat

    def jac_constr_capacity(z: np.ndarray):
        jac = -1 * (r_app_mems * r_num_instances)
        jac = jac.reshape(-1, 1)
        jac = jac.repeat(repeats=num_servers, axis=0)
        jac = jac.repeat(repeats=num_servers, axis=1)
        # print("cap:jac", jac.shape)
        # jac_flat = flatten(jac, num_servers * NUM_REQS * num_servers)
        return jac.T

    def constr_delay(z: np.ndarray):
        z_mat = matify(z, NUM_REQS, num_servers)
        preds = np.zeros(shape=(NUM_REQS, num_servers))
        for s in range(num_servers):
            z_s = z_mat[:, s]

            dl = get_predictor_inputs(z_s, requests_df)

            preds_s = []
            with th.no_grad():
                for (x, _) in dl:
                    ts, _ = service_predictor(x)
                    preds_s.append(ts.item())

            preds_s = np.array(preds_s)
            preds_s = preds_s.reshape(NUM_REQS, 1)
            preds_s *= r_apps_base_exec

            preds[:, s] = preds_s.reshape(-1,)

        # print("delay eval", np.count_nonzero(z * preds > r_delays))
        constr = -z_mat * preds + r_delays

        constr_flat = flatten(constr, NUM_REQS * num_servers)
        return constr_flat

    def jac_constr_delay(z: np.ndarray):
        z_mat = matify(z, NUM_REQS, num_servers)
        preds = np.zeros(shape=(NUM_REQS, num_servers))
        grads = np.zeros(shape=(NUM_REQS, num_servers))

        for s in range(num_servers):
            z_s = z_mat[:, s]
            dl = get_predictor_inputs(z_s, requests_df)

            preds_s = []
            grads_s = []
            for (x, _) in dl:
                for k, v in x.items():
                    if k != "lengths":
                        v.requires_grad_(True)
                ts, _ = service_predictor(x)
                preds_s.append(ts.item())
                grad = th.autograd.grad(ts, x["x_features"], retain_graph=True)
                grads_s.append(grad[0][0][1])

            preds_s = np.array(preds_s)
            preds_s = preds_s.reshape(NUM_REQS, 1)
            preds_s *= r_apps_base_exec
            grads_s = np.array(grads_s)
            grads_s = grads_s.reshape(NUM_REQS, 1)
            grads_s *= r_apps_base_exec

            preds[:, s] = preds_s.reshape(-1,)
            grads[:, s] = grads_s.reshape(-1,)

        jac = -1 * (preds + z_mat * grads * r_num_instances.repeat(2, axis=1))
        jac_flat = flatten(jac, NUM_REQS * num_servers)

        return jac_flat

    def constr_service(z: np.array):
        z_mat = matify(z, NUM_REQS, num_servers)
        constr = -np.sum(z_mat, axis=1) + 1
        constr = constr.reshape(NUM_REQS, 1)
        print("ser:constr", constr.shape)
        # constr = np.sum(z_mat, axis=1) - 0.1
        return constr

    def jac_constr_service(z: np.array):
        # jac = -np.ones(shape=(NUM_REQS, num_servers))
        # jac = np.ones(shape=(NUM_REQS, num_servers))
        # jac_flat = flatten(jac, NUM_REQS * num_servers)
        jac = np.kron(np.eye(NUM_REQS), -1.0 * np.ones((1, num_servers)))
        print("ser:jac", jac.shape)
        return jac

    z0 = np.zeros(shape=(NUM_REQS, num_servers))
    bounds = [[(0, 1) for _ in range(num_servers)] for _ in range(NUM_REQS)]

    for r, s in product(range(NUM_REQS), range(num_servers)):
        if constr_solution[r, s] != 0:
            z0[r, s] = constr_solution[r, s]

    z0 = np.array(z0)
    bounds = np.array(bounds)

    z0 = flatten(z0, NUM_REQS * num_servers)
    bounds = flatten(bounds, NUM_REQS * num_servers, 2)

    constraints = [
        {"type": "ineq", "fun": constr_capacity, "jac": jac_constr_capacity},
        # {"type": "ineq", "fun": constr_delay, "jac": jac_constr_delay},
        {"type": "ineq", "fun": constr_service, "jac": jac_constr_service},
    ]

    start = time.perf_counter()
    result = minimize_ipopt(
        fun=objective,
        x0=z0,
        jac=grad_objective,
        constraints=constraints,
        options={
            "max_iter": 10_000,
        },
        bounds=bounds,
    )
    end = time.perf_counter()

    opt_result = OptResult(
        status=result.status,
        message=result.message,
        obj_raw=result.fun,
        exec_time=end - start,
        solution=result.x,
        num_iters=result.nit,
    )

    return opt_result