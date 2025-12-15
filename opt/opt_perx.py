import os
import time
import math
from tqdm import tqdm
import json
import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from itertools import product
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import pandas as pd
import copy
import gc
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from kneed import KneeLocator
import matplotlib.pyplot as plt

from models.types import *

SERVER = 1
SCENARIO = "s1"

C_S = 32.0

class ElbowPredictor:
    def __init__(self, x, y):
        x = np.array(x)
        y = np.array(y)
        ix = np.argsort(x)
        x = x[ix]
        y = y[ix]

        self.x = x
        self.y = y

        self.elbow_point = 0.0
        self.elbow_point_y = 0.0
        self.theta1 = None
        self.theta2 = None
        self.lr1 = None
        self.lr2 = None

        self.single_seg = False
        self.fit_pc_lines()
    
    def fit_pc_lines(self):
        x = self.x
        y = self.y

        elbow_locator = KneeLocator(x, y, S=1.5, curve="convex", direction="increasing")
        elbow = elbow_locator.knee

        if elbow is None:
            self.lr1 = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
            self.single_seg = True
        else:
            self.elbow_point = elbow
            self.elbow_point_y = y[x == elbow][0]

            x1 = x[x < elbow].reshape((-1, 1))
            x2 = x[x >= elbow].reshape((-1, 1))

            y1 = y[x < elbow].reshape((-1, 1))
            y2 = y[x >= elbow].reshape((-1, 1))

            self.lr1 = LinearRegression().fit(x1, y1)
            self.lr2 = LinearRegression().fit(x2, y2)

    def predict(self, x):
        if x <= self.elbow_point or self.single_seg:
            # return self.theta1.T @ np.array([[x], [1]])
            return self.lr1.predict(np.array([x]).reshape(-1, 1))[0]
        else:
            # return self.theta2.T @ np.array([[x], [1]])
            return self.lr2.predict(np.array([x]).reshape(-1, 1))[0]

colors = {
    "cnn": "orange",
    "dnn": "red",
    "lstm": "purple",
    "rf": "green",
}
met_path = "/home/arshad/code/pa_res_alloc/reports/metrics"
fig_path = "/home/arshad/code/pa_res_alloc/reports/figures"

def solve_server_instance(requests_df: pd.DataFrame, num_servers: int=1, time_limit: int=1, num_iters: int=1):
    SIGMA = 0.165 # $ / kWh

    Es = 100.0 # w
    Eas = {
        "cnn": 8.77,
        "lstm": 16,
        "dnn": 22,
        "rf": 5.0,
    }

    # NRT
    # MUL = 1.0
    # AS (dnn/lstm)
    match SCENARIO:
        case "s1":
            if SERVER == 1:
                MUL_delta = 10 #s1
                MUL_base = 0.5 #s1
            elif SERVER == 2:
                MUL_delta = 2.0 #s1
                MUL_base = 1.0 #s1
        case "s2":
            MUL_delta = 1.5 #s2
            MUL_base = 1.0 #s2
        case "s3":
            MUL_delta = 3.0 #s3
            MUL_base = 1.0 #s3
    # AS50
    # MUL = 1.33

    ELBOW_METRIC = "p95"

    LAMBDA_A = {
        # "cnn-v1": 8,
        # "lstm-v1": 8,
        # "dnn-v1": 8,
        # "rf-v1": 8,

        "tclf-gcn": 8,
        "tclf-rf": 8,
        "tstr-lstm": 8,
        "kalman-gru": 8,
        "iclf-mnet": 8,
        "text-bert": 8,
        "iclf-efnet": 8,
        "text-tbert": 8,
        "iclf-mvit": 8,
    }

    APPLICATIONS = APPLICATIONS_E

    # NUM_R = len(requests)
    NUM_ITERS = num_iters
    NUM_A = len(APPLICATIONS)
    NUM_S = num_servers
    # TIME_LIMIT = max(num_servers // 2, 1)
    TIME_LIMIT = time_limit
    # NUM_REQUESTS = int(NUM_S * 100 / 3)
    # OUT_NAME = f"s"

    def parse_resquests(requests_df: pd.DataFrame):
        parsed_requests = []

        for ix, request in requests_df.iterrows():
            app = request.application

            rho = request.payment

            lambda_r = request.arrival_rate

            n_i = request.num_instances

            tau = request.delay

            sla_type = request.sla_type

            parsed_requests.append((rho, tau, lambda_r, n_i, app, sla_type))

        return parsed_requests

    def load_elbow_params(lambda_a):
        for target_ix in range(len(APP_INFERENCE_TARGETS)):
            for app in lambda_a:
                app_rate = lambda_a[app]
                a = APPLICATIONS.index(app)
                a1 = contention_params_sig[target_ix][(a, a)]["alpha1"]
                b1 = contention_params_sig[target_ix][(a, a)]["beta1"]
                a2 = contention_params_sig[target_ix][(a, a)]["alpha2"]
                b2 = contention_params_sig[target_ix][(a, a)]["beta2"]
                a3 = contention_params_sig[target_ix][(a, a)]["alpha3"]
                b3 = contention_params_sig[target_ix][(a, a)]["beta3"]

                nis = list(range(4, 16, 2))
                y = []
                for ni in nis:
                    t1 = a1 + b1 * ni
                    t2 = a2 + b2 * ni
                    t3 = a3 + b3 * ni

                    sig = t1 / (1 + np.exp(-t2 * (app_rate - t3)))
                    y.append(sig)

                elbow_predictor = ElbowPredictor(nis, y)
                yp = []

                for ni in nis:
                    yp.append(elbow_predictor.predict(ni))

                elbow_x = float(elbow_predictor.elbow_point)
                elbow_y = float(elbow_predictor.elbow_point_y)

                c1 = float(elbow_predictor.lr1.intercept_[0])
                m1 = float(elbow_predictor.lr1.coef_[0][0])

                x_vals = [4.0]
                y_vals = [m1 * 4 + c1]

                if elbow_x > 4.0:
                    x_vals.append(elbow_x)
                    y_vals.append(elbow_y)

                if elbow_x != 16 and elbow_predictor.lr2 is not None:
                    c2 = float(elbow_predictor.lr2.intercept_[0])
                    m2 = float(elbow_predictor.lr2.coef_[0][0])
                    x_vals.append(16.0)
                    y_vals.append(m2 * 16 + c2)
                else:
                    x_vals.append(16.0)
                    y_vals.append(m1 * 16 + c1)

                contention_params_elbow[target_ix][a] = {
                    "x_vals": x_vals,
                    "y_vals": y_vals,
                }

    paths_target_ix = {
        0: "models_m",
        1: "models_p75",
        2: "models_p95",
    }

    server = f"server_{SERVER}"
    path_baseline = f"{HOME_PATH}" + f"/{server}" + "/results/{}/perx/baselines.json"

    path_sig = f"{HOME_PATH}" + f"/{server}" + "/results/{}/perx/sig_params.json"
    path_elbow = f"{HOME_PATH}" + f"/{server}" + "/data/pairwise_exps/contention_params_elbow_{}_{}.json"
    path_elbow_pairs = f"{HOME_PATH}" + f"/{server}" + "/data/pairwise_exps/contention_params_elbow_pairs_{}_{}.json"

    contention_params_sig = defaultdict(lambda: {})
    contention_params_elbow = defaultdict(lambda: {})
    contention_params_elbow_pairs = defaultdict(lambda: {})

    baselines = []
    for target_ix in range(len(APP_INFERENCE_TARGETS)):
        baselines_json = json.load(fp=(open(path_baseline.format(paths_target_ix[target_ix]))))

        baselines_target = [baselines_json[app] * MUL_base for app in APPLICATIONS]
        baselines.append(baselines_target)

    for target_ix in range(len(APP_INFERENCE_TARGETS)):
        for app in APPLICATIONS:
            contention_params = json.load(fp=(open(path_sig.format(paths_target_ix[target_ix]))))

            for p0, vals in contention_params.items():
                for p1 in vals.keys():
                    # for pair in contention_params.keys():
                    ix0 = APPLICATIONS.index(p0)
                    ix1 = APPLICATIONS.index(p1)

                    contention_params_sig[target_ix][(ix0, ix1)] = contention_params[p0][p1]

    # for ix, app in enumerate(APPLICATIONS):
    #     app_rate = LAMBDA_A[app]
    #     contention_params = json.load(fp=(open(path_elbow.format(app, ELBOW_METRIC))))

    #     keys = list(contention_params.keys())
    #     keys = np.array(list(map(lambda x: float(x), keys)))
    #     key_ix = np.argmin(np.abs((keys - app_rate)))
    #     closest_app_rate = float(keys[key_ix])
    #     contention_params = contention_params[str(closest_app_rate)]

    #     m1 = contention_params["beta1"]
    #     c1 = contention_params["alpha1"]
    #     m2 = contention_params["beta2"]
    #     c2 = contention_params["alpha2"]
    #     x_vals = [2.0, contention_params["elbow_x"]]
    #     y_vals = [m1 * 2 + c1, contention_params["elbow_y"]]

    #     if contention_params["elbow_x"] != 32:
    #         x_vals.append(32.0)
    #         y_vals.append(m2 * 32 + c2)

    #     contention_params_elbow[ix] = {
    #         "x_vals": x_vals,
    #         "y_vals": y_vals,
    #     }

    # for ix, app in enumerate(APPLICATIONS):
    #     app_rate = LAMBDA_A[app]
    #     contention_params_pairs = json.load(fp=(open(path_elbow_pairs)))

    #     keys = list(contention_params_pairs.keys())
    #     keys = np.array(list(map(lambda x: float(x), keys)))
    #     key_ix = np.argmin(np.abs((keys - app_rate)))
    #     closest_app_rate = float(keys[key_ix])
    #     contention_params_pairs = contention_params_pairs[str(closest_app_rate)]

    #     # for pair_key, contention_params in contention_params_pairs.items():
    #     for k1, vals in contention_params_pairs.keys():
    #         for k2, contention_params in vals.item():
    #             k1_ix = APPLICATIONS.index(k1)
    #             k2_ix = APPLICATIONS.index(k2)

    #             m1 = contention_params["beta1"]
    #             c1 = contention_params["alpha1"]
    #             m2 = contention_params["beta2"]
    #             c2 = contention_params["alpha2"]
    #             elbow_x = contention_params["elbow_x"]
    #             x_vals = [2.0, contention_params["elbow_x"]]
    #             y_vals = [m1 * 2 + c1, contention_params["elbow_y"]]

    #             if contention_params["elbow_x"] != 32:
    #                 x_vals.append(32.0)
    #                 y_vals.append(m2 * 32 + c2)

    #             contention_params_elbow_pairs[(k1_ix, k2_ix)] = {
    #                 "alpha1": c1,
    #                 "beta1": m1,
    #                 "alpha2": c2,
    #                 "beta2": m2,
    #                 "elbow_x": elbow_x,
    #                 "x_vals": x_vals,
    #                 "y_vals": y_vals,
    #             }

    def opt_deployments(requests, method, lambda_as=None):
        NUM_R = len(requests)
        model = gp.Model("oran_scaling")

        time_limit = TIME_LIMIT

        model.setParam('OutputFlag', 0)
        model.setParam("TimeLimit", 60 * time_limit)
        model.setParam("NodefileStart", 32)

        X_r_as = {}
        for (r, a, s) in product(range(NUM_R), range(NUM_A), range(NUM_S)):
            key = f"x_{r}_{a},{s}"
            x_r_as = model.addVar(vtype=GRB.INTEGER, lb=0, name=key)
            X_r_as[key] = x_r_as

        Z_r = {}
        for r in range(NUM_R):
            key = f"z_{r}"
            z_r = model.addVar(vtype=GRB.BINARY, name=key)
            Z_r[key] = z_r

        W_s = {}
        for s in range(NUM_S):
            key = f"w_{s}"
            w_s = model.addVar(vtype=GRB.BINARY, name=key)
            W_s[key] = w_s

        W_r_as = {}
        for r, a, s in product(range(NUM_R), range(NUM_A), range(NUM_S)):
            req = requests[r]
            # a = APPLICATIONS.index(req[4])
            key = f"w_{r}_{a},{s}"
            w_r_as = model.addVar(vtype=GRB.BINARY, name=key)
            W_r_as[key] = w_r_as

        W_as = {}
        for a, s in product(range(NUM_A), range(NUM_S)):
            key = f"w_{a},{s}"
            w_as = model.addVar(vtype=GRB.BINARY, name=key)
            W_as[key] = w_as

        M1 = 1e3
        # if server s has at least one app hosted
        for s in range(NUM_S):
            key = f"w_{s}"
            w_s = W_s[key]

            terms = []
            for (r, a) in product(range(NUM_R), range(NUM_A)):
                terms.append(X_r_as[f"x_{r}_{a},{s}"])

            model.addConstr(quicksum(terms) >= 1 - M1 * (1 - w_s))
            model.addConstr(quicksum(terms) <= M1 * (w_s))

        # if request r has an application instance a on server s
        for r, s in product(range(NUM_R), range(NUM_S)):
            req = requests[r]
            a = APPLICATIONS.index(req[4])
            key = f"w_{r}_{a},{s}"
            w_r_as = W_r_as[key]

            terms = []
            terms.append(X_r_as[f"x_{r}_{a},{s}"])

            model.addConstr(quicksum(terms) >= 1 - M1 * (1 - w_r_as))
            model.addConstr(quicksum(terms) <= M1 * (w_r_as))

        # if sever s has application a hosted
        for a, s in product(range(NUM_A), range(NUM_S)):
            key = f"w_{a},{s}"
            w_as = W_as[key]

            terms = []
            for r in range(NUM_R):
                terms.append(X_r_as[f"x_{r}_{a},{s}"])

            model.addConstr(quicksum(terms) >= 1 - M1 * (1 - w_as))
            model.addConstr(quicksum(terms) <= M1 * (w_as))

        M2 = 1e5
        constr_n_ra = {}
        for (r, a) in product(range(NUM_R), range(NUM_A)):
            req = requests[r]
            n_r_a = req[3]
            ar = APPLICATIONS.index(req[4])

            terms = []
            for s in range(NUM_S):
                terms.append(X_r_as[f"x_{r}_{a},{s}"])

            #! both work
            if ar == a:
                model.addConstr(quicksum(terms) >= n_r_a - M2 * (1 - Z_r[f"z_{r}"]))
                model.addConstr(quicksum(terms) <= n_r_a + M2 * (1 - Z_r[f"z_{r}"]))
            else:
                model.addConstr(quicksum(terms) == 0)

        revenue = []
        for r, key in enumerate(Z_r.keys()):
            revenue.append(requests[r][0] * Z_r[key])

        static_energy = []
        for s in range(NUM_S):
            key = f"w_{s}"
            w_s = W_s[key]
            static_energy.append(w_s * Es)

        model.setObjective(
            # quicksum(revenue) - SIGMA * (quicksum(static_energy) + quicksum(dynamic_energy) * 60 / (3.6 * 1e6)),
            quicksum(revenue),
            GRB.MAXIMIZE
        )

        ## Scal-ORAN approach
        if method == "scaloran":
            for r in range(NUM_R):
                req = requests[r]
                tau = req[1]
                a = APPLICATIONS.index(req[4])
                sla_type = req[5]

                for s in range(NUM_S):
                    w_r_as = W_r_as[f"w_{r}_{a},{s}"]

                    terms = []
                    terms2 = [] # Y_s
                    terms2_b = [] # pi_a,s
                    for ad in range(NUM_A):
                        terms2_b.append(W_as[f"w_{ad},{s}"])
                        for rd in range(NUM_R):

                            terms2.append(X_r_as[f"x_{rd}_{ad},{s}"])

                    # avg. delay ->
                    n_ads = quicksum(terms2)

                    n_ads_var = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"n_ads_var_{r}")
                    model.addConstr(n_ads_var == n_ads)

                    for ad in range(NUM_A):
                        x_vals = contention_params_elbow[sla_type][ad]["x_vals"]
                        y_vals = contention_params_elbow[sla_type][ad]["y_vals"]

                        tau_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                        model.addGenConstrPWL(n_ads_var, tau_var, x_vals, y_vals)

                        terms.append(W_as[f"w_{ad},{s}"] * tau_var)
                    # avg. delay ->

                    model.addConstr(quicksum(terms) <= (tau) + M2 * (1 - W_r_as[f"w_{r}_{a},{s}"]))

        elif method == "scaloran-var":
            for r in range(NUM_R):
                req = requests[r]
                tau = req[1]
                a = APPLICATIONS.index(req[4])
                sla_type = req[5]

                for s in range(NUM_S):
                    w_r_as = W_r_as[f"w_{r}_{a},{s}"]

                    terms = []
                    terms2 = []
                    for ad in range(NUM_A):
                        for rd in range(NUM_R):
                            # reqd = REQUESTS[rd]
                            # ad = APPLICATIONS.index(req[4])

                            terms2.append(X_r_as[f"x_{rd}_{ad},{s}"])

                    # avg. delay ->
                    n_ads = quicksum(terms2)

                    n_ads_var = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"n_ads_var_{r}")
                    model.addConstr(n_ads_var == n_ads)

                    x_vals = contention_params_elbow[sla_type][a]["x_vals"]
                    y_vals = contention_params_elbow[sla_type][a]["y_vals"]

                    tau_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"tau_var_{r}_{ad}")
                    model.addGenConstrPWL(n_ads_var, tau_var, x_vals, y_vals)

                    terms.append(tau_var)
                    # avg. delay ->

                    # model.addConstr(quicksum(terms) <= (tau) + M2 * (1 - Z_r[f"z_{r}"]))
                    model.addConstr(quicksum(terms) <= 0.6 * tau + M2 * (1 - W_r_as[f"w_{r}_{a},{s}"]))

        elif method == "prop":
            for r in range(NUM_R):
                req = requests[r]
                tau = req[1]
                a = APPLICATIONS.index(req[4])
                sla_type = req[5]

                for s in range(NUM_S):
                    w_r_as = W_r_as[f"w_{r}_{a},{s}"]

                    terms = []

                    for ad in range(NUM_A):
                        # app_rate = LAMBDA_A[APPLICATIONS[ad]]
                        app_rate = lambda_as[s][APPLICATIONS[ad]]

                        terms2 = []
                        terms3_a = []
                        terms3_b = []
                        w_ads = W_as[f"w_{ad},{s}"]
                        for rd in range(NUM_R):
                            req = requests[rd]
                            lam = req[2]

                            terms2.append(W_r_as[f"w_{rd}_{ad},{s}"] * X_r_as[f"x_{rd}_{ad},{s}"])
                            terms3_b.append(W_r_as[f"w_{rd}_{ad},{s}"])
                            terms3_a.append(W_r_as[f"w_{rd}_{ad},{s}"] * lam)

                        n_ads = quicksum(terms2)
                        n_ads_var = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"n_ads_var_{ad},{s}")
                        model.addConstr(n_ads_var == n_ads)

                        a1 = contention_params_sig[sla_type][(a, ad)]["alpha1"]
                        b1 = contention_params_sig[sla_type][(a, ad)]["beta1"]
                        a2 = contention_params_sig[sla_type][(a, ad)]["alpha2"]
                        b2 = contention_params_sig[sla_type][(a, ad)]["beta2"]
                        a3 = contention_params_sig[sla_type][(a, ad)]["alpha3"]
                        b3 = contention_params_sig[sla_type][(a, ad)]["beta3"]

                        # ---

                        nis = list(range(4, 16, 1))
                        y = []
                        for ni in nis:
                            t1 = a1 + b1 * ni
                            t2 = a2 + b2 * ni
                            t3 = a3 + b3 * ni

                            sig = t1 / (1 + np.exp(-t2 * (app_rate - t3)))
                            y.append(sig)

                        elbow_predictor = ElbowPredictor(nis, y)
                        yp = []

                        for ni in nis:
                            yp.append(elbow_predictor.predict(ni))

                        elbow_x = float(elbow_predictor.elbow_point)
                        elbow_y = float(elbow_predictor.elbow_point_y)

                        c1 = float(elbow_predictor.lr1.intercept_[0])
                        m1 = float(elbow_predictor.lr1.coef_[0][0])

                        x_vals = [4.0]
                        y_vals = [m1 * 4 + c1]

                        if elbow_x > 4.0:
                            x_vals.append(elbow_x)
                            y_vals.append(elbow_y)

                        if elbow_x != 16 and elbow_predictor.lr2 is not None:
                            c2 = float(elbow_predictor.lr2.intercept_[0])
                            m2 = float(elbow_predictor.lr2.coef_[0][0])
                            x_vals.append(16.0)
                            y_vals.append(m2 * 16 + c2)
                        else:
                            x_vals.append(16.0)
                            y_vals.append(m1 * 16 + c1)

                        tau_var = model.addVar(vtype=GRB.CONTINUOUS, lb=-100, name=f"tau_var_{r}_{ad}")
                        model.addGenConstrPWL(n_ads_var, tau_var, x_vals, y_vals)

                        mul = MUL_delta
                        # mul = MUL if APPLICATIONS[a] == "dnn" or APPLICATIONS[a] == "lstm" else 1.0
                        # mul = MUL if APPLICATIONS[a] == "dnn" or APPLICATIONS[a] == "lstm" or APPLICATIONS[ad] == "dnn" or APPLICATIONS[ad] == "lstm" else 1.0

                        terms.append(w_ads * (tau_var - baselines[sla_type][a]) * mul)

                    model.addConstr(quicksum(terms) <= tau - baselines[sla_type][a] + M2 * (1 - w_r_as))

        ## limit number of instances per server
        for s in range(NUM_S):
            terms  = []
            for (r, a) in product(range(NUM_R), range(NUM_A)):
                # for r in range(NUM_R):
                req = requests[r]
                # a = APPLICATIONS.index(req[4])
                mem = APP_MEM_REQS[req[4]]
                terms.append(mem * X_r_as[f"x_{r}_{a},{s}"])
                # terms.append(X_r_as[f"x_{r}_{a},{s}"])

            # temp = model.addVar(vtype=GRB.INTEGER, lb=0)
            # model.addConstr(temp == 3)
            # if method == "scaloran-var":
            #     mul = 0.75
            # else:
            #     mul = 1.0

            model.addConstr(quicksum(terms) <= C_S)

        model.optimize()
        print(f"{model.Status=}, {model.ObjVal=}")

        return model

    def grab_served_reqs(requests, model: gp.Model):
        served_reqs_ix = []
        placement = {}

        for r in range(len(requests)):
            placement[r] = {}

            req = requests[r]
            a = APPLICATIONS.index(req[4])
            var = model.getVarByName(f"z_{r}")

            if var is not None and var.x == 1:
                served_reqs_ix.append(r)
                for s in range(NUM_S):
                    var_s = model.getVarByName(f"x_{r}_{a},{s}")
                    placement[r][s] = 0
                    if var_s is not None and var_s.x > 0:
                        placement[r][s] += var_s.x

        return served_reqs_ix, placement

    def grab_placements(requests, model: gp.Model) -> pd.DataFrame:
        z = pd.DataFrame(data={f"z_{s}": np.zeros(shape=(len(requests_df))) for s in range(num_servers)})
        for s in range(NUM_S):
            z_s = []
            for r in range(len(requests)):
                req = requests[r]
                a = APPLICATIONS.index(req[4])
                n = req[3]
                x_r_as = model.getVarByName(f"x_{r}_{a},{s}").x
                z_s.append(x_r_as / n)

            z[f"z_{s}"] = np.array(z_s)

        return z

    def grab_deployment(requests, served_reqs_ix, model:gp.Model):
        deployment = {s: {
            app: 0 for app in APPLICATIONS
        } for s in range(NUM_S)}

        for r in served_reqs_ix:
            req = requests[r]
            a = APPLICATIONS.index(req[4])

            for s in range(NUM_S):
                var = model.getVarByName(f"x_{r}_{a},{s}")

                if var is not None and var.x > 0:
                    deployment[s][APPLICATIONS[a]] += int(var.x)

        return deployment

    def get_energy_consumption(requests, served_reqs_ix, model:gp.Model):
        static_energy = 0.0
        for s in range(NUM_S):
            key = f"w_{s}"
            var = model.getVarByName(key)
            if var is not None and var.x > 0:
                static_energy += var.x * Es

        dynamic_energy = 0.0

        return static_energy, dynamic_energy

    def verify_n_feasibility(requests, served_reqs_ix, model: gp.Model):
        feasible = True
        for r in range(len(requests)):
            req = requests[r]
            n_r_a = req[3]
            a = APPLICATIONS.index(req[4])

            # sigma_{s} x_{r}_{a}{s} == n_r_a

            num_instances = 0.0
            for s in range(NUM_S):
                var = model.getVarByName(f"x_{r}_{a},{s}")

                if var is not None and var.x > 0:
                    num_instances += var.x

            # try:
            #     if r in served_reqs_ix:
            #         assert num_instances == n_r_a, f"constraint violated, number of instances for served req, , {r=}, {requests[r]=}, {num_instances=}"
            #     # else:
            #     #     assert num_instances == 0, f"constraint violated, number of instances for req not served, {r=}, {requests[r]=}, {num_instances=}"
            # except AssertionError as e:
            #     print(e)
            #     continue
            if r in served_reqs_ix:
                if num_instances != int(n_r_a):
                    feasible = False
                    print(f"constraint violated, number of instances for served req, , {r=}, {requests[r]=}, {num_instances=}")

        return feasible

    def get_delay_constraints(requests, served_reqs_ix, deployment, model: gp.Model):
        min_delays = {s: {
        } for s in range(NUM_S)}

        for s in range(NUM_S):
            for app, n_app in deployment[s].items():
                if n_app >= 1:
                    taus = []
                    for r in served_reqs_ix:
                        req = requests[r]
                        tau = req[1]
                        a = req[4]

                        if a == app and model.getVarByName(f"w_{r}_{APPLICATIONS.index(a)},{s}").x == 1:
                            taus.append(tau)
                    min_delays[s][app] = min(taus)

        return min_delays

    def get_min_delays_deployment_scaloran(deployment, min_delays):
        app_delay_constrs = {s:{} for s in min_delays}

        for s in range(NUM_S):
            for app in min_delays[s].keys():
                app_rate = LAMBDA_A[app]
                app_delay = 0.0
                count = 0
                for app_j, app_j_n_i in deployment[s].items():
                    if app_j_n_i < 1:
                        continue

                    count += 1
                    contention_params = json.load(fp=(open(path_elbow.format(app, ELBOW_METRIC))))

                    keys = list(contention_params.keys())
                    keys = np.array(list(map(lambda x: float(x), keys)))
                    key_ix = np.argmin(np.abs((keys - app_rate)))
                    closest_app_rate = float(keys[key_ix])
                    contention_params = contention_params[str(closest_app_rate)]

                    m1 = contention_params["beta1"]
                    c1 = contention_params["alpha1"]
                    m2 = contention_params["beta2"]
                    c2 = contention_params["alpha2"]
                    elbow = contention_params["elbow_x"]

                    if app_j_n_i <= elbow:
                        app_delay += m1 * app_j_n_i + c1
                    else:
                        app_delay += m2 * app_j_n_i + c2

                app_delay /= count

                # app_delay_constrs[s][app] = wait_time + app_delay
                app_delay_constrs[s][app] = app_delay

        return app_delay_constrs

    def get_min_delays_deployment_prop(deployment, min_delays):
        app_delay_constrs = {s:{} for s in min_delays}

        for s in range(NUM_S):
            for app in min_delays[s].keys():
                app_rate = LAMBDA_A[app]
                # if s != 1 or app != "lstm":
                #     continue
                app_ix = APPLICATIONS.index(app)
                app_delay = baselines[app_ix]

                for app_j, app_j_n_i in deployment[s].items():
                    if app_j_n_i == 0:
                        continue

                    app_j_ix = APPLICATIONS.index(app_j)

                    contention_params = contention_params_elbow_pairs[(app_ix, app_j_ix)]

                    m1 = contention_params["beta1"]
                    c1 = contention_params["alpha1"]
                    m2 = contention_params["beta2"]
                    c2 = contention_params["alpha2"]
                    elbow = contention_params["elbow_x"]

                    if app_j_n_i <= elbow:
                        contention_delay = m1 * app_j_n_i + c1
                    else:
                        contention_delay = m2 * app_j_n_i + c2

                    # mul = MUL
                    mul = MUL_delta if app == "dnn" or app == "lstm" else 1.0

                    # app_delay += max((contention_delay - baselines[app_ix]), 0.0)
                    app_delay += (contention_delay - baselines[app_ix] * mul)

                    # print(app, app_j, contention_delay - baselines[app_ix])

                # app_delay_constrs[s][app] = wait_time + app_delay
                app_delay_constrs[s][app] =  app_delay

        return app_delay_constrs

    def verify_delay_constraints(min_delays, delay_constraints):
        feasible = True
        max_violation = 0.0
        for s in min_delays:
            for app in min_delays[s]:
                # if min_delays[s][app] - delay_constraints[s][app] < 1e-2:
                if min_delays[s][app] < delay_constraints[s][app] - 2e-2:
                    feasible = False

                    max_violation = max(max_violation, delay_constraints[s][app] - min_delays[s][app])

        return feasible, max_violation

    def compute_revenue(requests, served_user_ix):
        revenue = 0.0

        for request_ix in served_user_ix:
            req = requests[request_ix]
            revenue += req[0]

        return revenue

    def compute_lambda_as(requests, served_reqs_ix, deployment, model: gp.Model, lambda_as):
        arrivals_as = defaultdict(lambda : {app: 0 for app in APPLICATIONS})
        new_lambda_as = copy.deepcopy(lambda_as)

        for r in served_reqs_ix:
            req = requests[r]
            app = req[4]
            a = APPLICATIONS.index(app)

            for s in range(NUM_S):
                arrivals_as[s][app] += int(model.getVarByName(f"x_{r}_{a},{s}").x) * req[2] / req[3]

        for s in range(NUM_S):
            for app in APPLICATIONS:
                if arrivals_as[s][app] > 0:
                    new_lambda_as[s][app] = arrivals_as[s][app] / deployment[s][app]

        return new_lambda_as

    def run(requests, index):
        requests_df = pd.DataFrame(requests)
        requests_df.index = index

        lambda_a = {}
        for app in APPLICATIONS:
            tenant_lambda_ins = requests_df[(requests_df.iloc[:, 4] == app)][2] / requests_df[(requests_df.iloc[:, 4] == app)][3]
            mean_arrival_rate = tenant_lambda_ins.mean()

            if np.isfinite(mean_arrival_rate):
                lambda_a[app] = mean_arrival_rate
            else:
                lambda_a[app] = 6.66

            lambda_as = {s_ix: lambda_a for s_ix in range(NUM_S)}

        load_elbow_params(lambda_a)
        t1 = time.time()
        print("*" * 5, "scaloran")
        model_s = opt_deployments(requests=requests, method="scaloran")
        t2 = time.time()
        print("*" * 5, "scaloran-var")
        model_sv = opt_deployments(requests=requests, method="scaloran-var")
        t3 = time.time()
        # model_p = opt_deployments(requests=requests, method="prop")

        old_served_reqs_ix = []

        for ix in range(NUM_ITERS):
            print("*" * 5, "perx iter:", ix)
            model_p = opt_deployments(requests, "prop", lambda_as)

            served_reqs_ix, placement = grab_served_reqs(requests, model_p)
            deployment = grab_deployment(requests, served_reqs_ix, model_p)

            converged = old_served_reqs_ix == served_reqs_ix

            new_lambda_as = compute_lambda_as(requests, served_reqs_ix, deployment, model_p, lambda_as)
            lambda_as = new_lambda_as
            old_served_reqs_ix = served_reqs_ix

            time.sleep(1)

            if converged:
                break

        t4 = time.time()

        served_reqs_ix_s, placement_s = grab_served_reqs(requests, model_s)
        z_s = grab_placements(requests, model_s)
        z_s.loc[z_s.sum(axis=1) != 1] = 0.0
        z_s.index = requests_df.index
        served_reqs_ix_sv, placement_sv = grab_served_reqs(requests, model_sv)
        z_sv = grab_placements(requests, model_sv)
        z_sv.loc[z_sv.sum(axis=1) != 1] = 0.0
        z_sv.index = requests_df.index
        served_reqs_ix_p, placement_p = grab_served_reqs(requests, model_p)
        z_p = grab_placements(requests, model_p)
        z_p.loc[z_p.sum(axis=1) != 1] = 0.0
        z_p.index = requests_df.index

        obj_s = requests_df.loc[z_s.sum(axis=1) == 1][0].sum()
        obj_sv = requests_df.loc[z_sv.sum(axis=1) == 1][0].sum()
        obj_p = requests_df.loc[z_p.sum(axis=1) == 1][0].sum()
        pct_prop_s = (obj_p - obj_s) / obj_s
        pct_prop_sv = (obj_p - obj_sv) / obj_sv

        rev_s = compute_revenue(requests, served_reqs_ix_s)
        rev_sv = compute_revenue(requests, served_reqs_ix_sv)
        rev_p = compute_revenue(requests, served_reqs_ix_p)
        pct_prop_s = (rev_p - rev_s) / rev_s
        pct_prop_sv = (rev_p - rev_sv) / rev_sv

        deployment_s = grab_deployment(requests, served_reqs_ix_s, model_s)
        deployment_sv = grab_deployment(requests, served_reqs_ix_sv, model_sv)
        deployment_p = grab_deployment(requests, served_reqs_ix_p, model_p)

        static_s, dynamic_s = get_energy_consumption(requests, served_reqs_ix_s, model_s)
        static_sv, dynamic_sv = get_energy_consumption(requests, served_reqs_ix_sv, model_sv)
        static_p, dynamic_p = get_energy_consumption(requests, served_reqs_ix_p, model_p)

        # min_delays_s = get_delay_constraints(requests, served_reqs_ix_s, deployment_s, model_s)
        # min_delays_sv = get_delay_constraints(requests, served_reqs_ix_sv, deployment_sv, model_sv)
        # min_delays_p = get_delay_constraints(requests, served_reqs_ix_p, deployment_p, model_p)

        # delay_constraints_s = get_min_delays_deployment_scaloran(deployment_s, min_delays_s)
        # delay_constraints_sv = get_min_delays_deployment_scaloran(deployment_sv, min_delays_sv)
        # delay_constraints_p = get_min_delays_deployment_prop(deployment_p, min_delays_p)

        # delay_feasibility_s, delay_violation_s = verify_delay_constraints(min_delays_s, delay_constraints_s)
        # delay_feasibility_sv, delay_violation_sv = verify_delay_constraints(min_delays_sv, delay_constraints_sv)
        # delay_feasibility_p, delay_violation_p = verify_delay_constraints(min_delays_p, delay_constraints_p)

        requests_mod = []
        for req in requests:
            req_mod = (req[0], req[1], req[2] / req[3], req[3], req[4])

            requests_mod.append(req_mod)

        results = {
            "requests": requests_mod,
            "scaloran": {
                "exec_time": t2 - t1,
                "obj": obj_s,
                "revenue": rev_s,
                "served_reqs_ix": served_reqs_ix_s,
                "placement": placement_s,
                "z": z_s,
                "deployment": deployment_s,
                # "min_delays": min_delays_s,
                # "delay_constrs": delay_constraints_s,
                # "delay_feasibility": delay_feasibility_s,
                "static_e": static_s,
                "dynamic_e": dynamic_s,
            },
            "scaloran-var": {
                "exec_time": t3 - t2,
                "obj": obj_sv,
                "revenue": rev_sv,
                "served_reqs_ix": served_reqs_ix_sv,
                "placement": placement_sv,
                "z": z_sv,
                "deployment": deployment_sv,
                # "min_delays": min_delays_sv,
                # "delay_constrs": delay_constraints_sv,
                # "delay_feasibility": delay_feasibility_sv,
                "static_e": static_sv,
                "dynamic_e": dynamic_sv,
            },
            "prop": {
                "exec_time": t4 - t3,
                "obj": obj_p,
                "revenue": rev_p,
                "served_reqs_ix": served_reqs_ix_p,
                "placement": placement_p,
                "z": z_p,
                "deployment": deployment_p,
                # "min_delays": min_delays_p,
                # "delay_constrs": delay_constraints_p,
                # "delay_feasibility": delay_feasibility_p,
                "static_e": static_p,
                "dynamic_e": dynamic_p,
                "num_iters": ix,
            },
        }

        return results

    requests = parse_resquests(requests_df)

    results = run(requests=requests, index=requests_df.index)

    sol_scaloran = np.zeros(shape=(len(requests),))
    np.put(sol_scaloran, results["scaloran"]["served_reqs_ix"], 1)

    sol_scaloran_cons = np.zeros(shape=(len(requests),))
    np.put(sol_scaloran_cons, results["scaloran-var"]["served_reqs_ix"], 1)

    sol_perx = np.zeros(shape=(len(requests),))
    np.put(sol_perx, results["prop"]["served_reqs_ix"], 1)

    opt_results: Dict[str, OptResult] = {
        "scaloran": OptResult(
            exec_time=results["scaloran"]["exec_time"],
            obj_raw=results["scaloran"]["obj"],
            obj=results["scaloran"]["obj"],
            # solution=sol_scaloran,
            solution=results["scaloran"]["z"],
        ),
        "scaloran-cons": OptResult(
            exec_time=results["scaloran-var"]["exec_time"],
            obj_raw=results["scaloran-var"]["obj"],
            obj=results["scaloran-var"]["obj"],
            # solution=sol_scaloran_cons,
            solution=results["scaloran-var"]["z"],
        ),
        "perx": OptResult(
            exec_time=results["prop"]["exec_time"],
            obj_raw=results["prop"]["obj"],
            obj=results["prop"]["obj"],
            # solution=sol_perx,
            solution=results["prop"]["z"],
            num_iters=results["prop"]["num_iters"]
        ),
    }

    return opt_results


if __name__ == "__main__":
    pass
    # main(2, 3, False)
    # main(5, 3, False)
    # main(10, 3, False)
    # main(20, 3, False)

    # main(5, 1, True, load_path="r5.pkl")
    # main(20, 1, True, load_path="r20.pkl")

    # main(5, 1, True)
    # main(20, 1, True)