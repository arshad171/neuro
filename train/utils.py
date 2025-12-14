import random
import os
import pandas as pd
import numpy as np
import json
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
import torch as th
from tqdm import tqdm

from models.types import *

def get_runs(path: str) -> List[Deployment]:
    ds_runs_folders = os.listdir(path)
    ds_runs: List[Deployment] = []
    for run_folder in tqdm(ds_runs_folders):
        path1 = os.path.join(path, run_folder)

        if not os.path.isdir(path1):
            continue

        deployed_tenant_requests: List[TenantRequest] = load_tenant_requests(os.path.join(path1, "tenant_requests.json"))

        application_deployments: List[ApplicationDeploymnent] = []
        skip_flag = False

        for tenant in deployed_tenant_requests:
            if skip_flag:
                break

            try:
                # represent the tenant deployment
                tenant_service_metrics = pd.read_csv(os.path.join(path1, f"tenant-{tenant.id}.csv"))
                tenant_service_metrics.dropna(inplace=True)
                # append to application_deployments

                application_deployment_low_metics: List[LowLevelMetric] = []
                # TODO: gather low level metrics
                if os.path.exists(os.path.join(path1, "proc_metrics", f"tenant_{tenant.id}")):
                    path2 = os.path.join(path1, "proc_metrics", f"tenant_{tenant.id}")
                else:
                    path2 = os.path.join(path1, "proc_metrics", f"{tenant.id}")
                tenant_low_metrics_files = os.listdir(path2)
                tenant_low_metrics_files = list(filter(lambda x: x.endswith(".csv"), tenant_low_metrics_files))

                low_metrics_dfs = []
                for file in tenant_low_metrics_files:
                    df = pd.read_csv(os.path.join(path2, file))

                    df_pivoted = df.pivot_table(
                        index='time', 
                        columns='event', 
                        values=['value'], 
                        aggfunc='first'
                    )
                    df_pivoted.dropna(inplace=True)

                    low_metrics_dfs.append(df_pivoted)
                
                if len(low_metrics_dfs) > 0:
                    low_metrics_df = pd.concat(low_metrics_dfs, axis=0)
                    low_metrics = low_metrics_df.describe().loc[["mean", "std"]].to_dict()

                    for metric, description in low_metrics.items():
                        application_deployment_low_metics.append(LowLevelMetric(
                            name=metric[1],
                            mean=description["mean"],
                            std=description["std"],
                        ))

                if len(tenant_service_metrics["service_times"]) > 0:
                    sts_mean = np.mean(tenant_service_metrics["service_times"])
                    sts_p75 = np.percentile(tenant_service_metrics["service_times"], 75)
                    sts_p95 = np.percentile(tenant_service_metrics["service_times"], 95)
                else:
                    print("exceed time limit", tenant.application)
                    sts_mean = sts_p75 = sts_p95 = 2 * 60

                application_deployment = ApplicationDeploymnent(
                    id=f"{run_folder}.{tenant.id}",
                    application=tenant.application,
                    arrival_rate=tenant.arrival_rate,
                    num_instances=tenant.num_instances,
                    # service_time=sts_p95,
                    service_time_m=sts_mean,
                    service_time_p75=sts_p75,
                    service_time_p95=sts_p95,
                    low_metrics=application_deployment_low_metics,
                )

                application_deployments.append(application_deployment)
            except FileNotFoundError as e:
                skip_flag = True
                print("file not found error", tenant.application)
                continue
        
        if not skip_flag:
            ds_runs.append(Deployment(application_deployments=application_deployments))
    
    return ds_runs

def get_datapoints(ds_runs:List[Deployment], augment_zero_pred=True, bootstrap=True, use_gpu=False) -> List[Datapoint]:

    if use_gpu:
        low_proc_metrics = LOW_METRICS_GPU
    else:
        low_proc_metrics = LOW_METRICS

    datapoints: List[Datapoint] = []
    skipped_counter = 0

    for ds_run in tqdm(ds_runs):
        # df_ds_run = pd.DataFrame(ds_run.get_list()).groupby(by="application").drop(labels=["service_time"], axis=1).reset_index().to_dict(orient="records")

        all_deployments_run = ds_run.get_list()

        for ix in range(len(all_deployments_run)):
            try: 
                deployment_x = all_deployments_run[ix]
                deployment_context = all_deployments_run[:ix] + all_deployments_run[ix+1:]

                assert deployment_x["application"] is not None, "app is none"

                df_deployment_context = pd.DataFrame(deployment_context).groupby(by="application").agg({"num_instances": "sum", "arrival_rate": "mean", "service_time_m": "max", "service_time_p75": "max", "service_time_p95": "max"}).reset_index()
                # the service times of the context does not matter
                # df_deployment_context["service_time"] = None

                app_dep_x = ApplicationDeploymnent(
                    id=deployment_x["id"],
                    application=deployment_x["application"],
                    arrival_rate=deployment_x["arrival_rate"],
                    num_instances=deployment_x["num_instances"],
                    # service_time=deployment_x["service_time"],
                    service_time_m=deployment_x["service_time_m"],
                    service_time_p75=deployment_x["service_time_p75"],
                    service_time_p95=deployment_x["service_time_p95"],
                    low_metrics=[LowLevelMetric(
                        name=metric,
                        mean=deployment_x[f"{metric}_mean"],
                        std=deployment_x[f"{metric}_std"],
                    ) for metric in low_proc_metrics]
                )
                server_context = []
                for dep in df_deployment_context.to_dict(orient="records"):
                    app_dep = ApplicationDeploymnent(
                        id=dep.get("id", None),
                        application=dep["application"],
                        arrival_rate=dep["arrival_rate"],
                        num_instances=dep["num_instances"],
                        # service_time=dep["service_time"],
                        service_time_m=dep["service_time_m"],
                        service_time_p75=dep["service_time_p75"],
                        service_time_p95=dep["service_time_p95"],
                        low_metrics=[]
                    )
                    server_context.append(app_dep)

                if len(server_context) == 0:
                    print("(1) skipping, no server context")
                else:
                    datapoints.append(Datapoint(
                        app_dep_x,
                        server_context=server_context,
                    ))

                if augment_zero_pred and random.random() < 0.5:
                    n_rep = 2 if app_dep_x.application in APPLICATIONS_E_LOW else 1

                    for _ in range(n_rep):
                        zero_app_dep_x = app_dep_x._replace(num_instances=random.uniform(0, 0.05))
                        zero_app_dep_x = zero_app_dep_x._replace(service_time=0, service_time_m=0, service_time_p75=0, service_time_p95=0)

                        if len(server_context) == 0:
                            print("(2) skipping, no server context")
                        else:
                            datapoints.append(Datapoint(
                                zero_app_dep_x,
                                server_context=drop_random(server_context),
                            ))

            except KeyError as e:
                print("skipped(2)", skipped_counter)
                print(e)
                skipped_counter += 1

        df_all_deployments_run = pd.DataFrame(all_deployments_run)
        df_all_deployments_run.reset_index(inplace=True)
        
        if df_all_deployments_run.empty or len(df_all_deployments_run) == 0:
            print("empty df")
            continue

        if bootstrap:
            for app in df_all_deployments_run["application"].unique():
                df_deployments_app = df_all_deployments_run[df_all_deployments_run['application'] == app]
                df_deployments_app_ix = df_deployments_app.index.to_list()
                df_deployments_app_ixc = subsets_bitwise(df_deployments_app_ix)

                for ixc in df_deployments_app_ixc:
                    try:
                        df_deployments_app_c = df_deployments_app.iloc[df_deployments_app.index.isin(ixc)]
                        df_deployments_app_other = df_deployments_app.iloc[~df_deployments_app.index.isin(ixc)]

                        # TODO: the aggregated arrival rate here must be the sum
                        # or the weighted mean over the merged instances and not the mean
                        aggr = {"num_instances": "sum", "arrival_rate": "mean", "service_time_m": "mean", "service_time_p75": "mean", "service_time_p95": "mean"}
                        for metric in low_proc_metrics:
                            aggr[f"{metric}_mean"] = "mean"
                            aggr[f"{metric}_std"] = "mean"

                        deployment_x = df_deployments_app_c.agg(aggr)
                        if deployment_x.isna().any():
                            print("nan!, skipping")
                            continue

                        deployment_x = deployment_x.to_dict()
                        deployment_x["application"] = app

                        df_deployment_context = pd.concat([
                            df_all_deployments_run[df_all_deployments_run['application'] != app],
                            df_deployments_app_other,
                        ])

                        df_deployment_context = df_deployment_context.groupby(by="application").agg({"num_instances": "sum", "arrival_rate": "mean", "service_time_m": "max", "service_time_p75": "max", "service_time_p95": "max"}).reset_index()

                        app_dep_x = ApplicationDeploymnent(
                            id=deployment_x.get("id", None),
                            application=deployment_x["application"],
                            arrival_rate=deployment_x["arrival_rate"],
                            num_instances=deployment_x["num_instances"],
                            # service_time=deployment_x["service_time"],
                            service_time_m=deployment_x["service_time_m"],
                            service_time_p75=deployment_x["service_time_p75"],
                            service_time_p95=deployment_x["service_time_p95"],
                            low_metrics=[LowLevelMetric(
                                name=metric,
                                mean=deployment_x[f"{metric}_mean"],
                                std=deployment_x[f"{metric}_std"],
                            ) for metric in low_proc_metrics]
                        )

                        server_context = []

                        for dep in df_deployment_context.to_dict(orient="records"):
                            app_dep = ApplicationDeploymnent(
                                id=dep.get("id", None),
                                application=dep["application"],
                                arrival_rate=dep["arrival_rate"],
                                num_instances=dep["num_instances"],
                                # service_time=dep["service_time"],
                                service_time_m=dep["service_time_m"],
                                service_time_p75=dep["service_time_p75"],
                                service_time_p95=dep["service_time_p95"],
                                low_metrics=[]
                            )
                            server_context.append(app_dep)
                        if len(server_context) == 0:
                            print("(3) skipping, no server context")
                        else:
                            datapoints.append(Datapoint(
                                x=app_dep_x,
                                server_context=server_context
                            ))
                    except KeyError as e:
                        print("skipped(1)", skipped_counter)
                        print(e)
                        skipped_counter += 1

    return datapoints

def get_datapoints_pairwise_exps(path) -> List[Datapoint]:
    datapoints = []
    for (a, ad) in zip(APPLICATIONS_E, APPLICATIONS_E):
        path1 = os.path.join(path, f"{a}_{ad}")
        folders = os.listdir(path1)
        for f in folders:
            path2 = os.path.join(path1, f)

            runs = get_runs(path2)
            dps = get_datapoints(runs, augment_zero_pred=True)
            datapoints.extend(dps)
    
    return datapoints


def drop_random(l):
    k = max(1, len(l) // 2)
    k = random.randint(1, k)
    ll = random.sample(l, k=len(l) - k)

    return ll

def subsets_bitwise(lst):
    n = len(lst)
    return [[lst[j] for j in range(n) if (i & (1 << j))] for i in range(1, 1 << n)]

def load_datapoints(path) -> List[Datapoint]:
    return pickle.load(open(path, "rb"))

class CustomDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        self.datapoints: List[Datapoint] = pickle.load(open(path, "rb"))

        for dp in self.datapoints:
            assert dp.x.application is not None, "app is none"


    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        datapoint: Datapoint = self.datapoints[index]

        feature_app = th.zeros(size=(len(APPLICATIONS),))

        feature_deployment = th.zeros(size=(2,))

        feature_server_context = th.zeros(size=(2 * len(APPLICATIONS),))

        # one hot enocde the application ID
        feature_app[APPLICATIONS.index(datapoint.x.application)] = 1

        # normalize
        feature_deployment[0] = datapoint.x.num_instances / 16
        feature_deployment[1] = datapoint.x.arrival_rate / 10

        for app_dep in datapoint.server_context:
            app_ix = APPLICATIONS.index(app_dep.application)
            feature_server_context[app_ix * 2] = app_dep.num_instances / 16
            feature_server_context[app_ix * 2 + 1] = app_dep.arrival_rate / 10
        
        x = {
            "feature_app": feature_app,
            "feature_deployment": feature_deployment,
            "feature_server_context": feature_server_context,
        }

        y = datapoint.x.service_time

        return x, th.tensor([y], dtype=th.float)

class CustomDatasetSetRepr(Dataset):
    def __init__(self, path):
        super().__init__()

        self.datapoints: List[Datapoint] = pickle.load(open(path, "rb"))

        for dp in self.datapoints:
            assert dp.x.application is not None, "app is none"


    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        datapoint: Datapoint = self.datapoints[index]

        # to predict
        y = datapoint.x.service_time
        x_feature = th.zeros(size=(1 + 2, ))

        x_feature[0] = APPLICATIONS.index(datapoint.x.application)
        x_feature[1] = datapoint.x.num_instances / 16
        x_feature[2] = datapoint.x.arrival_rate / 10
        # a set of application deployments
        features = []

        length = 0
        for app_dep in datapoint.server_context:
            length += 1
            feature_app = th.zeros(size=(1,))
            feature_load = th.zeros(size=(2,))

            # one hot enocde the application ID
            feature_app[0] = APPLICATIONS.index(app_dep.application)

            feature_load[0] = app_dep.num_instances / 16
            feature_load[1] = app_dep.arrival_rate / 10

            feature = th.hstack([feature_app, feature_load])
            feature = feature.unsqueeze(1)

            features.append(feature)

        for _ in range(len(APPLICATIONS) - length):
            features.append(th.zeros(3, 1))

        x = {
            "x_features": x_feature,
            "features": th.hstack(features),
            "lengths": th.tensor([length]),
        }

        return x, th.tensor([y], dtype=th.float)

class CustomDatasetSetReprHydra(Dataset):
    def __init__(self, path, use_base_param: bool = False, datapoints=None, scaling_stats_name=None, use_gpu=False):
        super().__init__()

        self.use_base_param = use_base_param
        if scaling_stats_name is None:
            self.app_exec_max = APP_EXEC_MAX
        else:
            stats = json.load(open(os.path.join("models", scaling_stats_name), "r"))
            self.app_exec_max = stats["max"]
            self.app_exec_min = stats["min"]
        
        self.low_metrics = LOW_METRICS if not use_gpu else LOW_METRICS_GPU
        

        if path is not None:
            self.datapoints: List[Datapoint] = pickle.load(open(path, "rb"))
        else:
            self.datapoints = datapoints

        for dp in self.datapoints:
            assert dp.x.application is not None, "app is none"
        
        self.low_metrics_norma = json.load(open("models/norma.json", "r"))
        self.low_metrics_norma = self.low_metrics_norma["low_metrics"]

        if use_gpu:
            self.applications = APPLICATIONS_E_GPU
        else:
            self.applications = APPLICATIONS_E



    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        datapoint: Datapoint = self.datapoints[index]

        # to predict
        # y = datapoint.x.service_time
        y1 = datapoint.x.service_time_m
        y2 = datapoint.x.service_time_p75
        y3 = datapoint.x.service_time_p95

        exec_scaling = self.app_exec_max[datapoint.x.application]

        if self.use_base_param:
            # y /= APP_BASE_EXEC[datapoint.x.application]
            # y /= APP_MAX_EXEC[datapoint.x.application]

            # y1 /= APP_EXEC_MAX[datapoint.x.application]
            # y2 /= APP_EXEC_MAX[datapoint.x.application]
            # y3 /= APP_EXEC_MAX[datapoint.x.application]

            # pass
            y1 /= exec_scaling[0]
            y2 /= exec_scaling[1]
            y3 /= exec_scaling[2]

            # y = (y - APP_BASE_EXEC[datapoint.x.application]) / (APP_MAX_EXEC[datapoint.x.application] - APP_BASE_EXEC[datapoint.x.application])

        x_feature = th.zeros(size=(1 + 2 + 1, ))

        lm_x_feature = th.zeros(size=(len(self.low_metrics),))

        # APPLICATIONS_E = APPLICATIONS_E1 + APPLICATIONS_E2
        x_feature[0] = self.applications.index(datapoint.x.application)
        # x_feature[0] = APPLICATIONS_V2_MAPPING.get(datapoint.x.application)
        x_feature[1] = datapoint.x.num_instances / 16
        x_feature[2] = datapoint.x.arrival_rate / 10

        # x_feature[3] = APPLICATION_ARCHS[datapoint.x.application]["size"] / 1_000_000
        x_feature[3] = 0.0

        for metric in datapoint.x.low_metrics:
            lm_x_feature[self.low_metrics.index(metric.name)] = metric.mean / self.low_metrics_norma[metric.name]

        # a set of application deployments
        features = []

        length = 0
        for app_dep in datapoint.server_context:
            length += 1
            feature_app = th.zeros(size=(1,))
            feature_load = th.zeros(size=(2,))

            # one hot enocde the application ID
            feature_app[0] = self.applications.index(app_dep.application)
            # x_feature[0] = APPLICATIONS_V2_MAPPING.get(datapoint.x.application)

            feature_load[0] = app_dep.num_instances / 16
            feature_load[1] = app_dep.arrival_rate / 10

            feature = th.hstack([feature_app, feature_load])
            feature = feature.unsqueeze(1)

            features.append(feature)

        for _ in range(len(self.applications) - length):
        # for _ in range(APPLICATIONS_V2_MAX - length):
            features.append(th.zeros(3, 1))

        x = {
            "x_features": x_feature,
            "features": th.hstack(features),
            "lengths": th.tensor([length]),
        }

        return x, (th.tensor([y1, y2, y3], dtype=th.float), lm_x_feature)
