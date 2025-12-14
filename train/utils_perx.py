import os
import pandas as pd
import numpy as np

from models.types import *


def get_p95_lam_ni_ret_Aten(app1, app2, lam, ni, path, metric="m"):
    path1 = os.path.join(path, f"{app1}_{app2}", f"{ni}", f"{lam}")

    tenant_requests: List[TenantRequest] = load_tenant_requests(os.path.join(path1, "tenant_requests.json"))
    tenant_requests_map : Dict[int, TenantRequest] = {}
    service_times_map: Dict[int, float] = {}

    for tenant in tenant_requests:
        t_id = tenant.id
        df = pd.read_csv(os.path.join(path1, f"tenant-{t_id}.csv"))

        df.dropna(inplace=True)

        if len(df) == 0:
            sts = 2 * 60
        else:
            match metric:
                case "m":
                    sts = np.mean(df["service_times"])
                case "p75":
                    sts = np.percentile(df["service_times"], 75)
                case "p95":
                    sts = np.percentile(df["service_times"], 95)

        tenant_requests_map[t_id] = tenant_requests_map.get(t_id, tenant)
        service_times_map[t_id] = sts

    return tenant_requests_map, service_times_map

def get_p95_lam_ret_all_ni_ten0(app1, app2, lam, path, all_max=False):
    path1 = os.path.join(path, f"{app1}_{app2}")

    num_instances = os.listdir(path1)

    num_instances = list(map(int, num_instances))
    num_instances.sort()
    ret_sts: List[Dict[int, float]] = []

    for ni in num_instances:
        teanant_requests_map, service_times_map = get_p95_lam_ni_ret_Aten(app1, app2, lam, ni, path)

        ret_sts.append({
            "ni": ni,
            # "sts": service_times_map[0],
            "sts": service_times_map[1] if not all_max else max(service_times_map.values()),
        })
    
    ret_df = pd.DataFrame(ret_sts)
    ret_df = ret_df.sort_values(by="ni")
    return ret_df


def get_p95_ni_ret_all_lam_ten0(app1, app2, ni, path):
    path1 = os.path.join(path, f"{app1}_{app2}", f"{ni}")

    lams = os.listdir(path1)

    lams = list(map(float, lams))
    lams.sort()
    ret_sts: List[Dict[str, Any]] = []

    for lam in lams:
        teanant_requests_map, service_times_map = get_p95_lam_ni_ret_Aten(app1, app2, lam, ni, path)

        ret_sts.append({
            "lam": lam,
            "sts": service_times_map[0],
        })
    
    ret_df = pd.DataFrame(ret_sts)
    ret_df = ret_df.sort_values(by="lam")
    return ret_df


def get_p95_app1_app2_all_ten0(app1, app2, path, metric="m"):
    path1 = os.path.join(path, f"{app1}_{app2}")

    num_instances = os.listdir(path1)

    num_instances = list(map(int, num_instances))
    num_instances.sort()
    ret_sts: List[Dict[int, float]] = []

    for ni in num_instances:
        lams = os.listdir(os.path.join(path1, f"{ni}"))
        lams = list(map(float, lams))
        lams.sort()

        for lam in lams:
            path2 = os.path.join(path1, f"{ni}", f"{lam}")
            teanant_requests_map, service_times_map = get_p95_lam_ni_ret_Aten(app1, app2, lam, ni, path, metric=metric)

            ret_sts.append({
                "ni": ni,
                "lam": lam,
                # tenant id of the first tenant (0 or 1)
                # "sts": service_times_map[0],
                "sts": service_times_map[1],
            })
    
    ret_df = pd.DataFrame(ret_sts)
    return ret_df

def get_ni_sts_curve_from_sig_params(sig_params, lam, ni_start=2, ni_end=16):
    a1 = sig_params["alpha1"]
    b1 = sig_params["beta1"]
    a2 = sig_params["alpha2"]
    b2 = sig_params["beta2"]
    a3 = sig_params["alpha3"]
    b3 = sig_params["beta3"]

    nis = list(range(ni_start, ni_end, 2))
    y = []

    for ni in nis:
        t1 = a1 + b1 * ni
        t2 = a2 + b2 * ni
        t3 = a3 + b3 * ni

        sig = t1 / (1 + np.exp(-t2 * (lam - t3)))
        y.append(sig)
    
    return nis, y

