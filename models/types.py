import json
import os
from typing import Any, NamedTuple, List, Dict, Optional
from enum import Enum
from collections import defaultdict

HOME_PATH = "/home/neuro"

METHODS = ["scaloran", "scaloran-cons", "perx", "perx++"]

APPLICATIONS = ["dnn", "lstm", "cnn", "rf"]

APPLICATIONS_V1 = ["dnn-v1", "lstm-v1", "cnn-v1", "rf-v1"]

APPLICATIONS_V21 = [
    "dnn-v1", "lstm-v1", "cnn-v1", "rf-v1",
    "dnn-v2", "lstm-v2", "cnn-v2", "rf-v2",
]

APPLICATIONS_V22 = [
    "dnn-v1", "lstm-v1", "cnn-v1", "rf-v1",
    "dnn-v2", "lstm-v2", "cnn-v2", "rf-v2",
    "bert"
]

APPLICATIONS_E1 = [
    "tclf-gcn",
    "tclf-rf",
    "tstr-lstm",
    "kalman-gru",
    "iclf-mnet",
    "text-bert",
]

APPLICATIONS_E2 = [
    "iclf-efnet",
    "text-tbert",
    "iclf-mvit",
]

APPLICATIONS_E = APPLICATIONS_E1 + APPLICATIONS_E2

APPLICATIONS_E_GPU = [
    "tclf-gcn",
    # "tclf-rf",
    "tstr-lstm",
    "kalman-gru",
    "iclf-mnet",
    "text-bert",
    "iclf-efnet",
    "text-tbert",
    "iclf-mvit",
]


APPLCIATIONS_E_SORTED = [
    "TCLF-RF",
    "TSTR-LSTM",
    "KNet-GRU",
    "TCLF-GCN",
    "ICLF-MNET",
    "TEXT-BERT",
    "ICLF-EFNET",
    "TEXT-TBERT",
    "ICLF-MVIT",
]

APPLICATIONS_V2_MAPPING = {
    "dnn-v1": 0,
    "lstm-v1": 1,
    "cnn-v1": 2,
    "rf-v1": 3,

    "dnn-v2": 0,
    "lstm-v2": 1,
    "cnn-v2": 2,
    "rf-v2": 3,

    "bert": 4,
}

APPLICATIONS_V2_MAX = 5

APPLICATION_ARCHS = {
    "dnn-v1": {"size": 152_010},
    "lstm-v1": {"size": 161_821},
    "cnn-v1": {"size": 50_000},
    "rf-v1": {"size": 100 * 10 * 100},

    "dnn-v2": {"size": 452_810},
    "lstm-v2": {"size": 446_601},
    "cnn-v2": {"size": 538_218},
    "rf-v2": {"size": 500 * 20 * 100},

    "bert": {"size": 28_763_649 / 1_000},
}

APP_MEM_REQS = {
    "dnn-v1": 0.9,
    "lstm-v1": 1.1,
    "cnn-v1": 0.65,
    "rf-v1": 0.65,

    "dnn-v2": 3.0,
    "lstm-v2": 2.5,
    "cnn-v2": 0.65,
    "rf-v2": 0.65,

    "bert": 4,

    ###
    "tclf-gcn": 0.5, # 50 ms
    "tclf-rf": 0.3, # 25 ms (can be a real-time app)
    "tstr-lstm": 1.0, # 200 ms (limit memory)
    "kalman-gru": 0.3, # 5 ms (can be a real-time app)
    "iclf-mnet": 0.8, # 20 ms
    "text-bert": 0.6, # 120 ms

    "iclf-efnet": 1.3, # 200 ms
    "text-tbert": 0.7, # 15 ms
    "iclf-mvit": 0.5, # 60 ms
    "iclf-resnet": 1.0, # 0.388
}

APP_MEM_REQS_GPU = {
    "tclf-gcn": 0.4, # 0.32
    # "tclf-rf": 0.3,
    "tstr-lstm": 0.4, # 0.304
    "kalman-gru": 0.4, # 0.330
    "iclf-mnet": 0.5, # 0.404
    "text-bert": 0.8, # 0.766
    "iclf-efnet": 0.85, # 0.786
    "text-tbert": 0.4, # 0.318
    "iclf-mvit": 0.45, # 0.388
    "iclf-resnet": 0.5, # 0.388
}

APP_INFERENCE_TARGETS = ["mean", "p75", "p95"]


def get_app_mem_req(app, gpu=False):
    app_mems = APP_MEM_REQS if not gpu else APP_MEM_REQS_GPU
    mem_limit = app_mems.get(app, None)

    if mem_limit is not None:
        mem_limit = mem_limit * 1.2 if not gpu else mem_limit * 0.9
    
    if not gpu:
        mem_limit = round(mem_limit, 1)

    return mem_limit

APP_TASK_ARRIVAL_RATES_E = {
    "tclf-gcn": 1 / 0.132,
    "tclf-rf": 1 / 0.0115,
    "tstr-lstm": 1 / 0.149,
    "kalman-gru": 1 / 0.00508,
    "iclf-mnet": 1 / 0.0602,
    "text-bert": 1 / 0.512,
}

APP_TASK_ARRIVAL_RATES_RANGE_E = {
    "tclf-gcn": [20, 100],
    "tclf-rf": [20, 100],
    "tstr-lstm": [20, 100],
    "kalman-gru": [20, 100],
    "iclf-mnet": [20, 100],
    "text-bert": [2, 20],
}

APP_BASE_EXEC = {
    "dnn-v1": 0.10,
    "lstm-v1": 0.15,
    "cnn-v1": 0.0625,
    "rf-v1": 0.0125,

    "dnn-v2": 0.10,
    "lstm-v2": 0.15,
    "cnn-v2": 0.0625,
    "rf-v2": 0.0125,

    "bert": 0.2,

    "tclf-gcn": 0.04,
    "tclf-rf": 0.01,
    "tstr-lstm": 0.1,
    "kalman-gru": 0.004,
    "iclf-mnet": 0.05,
    "text-bert": 0.1,

    "iclf-efnet": 0.1,
    "text-tbert": 0.01,
    "iclf-mvit": 0.05,
}

APP_MAX_EXEC = {
    # e1e2h
    # 'tclf-gcn': 0.8579182028770447,
    # 'tclf-rf': 0.3032711446285248,
    # 'tstr-lstm': 0.7533926963806152,
    # 'kalman-gru': 0.40500426292419434,
    # 'iclf-mnet': 0.9251695275306702,
    # 'text-bert': 0.7965897917747498,
    # 'iclf-efnet': 1.0,
    # 'text-tbert': 0.7301825284957886,
    # 'iclf-mvit': 1.020246982574463,

    # e1e1oe2h
    'tclf-gcn': 22.0528507232666,
    'tclf-rf': 0.0415102019906044,
    'tstr-lstm': 0.7699626684188843,
    'kalman-gru': 0.08132541179656982 / 3,
    'iclf-mnet': 108.42182922363281,
    'text-bert': 67.54591369628906,
    'iclf-efnet': 120.0,
    'text-tbert': 13.742752075195312,
    'iclf-mvit': 95.1012954711914,
}


def get_app_exec_max(server_rank):
    with open(os.path.join(f"{HOME_PATH}", "models", f"sts_server{server_rank}_cpu.json"), "r") as file:
        stats = json.load(file)
        return stats["max"]
    # if server_rank == 0:
    #     return {
    #         "tclf-gcn": [13.368155479431152, 17.26291275024414, 22.0528507232666],
    #         "tclf-rf": [0.011183061636984348, 0.01432347297668457, 0.0415102019906044],
    #         "tstr-lstm": [0.525345504283905, 0.6639305353164673, 0.7699626684188843],
    #         "kalman-gru": [0.012417145073413849, 0.003949642181396484, 0.08132541179656982],
    #         "iclf-mnet": [102.6425552368164, 105.85325622558594, 108.42182922363281],
    #         "text-bert": [41.83883285522461, 46.11003112792969, 56.48748016357422],
    #         "iclf-efnet": [120.0, 118.8111572265625, 120.0],
    #         "text-tbert": [8.286052703857422, 9.772259712219238, 13.742752075195312],
    #         "iclf-mvit": [81.0706787109375, 88.86546325683594, 95.1012954711914],
    #     }
    # elif server_rank == 1:
    #     return {
    #         "tclf-gcn": [38.02870559692383, 43.92634201049805, 46.85285186767578],
    #         "tclf-rf": [0.01768096722662449, 0.021692097187042236, 0.06558501720428467],
    #         "tstr-lstm": [1.1152732372283936, 1.3969464302062988, 1.8659656047821045],
    #         "kalman-gru": [0.02586260624229908, 0.06358623504638672, 0.11915206909179688],
    #         "iclf-mnet": [121.47722625732422, 121.47722625732422, 121.47722625732422],
    #         "text-bert": [120.0, 120.0, 120.0],
    #         "iclf-efnet": [122.04595947265625, 122.04595947265625, 122.04595947265625],
    #         "text-tbert": [22.31871795654297, 29.796571731567383, 31.28729248046875],
    #         "iclf-mvit": [120.96502685546875, 120.96502685546875, 120.96502685546875],
    #     }

### server_1
# e1e1oe2 (*)
APP_EXEC_MAX = {'tclf-gcn': [13.368155479431152, 17.26291275024414, 22.0528507232666], 'tclf-rf': [0.011183061636984348, 0.01432347297668457, 0.0415102019906044], 'tstr-lstm': [0.525345504283905, 0.6639305353164673, 0.7699626684188843], 'kalman-gru': [0.012417145073413849, 0.003949642181396484, 0.08132541179656982], 'iclf-mnet': [102.6425552368164, 105.85325622558594, 108.42182922363281], 'text-bert': [41.83883285522461, 46.11003112792969, 56.48748016357422], 'iclf-efnet': [120.0, 118.8111572265625, 120.0], 'text-tbert': [8.286052703857422, 9.772259712219238, 13.742752075195312], 'iclf-mvit': [81.0706787109375, 88.86546325683594, 95.1012954711914]}
APP_EXEC_MIN = {'tclf-gcn': [0.04118216037750244, 0.04308032989501953, 0.049125004559755325], 'tclf-rf': [0.005355142056941986, 0.005410194396972656, 0.005952763371169567], 'tstr-lstm': [0.13362009823322296, 0.13778185844421387, 0.1463068723678589], 'kalman-gru': [0.0021471274085342884, 0.0022356510162353516, 0.0026605844032019377], 'iclf-mnet': [0.07093993574380875, 0.06460070610046387, 0.08770449459552765], 'text-bert': [0.16787423193454742, 0.1311308741569519, 0.2003573179244995], 'iclf-efnet': [0.5715664625167847, 0.003205239772796631, 0.44005441665649414], 'text-tbert': [0.011626155115664005, 0.012388646602630615, 0.01636815071105957], 'iclf-mvit': [0.05962417274713516, 0.044938087463378906, 0.06649579852819443]}

### server_2
# cap1 - v1v2v3v4 (*)
# cap2
# u
# APP_EXEC_MAX = {'tclf-gcn': [38.02870559692383, 43.92634201049805, 46.85285186767578], 'tclf-rf': [0.01768096722662449, 0.021692097187042236, 0.06558501720428467], 'tstr-lstm': [1.1152732372283936, 1.3969464302062988, 1.8659656047821045], 'kalman-gru': [0.02586260624229908, 0.06358623504638672, 0.11915206909179688], 'iclf-mnet': [121.47722625732422, 121.47722625732422, 121.47722625732422], 'text-bert': [120.0, 120.0, 120.0], 'iclf-efnet': [122.04595947265625, 122.04595947265625, 122.04595947265625], 'text-tbert': [22.31871795654297, 29.796571731567383, 31.28729248046875], 'iclf-mvit': [120.96502685546875, 120.96502685546875, 120.96502685546875]}
# APP_EXEC_MIN = {'tclf-gcn': [1.1413099765777588, 1.19663667678833, 1.2988543510437012], 'tclf-rf': [0.004654435440897942, 0.0044893622398376465, 0.005244839005172253], 'tstr-lstm': [0.1300969272851944, 0.13441157341003418, 0.16747084259986877], 'kalman-gru': [0.003384184092283249, 0.0033069849014282227, 0.0038050650618970394], 'iclf-mnet': [5.136005401611328, 6.23702335357666, 19.141199111938477], 'text-bert': [3.174311876296997, 3.173732280731201, 13.097439765930176], 'iclf-efnet': [3.9456517696380615, 3.9456517696380615, 3.9456517696380615], 'text-tbert': [0.7971596717834473, 0.44602251052856445, 0.9508369565010071], 'iclf-mvit': [0.8418784141540527, 0.11757850646972656, 0.8418784141540527]}

TENANT_PRICING_SCHEME = {
    "rf-v1": 3.0,
    "cnn-v1": 4.0,
    "lstm-v1": 5.0,
    "dnn-v1": 6.0,

    "rf-v2": 4.0,
    "cnn-v2": 5.0,
    "lstm-v2": 6.0,
    "dnn-v2": 7.0,

    "bert": 8.0,

    "tclf-gcn": 6, #7
    "tclf-rf": 2,
    "tstr-lstm": 5,
    "kalman-gru": 3,
    "iclf-mnet": 10,
    "text-bert": 9,
    "iclf-efnet": 11,
    "text-tbert": 7, #8
    "iclf-mvit": 12,
}

PAYMENT_SLA_MUL = {
    0: 0.8,
    1: 0.9,
    2: 1.0,
}

TENANT_DELAYS_E = {
    "tclf-gcn": [0.13, 0.13*5],
    "tclf-rf": [0.011, 0.011*5],
    "tstr-lstm": [0.14, 0.14*5],
    "kalman-gru": [0.005, 0.005*5],
    "iclf-mnet": [0.06, 0.06*5],
    "text-bert": [0.51, 0.51*5]
}

TENANT_DELAYS_1 = {
    "rf-v1": [0.2, 0.5],
    "cnn-v1": [0.2, 0.5],
    "lstm-v1": [0.2, 0.5],
    "dnn-v1": [0.2, 0.5],

    "rf-v2": [0.2, 0.5],
    "cnn-v2": [0.2, 0.5],
    "lstm-v2": [0.2, 0.5],
    "dnn-v2": [0.2, 0.5],

    "bert": [0.2, 0.5],
}

TENANT_DELAYS_2 = {
    "rf-v1": [0.02, 0.1],
    "cnn-v1": [0.13, 0.25],
    "lstm-v1": [0.4, 0.6],
    "dnn-v1": [0.25, 0.5],
    "rf-v2": [0.03, 0.08],
    "cnn-v2": [0.26, 0.5],
    "lstm-v2": [0.5, 0.8],
    "dnn-v2": [0.5, 0.8],
    "bert": [0.4, 0.8],

    "tclf-gcn": [0.5, 1.0],
    "tclf-rf": [0.015, 0.03],
    "tstr-lstm": [0.5, 1],
    "kalman-gru": [0.005, 0.01],
    "iclf-mnet": [1, 5],
    "text-bert": [1, 5],
    "iclf-efnet": [1, 5],
    "text-tbert": [1, 5],
    "iclf-mvit": [1, 5],
}

APPLICATIONS_RT = [
    "tclf-rf",
    "knet-gru",
    "tstr-lstm"
]

APPLICATIONS_E_LOW = ["tclf-gcn", "text-bert", "iclf-mnet","iclf-efnet", "text-tbert", "iclf-mvit"]
APPLICATIONS_E_LOW_EXCL = ["text-tbert", "tclf-gcn"]
APPLICATIONS_EXECLUDE_GPU = ["tclf-rf"]
APPLICATIONS_RT = ["tstr-lstm", "tclf-rf", "knet-gru"]

LOW_METRICS = [
    "cpu-clock",
    "cache-misses",
    "page-faults",
]

LOW_METRICS_GPU = [
    "sm_util",
]


# lambda: min =1.33, max = 33, mean = 6
# avg. lambda  = 6
ARRIVAL_RATES = [20, 100] # per instance
NUM_INSTANCES = [2, 10]


ARRIVAL_RATES_LOW = [1, 5]
NUM_INSTANCES_LOW = [1, 3]

class TenantRequest(NamedTuple):
    ### request paramters
    id: int
    application: str
    # avg. per instance
    arrival_rate: float
    num_instances: int
    payment: float = 0.0
    delay: float = 0.0
    sla_type: int = 1

    ### additional attrs for processing
    port: int | List = None
    pids: List = []

class LowLevelMetric(NamedTuple):
    name: str
    mean: float
    std: float

class ApplicationDeploymnent(NamedTuple):
    id: str
    application: str
    # avg. per instance
    arrival_rate: float
    num_instances: int
    # possibly other load paramters
    # predition variables
    service_time: float = 0
    service_time_m: float = 0
    service_time_p75: float = 0
    service_time_p95: float = 0

    low_metrics: List[LowLevelMetric] = []

class Deployment(NamedTuple):
    application_deployments: List[ApplicationDeploymnent]

    def get_list(self):
        ret = []
        for dep in self.application_deployments:
            rec = {}
            for k, v in dep._asdict().items():
                if k == "low_metrics":
                    for l_metric in v:
                        rec[f"{l_metric.name}_mean"] = l_metric.mean
                        rec[f"{l_metric.name}_std"] = l_metric.std
                else:
                    rec[k] = v
                
            ret.append(rec)
        # return [obj._asdict() for obj in self.application_deployments]
        return ret

class Datapoint(NamedTuple):
    x: ApplicationDeploymnent
    server_context: List[ApplicationDeploymnent]

def load_tenant_requests(path) -> List[TenantRequest]:
    data = json.load(open(path, "r"))
    tenant_requests: List[TenantRequest] = [TenantRequest(**req) for req in data["tenant_requests"]]

    return tenant_requests

def dump_tenant_requests(tenant_requests, path) -> List[TenantRequest]:
    out_data = {
        "tenant_requests": [req._asdict() for req in tenant_requests]
        # "tenant_requests": tenant_requests
    }

    json.dump(out_data, open(path, "w"), indent=4)


class EmbeddingTypes(Enum):
    USE_EMBEDDING_LAYER = 1
    USE_ONEHOT = 2

class ApplicationCatalog(Enum):
    V1 = 1
    V21 = 2
    V22 = 3
    E1 = 4
    E2 = 5
    E = 6

class OptResult(NamedTuple):
    status: Optional[int] = None
    message: Optional[str] = None
    exec_time: Optional[float] = None
    obj_raw: Optional[float] = None
    obj: Optional[float] = None
    solution: Optional[float] = None
    solution_x: Optional[float] = None
    info: Any = None
    num_iters: Optional[int] = None
