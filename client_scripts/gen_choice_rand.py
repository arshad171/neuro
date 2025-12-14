import json
import os
import random
import numpy as np
import argparse
from models.types import *
from docker.utils import get_app_arrival_rate, get_app_num_instances

OUT_FILE = "tenant_requests.json"
MIN_NUM_INSTANCES = min(NUM_INSTANCES)
# MIN_MEM_REQ = min(list(APP_MEM_REQS.values()))

parser = argparse.ArgumentParser(description="kwargs")
parser.add_argument("--out-folder", type=str, help="output folder to dump the config", default=".")
parser.add_argument("--config", type=str, help="explicit config to deploy, else random_u", default=None)
parser.add_argument("--mem-limit", type=int, help="memory limit", default=-1)
parser.add_argument("--rand-app-cat", type=str, help="v1|v2", default="v2")
parser.add_argument("--use-gpu", action='store_true', help="uses mem reqs of GPU apps", default=False)

args = parser.parse_args()
arg_out_folder = args.out_folder
arg_config = args.config
arg_rand_app_cat = args.rand_app_cat
arg_use_gpu = args.use_gpu
arg_mem_limit = int(args.mem_limit)

tenant_requests: List[TenantRequest] = []

# available_capacity = 40
# cap1: 24 - 32, 17 - 23
# cap2: 16 - 24, 8, 16
# cap3: 8 - 16
if arg_mem_limit == -1:
    available_capacity = random.randint(8, 16)
else:
    available_capacity = arg_mem_limit

print("capacity limit", available_capacity)

if arg_config:
    deployments = arg_config.split(",")
    tenant_requests = []
    id = 1
    for deployment in deployments:
        app_deployment_config = deployment.split("|")
        app_deployment_config_dict = {}
        for config in app_deployment_config:
            key, val = config.split("=")
            app_deployment_config_dict[key] = val
        
        app = app_deployment_config_dict.get("app", None)
        lam = app_deployment_config_dict.get("lam", None)
        ni = app_deployment_config_dict.get("ni", None)

        if lam is None and ni != "none":
            ni = float(ni)

            lam = get_app_arrival_rate(app, ni, exact=True)
        elif lam is not None and ni == "none":
            lam = float(lam)

            ni = get_app_num_instances(app, lam)

        else:
            lam = float(lam)
            ni = int(ni)

        # deployments_list.append(app_deployment_config_dict)
        tenant_requests.append(TenantRequest(
            id=id,
            application=app,
            arrival_rate=lam,
            num_instances=ni,
        ))

        id += 1

else:
    id = 1
    if arg_rand_app_cat == "v1":
        applications = APPLICATIONS_E1
    elif arg_rand_app_cat == "v2":
        applications = APPLICATIONS_E1 + APPLICATIONS_E2

    applications = ["tstr-lstm", "iclf-mnet"]
    
    if arg_use_gpu:
        applications = list(set(applications) - set(APPLICATIONS_EXECLUDE_GPU))
    
    print(applications)

    for _ in range(500):
        if (available_capacity <= 0):
            break
        
        t_app = random.choice(applications)

        if t_app in APPLICATIONS_E_LOW:
            if not arg_use_gpu:
                if t_app in APPLICATIONS_E_LOW_EXCL:
                    if random.random() < 0.4:
                        continue

                elif random.random() < 0.4:
                    continue

            # t_num_instances = int(random.randint(*NUM_INSTANCES_LOW))
            # t_arrival_rate = t_num_instances

            t_num_instances = int(random.randint(*NUM_INSTANCES_LOW))
            if arg_use_gpu:
                t_arrival_rate = random.randint(*ARRIVAL_RATES)
            else:
                t_arrival_rate = random.randint(*ARRIVAL_RATES_LOW)

            t_arrival_rate /= t_num_instances
        else:
            t_arrival_rate = random.randint(*ARRIVAL_RATES)
            # t_arrival_rate = int(t_arrival_rate / 2)
            t_num_instances = int(random.randint(*NUM_INSTANCES))
            # t_num_instances = max(1, int(t_num_instances / 2))
            t_arrival_rate /= t_num_instances
        

        app_mem_req = t_num_instances * get_app_mem_req(t_app, arg_use_gpu)
        if available_capacity - app_mem_req <= 0:
            continue

        
        tenant_requests.append(TenantRequest(id=id, application=t_app, arrival_rate=t_arrival_rate, num_instances=t_num_instances))
        id += 1

        # available_capacity -= t_num_instances
        available_capacity -= app_mem_req

    print(f"{available_capacity=}")
    total_mem = 0
    total_ni = 0


    for req in tenant_requests:
        print(f"{req.application:<10} \t {req.num_instances:>5}")
        total_mem += req.num_instances *  get_app_mem_req(req.application, arg_use_gpu)
        total_ni += req.num_instances

    print(f"{total_mem=} | {total_ni=}")

out_data = {
    "tenant_requests": [req._asdict() for req in tenant_requests]
    # "tenant_requests": tenant_requests
}

json.dump(out_data, open(os.path.join(arg_out_folder, "tenant_requests.json"), "w"), indent=4)
