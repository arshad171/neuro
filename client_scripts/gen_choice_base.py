import json
import os
import random
import numpy as np
import argparse
from models.types import *

OUT_FILE = "tenant_requests.json"
MIN_NUM_INSTANCES = min(NUM_INSTANCES)
MIN_MEM_REQ = min(list(APP_MEM_REQS.values()))

parser = argparse.ArgumentParser(description="kwargs")
parser.add_argument("--out-folder", type=str, help="output folder to dump the config", default=".")
parser.add_argument("--app", type=str, help="application name", default=".")
parser.add_argument("--use-gpu", action='store_true', help="uses mem reqs of GPU apps", default=False)

args = parser.parse_args()
arg_out_folder = args.out_folder
arg_app = args.app
arg_use_gpu = args.use_gpu

tenant_requests: List[TenantRequest] = []

# available_capacity = 40
available_capacity = 32

id = 0
t_app = arg_app
t_arrival_rate = random.randint(*ARRIVAL_RATES) / 2
t_num_instances = 1
t_arrival_rate /= t_num_instances

id = 1

tenant_requests.append(TenantRequest(id=id, application=t_app, arrival_rate=t_arrival_rate, num_instances=t_num_instances))

# available_capacity -= t_num_instances
available_capacity -= t_num_instances * get_app_mem_req(t_app, arg_use_gpu)

print(f"{available_capacity=}")
total_mem = 0
total_ni = 0
for req in tenant_requests:
    print(f"{req.application:<10} \t {req.num_instances:>5}")
    total_mem += req.num_instances * get_app_mem_req(t_app, arg_use_gpu)
    total_ni += req.num_instances

print(f"{total_mem=}")

out_data = {
    "tenant_requests": [req._asdict() for req in tenant_requests]
    # "tenant_requests": tenant_requests
}

json.dump(out_data, open(os.path.join(arg_out_folder, "tenant_requests.json"), "w"), indent=4)
