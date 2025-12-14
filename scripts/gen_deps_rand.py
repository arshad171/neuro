import json
import os
import sys
import shutil
import yaml
from models.types import *
from .utils import get_sample_kube_dep

import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="kwargs")

# Define keyword arguments (options)
parser.add_argument("--out-folder", type=str, help="output folder", default=".")

# Parse the arguments
args = parser.parse_args()

arg_out_folder = args.out_folder

dir_path = f"{arg_out_folder}/kube_deps_p5"

if os.path.exists(dir_path):
    shutil.rmtree(dir_path)

os.mkdir(dir_path)

data = json.load(open(os.path.join(arg_out_folder, "tenant_requests.json")))
tenant_requests: List[TenantRequest] = [TenantRequest(**req) for req in data["tenant_requests"]]

kube_deps = []
port_counter = 31000
emul_tenants = []
for tenant_request in tenant_requests:
    t_id = tenant_request.id
    t_app = tenant_request.application
    t_num_instances = tenant_request.num_instances

    if mem_limit := APP_MEM_REQS.get(t_app, None):
        mem_limit = round(mem_limit * 1.2, 1)
    else:
        mem_limit = None
    print(f"{t_app}, {mem_limit=}")
    kube_dep = get_sample_kube_dep(t_app, t_num_instances, port_counter, t_id, env_proc_title=f"kube_{t_id}", memory_limit=mem_limit)
    # kube_dep = get_sample_kube_dep(t_app, t_num_instances, port_counter, t_id, env_proc_title=f"kube_{t_id}")

    emul_tenant = tenant_request._replace(port=port_counter)
    emul_tenants.append(emul_tenant)


    port_counter += 1
    kube_deps.extend(kube_dep)


with open(os.path.join(dir_path, f"kube_deps.yaml"), "w") as out_file:
    for i, doc in enumerate(kube_deps):
        if i > 0:
            out_file.write('---\n')
        yaml.safe_dump(doc, out_file)

out_data = {
    "tenant_requests": [req._asdict() for req in emul_tenants]
    # "tenant_requests": tenant_requests
}

json.dump(out_data, open(os.path.join(arg_out_folder, "tenant_requests.json"), "w"), indent=4)