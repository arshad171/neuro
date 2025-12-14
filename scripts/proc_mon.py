import shlex
import json
import os
import argparse
import psutil
import shutil
import setproctitle
import subprocess

from models.types import *

setproctitle.setproctitle("my_pref_logger")

def get_pid_by_name(process_name):
    pids = []
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if proc.info['name'] == process_name:
            pids.append(proc.info['pid'])

    return pids

def run_perf_profiler(pid, out_file):
    # command = f'sleep 5 && sudo perf stat -p {pid} -I 1000 -e cpu-clock,cache-references,cache-misses,major-faults,minor-faults -x"|" 2>{out_file} 1>{out_file}'
    command = f"echo hitman12345 | sudo -S perf stat -p {pid} -I 1000 -e cpu-clock,cache-references,cache-misses,page-faults -o {out_file} -D 5 --no-big-num"
    print(command)
    print(shlex.split(command))

    try:
        # process = subprocess.Popen(shlex.split(command), shell=True)
        process = subprocess.Popen(command, shell=True)

        # stdout, stderr = process.communicate()
        # print(stdout, stderr)

    except Exception as e:
        print(e)

# Initialize the parser
parser = argparse.ArgumentParser(description="kwargs")
parser.add_argument("--out-folder", type=str, help="output folder", default=".")

args = parser.parse_args()

arg_out_folder = args.out_folder

dir_path = f"{arg_out_folder}/proc_metrics"

if os.path.exists(dir_path):
    shutil.rmtree(dir_path)

os.mkdir(dir_path)

tenant_requests: List[TenantRequest] = load_tenant_requests(os.path.join(arg_out_folder, "tenant_requests.json"))

for ix, tenant in enumerate(tenant_requests):
    tenant_pids = get_pid_by_name(process_name=f"kube_{tenant.id}")

    temp: TenantRequest = tenant._replace(pids=tenant_pids)
    tenant_requests[ix] = temp

dump_tenant_requests(tenant_requests, os.path.join(arg_out_folder, "tenant_requests.json"))

for tenant in tenant_requests:
    path1 = os.path.join(dir_path, f"tenant_{tenant.id}")
    os.mkdir(path1)
    for pid in tenant.pids:
        path2 = os.path.join(path1, f"{pid}.raw")

        run_perf_profiler(pid, path2)

# pid = "1680308"

# path = "log"
# run_perf_profiler(pid, path)

