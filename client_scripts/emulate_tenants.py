from collections import defaultdict
import itertools
import os
import subprocess
import json
import sys
import pandas as pd
import numpy as np

import time
import requests
import aiohttp
import asyncio
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

from models.types import *


batch_size = 2

FLUSH_URL = "http://{}:{}/flush"
REQ_URL = "http://{}:{}/request-static"
SERVICE_URL = "http://{}:{}/service-times"

parser = argparse.ArgumentParser(description="kwargs")

parser.add_argument("--out-folder", type=str, help="output folder", default="output")
parser.add_argument("--host", type=str, help="server running the apps / containers. localhost | minikube", required=True)
parser.add_argument("--time-limit", type=float, help="server deployment to emulate (mins)", default=1)
parser.add_argument("--batch-size", type=int, help="batch size", default=1)

args = parser.parse_args()
arg_out_folder = args.out_folder
arg_time_limit = args.time_limit
arg_host = args.host
arg_batch_size = args.batch_size

batch_size = arg_batch_size

if arg_host == "localhost":
    HOST = "0.0.0.0"
elif arg_host == "minikube":
    HOST = "192.168.49.2"

out_path = f"{arg_out_folder}/"
os.makedirs(out_path, exist_ok=True)

TIME_CAP = 60 * arg_time_limit

async def fetch_url(session, url, rate, app_name):
    wait_time = np.random.exponential(scale=1/rate)

    # wait_time = np.random.uniform(low=0.8 * 1/rate, high=1.2 * 1/rate)

    # ret = await wait(rate)
    # wait_time = 1/rate

    start_time = time.perf_counter()  # Start time
    try:
        # async with session.get(url, json={"num_samples": num_samples}) as response:
        async with session.post(url, json={"batch_size": batch_size}) as response:
            await response.text()  # Wait for the response body (optional)
            end_time = time.perf_counter()  # End time
            response_time = end_time - start_time  # Measure time taken
            return url, response.status, response_time, wait_time
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print("skipping request", e)


async def fetch(ports, tenant_ix, rate):
    global wait_till_complete

    reqs_sent = 0
    timeout = aiohttp.ClientTimeout(total=24 * 60 * 60)
    start_time = time.time()
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        results = []

        # while (app_ix == 0 and reqs_to_send > 0) or (app_ix != 0 and wait_till_complete):
        count = 0
        while time.time() - start_time < TIME_CAP:
            count += 1
            wait_time = np.random.exponential(scale=1/rate)
            # wait_time = np.random.uniform(0, 2/rate)
            await asyncio.sleep(wait_time)
            # time.sleep(wait_time)
            if count % 100 == 0:
                print(f"{tenant_ix}: {reqs_sent}")
            # url = urls[count % len(urls)]
            port = ports[count % len(ports)]
            url = REQ_URL.format(HOST, port)
            task = asyncio.create_task(fetch_url(session, url, rate, tenant_ix))
            tasks.append(task)
            # tasks.append(fetch_url(session, url, rate, app_name))
            reqs_sent = reqs_sent + 1
        
        print(tenant_ix, "*** waiting")
        # results = await asyncio.gather(*tasks)

        done, pending = await asyncio.wait(tasks, timeout=TIME_CAP)

        # Gather results from completed tasks
        results = [task.result() for task in done if not task.cancelled() and not task.exception()]

        # Optionally cancel the pending ones
        for task in pending:
            task.cancel()

        for port in ports:
            try:
                async with session.post(FLUSH_URL.format(HOST, port)) as response:
                    res = await response.text()
                    print(res)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print("skipping flush request", e)

        print(tenant_ix, "*** done")

        return results

def run_event_loop(*args, **kwargs):
    asyncio.run(fetch(*args, **kwargs))
    print("finished event_loop")


data = json.load(open(os.path.join(arg_out_folder, "tenant_requests.json")))
tenant_requests: List[TenantRequest] = [TenantRequest(**req) for req in data["tenant_requests"]]

print(tenant_requests)

with ThreadPoolExecutor(max_workers=len(tenant_requests)) as executor:
    futures = []
    for ix, tenant_request in enumerate(tenant_requests):
        tenant_id = tenant_request.id
        replicas = tenant_request.num_instances
        rate = tenant_request.arrival_rate * replicas
        ports = tenant_request.port
        # if isinstance(ports, list):
        #     urls = [REQ_URL.format(HOST, port) for port in ports]
        # else:
        #     urls = [REQ_URL.format(HOST, ports)]

        if not isinstance(ports, list):
            ports = [ports]
        
        future = executor.submit(run_event_loop, ports=ports, tenant_ix=tenant_id, rate=rate)

        futures.append(future)
    
    results = [future.result() for future in futures]


print("***** gather stats")
for ix, tenant_request in enumerate(tenant_requests):
    tenant_id = tenant_request.id
    replicas = tenant_request.num_instances
    ports = tenant_request.port
    rate = tenant_request.arrival_rate * replicas

    if isinstance(ports, list):
        service_urls = [SERVICE_URL.format(HOST, port) for port in ports]
    else:
        service_urls = [SERVICE_URL.format(HOST, ports)]


    num_iters = 100

    dfs = {}
    while num_iters > 0:
        try:
            data = requests.get(random.choice(service_urls), timeout=0.5)
        except requests.exceptions.Timeout:
            print("timeout error", tenant_id)
            continue

        raw_data = data.json()
        print("got data", tenant_id, raw_data["uid"])

        if len(raw_data["service_times"]) == 0:
            print(f"----{num_iters=}")
            print("got no data", tenant_id, raw_data["uid"])

        else:
            uid = raw_data["uid"]
            if not uid in dfs:
                nest = [
                    raw_data["service_times"],
                    raw_data["arrival_times"],
                    raw_data["departure_times"],
                    raw_data["interarrival_times"],
                    raw_data["queueing_times"],
                    raw_data["queue_lens"],
                ]
                df = pd.DataFrame((_ for _ in itertools.zip_longest(*nest)), columns=['service_times', 'arrival_times', "departure_times", "interarrival_times", "queueing_times", "queue_lens"])

                dfs[uid] = df

        num_iters -= 1

    if len(dfs) > 0:
        all_df = pd.concat(list(dfs.values()), axis=0)
        all_df.to_csv(os.path.join(out_path, f"tenant-{tenant_id}.csv"))
