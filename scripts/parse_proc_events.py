import os
import re
import argparse
import json
import pandas as pd

from models.types import *

def parse_raw_data(path):
    parsed_data = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue

            line = line.strip()
            line = re.sub(r'\s+', ' ', line)

            line = line.split(" ")
            ix_hash = line.index("#")
            line_len = len(line[0: ix_hash])
            parsed_row = {
                "time": line[0],
                "count": float(line[1].replace(",", ".")),
                "count_unit": line[2] if line_len == 4 else "",
                "event": line[3 if line_len == 4 else 2],
                "value": float(line[ix_hash + 1].replace(",", ".")),
                "value_unit": " ".join(line[ix_hash + 2:])

            }

            parsed_data.append(parsed_row)
    
    return parsed_data


parser = argparse.ArgumentParser(description="kwargs")
parser.add_argument("--out-folder", type=str, help="output folder", default=".")

args = parser.parse_args()

arg_out_folder = args.out_folder

dir_path = f"{arg_out_folder}/proc_metrics"

tenant_requests: List[TenantRequest] = load_tenant_requests(os.path.join(arg_out_folder, "tenant_requests.json"))

for tenant in tenant_requests:
    path1 = os.path.join(dir_path, f"tenant_{tenant.id}")
    for pid in tenant.pids:
        parsed_data = parse_raw_data(os.path.join(path1, f"{pid}.raw"))

        df = pd.DataFrame(parsed_data)

        df.to_csv(os.path.join(path1, f"{pid}.parsed.csv"))