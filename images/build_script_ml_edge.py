import subprocess
import sys

ML_MODEL_CONFIGS = [
    {"name": "tclf-gcn", "port": "8000", "module": "ml_tclf_gcn"},
    {"name": "tclf-rf", "port": "8001", "module": "ml_tclf_rf"},
    {"name": "tstr-lstm", "port": "8002", "module": "ml_tstr_lstm"},
    {"name": "kalman-gru", "port": "8003", "module": "ml_kalman_gru"},
    {"name": "iclf-mnet", "port": "8004", "module": "ml_iclf_mnet"},
    {"name": "text-bert", "port": "8005", "module": "ml_text_bert"},

    {"name": "iclf-efnet", "port": "8006", "module": "ml_iclf_efnet"},
    {"name": "text-tbert", "port": "8007", "module": "ml_text_tbert"},
    {"name": "iclf-mvit", "port": "`8008", "module": "ml_iclf_mvit"},
]

for config in ML_MODEL_CONFIGS:
    name = config["name"]
    port = config["port"]
    module = config["module"]
    arg_device_type = config.get("arg_device_type", "cpu")

    print("*" * 10, "building", name)


    subprocess.run(["rm", "-rf", "module"])
    subprocess.run(["cp", "-r", f"./{module}", "module"])

    result = subprocess.run([
        "docker", "build",
        "--build-arg", f'ARG_MODULE={module}',
        "--build-arg", f'APP_PORT={port}',
        "--build-arg", f'APP_MODEL_URL={name}',
        "--build-arg", f'ARG_TF_USE_LEGACY_KERAS={config.get("TF_USE_LEGACY_KERAS", "0")}',
        "--build-arg", f'APP_MODEL_BATCH_SIZE=16',
        "--build-arg", f'USE_GPU=0',
        "--build-arg", f'ARG_DEVICE_TYPE={arg_device_type}',
        "-t", f"{name}", "."
    ], stdout=sys.stdout, stderr=sys.stderr)

    subprocess.run(["rm", "-rf", "module"])

    print(result)

