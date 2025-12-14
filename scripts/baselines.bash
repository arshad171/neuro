ulimit -n 5000;

apps=("tclf-gcn" "tclf-rf" "tstr-lstm" "kalman-gru" "iclf-mnet" "text-bert" "iclf-efnet" "text-tbert" "iclf-mvit")

OUT_FOLDER_1="/home/neuro/data/base"
mkdir -p $OUT_FOLDER_1

for app in "${apps[@]}"; do

    OUT_FOLDER_2="$OUT_FOLDER_1/${app}"
    mkdir -p $OUT_FOLDER_2;
    echo $OUT_FOLDER_2

    kubectl delete deployments --all -n default;
    kubectl delete services --all -n default;

    python -m client_scripts.gen_choice_rand --out-folder="$OUT_FOLDER_2" --config="app=$app|lam=10|ni=1";
    python -m scripts.gen_deps_rand --out-folder="$OUT_FOLDER_2";
    kubectl apply -f "$OUT_FOLDER_2/kube_deps_p5";


    # # sleep_time=$(( ni > 10 ? ni : 10 ))
    # sleep_time=$(( ni + 12 ))
    # sleep "$sleep_time"
    sleep 30;

    python -m scripts.proc_mon --out-folder="$OUT_FOLDER_2";
    python -m client_scripts.emulate_tenants --out-folder="$OUT_FOLDER_2";
    python -m scripts.parse_proc_events --out-folder="$OUT_FOLDER_2";
done
