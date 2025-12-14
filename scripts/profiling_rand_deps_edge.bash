ulimit -n 5000;

APP_CAT="v2"
OUT_FOLDER="/home/arshad/code/pa_res_alloc_2/server1_cpu_temp"
mkdir -p $OUT_FOLDER

for ((i=0; i<=0; i = i + 1)); do
    echo $i

    out_folder="$OUT_FOLDER/$i"; 

    rm -rf $out_folder; mkdir -p $out_folder;

    python -m client_scripts.gen_choice_rand --out-folder="$out_folder" --rand-app-cat=$APP_CAT --mem-limit=28;
    python -m scripts.gen_deps_rand --out-folder="$out_folder"

    kubectl delete deployments --all -n default;
    kubectl delete services --all -n default;

    docker system prune -f;

    sleep 5;
    kubectl apply -f "$out_folder/kube_deps_p5";
    sleep 45;

    python -m scripts.proc_mon --out-folder="$out_folder";
    python -m client_scripts.emulate_tenants --out-folder="$out_folder" --host="minikube" --time-limit=30;
    python -m scripts.parse_proc_events --out-folder="$out_folder";

    # pkill $(pgrep "my_pref_logger");

    kubectl delete deployments --all -n default;
    kubectl delete services --all -n default;
    sleep 5;
done
