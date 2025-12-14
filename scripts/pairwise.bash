ulimit -n 5000;

apps1=("tclf-gcn" "tclf-rf" "tstr-lstm" "kalman-gru" "iclf-mnet" "text-bert" "iclf-efnet" "text-tbert" "iclf-mvit")
apps2=("tclf-gcn" "tclf-rf" "tstr-lstm" "kalman-gru" "iclf-mnet" "text-bert" "iclf-efnet" "text-tbert" "iclf-mvit")
lams=(1.0 2.0 3.0 4.0)
nis_max=(
    ["tclf-gcn"]=16
    ["tclf-rf"]=16
    ["tstr-lstm"]=16
    ["kalman-gru"]=16
    ["iclf-mnet"]=16
    ["text-bert"]=16
    ["iclf-efnet"]=16
    ["text-tbert"]=16
    ["iclf-mvit"]=16
)

# apps1=("tclf-gcn")
# apps2=("tclf-gcn")
# lams=(8.0)
# nis=(4)

OUT_FOLDER_1="/home/neuro/data/pair"
mkdir -p $OUT_FOLDER_1

for app1 in "${apps1[@]}"; do
    for app2 in "${apps2[@]}"; do
        ni_max=${nis_max[$app2]};
        nis=(4 8 $ni_max);
        for ni in "${nis[@]}"; do
            for lam in "${lams[@]}"; do
                echo "$app1 - $app2 ($ni, $lam) $ni_max"
                OUT_FOLDER_2="$OUT_FOLDER_1/${app1}_${app2}/$ni/$lam"
                mkdir -p $OUT_FOLDER_2;
                echo $OUT_FOLDER_2

                kubectl delete deployments --all -n default;
                kubectl delete services --all -n default;

                python -m client_scripts.gen_choice_rand --out-folder="$OUT_FOLDER_2" --config="app=$app1|lam=5|ni=1,app=$app2|lam=$lam|ni=$ni";
                python -m scripts.gen_deps_rand --out-folder="$OUT_FOLDER_2";
                kubectl apply -f "$OUT_FOLDER_2/kube_deps_p5";


                # sleep_time=$(( ni > 10 ? ni : 10 ))
                sleep_time=$(( ni + 12 ))
                sleep "$sleep_time"
                # sleep 40;

                python -m scripts.proc_mon --out-folder="$OUT_FOLDER_2";
                python -m client_scripts.emulate_tenants --out-folder="$OUT_FOLDER_2";
                python -m scripts.parse_proc_events --out-folder="$OUT_FOLDER_2";
            done
        done
    done
done