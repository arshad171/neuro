# NeuRO: Inference-time Profiling and Orchestration of ML Applications at the Edge

## Highlights :rocket:

- NeuRO Profiler architecture. see `train/predictors.py:55`
- Solving NeuRO optimizer with embedded neural network constraints `opt/nlp.py:730`

## Requirements

- OS: any Linux distro running Docker and Kubernets
- Install the python reqiurements `pip install -r requirements.txt`

## 1. Docker Images

1. A set of 9 ML applications are provided under the [images](./images) directory. The applications are as packages Docker containers intended to to be run as services. More applications can be added following the format.

2. Make sure to download the repective weights for the ML architectures by running the [download_weights](./images/download_weights.py). This is requimrent for `ml_iclf_mvit, ml_text_tbert, ml_text_bert` applications.

3. Build the images

```bash
# TODO: activate the kube docker registry
cd images;
python build_script_ml_edge.py
```

## 2. Performance Profiling

The [scripts](./scripts) folder contains the following sample scripts to aid in performance profiling

```bash
# TODO: activate the kube docker registry
cd $PROJECT_ROOT;
```

1. Capture baseline measurements (under no resource contention) `bash scripts/baselines.bash`
2. Pairwise experiments (pairwise resource contention) `bash scripts/pairwise.bash`
3. NeuRO profiling experiments (random application deployments) `bash scripts/rand.bash`


## 3. Data Processing

Run the [data_analysis](./data_analysis.ipynb) notebook to read the data and create data preprocessing artifacts
-  `DS_TRAIN`: paths to folders containing the raw data

## 4. Training Performance Profilers

### NeuRO

First update the following variables before running [training.ipynb](./training.ipynb)
-  `DS_TRAIN`: paths to folders containing the raw data

### Benchmarks

First update the following variables before running [training_perx.ipynb](./training_perx.ipynb)
-  `DS_TRAIN`: paths to folders containing the raw data

## 5. Optimization

- Steps (1)-(4) may be repeated to profile multiple servers.
- Make sure to update the output folders each of the steps when repeating the process for different servers so as to not override the results.
- The more powerful servers have higher rank (or a lower index)

Run the [opt_neuro](./opt_neuro.py) script.
- `NUM_SERVERS`: the number of servers
- `TIME_LIMITS`: the limit for NeuRO (in mins)
- `C_S`: memory limit (GB)
- `SERVER_RANK_L_FRAC`: fraction of low to high rank servers

## Citation



