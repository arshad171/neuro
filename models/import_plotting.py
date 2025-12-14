import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import json
from matplotlib.patches import Patch
import altair as alt
from scipy.stats import gaussian_kde
import numpy as np
from typing import NamedTuple
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D

CMAP_APPLICATIONS_E = cm.get_cmap('Set3')

# FIGS_PATH = "/Users/arshadjaveed/My Data/Workspace/pa_res_alloc/results/figures_neuro"
FIGS_PATH = "/home/arshad/code/pa_res_alloc_2/results/figures_neuro"

plt.rcParams.update({
    'legend.fontsize': 16,      # Legend font size
    "text.usetex": True,            # Use LaTeX to render text
    "font.family": "serif",         # Use a serif font family
    "font.serif": ["Computer Modern"],  # Set the default LaTeX font to Computer Modern
    "axes.labelsize": "26",           # Font size for axis labels
    'axes.labelweight': 'bold',
    'axes.labelweight': 'bold',  # Make the label bold
    'axes.titleweight': 'bold',  # Make the title bold
    "font.size": 12,                # General font size
    "legend.fontsize": 12,          # Font size for legends
    "xtick.labelsize": 22,          # Font size for x-tick labels
    "ytick.labelsize": 22,          # Font size for y-tick labels
    "lines.linewidth": 2,
    "lines.markersize": 8,
    'grid.linestyle': '--',
    'grid.color': 'gray',
    'grid.alpha': 0.5,
})


COLOR_PALETTE = {
    "cnn": "orange",
    "dnn": "red",
    "lstm": "purple",
    "rf": "green",
    ###
    "cnn-v1": "orange",
    "dnn-v1": "red",
    "lstm-v1": "purple",
    "rf-v1": "green",
    ###
    "cnn-v2": "orange",
    "dnn-v2": "red",
    "lstm-v2": "purple",
    "rf-v2": "green",
    ###
    "bert": "black",
    ###
    "scaloran": "skyblue",
    "scaloran-cons": "darkorange",
    "perx": "salmon",
    "perx++": "crimson",
    "perx++ (rel)": "black",
    "perx++ (rnd)": "crimson",
    "NeuRO (rel)": "black",
    "NeuRO (rnd)": "crimson",
    ###
    "neuro": "crimson",
    "NeuRO (rel) - m1": "black",
    "NeuRO (rnd) - m1": "crimson",
    "NeuRO (rel) - m2": "gray",
    "NeuRO (rnd) - m2": "teal",
    "neuro (l)": "gray",
    "neuro (h)": "black",
    ###
    "tclf-gcn": mcolors.to_hex(CMAP_APPLICATIONS_E(0)),
    "tclf-rf": mcolors.to_hex(CMAP_APPLICATIONS_E(1)),
    "tstr-lstm": mcolors.to_hex(CMAP_APPLICATIONS_E(2)),
    "kalman-gru": mcolors.to_hex(CMAP_APPLICATIONS_E(3)),
    "knet-gru": mcolors.to_hex(CMAP_APPLICATIONS_E(3)),
    "iclf-mnet": mcolors.to_hex(CMAP_APPLICATIONS_E(4)),
    "text-bert": mcolors.to_hex(CMAP_APPLICATIONS_E(5)),
    "iclf-efnet": mcolors.to_hex(CMAP_APPLICATIONS_E(6)),
    "text-tbert": mcolors.to_hex(CMAP_APPLICATIONS_E(7)),
    "iclf-mvit": mcolors.to_hex(CMAP_APPLICATIONS_E(8)),
}

MARKERS = {
    "cnn-v1": "^",
    "dnn-v1": "o",
    "lstm-v1": "v",
    "rf-v1": "s",
    ###
    "scaloran": "s",
    "scaloran-cons": "^",
    "perx": "o",
    "perx++": "D",
    "perx++ (rel)": "D",
    "perx++ (rnd)": "v",
    "NeuRO (rel)": "D",
    "NeuRO (rnd)": "v",
    ###
    "NeuRO (rel) - m1": "D",
    "NeuRO (rnd) - m1": "v",
    "NeuRO (rel) - m2": "D",
    "NeuRO (rnd) - m2": "v",
}

class Description(NamedTuple):
    text: str = ""
    extra: str = ""
    unit: str = ""

DESCRIPTIONS = {
    "arr": Description(text="Arrival rate", unit="[/sec]"),
    "cpu-clock": Description(text="CPU clock", unit="[msec]"),
    "cache-references": Description(text="Cache references", unit="[M/sec]"),
    "cache-misses": Description(text="Cache miss rate", unit="[\%]"),
    "page-faults": Description(text="Page faults", unit="[K/sec]"),
    "dnn": Description(text="FF"),
    "lstm": Description(text="LSTM"),
    "cnn": Description(text="CNN"),
    "rf": Description(text="RF"),
    "dnn-v1": Description(text="FF-v1"),
    "lstm-v1": Description(text="LSTM-v1"),
    "cnn-v1": Description(text="CNN-v1"),
    "rf-v1": Description(text="RF-v1"),
    "dnn-v2": Description(text="FF-v2"),
    "lstm-v2": Description(text="LSTM-v2"),
    "cnn-v2": Description(text="CNN-v2"),
    "rf-v2": Description(text="RF-v2"),
    "bert": Description(text="BERT"),
    "perx": Description(text="PERX"),
    "scaloran": Description(text="Scal-ORAN"),
    # "scaloran-cons": Description(text="Scal-ORAN-cons"),
    "perx-nn": Description(text="PERX - NN"),
    "perx-boot-nn": Description(text="PERX - Boot, NN"),
    "perx-boot-deepsets": Description(text="PERX - DeepSets"),
    "perx-hydra": Description(text="PERX - Hydra (perf)"),
    "perx-hydra-param": Description(text="PERX - Param, Hydra (perf)"),
    ###
    # "revenue": Description(text="Revenue", extra="($U$)", unit="[\$]"),
    "revenue": Description(text="Revenue", extra="", unit="[\$]"),
    "lost_revenue": Description(text="Lost Revenue", extra="($U$)", unit="[\$]"),
    "exec_time": Description(text="Execution time", unit="[s]"),
    "inf_time": Description(text="Inference time", unit="[s]"),
    "acc_ratio": Description(text="Acceptance ratio"),
    "cdf": Description(text="CDF"),
    "num_reqs": Description(text="No. of requests", extra="($R$)"),
    "num_servers": Description(text="No. of servers", extra="($S$)"),
    "num_instances": Description(text="No. of instances"),
    "num_served_user_reqs": Description(text="No. of served user requests"),
    ###
    "scaloran": Description(text="Scal-ORAN"),
    # "scaloran-cons": Description("Scal-ORAN-cons"),
    "scaloran-cons": Description(text="App-agnostic"),
    "perx": Description(text="PERX"),
    "perx++": Description(text="PERX++"),
    "perx++ (rel)": Description(text="PERX++ (rel)"),
    "perx++ (rnd)": Description(text="PERX++ (rnd)"),
    "NeuRO": Description(text="NeuRO"),
    "neuro": Description(text="NeuRO"),
    "NeuRO (rel)": Description(text="NeuRO (rel)"),
    "NeuRO (rnd)": Description(text="NeuRO (rnd)"),
    ###
    "NeuRO (rel) - m1": Description(text="NeuRO (rel) - m1"),
    # "NeuRO (rnd) - m1": Description(text="NeuRO (rnd) - m1"),
    "NeuRO (rnd) - m1": Description(text="NeuRO"),
    "NeuRO (rel) - m2": Description(text="NeuRO (rel) - m2"),
    "NeuRO (rnd) - m2": Description(text="NeuRO (rnd) - m2"),
    #
    "tclf-gcn": Description(text="TCLF-GCN"),
    "tclf-rf": Description(text="TCLF-RF"),
    "tstr-lstm": Description(text="TSTR-LSTM"),
    "kalman-gru": Description(text="KNet-GRU"),
    "iclf-mnet": Description(text="ICLF-MNET"),
    "text-bert": Description(text="TEXT-BERT"),
    "iclf-efnet": Description(text="ICLF-EFNET"),
    "text-tbert": Description(text="TEXT-TBERT"),
    "iclf-mvit": Description(text="ICLF-MVIT"),
    #
    "pct_vio": Description(text="Task latency violation (\%)"),
    "sla": Description(text="Latency SLA"),
    "norm_sla": Description(text="Normalized SLA"),
    "scenario_re": Description(text="R+E"),
    "scenario_edge": Description(text="EDGE"),
    "scenario_rt": Description(text="RT"),
    "scenario_s1": Description(text="R+E"),
    "scenario_s2": Description(text="EA"),
    "scenario_s3": Description(text="RA"),
    "e_app_presence": Description(text="Edge app presence"),
    "sla_vio": Description(text="SLA metric violation (\%)"),
}

SNS_BARPLOT_OPTS = {
    "saturation": 0.9
}

def get_label(key, raw=False):
    if key not in DESCRIPTIONS:
        return key

    des =  DESCRIPTIONS[key]
    if raw:
        return des.text

    label = r"\textbf{" + des.text + r"}"

    if des.extra != "":
        label += " "
        label += des.extra

    if des.unit != "":
        label += " "
        label += des.unit
    
    return label
    # if des.unit != "":
    #     return r"\textbf{" + des.text+ r"}" + " " + des.extra + " " + des.unit
    # else:
    #     return r"\textbf{" + des.text+ r"}"

def get_upper_legend(key):
    des =  DESCRIPTIONS[key]

    return des.text.upper()

def save_fig(name):
    plt.savefig(os.path.join(FIGS_PATH, f"{name}.pdf"), format="pdf", bbox_inches='tight')


def remove_outliers_iqr(x):
    q1 = np.percentile(x, 30)
    q3 = np.percentile(x, 70)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return x[(x >= lower) & (x <= upper)]

def get_kde(x, bw=0.3, remove_fliers=False):
    x = np.array(x)
    if remove_fliers:
        x = remove_outliers_iqr(x)

    kde = gaussian_kde(x, bw_method=bw)
    x_vals1 = np.linspace(min(x), max(x), len(x))
    kde_vals1 = kde(x_vals1)
    # kde_vals1 /= kde_vals1.sum()

    return x_vals1, kde_vals1
