import json
import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.types import *

class ServicePredictorNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        input_size = 14

        self.lin1 = nn.Linear(input_size, 100)
        self.lin2 = nn.Linear(100, 50)
        self.lin3 = nn.Linear(50, 25)
        self.lin4 = nn.Linear(25, 1)

        self.relu_act = nn.ReLU()

    def forward(self, inputs):
        x = th.hstack([inputs["feature_app"], inputs["feature_deployment"], inputs["feature_server_context"]])

        h = self.lin1(x)
        h = self.relu_act(h)

        h = self.lin2(h)
        h = self.relu_act(h)

        h = self.lin3(h)
        h = self.relu_act(h)

        h = self.lin4(h)
        h = self.relu_act(h)

        return h


class ExpLayer(nn.Module):
    def __init__(self):
        super(ExpLayer, self).__init__()

    def forward(self, x):
        return th.exp(x)

class ClippedLin(nn.Module):
    def __init__(self):
        super(ClippedLin, self).__init__()

    def forward(self, x):
        return 10 * th.tanh(x)

class ServicePredictorNNSetReprHydra(nn.Module):
    def __init__(self, application_catalog:ApplicationCatalog, use_base_param: bool=False, embedding_type: EmbeddingTypes=EmbeddingTypes.USE_EMBEDDING_LAYER, use_gpu=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_base_param = use_base_param
        self.application_catalog = application_catalog
        self.embedd_dim = 5
        self.load_feat_dim = 2
        self.app_arch_dim = 1
        self.num_inf_targets = len(APP_INFERENCE_TARGETS)

        match application_catalog:
            case ApplicationCatalog.V1:
                self.num_embedds = len(APPLICATIONS)
            case ApplicationCatalog.V21:
                self.num_embedds = len(APPLICATIONS_V21)    
            case ApplicationCatalog.V22:
                self.num_embedds = len(APPLICATIONS_V22)    
            case ApplicationCatalog.E1:
                self.num_embedds = len(APPLICATIONS_E1)
            case ApplicationCatalog.E2:
                self.num_embedds = len(APPLICATIONS_E2)
            case ApplicationCatalog.E:
                if use_gpu:
                    self.num_embedds = len(APPLICATIONS_E_GPU)
                else:
                    self.num_embedds = len(APPLICATIONS_E)
        if embedding_type == EmbeddingTypes.USE_EMBEDDING_LAYER:
            self.embedding = nn.Embedding(num_embeddings=self.num_embedds, embedding_dim=self.embedd_dim)
        elif embedding_type == EmbeddingTypes.USE_ONEHOT:

            self.embedding = lambda x: F.one_hot(x, num_classes=self.num_embedds)

        self.feat_dim1 = self.embedd_dim + self.load_feat_dim
        self.feat_dim_fx_out = 5
        self.num_lm_out = len(LOW_METRICS) if not use_gpu else len(LOW_METRICS_GPU)

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.feat_dim1, 50),
            # nn.Tanh() if self.use_base_param else nn.ReLU(),
            nn.GELU(),
            nn.Linear(50, 25),
            # nn.Tanh() if self.use_base_param else nn.ReLU(),
            nn.GELU(),
            nn.Linear(25, self.feat_dim_fx_out),
        )

        # out + dim(x_feat)
        self.regressor = nn.Sequential(
            nn.Linear(1 * self.feat_dim_fx_out + self.feat_dim1, 50),
            # nn.Tanh() if self.use_base_param else nn.ReLU(),
            nn.GELU(),
            nn.Linear(50, 25),
            # nn.Tanh() if self.use_base_param else nn.ReLU(),
            nn.GELU(),
            nn.Linear(25, self.num_inf_targets),
            # nn.Sigmoid() if self.use_base_param else nn.ReLU(),
            # (
            #     ExpLayer()
            #     if self.use_base_param
            #     else nn.Linear(self.num_inf_targets, self.num_inf_targets)
            # ),
            (
                ExpLayer()
                if self.use_base_param
                else ClippedLin()
            ),
        )

        self.embedding_lm_head = nn.Sequential(
            nn.Linear(self.feat_dim1, 50),
            # nn.Tanh() if self.use_base_param else nn.ReLU(),
            nn.GELU(),
            nn.Linear(50, 10),
            # nn.Tanh() if self.use_base_param else nn.ReLU(),
            nn.GELU(),
            nn.Linear(10, self.num_lm_out),
            # nn.Tanh() if self.use_base_param else nn.ReLU(),
            nn.GELU(),
        )

        self.add_module("feature_extractor", self.feature_extractor)
        self.add_module("regressor", self.regressor)
        self.add_module("embedding", self.embedding)
        self.add_module("embedding_lm_head", self.embedding_lm_head)

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, inputs):
        x_features = inputs["x_features"]
        features = inputs["features"]
        lenghts = inputs["lengths"]

        preds_sts = []
        preds_lms = []
        batch_size = features.shape[0]
        for ix in range(batch_size):
            # server context
            xf = features[ix]
            xl = lenghts[ix]

            h1 = xf[:, :xl]
            h1_app = xf[0, :xl]
            h1_load = xf[1:, :xl]

            h1_app = self.embedding(h1_app.long())
            h1 = th.vstack([h1_app.T, h1_load])

            hh = h1

            # h2 = x_features[ix].repeat(xl, 1)
            # hh = th.vstack([h1, h2.T])

            h = self.feature_extractor(hh.T)

            h1 = h.sum(dim=0)
            # h2 = h.max(dim=0).values

            x_app = x_features[ix][0]
            x_load = x_features[ix][1:3]
            # x_arch = x_features[ix][3]

            x_app = self.embedding(x_app.long())

            x_f = th.hstack([x_app, x_load])
            hh = th.hstack([h1, x_f])

            preds_sts.append(hh)

            preds_lms.append(x_f)

        preds_sts = th.vstack(preds_sts)
        preds_sts = self.regressor(preds_sts)

        # if self.use_base_param:
        #     preds_sts = 1 / preds_sts

        preds_lms = th.vstack(preds_lms)
        preds_lms = self.embedding_lm_head(preds_lms)

        return preds_sts, preds_lms

    def save_weights(self, path):
        th.save(self.embedding.state_dict(), os.path.join(path, f"weights_embedding.pth"))
        th.save(self.feature_extractor.state_dict(), os.path.join(path, f"weights_feature_extractor.pth"))
        th.save(self.regressor.state_dict(), os.path.join(path, f"weights_regressor.pth"))
        th.save(self.embedding_lm_head.state_dict(), os.path.join(path, f"weights_embedding_lm_head.pth"))

    def load_weights(self, path, expand_embedd=True):
        if expand_embedd:
            print("old")
            print(self.embedding.weight.data)
            embedding_layer_v1 = nn.Embedding(num_embeddings=len(APPLICATIONS_E1), embedding_dim=self.embedd_dim)
            embedding_layer_v1.load_state_dict(th.load(os.path.join(path, f"weights_embedding.pth")))
            weights_v1 = embedding_layer_v1.weight.data
            weights_v2 = self.embedding.weight.data
            weights_v2[:len(APPLICATIONS_E1)] = weights_v1.data
            self.embedding.weight.data = weights_v2

            # def freeze_old_weights(grad):
            #     grad[:len(APPLICATIONS)].zero_()
            #     return grad

            # self.embedding.weight.register_hook(freeze_old_weights)

            # self.embedding = nn.Embedding.from_pretrained(weights_v2)
            print("new")
            print(self.embedding.weight.data)
        else:
            print("loaded")
            self.embedding.load_state_dict(th.load(os.path.join(path, f"weights_embedding.pth")))

        self.feature_extractor.load_state_dict(th.load(os.path.join(path, f"weights_feature_extractor.pth")))
        self.regressor.load_state_dict(th.load(os.path.join(path, f"weights_regressor.pth")))
        self.embedding_lm_head.load_state_dict(th.load(os.path.join(path, f"weights_embedding_lm_head.pth")))

class ServicePredictorNNSetRepr(nn.Module):
    def __init__(self, embedding_type: EmbeddingTypes=EmbeddingTypes.USE_EMBEDDING_LAYER, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if embedding_type == EmbeddingTypes.USE_EMBEDDING_LAYER:
            embedd_dim = 2
            self.embedding = nn.Embedding(num_embeddings=len(APPLICATIONS), embedding_dim=embedd_dim, max_norm=1.0)
        elif embedding_type == EmbeddingTypes.USE_ONEHOT:
            embedd_dim = len(APPLICATIONS)
            self.embedding = lambda x: F.one_hot(x, num_classes=len(APPLICATIONS))

        feat_in = embedd_dim + 2
        feat_dim = 5

        self.feature_extractor = nn.Sequential(
            nn.Linear(feat_in, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, feat_dim),
        )

        # out + dim(x_feat)
        self.regressor = nn.Sequential(
            nn.Linear(2 * feat_dim + embedd_dim + 2, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )

        self.add_module("feature_extractor", self.feature_extractor)
        self.add_module("regressor", self.regressor)

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
    
    def forward(self, inputs):
        # TODO: fix: assuming a batch size of 1
        x_features = inputs["x_features"]
        features = inputs["features"]
        lenghts = inputs["lengths"]

        hs = []
        batch_size = features.shape[0]
        for ix in range(batch_size):
            xf = features[ix]
            xl = lenghts[ix]

            h1 = xf[:, :xl]
            h1_app = xf[0, :]
            h1_load = xf[1:, :]


            h1_app = self.embedding(h1_app.long())
            h1 = th.vstack([h1_app.T, h1_load])

            hh = h1

            # h2 = x_features[ix].repeat(xl, 1)
            # hh = th.vstack([h1, h2.T])

            h = self.feature_extractor(hh.T)

            h1 = h.sum(dim=0)
            h2 = h.max(dim=0).values

            x_app = x_features[ix][0]
            x_load = x_features[ix][1:]
            x_app = self.embedding(x_app.long())
            x_f = th.hstack([x_app, x_load])
            hh = th.hstack([h1, h2, x_f])

            hs.append(hh)

        hs = th.vstack(hs)
        hs = self.regressor(hs)

        return hs

# PERX (v1)
class PERXPredictor:
    def __init__(self, baselines_path, sig_path, app_name):
        self.app_name = app_name
        self.baselines = json.load(open(baselines_path, "r"))
        self.sig_app_params = json.load(open(sig_path, "r"))
    
    def predict(self, deployment: Deployment, mul=1.0):
        predicted_app_service_time = 0.0

        # deployment = datapoint.server_context + [datapoint.x]
        # application_deployments = [obj._asdict() for obj in deployment]
        application_deployments = deployment.get_list()

        df_app_deps = pd.DataFrame(application_deployments).groupby(by="application").agg({"num_instances": "sum", "arrival_rate": "mean"})

        for app, row in df_app_deps.iterrows():
            a1 = self.sig_app_params[f"{self.app_name}"][f"{app}"]["alpha1"]
            b1 = self.sig_app_params[f"{self.app_name}"][f"{app}"]["beta1"]
            a2 = self.sig_app_params[f"{self.app_name}"][f"{app}"]["alpha2"]
            b2 = self.sig_app_params[f"{self.app_name}"][f"{app}"]["beta2"]
            a3 = self.sig_app_params[f"{self.app_name}"][f"{app}"]["alpha3"]
            b3 = self.sig_app_params[f"{self.app_name}"][f"{app}"]["beta3"]

            t1 = a1 + b1 * row["num_instances"]
            t2 = a2 + b2 * row["num_instances"]
            t3 = a3 + b3 * row["num_instances"]

            pred = t1 / (1 + np.exp(-t2 * (row["arrival_rate"] - t3)))
            pred -= self.baselines[self.app_name]
            pred = max(0, pred)

            predicted_app_service_time += pred * mul
        
        predicted_app_service_time += self.baselines[self.app_name]
        return predicted_app_service_time


class ScalORANPredictor:
    def __init__(self, sig_path_fmt):
        self.sig_path_fmt = sig_path_fmt

    def predict(self, deployment: Deployment):
        predicted_app_service_time = []

        application_deployments = deployment.get_list()

        df_app_deps = pd.DataFrame(application_deployments).groupby(by="application").agg({"num_instances": "sum", "arrival_rate": "mean"})

        num_instances = df_app_deps["num_instances"].sum()

        for app, row in df_app_deps.iterrows():
            # num_instances = row["num_instances"]
            sig_app_params = json.load(open(self.sig_path_fmt.format(app), "r"))

            a1 = sig_app_params[f"{app}"][f"{app}"]["alpha1"]
            b1 = sig_app_params[f"{app}"][f"{app}"]["beta1"]
            a2 = sig_app_params[f"{app}"][f"{app}"]["alpha2"]
            b2 = sig_app_params[f"{app}"][f"{app}"]["beta2"]
            a3 = sig_app_params[f"{app}"][f"{app}"]["alpha3"]
            b3 = sig_app_params[f"{app}"][f"{app}"]["beta3"]

            t1 = a1 + b1 * num_instances
            t2 = a2 + b2 * num_instances
            t3 = a3 + b3 * num_instances

            pred = t1 / (1 + np.exp(-t2 * (row["arrival_rate"] - t3)))

            predicted_app_service_time.append(pred)
        
        predicted_app_service_time = np.mean(predicted_app_service_time)
        return predicted_app_service_time

class ScalORANConsPredictor:
    def __init__(self, sig_path, app_name):
        self.app_name = app_name
        self.sig_app_params = json.load(open(sig_path, "r"))

    def predict(self, deployment: Deployment):
        predicted_app_service_time = []

        application_deployments = deployment.get_list()

        df_app_deps = pd.DataFrame(application_deployments).groupby(by="application").agg({"num_instances": "sum", "arrival_rate": "mean"})

        num_instances = df_app_deps["num_instances"].sum()

        a1 = self.sig_app_params[f"{self.app_name}"][f"{self.app_name}"]["alpha1"]
        b1 = self.sig_app_params[f"{self.app_name}"][f"{self.app_name}"]["beta1"]
        a2 = self.sig_app_params[f"{self.app_name}"][f"{self.app_name}"]["alpha2"]
        b2 = self.sig_app_params[f"{self.app_name}"][f"{self.app_name}"]["beta2"]
        a3 = self.sig_app_params[f"{self.app_name}"][f"{self.app_name}"]["alpha3"]
        b3 = self.sig_app_params[f"{self.app_name}"][f"{self.app_name}"]["beta3"]

        t1 = a1 + b1 * num_instances
        t2 = a2 + b2 * num_instances
        t3 = a3 + b3 * num_instances

        pred = t1 / (1 + np.exp(-t2 * (df_app_deps.loc[self.app_name]["arrival_rate"] - t3)))

        predicted_app_service_time = pred
        
        return predicted_app_service_time
