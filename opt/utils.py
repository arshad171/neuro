import random


from models.types import *
from train.predictors import *
from train.utils import *

def get_num_reqs(scenario, num_servers):
    match scenario:
        case "s1":
            num_reqs = 15 * num_servers #s1
        case "s2":
            num_reqs = 15 * num_servers #s2
        case "s3":
            num_reqs = 15 * num_servers #s3
        
    return num_reqs

def generate_requests_s1(num: int, application_catalog:ApplicationCatalog, seed=None) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    match application_catalog:
        case ApplicationCatalog.V1:
            applications = APPLICATIONS_V1
        case ApplicationCatalog.V22:
            applications = APPLICATIONS_V22   
        case ApplicationCatalog.E1:
            applications = APPLICATIONS_E1
        case ApplicationCatalog.E2:
            applications = APPLICATIONS_E2
        case ApplicationCatalog.E:
            applications = APPLICATIONS_E     

    requests: List[TenantRequest] = []

    for ix in range(1, num+1):
        app = random.choice(applications)

        payment = TENANT_PRICING_SCHEME[app]

        sla_type = np.random.randint(low=0, high=len(APP_INFERENCE_TARGETS))
        payment_mul = PAYMENT_SLA_MUL[sla_type]

        if app in APPLICATIONS_E_LOW:
            # arrival_rate = np.random.randint(*ARRIVAL_RATES_LOW)
            arrival_rate = num_instances = round(np.random.randint(*NUM_INSTANCES_LOW))
            payment += 0.1 * TENANT_PRICING_SCHEME[app] * arrival_rate / (ARRIVAL_RATES_LOW[1] / 2)
            payment += 0.1 * TENANT_PRICING_SCHEME[app] * num_instances / (NUM_INSTANCES_LOW[1] / 2)
        else:
            arrival_rate = np.random.randint(*ARRIVAL_RATES)
            num_instances = round(np.random.randint(*NUM_INSTANCES))
            payment += 0.15 * TENANT_PRICING_SCHEME[app] * arrival_rate / (ARRIVAL_RATES[1] / 2)
            payment += 0.15 * TENANT_PRICING_SCHEME[app] * num_instances / (NUM_INSTANCES[1] / 2)

        payment *= payment_mul

        delay = round(np.random.uniform(*TENANT_DELAYS_2[app]), 3)

        request = TenantRequest(
            id=ix,
            application=app,
            arrival_rate=arrival_rate,
            num_instances=num_instances,
            payment=payment,
            delay=delay,
            sla_type=sla_type,
        )

        requests.append(request)

    requests_df = pd.DataFrame(requests)
    requests_df.drop(labels=["port", "pids"], inplace=True, axis=1)
    requests_df.set_index("id", inplace=True)
    requests_df = requests_df.apply(add_app_mem_col, axis=1)
    requests_df["arrival_rate_per"] = requests_df["arrival_rate"] / requests_df["num_instances"]
    requests_df["g_tau"] = 1 / requests_df["num_instances"]

    return requests_df

def generate_requests_s2(num: int, application_catalog:ApplicationCatalog, seed=None, part_rt=0.4) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    match application_catalog:
        case ApplicationCatalog.V1:
            applications = APPLICATIONS_V1
        case ApplicationCatalog.V22:
            applications = APPLICATIONS_V22   
        case ApplicationCatalog.E1:
            applications = APPLICATIONS_E1
        case ApplicationCatalog.E2:
            applications = APPLICATIONS_E2
        case ApplicationCatalog.E:
            applications = APPLICATIONS_E     

    requests: List[TenantRequest] = []


    part_rt = int(part_rt * num)
    for ix in range(1, part_rt):
        app = random.choice(APPLICATIONS_RT)
        # app = random.choice(["kalman-gru", "tclf-rf"])

        payment = TENANT_PRICING_SCHEME[app]

        sla_type = np.random.randint(low=0, high=len(APP_INFERENCE_TARGETS))
        payment_mul = PAYMENT_SLA_MUL[sla_type]

        num_instances = round(np.random.randint(*NUM_INSTANCES))
        arrival_rate = num_instances
        payment += 0.15 * TENANT_PRICING_SCHEME[app] * arrival_rate / (ARRIVAL_RATES[1] / 2)
        payment += 0.15 * TENANT_PRICING_SCHEME[app] * num_instances / (NUM_INSTANCES[1] / 2)
        
        payment *= payment_mul

        delay = round(np.random.uniform(*TENANT_DELAYS_2[app]), 3)

        request = TenantRequest(
            id=ix,
            application=app,
            arrival_rate=arrival_rate,
            num_instances=num_instances,
            payment=payment,
            delay=delay,
            sla_type=sla_type,
        )

        requests.append(request)

    for ix in range(part_rt+1, num+1):
        app = random.choice(APPLICATIONS_E_LOW)
        # app = random.choice(["text-bert", "iclf-mnet"])

        payment = TENANT_PRICING_SCHEME[app]

        sla_type = np.random.randint(low=0, high=len(APP_INFERENCE_TARGETS))
        payment_mul = PAYMENT_SLA_MUL[sla_type]

        num_instances = arrival_rate = random.randint(1, 3)
        payment += 0.1 * TENANT_PRICING_SCHEME[app] * arrival_rate / (ARRIVAL_RATES_LOW[1] / 2)
        payment += 0.1 * TENANT_PRICING_SCHEME[app] * num_instances / (NUM_INSTANCES_LOW[1] / 2)

        payment *= payment_mul

        delay = round(np.random.uniform(3, 5), 3)

        request = TenantRequest(
            id=ix,
            application=app,
            arrival_rate=arrival_rate,
            num_instances=num_instances,
            payment=payment,
            delay=delay,
            sla_type=sla_type,
        )

        requests.append(request)

    requests_df = pd.DataFrame(requests)
    requests_df.drop(labels=["port", "pids"], inplace=True, axis=1)
    requests_df.set_index("id", inplace=True)
    requests_df = requests_df.apply(add_app_mem_col, axis=1)
    requests_df["arrival_rate_per"] = requests_df["arrival_rate"] / requests_df["num_instances"]
    requests_df["g_tau"] = 1 / requests_df["num_instances"]

    return requests_df


def generate_requests_s3(num: int, application_catalog:ApplicationCatalog, seed=None, part_rt=0.9):
    return generate_requests_s2(num, application_catalog, seed, part_rt)

GEN_REQS_FUNCS = {
    "s1": generate_requests_s1,
    "s2": generate_requests_s2,
    "s3": generate_requests_s3,
}


def add_app_mem_col(row):
    row["app_mem"] = APP_MEM_REQS[row["application"]]
    # row["exec_scaling"] = APP_BASE_EXEC[row["application"]] * APP_MAX_EXEC[row["application"]]
    # row["exec_scaling"] =  APP_MAX_EXEC[row["application"]]
    row["exec_scaling"] =  APP_EXEC_MAX[row["application"]][row["sla_type"]]

    return row

def get_predictor_inputs(z, requests_df: pd.DataFrame, trim=False):
    requests_df = requests_df.copy(deep=True)
    requests_df["num_instances"] *= z
    # requests_df["num_instances"] = requests_df["num_instances"].astype(int)
    # requests_df["num_instances"] = requests_df["num_instances"].round()
    requests_df["service_time"] = 0

    datapoints: List[Datapoint] = []
    for ix in requests_df.index:
        dep = requests_df.loc[ix]
        context = requests_df.drop(index=ix)

        # if trim == True:
        context = context[context["num_instances"] > 0]

        if len(context) == 0:
            context = pd.concat([context, dep], axis=0)
            context["num_instances"] == 0

        context = pd.DataFrame(context).groupby(by="application").agg({"num_instances": "sum", "arrival_rate": "mean", "service_time": "max"}).reset_index()

        app_dep = ApplicationDeploymnent(
            id=dep.name,
            application=dep["application"],
            arrival_rate=dep["arrival_rate"],
            num_instances=dep["num_instances"],
            service_time=dep["service_time"],
            low_metrics=[
                LowLevelMetric(
                    name=metric,
                    mean=0,
                    std=0,
                ) for metric in LOW_METRICS
            ]
        )

        app_context : List[ApplicationDeploymnent] = []

        for c_dep in context.to_dict(orient="records"):
            c_app_dep = ApplicationDeploymnent(
                id=c_dep.get("id", None),
                application=c_dep["application"],
                arrival_rate=c_dep["arrival_rate"],
                num_instances=c_dep["num_instances"],
                service_time=c_dep["service_time"],
                low_metrics=[]
            )
            app_context.append(c_app_dep)

        datapoints.append(Datapoint(
            app_dep,
            server_context=app_context
        ))

    ds = CustomDatasetSetReprHydra(path=None, datapoints=datapoints)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    return dl