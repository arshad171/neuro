import numpy as np
import torch as th
import torchvision.models as tv_models

batch_size = 1


class Model:
    def __init__(self, device_type="cpu", batch_size=batch_size):
        self.batch_size = batch_size
        # weights = th.load("module/weights.pth")
        print("cuda available:", th.cuda.is_available())
        self.device = th.device(device_type)
        # resnet
        self.model = tv_models.resnet152(weights=None).to(self.device) # 900 MB
        # self.model = tv_models.convnext_large(weights=None).to(self.device) # 850 MB
        # self.model = tv_models.vit_h_14(weights=None).to(self.device) # 850 MB

    def get_data(self, batch_size):
        return th.randn(batch_size, 3, 224, 224, device=self.device)

    def predict(self, batch_size=None):
        batch_size = self.batch_size if not batch_size else batch_size
        print(batch_size)
        data = self.get_data(batch_size)

        with th.no_grad():
            preds = self.model(data)

            return preds.cpu().detach().numpy().tolist()


if __name__ == "__main__":
    import time
    import argparse
    import json
    import os

    import setproctitle
    setproctitle.setproctitle("my_proc")

    time_cap = 1 * 30

    time.sleep(10)

    parser = argparse.ArgumentParser(description="kwargs")

    parser.add_argument("--batch-size", type=int, help="batch size", default=1)
    parser.add_argument("--out-folder", type=str, help="output folder", default=".")

    args = parser.parse_args()
    arg_batch_size = args.batch_size
    arg_out_folder = args.out_folder

    json.dump({"pid": os.getpid()}, open(os.path.join(arg_out_folder, "pid.json"), "w"))

    batch_size = arg_batch_size

    model = Model(device_type="cuda", batch_size=arg_batch_size)

    sts = []
    start_time = time.time()
    while time.time() - start_time < time_cap:
    # while True:
        t1 = time.perf_counter()
        model.predict()
        t2 = time.perf_counter()

        exec_time = t2 - t1
        print(exec_time, np.mean(sts[:-200]), flush=True)
        sts.append(exec_time)

        while (time.perf_counter() - t2) < 1 / 1_000:
            pass
        # time.sleep(np.random.exponential(scale=1 / 100))
        # time.sleep(np.random.uniform(0, 1 / 50))

    # while True:
    #     print("alive", flush=True)
    #     time.sleep(1)
    
    json.dump({"sts": sts}, open(os.path.join(arg_out_folder, "sts.json"), "w"))


