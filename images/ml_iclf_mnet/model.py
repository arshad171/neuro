import torch as th
import torchvision.models as tv_models

batch_size = 8


class Model:
    def __init__(self, device_type="cpu", batch_size=batch_size):
        self.batch_size = batch_size
        # weights = th.load("module/weights.pth")
        print("cuda available:", th.cuda.is_available())
        self.device = th.device(device_type)
        self.model = tv_models.mobilenet_v3_large(weights=None).to(self.device)

    def get_data(self, batch_size):
        return th.randn(batch_size, 3, 224, 224, device=self.device)

    def predict(self, batch_size=batch_size):
        batch_size = self.batch_size if not batch_size else batch_size
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

    # time.sleep(10)

    parser = argparse.ArgumentParser(description="kwargs")

    parser.add_argument("--batch-size", type=int, help="batch size", default=1)
    parser.add_argument("--out-folder", type=str, help="output folder", default=".")

    args = parser.parse_args()
    arg_batch_size = args.batch_size
    arg_out_folder = args.out_folder

    # json.dump({"pid": os.getpid()}, open(os.path.join(arg_out_folder, "pid.json"), "w"))

    batch_size = arg_batch_size

    model = Model(device_type="cuda", batch_size=arg_batch_size)

    sts = []
    # for ix in range(1000):
    start_time = time.time()
    # while time.time() - start_time < time_cap:
    while True:
    # for _ in range(100):
        t1 = time.time()
        model.predict()
        print(time.time() - t1, flush=True)
        sts.append(time.time() - t1)
        # time.sleep(0.1)

    while True:
        print("alive", flush=True)
        time.sleep(1)
    
    # json.dump({"sts": sts}, open(os.path.join(arg_out_folder, "sts.json"), "w"))


