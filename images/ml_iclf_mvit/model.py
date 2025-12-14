import random
import string
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
import torch as th

batch_size = 2
# weights_path = "/app/module/weights"
weights_path = "/home/arshad/code/pa_res_alloc_2/ml_iclf_mvit/weights"



class Model:
    def __init__(self, device_type="cpu", batch_size=batch_size):
        self.batch_size = batch_size
        self.model_id = "apple/mobilevit-small"
        print("cuda available:", th.cuda.is_available())
        self.device = th.device(device_type)
        self.tokenizer = MobileViTImageProcessor.from_pretrained(weights_path)
        self.model = MobileViTForImageClassification.from_pretrained(weights_path).to(self.device)

    def get_data(self, batch_size=None, word_length=6):
        data = {"pixel_values": th.randn(size=(batch_size, 3, 256, 256), device=self.device)}
        return data

    def predict(self, batch_size=batch_size):
        batch_size = self.batch_size if not batch_size else batch_size

        preds = []
        data = self.get_data(batch_size)

        with th.no_grad():
            outputs = self.model(**data)

        logits = outputs.logits

        preds = th.argmax(logits, dim=1)

        return preds.cpu().numpy().tolist()


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
    # for ix in range(1000):
    start_time = time.time()

    while time.time() - start_time < time_cap:
        t1 = time.time()
        model.predict()
        print(time.time() - t1)
        sts.append(time.time() - t1)
        # time.sleep(0.1)
    
    json.dump({"sts": sts}, open(os.path.join(arg_out_folder, "sts.json"), "w"))


