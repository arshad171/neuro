import random
import string
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch as th

batch_size = 1
# weights_path = "/app/module/weights"
weights_path = "/home/arshad/code/pa_res_alloc_2/ml_text_bert/weights"

class Model:
    def __init__(self, device_type="cpu", batch_size=batch_size):
        self.batch_size = batch_size
        print("cuda available:", th.cuda.is_available())
        self.device = th.device(device_type)
        self.model_id = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(weights_path)
        self.model = BertForSequenceClassification.from_pretrained(weights_path).to(self.device)

    def get_data(self, batch_size, word_length=6):
        sentences = []
        for _ in range(batch_size):
            words = []
            for _ in range(128):
                word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
                words.append(word)
            
            sentences.append(' '.join(words))

        return sentences
        # return ' '.join(words)

    def predict(self, batch_size=None):
        batch_size = self.batch_size if not batch_size else batch_size

        # for _ in range(batch_size):
        data = self.get_data(batch_size)
        inputs = self.tokenizer(data, return_tensors="pt", max_length=512, truncation=True, padding=True)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}


        with torch.no_grad():
            outputs = self.model(**inputs)

            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().tolist()

            # return preds
            return preds


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

    json.dump({"pid": os.getpid()}, open(os.path.join(arg_out_folder, "pid.json"), "w"))

    batch_size = arg_batch_size

    model = Model(device_type="cuda", batch_size=arg_batch_size)

    sts = []
    # for ix in range(1000):
    start_time = time.time()
    while time.time() - start_time < time_cap:
    # for _ in range(100):
        t1 = time.time()
        model.predict()
        print(time.time() - t1, flush=True)
        sts.append(time.time() - t1)
        # time.sleep(0.1)

    # while True:
    #     print("alive", flush=True)
    #     time.sleep(1)
    
    json.dump({"sts": sts}, open(os.path.join(arg_out_folder, "sts.json"), "w"))



