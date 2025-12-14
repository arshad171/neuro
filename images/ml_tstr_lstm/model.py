import tensorflow as tf


batch_size = 16
input_dim = 784
window = 100

class Model:
    def __init__(self, device_type="cpu", batch_size=batch_size):
        self.batch_size = batch_size
        # weights = th.load("module/weights.pth")
        if device_type == "cpu":
            tf.config.set_visible_devices([], 'GPU')
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(input_dim, window), batch_size=16),
            tf.keras.layers.LSTM(units=50, return_sequences=True),
            tf.keras.layers.LSTM(units=50, activation="relu"),
            tf.keras.layers.Dense(units=10, activation="relu"),
            tf.keras.layers.Dense(units=10, activation="relu"),
            tf.keras.layers.Dense(units=1, activation="relu"),
        ])


    def get_data(self, batch_size):
        data = tf.random.uniform(shape=(batch_size, input_dim, window))

        return data

    def predict(self, batch_size=batch_size):
        batch_size = self.batch_size if not batch_size else batch_size
        data = self.get_data(batch_size)

        preds = self.model.predict(data)

        return preds.tolist()

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
    while time.time() - start_time < time_cap:
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
