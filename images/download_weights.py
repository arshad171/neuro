from torch import *
from torchvision import *
from transformers import *

model_id = "apple/mobilevit-small"
tokenizer = BertTokenizer.from_pretrained(model_id)
model = BertForSequenceClassification.from_pretrained(model_id)

weights_path = "./ml_text_tbert/weights"

tokenizer.save_pretrained(weights_path)
model.save_pretrained(weights_path)

# tokenizer = BertTokenizer.from_pretrained(weights_path)
# model = BertForSequenceClassification.from_pretrained(weights_path)