import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F

# Load and Convert Hugging Face Model
tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
model = AutoModel.from_pretrained('deepset/sentence_bert')

# dummy inputs for tracing
sentence = 'Who are you voting for in 2020?'
labels = ['business', 'art & culture', 'politics']

# run inputs through model and mean-pool over the sequence
# dimension to get sequence-level representations
inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                     return_tensors='pt', max_length=256,
                                     truncation=True, padding='max_length')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

class PyTorch_to_TorchScript(torch.nn.Module):
    def __init__(self):
        super(PyTorch_to_TorchScript, self).__init__()
        self.model = AutoModel.from_pretrained('deepset/sentence_bert')
    def forward(self, data, attention_mask=None):
        return self.model(data, attention_mask)[0]

pt_model = PyTorch_to_TorchScript().eval()

remove_attributes = []
for key, value in vars(pt_model).items():
    if value is None:
        remove_attributes.append(key)

for key in remove_attributes:
    delattr(pt_model, key)

traced_script_module = torch.jit.trace(pt_model, (input_ids, attention_mask), strict=False)
traced_script_module.save("./model.pt")

import shutil
import os
os.mkdir('./model_repository/deepset')
os.mkdir('./model_repository/deepset/1')
shutil.copy('model.pt', './model_repository/deepset/1')

'''
name: "deepset"
platform: "pytorch_libtorch"
input [
 {
    name: "input__0"
    data_type: TYPE_INT32
    dims: [4, 256]
  } ,
{
    name: "input__1"
    data_type: TYPE_INT32
    dims: [4, 256]
  }
]
output {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [4, 256, 768]
  }
'''