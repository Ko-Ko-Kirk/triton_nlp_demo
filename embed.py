import numpy as np
import sys
from functools import partial
import os
import tritongrpcclient
import tritongrpcclient.model_config_pb2 as mc
import tritonhttpclient
from tritonclientutils import triton_to_np_dtype
from tritonclientutils import InferenceServerException
import torch
from transformers import AutoTokenizer
from torch.nn import functional as F

tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
VERBOSE = False

sentence1 = 'Who are you voting for 2021?'
sentence2 = 'Jupiter’s Biggest Moons Started as Tiny Grains of Hail'
sentence3 = 'Hi Ko Ko, send me your invoice, thankyou!'
labels = ['business', 'space and science', 'politics']
input_name = ['input__0', 'input__1']
output_name = 'output__0'


def run_inference(sentence, model_name='deepset', url='127.0.0.1:8000', model_version='1'):
    triton_client = tritonhttpclient.InferenceServerClient(
        url=url, verbose=VERBOSE)
    model_metadata = triton_client.get_model_metadata(
        model_name=model_name, model_version=model_version)
    model_config = triton_client.get_model_config(
        model_name=model_name, model_version=model_version)
    # I have restricted the input sequence length to 256
    inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                     return_tensors='pt', max_length=256,
                                     truncation=True, padding='max_length')
    
    input_ids = inputs['input_ids']
    input_ids = np.array(input_ids, dtype=np.int32)
    mask = inputs['attention_mask']
    mask = np.array(mask, dtype=np.int32)
    mask = mask.reshape(4, 256) 
    input_ids = input_ids.reshape(4, 256)
    input0 = tritonhttpclient.InferInput(input_name[0], (4, 256), 'INT32')
    input0.set_data_from_numpy(input_ids, binary_data=False)
    input1 = tritonhttpclient.InferInput(input_name[1], (4, 256), 'INT32')
    input1.set_data_from_numpy(mask, binary_data=False)
    output = tritonhttpclient.InferRequestedOutput(output_name,  binary_data=False)
    response = triton_client.infer(model_name, model_version=model_version, inputs=[input0, input1], outputs=[output])
    embeddings = response.as_numpy('output__0')
    embeddings = torch.from_numpy(embeddings)
    sentence_rep = embeddings[:1].mean(dim=1)
    label_reps = embeddings[1:].mean(dim=1)
    similarities = F.cosine_similarity(sentence_rep, label_reps)
    closest = similarities.argsort(descending=True)
    for ind in closest:
        print(f'label: {labels[ind]} \t similarity: {similarities[ind]}')


print("Input sentence:", sentence1)
print('\n')
run_inference(sentence1)
print('\n')
print("Input sentence:", sentence2)
print('\n')
run_inference(sentence2)
print('\n')
print("Input sentence:", sentence3)
print('\n')
run_inference(sentence3)