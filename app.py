import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os
import torch

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
    MT5TokenizerFast as MT5Tokenizer,
)

app = Flask(__name__)
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_dir = 'model/'

tokenizer = T5Tokenizer.from_pretrained('t5-small',force_download=True)
print("model dir is :: ",model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(torch_device)
model.resize_token_embeddings(len(tokenizer))
model.eval()
    

@app.route('/')
def home():
    return 'hello world'
	
@app.route('/ping')
def ping():
    return jsonify('autocomplete app is up and running')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    output = []
    query = data['query']
    test_tokenized = tokenizer.encode_plus(query, return_tensors="pt")
    test_input_ids  = test_tokenized["input_ids"]
    test_attention_mask = test_tokenized["attention_mask"]
    
    beam_outputs = model.generate(
    input_ids=test_input_ids,attention_mask=test_attention_mask,
    max_length=96,
    early_stopping=True,
    num_beams=20,
    num_return_sequences=5,
    no_repeat_ngram_size=2
    )

    for beam_output in beam_outputs:
        output.append(tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True))
        
    return jsonify(output)



if __name__ == '__main__':
    
    app.run(debug=True)
