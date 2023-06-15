# import streamlit as st
import time
import torch
from fastapi import FastAPI, HTTPException
from utils import load_starcoder
from models import Prompt
import logging
# from mii_utils import generator

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.model = None
app.tokenizer = None
app.model, app.tokenizer = load_starcoder()

@app.post('/prompt/', status_code=200)
def prompt(prompt: Prompt):
    if app.model is None or app.tokenizer is None:
        raise HTTPException(status_code=503, detail='StarChat model and/or tokenizer not initialized')
    if prompt.temp <= 0 or prompt.temp > 1:
        raise HTTPException(status_code=400, detail=f'Temperature must be in this range: 0 <= temp >= 1, not {prompt.temp}')

    prompt_strings = []
    for user_prompt in prompt.user:
        prompt_string = f"<|system|>\n{prompt.system}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>"
        prompt_strings.append(prompt_string)

    # outputs = generator.query({'query': prompt_strings}, 
    #                           max_new_tokens=500, 
    #                           do_sample=True, 
    #                           temperature=prompt.temp, 
    #                           top_k=50, 
    #                           top_p=0.95, 
    #                           eos_token_id=49155, 
    #                           pad_token_id=49155, 
    #                           )
    # print(outputs.response)
    tokenized = app.tokenizer(prompt_strings, return_tensors='pt')
    input_ids = tokenized.input_ids
    input_ids = input_ids.to('cuda')

    if len(input_ids[0]) > app.tokenizer.model_max_length:
        raise HTTPException(status_code=400, detail=f'Prompt must be under {app.tokenizer.model_max_length} tokens, not {len(input_ids[0])}')

    # print(inputs)
    # print(len(inputs))
    _logger.info('Running inference...')
    start = time.perf_counter()
    outputs = app.model.generate(input_ids=input_ids, 
                                #  return_dict_in_generate=True,
                                 max_new_tokens=300, 
                                 min_new_tokens=100, 
                                 do_sample=True, 
                                 temperature=prompt.temp, 
                                 top_k=50, 
                                 top_p=0.95, 
                                 eos_token_id=49155, 
                                 pad_token_id=49155, 
                                 attention_mask=tokenized.attention_mask,
                                 )
    elapsed = time.perf_counter()-start
    _logger.info(f'Ran inference ({elapsed: .3f}s)')
    # return {f'r{i+1}': output for i, output in enumerate(app.tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=False))}
    return {'response': app.tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False)}
