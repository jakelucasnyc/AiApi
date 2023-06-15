# import streamlit as st
import time
import torch
from fastapi import FastAPI, HTTPException
from utils import load_starcoder
from models import Prompt
import logging

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

    tokenized = app.tokenizer(prompt_strings, return_tensors='pt')
    input_ids = tokenized.input_ids
    input_ids = input_ids.to(app.model.device)

    if len(input_ids[0]) > app.tokenizer.model_max_length:
        raise HTTPException(status_code=400, detail=f'Prompt must be under {app.tokenizer.model_max_length} tokens, not {len(input_ids[0])}')

    # print(inputs)
    # print(len(inputs))
    _logger.info('Running inference...')
    start = time.perf_counter()
    outputs = app.model.generate(input_ids=input_ids, 
                                #  return_dict_in_generate=True,
                                 max_new_tokens=500, 
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
    return {f'r{i+1}': output for i, output in enumerate(app.tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=False))}

# # if st.checkbox('Load pipeline'):
# pipeline_load_state = st.text('Loading pipeline...')
# start = time.perf_counter()
# pipe = load_starcoder()
# # time.sleep(1)
# elapsed = time.perf_counter()-start
# pipeline_load_state.text(f'Loading pipeline...done ({elapsed: .3f}s)')

# We use a special <|end|> token with ID 49155 to denote ends of a turn

# st.text_area('Prompt:', key='prompt')
# if st.session_state.prompt:
#     prompt = prompt_template.format(query=st.session_state.prompt)
#     inference_state = st.text('Running inference...')
#     start = time.perf_counter()
#     # time.sleep(1)
#     outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, eos_token_id=49155)
#     # output = f'Test output from {prompt}'
#     elapsed = time.perf_counter()-start
#     inference_state.text(f'Running inference...done ({elapsed: .3f}s)')
#     output_text = st.text(outputs[0]['generated_text'])

# while True:
#     user_input = input('What is your query?:\n')
#     prompt = prompt_template.format(query=user_input)
#     print('Running inference...')
#     start = time.perf_counter()
#     outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, eos_token_id=49155)
#     elapsed = time.perf_counter()-start
#     print(f'Ran inference in {elapsed: .3f}s')
#     print(outputs[0]['generated_text'])