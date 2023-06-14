# import streamlit as st
import time
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI, HTTPException
from app.utils import load_starcoder
from app.models import Prompt
import logging

_logger = logging.getLogger(__name__)

app = FastAPI()
app.pipe = None
app.pipe = load_starcoder()

@app.post('/prompt/', status_code=200)
async def prompt(prompt: Prompt):
    if app.pipe is None:
        raise HTTPException(status_code=503, detail='StarChat pipeline not initialized')
    if not (0 <= prompt.temp >= 1):
        raise HTTPException(status_code=400, detail=f'Temperature must be in this range: 0 <= temp >= 1, not {prompt.temp}')

    prompt = f"<|system|>\n{prompt.system}<|end|>\n<|user|>\n{prompt.user}<|end|>\n<|assistant|>"
    _logger.info('Running inference...')
    start = time.perf_counter()
    outputs = app.pipe(prompt, do_sample=True, temperature=prompt.temp, top_k=50, top_p=0.95, eos_token_id=49155)
    elapsed = time.perf_counter()-start
    _logger.info(f'Ran inference ({elapsed: .3f}s)')
    return {'response': outputs[0]['generated_text']}

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