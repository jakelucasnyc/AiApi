import streamlit as st
import time
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_data
def load_starcoder():
    checkpoint = 'HuggingFaceH4/starchat-beta'
    # print('Loading pipeline...')
    pipe = pipeline('text-generation', model=checkpoint, torch_dtype=torch.bfloat16, device=0)
    # print(f'Loaded pipeline in {elapsed: .3f}s')
    return pipe

pipeline_load_state = st.text('Loading pipeline...')
start = time.perf_counter()
pipe = load_starcoder()
# time.sleep(1)
elapsed = time.perf_counter()-start
pipeline_load_state.text(f'Loading pipeline...done ({elapsed: .3f}s)')

prompt_template = "<|system|>\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
# We use a special <|end|> token with ID 49155 to denote ends of a turn

st.text_area('Prompt:', key='prompt')
if st.session_state.prompt:
    prompt = prompt_template.format(query=st.session_state.prompt)
    inference_state = st.text('Running inference...')
    start = time.perf_counter()
    # time.sleep(1)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, eos_token_id=49155)
    # output = f'Test output from {prompt}'
    elapsed = time.perf_counter()-start
    inference_state.text(f'Running inference...done ({elapsed: .3f}s)')
    output_text = st.text(outputs[0]['generated_text'])

# while True:
#     user_input = input('What is your query?:\n')
#     prompt = prompt_template.format(query=user_input)
#     print('Running inference...')
#     start = time.perf_counter()
#     outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, eos_token_id=49155)
#     elapsed = time.perf_counter()-start
#     print(f'Ran inference in {elapsed: .3f}s')
#     print(outputs[0]['generated_text'])