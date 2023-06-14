from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
_logger = logging.getLogger(__name__)
import torch
import time

def _load_starcoder():
    checkpoint = 'HuggingFaceH4/starchat-beta'
    _logger.info('Loading pipeline...')
    start = time.perf_counter()
    pipe = pipeline('text-generation', model=checkpoint, torch_dtype=torch.bfloat16, device=0)
    elapsed = time.perf_counter() - start
    _logger.info(f'Loaded pipeline ({elapsed: .3f}s)')
    return pipe

def load_starcoder():
    checkpoint = 'HuggingFaceH4/starchat-beta'
    _logger.info('Loading model...')
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=7500)
    elapsed = time.perf_counter() - start
    _logger.info(f'Loaded model ({elapsed: .3f}s)')
    return model, tokenizer
