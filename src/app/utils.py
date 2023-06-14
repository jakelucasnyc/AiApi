from transformers import pipeline
import logging
_logger = logging.getLogger(__name__)
import torch
import time

def load_starcoder():
    checkpoint = 'HuggingFaceH4/starchat-beta'
    _logger.info('Loading pipeline...')
    start = time.perf_counter()
    pipe = pipeline('text-generation', model=checkpoint, torch_dtype=torch.bfloat16, device=0)
    elapsed = time.perf_counter() - start
    _logger.info(f'Loaded pipeline ({elapsed: .3f}s)')
    return pipe