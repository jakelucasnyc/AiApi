from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from optimum.onnxruntime import ORTModelForCausalLM
import logging
_logger = logging.getLogger(__name__)
import torch
import time
import deepspeed

# def _load_starcoder():
#     checkpoint = 'HuggingFaceH4/starchat-beta'
#     _logger.info('Loading pipeline...')
#     start = time.perf_counter()
#     pipe = pipeline('text-generation', model=checkpoint, torch_dtype=torch.bfloat16, device=0)
#     elapsed = time.perf_counter() - start
#     _logger.info(f'Loaded pipeline ({elapsed: .3f}s)')
#     return pipe

def load_starcoder():
    checkpoint = 'HuggingFaceH4/starchat-beta'
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    _logger.info('Loading model...')
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, 
                                                 device_map='auto', 
                                                 torch_dtype=torch.bfloat16,
                                                 quantization_config=config, 
                                                #  load_in_8bit=True,

                                                 local_files_only=True
                                                 )
    # model.eval()
    # model = deepspeed.init_inference(model,
    #                                 #  mp_size=1,
    #                                  dtype=torch.int8,
    #                                 #  replace_method='auto',
    #                                  replace_with_kernel_inject=True,
    #                                  enable_cuda_graph=True,
    #                                  )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=7500, device_map='auto')
    tokenizer.pad_token = '<|pad|>'
    elapsed = time.perf_counter() - start
    _logger.info(f'Loaded model ({elapsed: .3f}s)')

    # _logger.info('Compiling model...')
    # start = time.perf_counter()
    # model = torch.compile(model)
    # elapsed = time.perf_counter() - start
    # _logger.info(f'Compiled model ({elapsed: .3f}s)')
    return model, tokenizer
