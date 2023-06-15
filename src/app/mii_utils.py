import mii

mii_configs = {'tensor_parallel': 1, 'dtype': 'fp16', 'port_number': 80}
mii.deploy(task='text-generation',
           model='HuggingFaceH4/starchat-beta',
           deployment_name='starchat-beta-deployment',
           mii_config=mii_configs)

generator = mii.mii_query_handle('starchat-beta-deployment')