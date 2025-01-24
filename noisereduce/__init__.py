import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from noisereduce.noisereduce import reduce_noise
