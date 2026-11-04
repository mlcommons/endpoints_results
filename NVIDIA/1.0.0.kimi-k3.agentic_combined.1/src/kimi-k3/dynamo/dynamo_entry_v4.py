"""Diagnostic decode entry point, keeping the installed public wheel intact."""
import runpy
import sys

from visible_eos_v4 import wrap_stop_mapper
from dynamo.trtllm.utils import request_utils

assert '--disaggregation-mode' in sys.argv
assert sys.argv[sys.argv.index('--disaggregation-mode') + 1] == 'decode'
request_utils.apply_stop_conditions_to_sampling_params = wrap_stop_mapper(
    request_utils.apply_stop_conditions_to_sampling_params)
runpy.run_module('dynamo.trtllm', run_name='__main__')
