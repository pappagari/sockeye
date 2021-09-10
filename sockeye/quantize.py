# Copyright 2020-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import logging
import os

import sockeye.constants as C
from sockeye.log import setup_main_logger, log_sockeye_version
import sockeye.model
from sockeye.utils import check_condition

logger = logging.getLogger(__name__)


def quantize(model_dir: str, dtype: str = C.DTYPE_FP16, force: bool = False):
    '''
    Quantize a trained Sockeye model in-place.

    :param model_dir: Trained Sockeye model directory.
    :param dtype: Target data type.
    :param force: Force-quantize model even if already quantized.
    '''
    log_sockeye_version(logger)

    params_best = os.path.join(model_dir, C.PARAMS_BEST_NAME)
    params_best_float32 = os.path.join(model_dir, C.PARAMS_BEST_NAME_FLOAT32)
    config = os.path.join(model_dir, C.CONFIG_NAME)
    config_float32 = os.path.join(model_dir, C.CONFIG_NAME_FLOAT32)

    if not force:
        for fname in params_best_float32, config_float32:
            check_condition(not os.path.exists(fname),
                            'File "%s" exists, indicating this model has already been quantized.' % fname)

    if dtype == C.DTYPE_FP16:
        # Load model and cast to float16
        model, _, _ = sockeye.model.load_model(model_dir, dtype=dtype)
    elif dtype == C.DTYPE_INT8:
        # Load model and compute int8 scaling factors
        model, _, __ = sockeye.model.load_model(model_dir, for_disk_saving=C.DTYPE_FP32, dtype=C.DTYPE_INT8)
    else:
        raise ValueError('Unknown quantization dtype: %s' % dtype)

    # Move original params and config files
    os.rename(params_best, params_best_float32)
    os.rename(config, config_float32)

    # Write new params
    model.save_parameters(params_best)
    # Write new config file. Float16 models are stored as float16. Int8 models
    # are stored as float32 with scaling factors.
    model.save_config(model_dir, dtype=dtype if dtype == C.DTYPE_FP16 else None)


def annotate_model_params(model_dir: str):
    '''
    Deprecated. Use `quantize()`. This function is kept for backward
    compatibility.
    '''
    logger.warn('The function `annotate_model_params()` is deprecated. Use `quantize()`.')
    quantize(model_dir, dtype=C.DTYPE_INT8)


def main():
    setup_main_logger(console=True, file_logging=False)
    params = argparse.ArgumentParser(description='Quantize trained Sockeye model')
    params.add_argument('--model', '-m', required=True, help='Trained Sockeye model directory.')
    params.add_argument('--dtype', '-d', default=C.DTYPE_INT8, choices=[C.DTYPE_FP16, C.DTYPE_INT8],
                        help='Target data type. Default: %(default)s.')
    params.add_argument('--force', '-f', action='store_true', default=False,
                        help='Force-quantize model even if already quantized. Default: %(default)s.')
    args = params.parse_args()

    quantize(args.model, dtype=args.dtype, force=args.force)


if __name__ == '__main__':
    main()
