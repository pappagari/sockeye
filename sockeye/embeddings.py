# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

"""
Command-line tool to inspect model embeddings.
"""
import argparse
import sys
import logging
from typing import Iterable, Tuple

import mxnet as mx
from mxnet import np, npx
import numpy as onp

from . import constants as C
from . import model
from . import utils
from .data_io import tokens2ids
from .log import setup_main_logger
from .utils import check_condition
from .vocab import reverse_vocab

logger = logging.getLogger(__name__)


def compute_sims(inputs: np.ndarray, normalize: bool) -> np.ndarray:
    """
    Returns a matrix with pair-wise similarity scores between inputs.
    Similarity score is (normalized) Euclidean distance. 'Similarity with self' is masked
    to large negative value.

    :param inputs: ndarray of inputs.
    :param normalize: Whether to normalize to unit-length.
    :return: ndarray with pairwise similarities of same shape as inputs.
    """
    if normalize:
        logger.info("Normalizing embeddings to unit length")
        inputs = inputs / np.linalg.norm(inputs, axis=-1)
    sims = np.dot(inputs, inputs)
    np.fill_diagonal(sims, -9999999.)
    return sims


def nearest_k(similarity_matrix: np.ndarray,
              query_word_id: int,
              k: int,
              gamma: float = 1.0) -> Iterable[Tuple[int, float]]:
    """
    Returns values and indices of k items with largest similarity.

    :param similarity_matrix: Similarity matrix.
    :param query_word_id: Query word id.
    :param k: Number of closest items to retrieve.
    :param gamma: Parameter to control distribution steepness.
    :return: List of indices and values of k nearest elements.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    values, indices = npx.topk(npx.softmax(similarity_matrix[query_word_id] / gamma), k=k, ret_typ='both')
    return zip(indices, values)


def main():
    """
    Command-line tool to inspect model embeddings.
    """
    setup_main_logger(file_logging=False)
    params = argparse.ArgumentParser(description='Shows nearest neighbours of input tokens in the embedding space.')
    params.add_argument('--model', '-m', required=True,
                        help='Model folder to load config from.')
    params.add_argument('--checkpoint', '-c', required=False, type=int, default=None,
                        help='Optional specific checkpoint to load parameters from. Best params otherwise.')
    params.add_argument('--side', '-s', required=True, choices=['source', 'target'], help='what embeddings to look at')
    params.add_argument('--norm', '-n', action='store_true', help='normalize embeddings to unit length')
    params.add_argument('-k', type=int, default=5, help='Number of neighbours to print')
    params.add_argument('--gamma', '-g', type=float, default=1.0, help='Softmax distribution steepness.')
    args = params.parse_args()
    embeddings(args)


def embeddings(args: argparse.Namespace):
    logger.info("Arguments: %s", args)

    sockeye_model, source_vocabs, target_vocabs = model.load_model(args.model,
                                                                   checkpoint=args.checkpoint,
                                                                   hybridize=False)

    if args.side == "source":
        vocab = source_vocabs[0]
    else:
        vocab = target_vocabs[0]
    vocab_inv = reverse_vocab(vocab)

    params = sockeye_model.collect_params()
    if args.side == "source":
        logger.info("Loading %s", sockeye_model.embedding_source.weight.name)
        weights = params[sockeye_model.embedding_source.weight.name].data()
    else:
        logger.info("Loading %s", sockeye_model.embedding_target.weight.name)
        weights = params[sockeye_model.embedding_target.weight.name].data()
    logger.info("Embedding size: %d", weights.shape[1])

    logger.info("Computing pairwise similarities...")
    sims = compute_sims(weights, args.norm)

    # weights (vocab, num_target_embed)
    check_condition(weights.shape[0] == len(vocab),
                    "vocab and embeddings matrix do not match: %d vs. %d" % (weights.shape[0], len(vocab)))

    logger.info("Reading from STDin...")
    for line in sys.stdin:
        tokens = list(utils.get_tokens(line))
        if not tokens:
            continue
        print("Input:", line.rstrip())
        ids = tokens2ids(tokens, vocab)
        for token, token_id in zip(tokens, ids):
            print("%s id=%d" % (token, token_id))
            neighbours = nearest_k(sims, token_id, args.k, args.gamma)
            for i, (neighbour_id, score) in enumerate(neighbours, 1):
                print("  %s id=%d sim=%.4f" % (vocab_inv[neighbour_id], neighbour_id, score))
        print()


if __name__ == '__main__':
    main()
