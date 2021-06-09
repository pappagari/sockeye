# Copyright 2017--2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Encoders for sequence-to-sequence models.
"""
import inspect
from itertools import chain
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

import mxnet as mx

from . import config
from . import constants as C
from . import layers
from . import transformer
from . import utils

logger = logging.getLogger(__name__)


ImageEncoderConfig = None


def get_encoder(config: 'EncoderConfig', prefix: str = '', dtype: str = C.DTYPE_FP32) -> 'Encoder':
    return get_transformer_encoder(config, prefix, dtype)


def get_transformer_encoder(config: transformer.TransformerConfig, prefix: str, dtype: str) -> 'Encoder':
    """
    Returns a Transformer encoder, consisting of an embedding layer with
    positional encodings and a TransformerEncoder instance.

    :param config: Configuration for transformer encoder.
    :param prefix: Prefix for variable names.
    :return: Encoder instance.
    """
    return TransformerEncoder(config=config, prefix=prefix + C.TRANSFORMER_ENCODER_PREFIX, dtype=dtype)


class Encoder(ABC, mx.gluon.HybridBlock):
    """
    Generic encoder interface.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        mx.gluon.HybridBlock.__init__(self, **kwargs)

    def forward(self, inputs, valid_length):  # pylint: disable=arguments-differ
        return mx.gluon.HybridBlock.forward(self, inputs, valid_length)

    def __call__(self, inputs, valid_length):  #pylint: disable=arguments-differ
        """
        Encodes inputs given valid lengths of individual examples.

        :param inputs: Input data.
        :param valid_length: Length of inputs without padding.
        :return: Encoded versions of input data (data, data_length).
        """
        return mx.gluon.HybridBlock.__call__(self, inputs, valid_length)

    @abstractmethod
    def get_num_hidden(self) -> int:
        """
        :return: The representation size of this encoder.
        """
        raise NotImplementedError()

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        :return: The size of the encoded sequence.
        """
        return seq_len

    def get_max_seq_len(self) -> Optional[int]:
        """
        :return: The maximum length supported by the encoder if such a restriction exists.
        """
        return None

@dataclass
class FactorConfig(config.Config):
    vocab_size: int
    num_embed: int
    combine: str  # From C.FACTORS_COMBINE_CHOICES
    share_embedding: bool


@dataclass
class EmbeddingConfig(config.Config):
    vocab_size: int
    num_embed: int
    dropout: float
    num_factors: int = field(init=False)
    factor_configs: Optional[List[FactorConfig]] = None
    allow_sparse_grad: bool = False

    def __post_init__(self):
        self.num_factors = 1
        if self.factor_configs is not None:
            self.num_factors += len(self.factor_configs)


class Embedding(Encoder):
    """
    Thin wrapper around MXNet's Embedding symbol. Works with both time- and batch-major data layouts.

    :param config: Embedding config.
    :param prefix: Name prefix for symbols of this encoder.
    :param dtype: Data type. Default: 'float32'.
    """

    def __init__(self,
                 config: EmbeddingConfig,
                 prefix: str,
                 embed_weight: Optional[mx.gluon.Parameter] = None,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(prefix=prefix)
        self.config = config
        self._dtype = dtype
        self._factor_weight_format_string = 'factor%d_weight'

        with self.name_scope():
            if embed_weight is None:
                self.embed_weight = self.params.get('weight',
                                                    shape=(self.config.vocab_size, self.config.num_embed),
                                                    grad_stype='row_sparse',
                                                    dtype=dtype)
                self._use_sparse_grad = self.config.allow_sparse_grad
            else:
                self.embed_weight = embed_weight  # adds to self._reg_params
                self.params.update({embed_weight.name: embed_weight})  # adds to self.params
                self._use_sparse_grad = embed_weight._grad_stype == 'row_sparse' and self.config.allow_sparse_grad

            if self.config.factor_configs is not None:
                for i, fc in enumerate(self.config.factor_configs, 1):
                    factor_weight_name = self._factor_weight_format_string % i
                    factor_weight = embed_weight if fc.share_embedding else \
                        self.params.get(factor_weight_name, shape=(fc.vocab_size, fc.num_embed), dtype=dtype)
                    # We set the attribute of the class to trigger the hybrid_forward parameter creation "magic"
                    setattr(self, factor_weight_name, factor_weight)

    def hybrid_forward(self, F, data, valid_length, embed_weight, **kwargs):  # pylint: disable=arguments-differ
        # We will catch the optional factor weights in kwargs
        average_factors_embeds = []  # type: List[Union[mx.sym.Symbol, mx.nd.ndarray]]
        concat_factors_embeds = []  # type: List[Union[mx.sym.Symbol, mx.nd.ndarray]]
        sum_factors_embeds = []  # type: List[Union[mx.sym.Symbol, mx.nd.ndarray]]
        if self.config.num_factors > 1 and self.config.factor_configs is not None:
            data, *data_factors = F.split(data=data,
                                          num_outputs=self.config.num_factors,
                                          axis=2,
                                          squeeze_axis=True)
            for i, (factor_data, factor_config) in enumerate(zip(data_factors,
                                                                 self.config.factor_configs), 1):
                factor_weight = kwargs[self._factor_weight_format_string % i]
                factor_embedding = F.Embedding(data=factor_data,
                                               input_dim=factor_config.vocab_size,
                                               weight=factor_weight,
                                               output_dim=factor_config.num_embed)
                if factor_config.combine == C.FACTORS_COMBINE_CONCAT:
                    concat_factors_embeds.append(factor_embedding)
                elif factor_config.combine == C.FACTORS_COMBINE_SUM:
                    sum_factors_embeds.append(factor_embedding)
                elif factor_config.combine == C.FACTORS_COMBINE_AVERAGE:
                    average_factors_embeds.append(factor_embedding)
                else:
                    raise ValueError("Unknown combine value for factors: %s" % factor_config.combine)
        else:
            data = F.squeeze(data, axis=2)

        embed = F.Embedding(data,
                            weight=embed_weight,
                            input_dim=self.config.vocab_size,
                            output_dim=self.config.num_embed,
                            dtype=self._dtype,
                            sparse_grad=self._use_sparse_grad)

        if self.config.num_factors > 1 and self.config.factor_configs is not None:
            if average_factors_embeds:
                embed = F.add_n(embed, *average_factors_embeds) / (len(average_factors_embeds) + 1)
            if sum_factors_embeds:
                embed = F.add_n(embed, *sum_factors_embeds)
            if concat_factors_embeds:
                embed = F.concat(embed, *concat_factors_embeds, dim=2)

        if self.config.dropout > 0:
            embed = F.Dropout(data=embed, p=self.config.dropout)

        return embed, F.identity(valid_length)  # identity: See https://github.com/apache/incubator-mxnet/issues/14228

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.num_embed


class EncoderSequence(Encoder, mx.gluon.nn.HybridSequential):
    """
    A sequence of encoders is itself an encoder.
    """

    def __init__(self, prefix: str = '') -> None:
        Encoder.__init__(self)
        mx.gluon.nn.HybridSequential.__init__(self, prefix=prefix)

    def add(self, *encoders):
        """Adds block on top of the stack."""
        for encoder in encoders:
            utils.check_condition(isinstance(encoder, Encoder), "%s is not of type Encoder" % encoder)
        mx.gluon.nn.HybridSequential.add(self, *encoders)

    def hybrid_forward(self, F, data, valid_length):  # pylint: disable=arguments-differ
        for block in self._children.values():
            data, valid_length = block(data, valid_length)
        return data, F.identity(valid_length)  # identity: See https://github.com/apache/incubator-mxnet/issues/14228

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return next(reversed(self._children.values())).get_num_hidden()

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        Returns the size of the encoded sequence.
        """
        for encoder in self._children.values():
            seq_len = encoder.get_encoded_seq_len(seq_len)
        return seq_len

    def get_max_seq_len(self) -> Optional[int]:
        """
        :return: The maximum length supported by the encoder if such a restriction exists.
        """
        max_seq_len = min((encoder.get_max_seq_len()
                           for encoder in self._children.values() if encoder.get_max_seq_len() is not None), default=None)
        return max_seq_len

    def append(self, cls, infer_hidden: bool = False, **kwargs) -> Encoder:
        """
        Extends sequence with new Encoder.

        :param cls: Encoder type.
        :param infer_hidden: If number of hidden should be inferred from previous encoder.
        :param kwargs: Named arbitrary parameters for Encoder.

        :return: Instance of Encoder.
        """
        params = dict(kwargs)
        if infer_hidden:
            params['num_hidden'] = self.get_num_hidden()

        sig_params = inspect.signature(cls.__init__).parameters
        encoder = cls(**params)
        self.add(encoder)
        return encoder


class TransformerEncoder(Encoder, mx.gluon.HybridBlock):
    """
    Non-recurrent encoder based on the transformer architecture in:

    Attention Is All You Need, Figure 1 (left)
    Vaswani et al. (https://arxiv.org/pdf/1706.03762.pdf).

    :param config: Configuration for transformer encoder.
    :param prefix: Name prefix for operations in this encoder.
    """

    def __init__(self,
                 config: transformer.TransformerConfig,
                 prefix: str = C.TRANSFORMER_ENCODER_PREFIX,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(prefix=prefix)
        self.config = config

        with self.name_scope():
            self.pos_embedding = layers.PositionalEmbeddings(weight_type=self.config.positional_embedding_type,
                                                             num_embed=self.config.model_size,
                                                             max_seq_len=self.config.max_seq_len_source,
                                                             prefix=C.SOURCE_POSITIONAL_EMBEDDING_PREFIX,
                                                             scale_up_input=True,
                                                             scale_down_positions=False)

            self.layers = mx.gluon.nn.HybridSequential()
            for i in range(config.num_layers):
                self.layers.add(transformer.TransformerEncoderBlock(config, prefix="%d_" % i, dtype=dtype))

            self.final_process = transformer.TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                     dropout=config.dropout_prepost,
                                                                     prefix="final_process_",
                                                                     num_hidden=self.config.model_size)
        if config.multiple_encoder_reps:
            self.layer_reps_to_concat = set(config.multiple_encoder_reps)
            utils.check_condition(config.num_layers in self.layer_reps_to_concat,
                                  'Specified encoder representations must include the last layer '
                                  f'({config.num_layers}): {sorted(self.layer_reps_to_concat)}')
            utils.check_condition(all(layer_num <= config.num_layers for layer_num in self.layer_reps_to_concat),
                                  'Specified encoder representations must refer to existing layers '
                                  f'{tuple(range(1, config.num_layers + 1))}: {sorted(self.layer_reps_to_concat)}')


    def hybrid_forward(self, F, data, valid_length):
        # positional embedding
        data = self.pos_embedding(data, None)

        if self.config.dropout_prepost > 0.0:
            data = F.Dropout(data=data, p=self.config.dropout_prepost)

        # (batch_size * heads, seq_len)
        att_valid_length = layers.prepare_source_valid_lengths(F, valid_length, data,
                                                               num_heads=self.config.attention_heads)

        data = F.transpose(data, axes=(1, 0, 2))

        data_to_concat = []
        for layer_num, block in enumerate(self.layers, 1):
            data = block(data, att_valid_length)
            if self.config.multiple_encoder_reps and layer_num in self.layer_reps_to_concat:
                data_to_concat.append(data)

        if self.config.multiple_encoder_reps:
            data = F.concat(*data_to_concat, dim=0)

        data = self.final_process(data, None)
        data = F.transpose(data, axes=(1, 0, 2))

        if self.config.multiple_encoder_reps:
            # Using encoder representations from N layers multiplies the
            # sequence length by N.
            valid_length = valid_length * len(self.layer_reps_to_concat)

            # NOTE: Multiple encoder representations must also be consolidated
            # for the decoder to use them correctly. This is implemented in the
            # separate `consolidate_encoder_reps` method as it requires NDArray
            # operations that break hybridization. When using multiple encoder
            # representations, call `consolidate_encoder_reps` immediately after
            # running this encoder.

        return data, valid_length

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.model_size


def consolidate_encoder_reps(data: mx.nd.NDArray, valid_length: mx.nd.NDArray, num_reps: int):
        """
        Consolidate multiple encoder representations that are concatenated in
        the seq_len dimension. This is required for batch_size > 1 because
        sequences shorter than the batch/bucket max length are padded. For
        example:

        input = [[a b <pad> <pad>]
                 [c d e <pad>]]

        valid_length = [2, 3]

        Concatenating encoder representations for layers 1 and 2 and updating
        valid_length accordingly yields:

        data = [[a_l1 b_l1 <pad>_l1 <pad>_l1 a_l2 b_l2 <pad>_l2 <pad>_l2]
                [c_l1 d_l1 e_l1 <pad>_l1 c_l2 d_l2 e_l2 <pad>_l2]]

        valid_length = [4, 6]

        For the decoder to work properly, all encodings of source content tokens
        must be consolidated to the left, followed by all encodings of padding
        tokens. This enables attention layers to correctly use content encodings
        and ignore padding encodings based on valid_length:

        consolidated_data =
            [[a_l1 b_l1 a_l2 b_l2 <pad>_l1 <pad>_l1 <pad>_l2 <pad>_l2]
             [c_l1 d_l1 e_l1 c_l2 d_l2 e_l2 <pad>_l1 <pad>_l2]]

        valid_length = [4, 6]

        :param data: Encoded data from `hybrid_forward`. Shape:
                     (batch_size, seq_len * num_reps, model_size)
        :param valid_length: Encoded data lengths from `hybrid_forward`. Shape:
                             (batch_size)
        :return: consolidated data. Shape:
                 (batch_size, seq_len * num_reps, model_size)
        """
        assert num_reps > 1, \
            'This method should only be called when using multiple encoder representations'
        if data.shape[0] == 1:
            # No padding when batch_size=1; no need to condense
            return data
        padded_rep_length = data.shape[1] // num_reps
        condensed_source_encoded = []
        # Process one encoded sequence at a time
        for reps, vlen in zip(data, valid_length):
            # Actual length of each representation in this sequence
            valid_rep_length = int(vlen.asscalar()) // num_reps
            if valid_rep_length == padded_rep_length:
                # No padding for batch/bucket max length; no need to condense
                condensed_source_encoded.append(reps)
            else:
                # Build list of indices that remaps alternating spans of data
                # and padding to all data followed by all padding, otherwise
                # maintaining order
                valid_indices = []
                pad_indices = []
                for rep_num in range(num_reps):
                    i = rep_num * padded_rep_length
                    j = i + valid_rep_length
                    k = j + (padded_rep_length - valid_rep_length)
                    valid_indices.append(range(i, j))
                    pad_indices.append(range(j, k))
                indices = mx.nd.array(list(chain(*valid_indices, *pad_indices)))
                # Remap (sequence finished)
                condensed_source_encoded.append(reps.take(indices))
        return mx.nd.stack(*condensed_source_encoded, axis=0)


EncoderConfig = Union[transformer.TransformerConfig]
