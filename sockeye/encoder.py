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

        # If specified, return representations from multiple layers. By default,
        # return representations from only the last layer.
        if config.multiple_encoder_reps:
            self.layer_reps_to_return = set(config.multiple_encoder_reps)
            utils.check_condition(config.num_layers in self.layer_reps_to_return,
                                  'Specified encoder representations must include the last layer '
                                  f'({config.num_layers}): {sorted(self.layer_reps_to_return)}')
            utils.check_condition(all(layer_num <= config.num_layers for layer_num in self.layer_reps_to_return),
                                  'Specified encoder representations must refer to existing layers '
                                  f'{tuple(range(1, config.num_layers + 1))}: {sorted(self.layer_reps_to_return)}')
        else:
            self.layer_reps_to_return = {config.num_layers}


    def hybrid_forward(self, F, data, valid_length):
        """
        Returns encoded data (NDArray/Symbol for single representation or list
        of NDArrays/Symbols for multiple representations) and valid sequence
        lengths
        """
        # positional embedding
        data = self.pos_embedding(data, None)

        if self.config.dropout_prepost > 0.0:
            data = F.Dropout(data=data, p=self.config.dropout_prepost)

        # (batch_size * heads, seq_len)
        att_valid_length = layers.prepare_source_valid_lengths(F, valid_length, data,
                                                               num_heads=self.config.attention_heads)

        # (seq_len, batch_size, model_size)
        data = F.transpose(data, axes=(1, 0, 2))

        # Run encoder layers, keeping references to layer representations (data)
        # that will be returned
        layer_data = []
        for layer_num, block in enumerate(self.layers, 1):
            data = block(data, att_valid_length)
            if layer_num in self.layer_reps_to_return:
                layer_data.append(data)

        # Apply final processing to each representation
        layer_data = [self.final_process(data, None) for data in layer_data]
        layer_data = [F.transpose(data, axes=(1, 0, 2)) for data in layer_data]

        if self.config.multiple_encoder_reps:
            # List of representations
            return layer_data, valid_length

        # Default: single representation
        return layer_data[0], valid_length

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.model_size


def concat_encoder_reps(layer_reps: List[mx.nd.NDArray], valid_length: mx.nd.NDArray):
        """
        Concatenate multiple encoder representations in the seq_len dimension
        and update valid_length to match. Reorder each resulting sequence so
        that all content encodings are consolidated to the left, followed by all
        padding encodings. Consolidation is required for valid_length to be
        meaningful and thus for decoder attention layers to work properly.

        Example:

        input = [[a b <pad> <pad>]
                 [c d e <pad>]]

        valid_length = [2, 3]

        Output of an encoder that returns representations from layers 1 and 2,
        where a_l1 is the encoding of token "a" from layer 1, etc.:

        layer_reps = [[[a_l1 b_l1 <pad>_l1 <pad>_l1]
                       [c_l1 d_l1 e_l1 <pad>_l1]],
                      [[a_l2 b_l2 <pad>_l2 <pad>_l2]
                       [c_l2 d_l2 e_l2 <pad>_l2]]]

        Output of this function:

        concatenated_layer_reps =
            [[a_l1 b_l1 a_l2 b_l2 <pad>_l1 <pad>_l1 <pad>_l2 <pad>_l2]
             [c_l1 d_l1 e_l1 c_l2 d_l2 e_l2 <pad>_l1 <pad>_l2]]

        updated_valid_length = [4, 6]

        :param layer_reps: Encoded data from `hybrid_forward`. List of NDArrays
                           with shape (batch_size, seq_len, model_size)
        :param valid_length: Encoded data lengths from `hybrid_forward`. NDArray
                             with shape (batch_size)
        :return: consolidated data. NDArray with shape
                 (batch_size, seq_len * num_reps, model_size) where num_reps is
                 len(layer_data)
        """
        num_reps = len(layer_reps)
        batch_size, padded_rep_length, model_size = layer_reps[0].shape
        if num_reps == 1:
            # Single representation; nothing to concat
            return layer_reps[0], valid_length

        # Concat N representations in seq_len dimension
        concat_reps = mx.nd.concat(*layer_reps, dim=1)
        concat_valid_length = valid_length * num_reps

        if concat_reps.shape[0] == 1:
            # Batch size 1: no padding encodings; nothing to remap
            return concat_reps, concat_valid_length

        # Build the list of indices that remaps encodings in a reshaped batch
        # (batch_size * seq_len). For each sequence (span of seq_len in reshaped
        # batch), all content embeddings are followed by all padding embeddings,
        # otherwise preserving order.
        remap = []  # type: List[int]
        for seq_num, vlen in enumerate(valid_length):
            # Actual length of each representation in the current sequence
            valid_rep_length = int(vlen.asscalar())
            # Indexes for remapping current sequence
            valid_indices = []  # type: List[int]
            pad_indices = []  # type: List[int]
            for rep_num in range(num_reps):
                # Track positions of encoded content vs padding for the current
                # representation within the current sequence
                i = (seq_num * padded_rep_length * num_reps) + rep_num * padded_rep_length
                j = i + valid_rep_length
                k = j + (padded_rep_length - valid_rep_length)
                valid_indices.extend(range(i, j))
                pad_indices.extend(range(j, k))
            remap.extend(valid_indices)
            remap.extend(pad_indices)
        # Use reshaping to apply a single take operation using the remapping
        # list
        concat_reps = concat_reps.reshape(shape=(batch_size * padded_rep_length * num_reps, model_size))
        concat_reps = concat_reps.take(mx.nd.array(remap))
        concat_reps = concat_reps.reshape(shape=(batch_size, padded_rep_length * num_reps, model_size))

        return concat_reps, concat_valid_length


EncoderConfig = Union[transformer.TransformerConfig]
