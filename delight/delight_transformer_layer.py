# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from delight.delight_modules.activation_layers import get_activation_layer
from delight.delight_modules.normalization_layers import get_norm_layer
from delight.delight_modules.nn_functions import get_weight_layer
from delight.delight_modules.config import DEFAULT_WIDTH_MULTIPLIER, DEFAULT_MIN_DEXTRA_LAYERS
from delight.delight_modules.dextra_unit import DExTraUnit

def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)

class SingleHeadAttention(nn.Module):
    """Single head attention as defined in DeLighT paper
    """

    def __init__(self, q_in_dim, kv_in_dim, proj_dim, out_dim,
                 dropout=0.0, bias=True,
                 self_attention=False, encoder_decoder_attention=False):
        '''
        :param embed_dim: Input dimension
        :param out_dim: Output dimension
        :param dropout: attention dropout
        :param bias: use bias or not
        :param self_attention: Using for self attention or not
        :param encoder_decoder_attention: Using for encoder-decoder attention or not
        :param qkv_proj: Project QKV or not. This is useful for projecting encoder output to query's dimensionality
        '''
        super(SingleHeadAttention, self).__init__()
        self.q_embed_dim = q_in_dim
        self.kv_embed_dim = kv_in_dim
        self.proj_dim = proj_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        if self.self_attention:
            assert q_in_dim == kv_in_dim
            self.linear_kqv = get_weight_layer(name='linear',
                                               in_features=self.q_embed_dim,
                                               out_features=self.proj_dim,
                                               use_bias=True,
                                               gates=3
                                               )
        elif self.encoder_decoder_attention:
            self.linear_q = get_weight_layer(name='linear',
                                             in_features=self.q_embed_dim,
                                             out_features=self.proj_dim,
                                             use_bias=True,
                                             gates=1
                                             )
            self.linear_kv = get_weight_layer(name='linear',
                                              in_features=self.kv_embed_dim,
                                              out_features=self.proj_dim,
                                              use_bias=True,
                                              gates=2
                                              )
        self.scaling = self.proj_dim ** -0.5
        self.out_proj = get_weight_layer(name='linear',
                                      in_features=self.proj_dim,
                                      out_features=self.out_dim,
                                      use_bias=True
                                      )

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def __repr__(self):
        s = '{name}(q_in_features={q_embed_dim}, kv_in_features={kv_embed_dim}, out_features={out_dim}, ' \
            'attn_dropout={dropout}, self_attention={self_attention}, ' \
            'encoder_decoder_attention={encoder_decoder_attention})'
        if self.self_attention:
            s += '\n  \t |---- KQV function: \t {}'.format(self.linear_kqv)
        elif self.encoder_decoder_attention:
            s += '\n  \t |---- KV function: \t {}'.format(self.linear_kv)
            s += '\n  \t |---- Q function: \t {}'.format(self.linear_q)
        s += '\n  \t |---- Proj: {}'.format(self.out_proj)
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(
            self,
            query,
            key_value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, q_embed_dim = query.size()
        assert q_embed_dim == self.q_embed_dim, 'Error in {}. {} != {}'.format(self.__class__.__name__, q_embed_dim,
                                                                               self.q_embed_dim)
        assert list(query.size()) == [tgt_len, bsz, q_embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key_value = None
        else:
            saved_state = None

        if self.self_attention:
            q, k, v = torch.chunk(self.linear_kqv(query), chunks=3, dim=-1)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.linear_q(query)
            if key_value is None:
                k = v = None
            else:
                k, v = torch.chunk(self.linear_kv(key_value), chunks=2, dim=-1)
        else:
            raise NotImplementedError

        q = q * self.scaling

        q = q.contiguous().transpose(0, 1)

        if k is not None:
            k = k.contiguous().transpose(0, 1)

        if v is not None:
            v = v.contiguous().transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, seq_len, head_dim)
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"]
                assert prev_key is not None
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"]
                assert prev_value is not None
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)

            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = SingleHeadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k
            saved_state["prev_value"] = v
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        # [B x T x C] x [B x C x S] --> [B x T x S]
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        attn_weights = SingleHeadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # key_padding_mask --> (B x Src_len)
            # don't attend to padding symbols
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool), float("-inf")
            )

        if before_softmax:
            return attn_weights, v

        # [B x T x S] --> [B x T x S]
        attn_weights_float = softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        # [B x T x S] x [B x S x F] --> [B x T x F]
        attn = torch.bmm(attn_probs, v)

        assert list(attn.size()) == [bsz, tgt_len, self.proj_dim]
        # [B x T x F] --> [T x B x F]
        attn = attn.transpose(0, 1).contiguous()

        # [T x B x F] --> [ T x B x F']
        attn = self.out_proj(attn)

        if need_weights:
            # [B x T x S] --> [T x B x S]
            attn_weights = attn_weights.transpose(1, 0)
            return attn, attn_weights
        else:
            attn_weights_tmp: Optional[Tensor] = None
            return attn, attn_weights_tmp

    @staticmethod
    def _append_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:

            filler = torch.zeros(batch_size, src_len - prev_key_padding_mask.size(1))
            if prev_key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1))
            if key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                                                           dim: 2 * dim
                                                           ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

    def compute_macs_params(self, T=1, S=1):
        macs = 0
        n_params = 0

        C = self.proj_dim

        # Scaled-dot-product macs
        # [B x T x C] x [B x C x S] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        num_macs_kq = T * S * C
        # [B x T x S] x [B x S x C] --> [B x T x C]
        num_macs_v = T * C * S

        macs += num_macs_kq + num_macs_v

        if self.self_attention:
            assert T == S
            q_params = sum([p.numel() for p in self.linear_kqv.parameters()])

            # multiply by Seq length
            macs += (q_params * T)
            n_params += q_params
        elif self.encoder_decoder_attention:
            q_params = sum([p.numel() for p in self.linear_q.parameters()])
            kv_params = sum([p.numel() for p in self.linear_kv.parameters()])

            macs += (q_params * T) + (kv_params * S)
            n_params += q_params + kv_params
        else:
            raise NotImplementedError

        out_params = sum([p.numel() for p in self.out_proj.parameters()])
        macs += (out_params * T)
        n_params += out_params

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params,
            'macs_attn': num_macs_kq + num_macs_v
        }

class DeLighTTransformerEncoderLayer(nn.Module):
    """DeLight Encoder layer
    """

    def __init__(self, args, embed_dim, width_multiplier=DEFAULT_WIDTH_MULTIPLIER, dextra_depth=DEFAULT_MIN_DEXTRA_LAYERS,
                 dextra_proj=2):
        super().__init__()
        self.embed_dim = embed_dim
        assert embed_dim % dextra_proj == 0

        self.proj_dim = embed_dim // dextra_proj
        self.dextra_layer = DExTraUnit(in_features=self.embed_dim,
                                       in_proj_features=self.proj_dim,
                                       out_features=self.proj_dim,
                                       width_multiplier=width_multiplier,
                                       dextra_depth=dextra_depth,
                                       dextra_dropout=args.delight_dropout,
                                       max_glt_groups=args.delight_enc_max_groups,
                                       act_type=args.act_type,
                                       use_bias=True,
                                       norm_type=args.norm_type,
                                       glt_shuffle=args.glt_shuffle,
                                       is_iclr_version=args.define_iclr
                                       )

        self.self_attn = SingleHeadAttention(q_in_dim=self.proj_dim,
                                             kv_in_dim=self.proj_dim,
                                             proj_dim=self.proj_dim,
                                             out_dim=self.embed_dim,
                                             dropout=args.attention_dropout,
                                             bias=True,
                                             self_attention=True,
                                             encoder_decoder_attention=False)

        self.self_attn_layer_norm = get_norm_layer(name=args.norm_type, out_features=self.embed_dim)
        self.dropout = args.dropout
        self.norm_fn = args.norm_type
        self.act_type = args.act_type
        self.activation_fn = get_activation_layer(name=args.act_type)
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.encoder_normalize_before

        # Light-weight FFN
        self.ffn_dropout = args.ffn_dropout
        ffn_red_factor = args.delight_enc_ffn_red
        assert self.embed_dim % ffn_red_factor == 0, '{}/{} should be a perfect divisor'.format(self.embed_dim,
                                                                                                ffn_red_factor)
        light_ffn_dim = self.embed_dim // ffn_red_factor
        self.fc1 = get_weight_layer(name='linear',
                                    in_features=self.embed_dim,
                                    out_features=light_ffn_dim,
                                    use_bias=True)
        self.fc2 = get_weight_layer(name='linear',
                                    in_features=light_ffn_dim,
                                    out_features=self.embed_dim,
                                    use_bias=True)

        self.final_layer_norm = get_norm_layer(name=args.norm_type, out_features=self.embed_dim)

    def __repr__(self):
        s = '{name}(in_features={embed_dim}, out_features={embed_dim}, dropout={dropout},' \
            'activation_dropout={activation_dropout}, ffn_dropout={ffn_dropout}, ' \
            'activation_fn={act_type}, norm_fn={norm_fn})'
        s += '\n \t Dextra Layer: \n \t \t {}'.format(self.dextra_layer)
        s += '\n \t Self Attention: \n \t \t {}'.format(self.self_attn)
        s += '\n \t     Light-weight FFN: \n \t     |---- {} \n \t     |---- {}'.format(self.fc1, self.fc2)
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        x = self.dextra_layer(x)

        x, _ = self.self_attn(
            query=x,
            key_value=None,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Light-weight FFN
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.ffn_dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

    def compute_macs_params(self, S=1):
        macs = 0
        n_params = 0
        macs_attn = 0

        # Layer Norms
        # MACS are zero for LayerNorm because they can be fused
        n_params += sum([p.numel() for p in self.self_attn_layer_norm.parameters()])

        # Dextra layer
        dextra_layer = self.dextra_layer.compute_macs_params()
        n_params += dextra_layer['params']
        macs += (dextra_layer['macs'] * S)

        # Attn
        self_attn_layer = self.self_attn.compute_macs_params(T=S, S=S)
        macs += self_attn_layer['macs']
        n_params += self_attn_layer['params']
        macs_attn += self_attn_layer['macs_attn']

        # FFN
        fc1_layer = self.fc1.compute_macs_params()
        # scale MACS by S because S tokens can be processed in parallel
        macs += (fc1_layer['macs'] * S)
        n_params += fc1_layer['params']

        fc2_layer = self.fc2.compute_macs_params()
        # scale MACS by S because S tokens can be processed in parallel
        macs += (fc2_layer['macs'] * S)
        n_params += fc2_layer['params']

        n_params += sum([p.numel() for p in self.final_layer_norm.parameters()])

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params,
            'macs_attn': macs_attn
        }


class DeLighTTransformerDecoderLayer(nn.Module):
    """Delight Decoder layer
    """

    def __init__(self, args, embed_dim, width_multiplier=DEFAULT_WIDTH_MULTIPLIER, dextra_depth=DEFAULT_MIN_DEXTRA_LAYERS,
                 no_encoder_attn=False, dextra_proj=2, *unused_args, **unused_kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        assert embed_dim % dextra_proj == 0
        self.proj_dim = embed_dim // dextra_proj

        self.norm_fn = args.norm_type
        self.act_type = args.act_type

        self.dextra_layer_sa = DExTraUnit(in_features=self.embed_dim,
                                          in_proj_features=self.proj_dim,
                                          out_features=self.proj_dim,
                                          width_multiplier=width_multiplier,
                                          dextra_depth=dextra_depth,
                                          dextra_dropout=args.delight_dropout,
                                          max_glt_groups=args.delight_dec_max_groups,
                                          act_type=args.act_type,
                                          use_bias=True,
                                          norm_type=args.norm_type,
                                          glt_shuffle=args.glt_shuffle,
                                          is_iclr_version=args.define_iclr
                                          )

        self.self_attn = SingleHeadAttention(q_in_dim=self.proj_dim,
                                             kv_in_dim=self.proj_dim,
                                             proj_dim=self.proj_dim,
                                             out_dim=self.embed_dim,
                                             dropout=args.attention_dropout,
                                             bias=True,
                                             self_attention=True,
                                             encoder_decoder_attention=False)

        self.dropout = args.dropout
        self.activation_fn = get_activation_layer(name=args.act_type)

        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = get_norm_layer(name=args.norm_type, out_features=self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            q_embed_dim = self.embed_dim
            self.encoder_attn = SingleHeadAttention(q_in_dim=q_embed_dim,
                                                    kv_in_dim=self.embed_dim,
                                                    proj_dim=self.proj_dim,
                                                    out_dim=self.embed_dim,
                                                    dropout=args.attention_dropout,
                                                    bias=True,
                                                    encoder_decoder_attention=True,
                                                    self_attention=False)

            self.encoder_attn_layer_norm = get_norm_layer(name=args.norm_type, out_features=self.embed_dim)

        self.ffn_dropout = args.ffn_dropout
        ffn_red_factor = args.delight_dec_ffn_red
        assert self.embed_dim % ffn_red_factor == 0, '{}/{} should be a perfect divisor'.format(self.embed_dim,
                                                                                                ffn_red_factor)

        # Feed forward network
        light_ffn_dim = self.embed_dim // ffn_red_factor
        self.fc1 = get_weight_layer(name='linear',
                                    in_features=self.embed_dim,
                                    out_features=light_ffn_dim,
                                    use_bias=True)
        self.fc2 = get_weight_layer(name='linear',
                                    in_features=light_ffn_dim,
                                    out_features=self.embed_dim,
                                    use_bias=True)
        self.final_layer_norm = get_norm_layer(name=args.norm_type, out_features=self.embed_dim)

        self.need_attn = True
        self.onnx_trace = False

    def __repr__(self):
        s = '{name}(in_features={embed_dim}, out_features={embed_dim}, dropout={dropout}, ' \
            'activation_dropout={activation_dropout}, ffn_dropout={ffn_dropout}, ' \
            'activation_fn={act_type}, norm_fn={norm_fn})'
        s += '\n \t     Dextra Layer (Query): \n \t \t {}'.format(self.dextra_layer_sa)
        s += '\n \t     Self Attention (Decoder): \n \t \t {}'.format(self.self_attn)
        if self.encoder_attn is not None:
            s += '\n \t     Encoder-Decoder Attention: \n \t \t {}'.format(self.encoder_attn)
        s += '\n \t     Light-weight FFN: \n \t     |---- {} \n \t     |---- {}'.format(self.fc1, self.fc2)
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # apply dextra layer
        x = self.dextra_layer_sa(x)

        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]

            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        x, attn = self.self_attn(
            query=x,
            key_value=None,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key_value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        #Light-weight FFN
        residual = x

        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.ffn_dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)


        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def compute_macs_params(self, T=1, S=1):
        macs = 0
        n_params = 0
        macs_attn = 0

        # LayerNorm
        n_params += sum([p.numel() for p in self.self_attn_layer_norm.parameters()])

        # self attention
        self_attn_layer = self.self_attn.compute_macs_params(T=T, S=T)
        dextra_layer = self.dextra_layer_sa.compute_macs_params()
        macs += self_attn_layer['macs'] + (dextra_layer['macs'] * T)
        n_params += self_attn_layer['params'] + dextra_layer['params']
        macs_attn += self_attn_layer['macs_attn']

        # Encoder-decoder attn
        if self.encoder_attn is not None:
            # self attention scaled-dot-product Attn
            n_params += sum([p.numel() for p in self.encoder_attn_layer_norm.parameters()])

            enc_attn = self.encoder_attn.compute_macs_params(T=T, S=S)
            macs += enc_attn['macs']
            n_params += enc_attn['params']
            macs_attn += enc_attn['macs_attn']

        # FFN
        fc1_layer = self.fc1.compute_macs_params()
        macs += (fc1_layer['macs'] * T)
        n_params += fc1_layer['params']

        fc2_layer = self.fc2.compute_macs_params()
        macs += (fc2_layer['macs'] * T)
        n_params += fc2_layer['params']

        n_params += sum([p.numel() for p in self.final_layer_norm.parameters()])

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params,
            'macs_attn': macs_attn
        }


if __name__ == '__main__':
    pass
