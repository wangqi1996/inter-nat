from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayer,
    MultiheadAttention,
    TransformerEncoderLayer,
    LayerNorm
)
from torch import Tensor

from .modules import FeedForward, RelativeSelfAttention, RelativePositionEmbeddings


def build_relative_embeddings(args, embedding_dim=None):
    if embedding_dim is None:
        embedding_dim = args.decoder_embed_dim // getattr(args, "decoder_attention_heads")
    return RelativePositionEmbeddings(
        max_rel_positions=getattr(args, "max_rel_positions", 4),
        embedding_dim=embedding_dim,
        direction=True,
        dropout=args.dropout
    )


class BlockedEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, relative_keys=None, relative_vals=None):
        super().__init__(args)
        self.ffn_block = FeedForward(
            d_model=self.embed_dim,
            d_hidden=args.decoder_ffn_embed_dim,
            dropout=args.dropout
        ) if args.enc_block_cls == "highway" else None

        self.relative_keys = relative_keys
        self.relative_vals = relative_vals

        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))
        self.final_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))

    def build_self_attention(self, embed_dim, args):
        if getattr(args, "enc_self_attn_cls", "abs") == "abs":
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        else:
            return RelativeSelfAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.relative_keys is None:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            index = utils.new_arange(x, x.size(0))
            pos_key_embed = self.relative_keys(index)
            pos_val_embed = self.relative_vals(index)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                pos_key=pos_key_embed,
                pos_val=pos_val_embed,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )

        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)

        if self.ffn_block is None:
            x = residual + x
        else:
            g = self.ffn_block(residual).sigmoid()
            x = residual * g + x * (1 - g)

        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class BlockedDecoderLayer(TransformerDecoderLayer):
    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False,
            relative_keys=None, relative_vals=None, **kwargs
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

        self.ffn_block = FeedForward(
            d_model=self.embed_dim,
            d_hidden=args.decoder_ffn_embed_dim,
            dropout=args.dropout
        ) if args.block_cls == "highway" else None

        self.relative_keys = relative_keys
        self.relative_vals = relative_vals

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            **kwargs
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))
        self.final_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))
        if self.encoder_attn_layer_norm is not None:
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False, **kwargs):
        if getattr(args, "self_attn_cls", "abs") == "abs":
            return MultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        else:
            return RelativeSelfAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

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
            **kwargs,
    ):

        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
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
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        if self.relative_keys is None:
            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
                **kwargs
            )
        else:
            index = utils.new_arange(x, x.size(0))
            pos_key_embed = self.relative_keys(index)
            pos_val_embed = self.relative_vals(index)
            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                pos_key=pos_key_embed,
                pos_val=pos_val_embed,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
                **kwargs
            )

        x = self.dropout_module(x)
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
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)

        if self.ffn_block is None:
            x = residual + x
        else:
            g = self.ffn_block(residual).sigmoid()
            x = residual * g + x * (1 - g)

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
