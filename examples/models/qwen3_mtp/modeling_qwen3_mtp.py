import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2Model

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3Model, Qwen3ForCausalLM
except ImportError:
    Qwen3Config = Qwen2Config
    Qwen3DecoderLayer = Qwen2DecoderLayer
    Qwen3RMSNorm = Qwen2RMSNorm
    Qwen3Model = Qwen2Model
    Qwen3ForCausalLM = Qwen2ForCausalLM

class Qwen3MTPConfig(Qwen3Config):
    model_type = "qwen3_mtp"
    def __init__(self, mtp_num_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.mtp_num_layers = mtp_num_layers

class Qwen3MTPLayer(nn.Module):
    """
    Corresponds to MultiTokenPredictionLayer in Megatron.
    """
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Normalization layers
        self.enorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Projection
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        # Transformer Block
        self.transformer_layer = Qwen3DecoderLayer(config, layer_idx)

        # Final Layer Norm
        self.final_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        decoder_input,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        **kwargs
    ):
        # Norms
        norm_h = self.hnorm(hidden_states)
        norm_e = self.enorm(decoder_input)

        # Concat & Project
        # Input shape: [batch, seq, hidden]
        combined = torch.cat([norm_e, norm_h], dim=-1)
        projected = self.eh_proj(combined)

        # Transformer Layer
        layer_outputs = self.transformer_layer(
            projected,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = layer_outputs[0]

        # Final Norm
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states

class Qwen3MTPForCausalLM(Qwen3ForCausalLM):
    config_class = Qwen3MTPConfig

    def __init__(self, config):
        super().__init__(config)

        # Create MTP Module List
        num_mtp_layers = getattr(config, "mtp_num_layers", 1)

        self.mtp = nn.ModuleList([
            Qwen3MTPLayer(config, layer_idx=i)
            for i in range(num_mtp_layers)
        ])

        # Initialize weights
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, **kwargs)

        return outputs