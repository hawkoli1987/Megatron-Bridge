import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge

class Qwen3MTPBridge(Qwen3Bridge):
    def mapping_registry(self) -> MegatronMappingRegistry:
        base_registry = super().mapping_registry()

        # MTP mappings
        mtp_mappings = [
            # Norms and Projections
            AutoMapping("mtp.layers.*.enorm.weight", "mtp.*.enorm.weight"),
            AutoMapping("mtp.layers.*.hnorm.weight", "mtp.*.hnorm.weight"),
            AutoMapping("mtp.layers.*.eh_proj.weight", "mtp.*.eh_proj.weight"),
            AutoMapping("mtp.layers.*.final_layernorm.weight", "mtp.*.final_layernorm.weight"),

            # Transformer Layer (Megatron-Core uses mtp_model_layer, HF uses transformer_layer)
            AutoMapping("mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight",
                        "mtp.*.transformer_layer.input_layernorm.weight"),
            AutoMapping("mtp.layers.*.mtp_model_layer.mlp.linear_fc1.layer_norm_weight",
                        "mtp.*.transformer_layer.post_attention_layernorm.weight"),

            # Qwen3 Specific Norms
            AutoMapping("mtp.layers.*.mtp_model_layer.self_attention.q_layernorm.weight",
                        "mtp.*.transformer_layer.self_attn.q_norm.weight"),
            AutoMapping("mtp.layers.*.mtp_model_layer.self_attention.k_layernorm.weight",
                        "mtp.*.transformer_layer.self_attn.k_norm.weight"),

            # Projections
            AutoMapping("mtp.layers.*.mtp_model_layer.self_attention.linear_proj.weight",
                        "mtp.*.transformer_layer.self_attn.o_proj.weight"),
            AutoMapping("mtp.layers.*.mtp_model_layer.mlp.linear_fc2.weight",
                        "mtp.*.transformer_layer.mlp.down_proj.weight"),
        ]

        # MTP Mappings
        special_mappings = [
            QKVMapping(
                megatron_param="mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.weight",
                q="mtp.*.transformer_layer.self_attn.q_proj.weight",
                k="mtp.*.transformer_layer.self_attn.k_proj.weight",
                v="mtp.*.transformer_layer.self_attn.v_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="mtp.layers.*.mtp_model_layer.mlp.linear_fc1.weight",
                gate="mtp.*.transformer_layer.mlp.gate_proj.weight",
                up="mtp.*.transformer_layer.mlp.up_proj.weight",
            ),
        ]

        combined_mappings = base_registry.mappings + mtp_mappings + special_mappings
        return MegatronMappingRegistry(*combined_mappings)

    @classmethod
    def register_for_custom_class(cls, custom_model_class):
        """
        Registers bridge for both the class type and the class name string.
        """
        print(f"🔗 Registering Qwen3MTPBridge for class: {custom_model_class.__name__}")

        # Register for the Class Type
        MegatronModelBridge.register_bridge(
            source=custom_model_class,
            target=GPTModel
        )(cls)

        # Register for the String Name
        # Required for AutoBridge validation with auto_map
        MegatronModelBridge.register_bridge(
            source=custom_model_class.__name__,
            target=GPTModel
        )(cls)

  