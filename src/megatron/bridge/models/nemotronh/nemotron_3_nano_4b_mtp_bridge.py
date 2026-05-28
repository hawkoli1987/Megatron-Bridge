"""Thin wrapper to register NemotronHBridge under custom MTP class names.

The actual mapping logic — including MTP flattening + QKV split — already
lives in NemotronHBridge. This shim exists so AutoBridge can find the bridge
when convert_checkpoints.py is run with --trust-remote-code on an exported
NemotronHMTPForCausalLM checkpoint.
"""
from megatron.core.models.mamba.mamba_model import MambaModel

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.nemotronh.nemotron_h_bridge import NemotronHBridge


class NemotronHMTPBridge(NemotronHBridge):
    """NemotronHBridge variant that can be registered for custom HF classes."""

    @classmethod
    def register_for_custom_class(cls, custom_model_class):
        """Register the bridge for both the class type and the class-name string."""
        print(f"Registering NemotronHMTPBridge for class: {custom_model_class.__name__}")
        MegatronModelBridge.register_bridge(
            source=custom_model_class,
            target=MambaModel,
        )(cls)
        MegatronModelBridge.register_bridge(
            source=custom_model_class.__name__,
            target=MambaModel,
        )(cls)
