from transformers import AutoConfig

try:
    from transformers import Qwen3Config
except ImportError:
    from transformers import Qwen2Config as Qwen3Config

class Qwen3MTPConfig(Qwen3Config):
    """Config for Qwen3 MTP draft model."""
    model_type = "mtp"

    def __init__(self, mtp_num_layers: int = 1, n_predict: int = 1, **kwargs):
        self.mtp_num_layers = mtp_num_layers
        self.n_predict = n_predict
        super().__init__(**kwargs)

def register_mtp_config():
    """Register MTP config with transformers AutoConfig."""
    try:
        AutoConfig.register("qwen3_mtp", Qwen3MTPConfig)
    except ValueError:
        pass

register_mtp_config()