
from transformers import T5Config, PretrainedConfig
from typing import List


# define Flan-T5 nest CNN autoencoder here
class DeTiMEAutoConfig(T5Config):
    model_type = "detime"

    def __init__(
        self,
        hidden_size1: int = 512,
        hidden_size3: int = 512,
        num_layer: int = 1,
        dropout: float = 0.1,
        max_length: int = 512,
        model_name: str = None,
        **kwargs,
    ):
        self.hidden_size1 = hidden_size1
        self.hidden_size3 = hidden_size3
        self.num_layer = num_layer
        self.dropout = dropout
        self.max_length = max_length
        self.model_name = model_name
        super().__init__(**kwargs)