"""
This module includes all the classes and functions for the nested autoencoder.
"""

from transformers import PreTrainedModel
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import datasets
import torch
import torch.nn.functional as F
from torch import nn
import random
import os
from configuration_detime import DeTiMEAutoConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# Define the CNN encoder and decoder model
class CNNEncoder(nn.Module):
    def __init__(self, hidden_size1, hidden_size3):
        super().__init__()
        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size1, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
#             nn.Conv1d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=hidden_size3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # Encode the input
        encoded = self.encoder(x)
        return encoded

class CNNDecoder(nn.Module):
        def __init__(self, hidden_size1, hidden_size3) -> None:
            super().__init__()

            # Define the decoder
            self.decoder = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size3, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
    #             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    #             nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=hidden_size1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            # Decode the encoding
            decoded = self.decoder(x)
            # decoded = decoded.permute(0, 2, 1)
            return decoded
        


class DeTiME(PreTrainedModel):
    config_class = DeTiMEAutoConfig

    def __init__(self, config):
        super().__init__(config)
        #change t5-small to config 
        model_name_or_path = config.model
        # peft_config = PrefixTuningConfig(peft_type="PREFIX_TUNING", task_type=TaskType.SEQ_2_SEQ_LM, 
        #                                  inference_mode=False, num_virtual_tokens=10)
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        # model = get_peft_model(model, peft_config)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

        #model.print_trainable_parameters()
        self.model = model
        self.config_model = 'CNN'
        if self.config_model == 'CNN':
            # self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
            self.encoder = CNNEncoder(
                                config.hidden_size1, config.hidden_size3)
            self.decoder = CNNDecoder(
                                config.hidden_size1, config.hidden_size3)
            self.encoder.main_input_name = self.model.main_input_name


            self.encoder.main_input_name = self.model.main_input_name
        self.main_input_name = self.model.main_input_name

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        output = self.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   #batch size * seq length * embedding size, 
        #print(output.shape)
        if self.config_model == 'CNN':
            encoder_output = self.encoder(output) #batch size * seq length * embedding size, 1 * batch size * hidden_size
            #print(encoder_output.shape)
            
            output = self.decoder(encoder_output) #1 batch_size, hidden_size

        return self.model.forward(input_ids=input_ids.contiguous(), encoder_outputs=(output.contiguous(), ), labels=labels.contiguous(),  **kwargs)

    def generate(self, input_ids, attention_mask, **kwargs):
        output = self.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   #batch size * seq length * embedding size, 
        #print(output.shape)
        # encoder_output = self.encoder(output) #batch size * seq length * embedding size, 1 * batch size * hidden_size
        # #print(encoder_output.shape)
        if self.config_model == 'CNN':
            encoder_output = self.encoder(output) #batch size * seq length * embedding size, 1 * batch size * hidden_size
            #print(encoder_output.shape)
            
            output = self.decoder(encoder_output) #1 batch_size, hidden_size
        elif self.config_model == 'RNN':
            output = self.encoder(output) #batch size * seq length * embedding size, 1 * batch size * hidden_size

        # output = self.decoder(encoder_output) #1 batch_size, hidden_size

        return self.model.generate(input_ids=input_ids.contiguous(), encoder_outputs=(output.contiguous(), ),  **kwargs)

