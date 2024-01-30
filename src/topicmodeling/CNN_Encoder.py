"""
This module includes all the classes and functions for the nested autoencoder.
"""

from transformers import PreTrainedModel
from transformers import T5ForConditionalGeneration
import datasets
import torch
import torch.nn.functional as F
from torch import nn
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

metric = datasets.load_metric('sacrebleu')

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


# class Decoder(torch.nn.Module):
#     def __init__(self, hidden_size1, hidden_size2, output_size, num_layers=1, dropout=0.1):
#         super().__init__()
#         self.fc = torch.nn.Linear(output_size, hidden_size1 * hidden_size2)
#         self.dropout = torch.nn.Dropout(dropout)
#         # self.batch_size = batch_size
#         self.hidden_size1 = hidden_size1
#         self.hidden_size2 = hidden_size2

#     def forward(self, x):
#         x = self.dropout(x)
#         x = self.fc(x)
#         x = torch.reshape(x, (-1, self.hidden_size1, self.hidden_size2))

#         return x


# class Encoder(torch.nn.Module):
#     def __init__(self, hidden_size1, hidden_size2, output_size, num_layers=1, dropout=0.1):
#         super().__init__()
#         self.fc = torch.nn.Linear(hidden_size1 * hidden_size2, output_size)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.hidden_size1 = hidden_size1
#         self.hidden_size2 = hidden_size2
#         # self.batch_size = batch_size

#     def forward(self, x):
#         # to do: Add softmax

#         # pad last dim by 1 on each side
#         p1d = (0, 0,  0, self.hidden_size1 - x.shape[1])
#         x = F.pad(x, p1d, "constant", 0)
#         x = torch.reshape(x, (-1, self.hidden_size1 * self.hidden_size2))
#         x = self.dropout(x)
#         x = self.fc(x)
#         # Add softmax to represent topic distributioon
#         x = F.softmax(x, dim=1)
#         return x


# class lstm_decoder(nn.Module):
#     ''' Decodes hidden state output by encoder '''
    
#     def __init__(self, hidden_size2, output_size, num_layers = 1):

#         '''
#         : param input_size:     the number of features in the input X
#         : param hidden_size:    the number of features in the hidden state h
#         : param num_layers:     number of recurrent layers (i.e., 2 means there are
#         :                       2 stacked LSTMs)
#         '''
        
#         super(lstm_decoder, self).__init__()
#         self.hidden_size2 = hidden_size2
#         self.output_size = output_size
#         self.num_layers = num_layers

#         self.lstm = nn.GRU(input_size = hidden_size2, hidden_size = output_size,
#                             num_layers = num_layers, batch_first = True)
#         self.linear = nn.Linear(output_size, hidden_size2)  
#         self.init_input = nn.Parameter(torch.randn(1, hidden_size2)) # Learned constant vector
         

#     def forward(self, x_input, encoder_hidden_states):
        
#         '''        
#         : param x_input:                    should be 2D (batch_size, input_size)
#         : param encoder_hidden_states:      hidden states
#         : return output, hidden:            output gives all the hidden states in the sequence;
#         :                                   hidden gives the hidden state and cell state for the last
#         :                                   element in the sequence 
 
#         '''
#         #print(x_input.shape, encoder_hidden_states.shape)
#         lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
#         output = self.linear(lstm_out)     
        
#         return output, self.hidden

#     def init_hidden(self, encoder_hidden):
#         # Set the initial hidden state of the decoder to be the last hidden state of the encoder
#         return encoder_hidden
    
#     def init_input_vector(self, batch_size):
#         # Create a batch of learned constant vectors for the initial input
#         return self.init_input.expand(batch_size, 1, -1) #Batch_size, sequence length, embeddings size


# class lstm_encoder(nn.Module):
#     ''' Encodes time-series sequence '''

#     def __init__(self, hidden_size2, output_size, num_layers = 1):
        
#         '''
#         : param input_size:     the number of features in the input X
#         : param input_size:     the number of features in the input X
#         : param hidden_size:    the number of features in the hidden state h
#         : param num_layers:     number of recurrent layers (i.e., 2 means there are
#         :                       2 stacked LSTMs)
#         '''
        
#         super(lstm_encoder, self).__init__()
        
#         self.hidden_size2 = hidden_size2
#         self.output_size = output_size
#         self.num_layers = num_layers

#         # define GRU layer
#         self.lstm = nn.GRU(input_size = hidden_size2, hidden_size = output_size,
#                             num_layers = num_layers, batch_first = True)
#         self.linear = nn.Linear(output_size, hidden_size2)  
        
#     def forward(self, x):
        
#         '''
#         : param x_input:               input of shape (# in batch, seq_len,  input_size)
#         : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
#         :                              hidden gives the hidden state and cell state for the last
#         :                              element in the sequence 
#         '''
        
#         lstm_out, self.hidden = self.lstm(x)
#         lstm_out = self.linear(lstm_out)     
#         return lstm_out, self.hidden   


class RNNEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.encoder = Encoder(config['hidden_size1'],
        #                        config['hidden_size2'], config['output_size'])
        # self.decoder = Decoder(config['hidden_size1'],
        #                        config['hidden_size2'], config['output_size'])
        self.encoder = lstm_encoder(
                               config.hidden_size2, config.output_size)
        self.decoder = lstm_decoder(
                               config.hidden_size2, config.output_size)
        # self.config = T5Config(

        #     vocab_size=self.model.config.vocab_size,
        #     d_model=self.model.config.d_model,
        #     d_ff=self.model.config.d_ff,
        #     num_heads=self.model.config.num_heads
        # )

    def forward(self, output, **kwargs):
        # Get the output of the T5 encoder
        # with torch.no_grad():
        encoder_output, encoder_hidden = self.encoder(output) #batch size * seq length * embedding size, 1 * batch size * hidden_size
        batch_size = output.size(0)
        decoder_input = self.decoder.init_input_vector(batch_size) #batch_size, 1, embedding size
        
        decoder_hidden = self.decoder.init_hidden(encoder_hidden) #1 batch_size, hidden_size
        target_length = encoder_output.size(1)
        outputs = []
        use_teacher_forcing = True if random.random() < 0 else False
            
        if use_teacher_forcing:
                # Use teacher forcing: feed the target sequence to the decoder one token at a time
                for i in range(target_length):
                    #print(decoder_input.shape)
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs.append(decoder_output.squeeze(1))
                    decoder_input = output[:, i:i+1]
                    #print(decoder_input.shape)

        else:
                # Use the previous output as the input to the decoder
                for i in range(target_length):
                    #print(decoder_input.shape, decoder_hidden.shape)
                    #batch size * seq length * embedding size, 1 * batch size * hidden_size
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs.append(decoder_output.squeeze(1))
                    decoder_input = decoder_output
                    
        output = torch.stack(outputs, dim=1)
        return output






# class T5NestAutoencoder(PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
#         self.encoder = Encoder(config.hidden_size1,
#                                config.hidden_size2, config.output_size)
#         self.decoder = Decoder(config.hidden_size1,
#                                config.hidden_size2, config.output_size)
#         self.encoder.main_input_name = self.model.main_input_name
#         self.main_input_name = self.model.main_input_name

#         # self.config = T5Config(

#         #     vocab_size=self.model.config.vocab_size,
#         #     d_model=self.model.config.d_model,
#         #     d_ff=self.model.config.d_ff,
#         #     num_heads=self.model.config.num_heads
#         # )

#     def forward(self, input_ids, attention_mask, **kwargs):
#         # Get the output of the T5 encoder
#         # with torch.no_grad():
#         output = self.model.encoder(
#             input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

#         output = self.encoder(output)
#         output = self.decoder(output)

#         return self.model.forward(input_ids=input_ids, encoder_outputs=(output, ), labels=input_ids, **kwargs)

#     # def generate(self, input_ids, attention_mask, max_length=20, num_return_sequences=1.0,
#     #              length_penalty=1.0, do_sample=True, early_stopping=True, num_beams = 1, num_beam_groups = 1,
#     #              synced_gpus=False):
#     #     output = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

#     #     output = self.encoder(output)
#     #     output = self.decoder(output)

#     #     return self.model.generate(input_ids=input_ids, encoder_outputs = (output, ), max_length = max_length,
#     #                                num_return_sequences = num_return_sequences, length_penalty=length_penalty,
#     #                                do_sample=do_sample, early_stopping=early_stopping, num_beams = num_beams,
#     #                                num_beam_groups = num_beam_groups, synced_gpus=synced_gpus)
#     def generate(self, input_ids, attention_mask, **kwargs):
#         output = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

#         output = self.encoder(output)
#         output = self.decoder(output)

#         return self.model.generate(input_ids=input_ids, encoder_outputs=(output,), **kwargs)


# class T5NestAutoencoderTrainT5Decoder(PreTrainedModel):
#     def __init__(self, model_train, config):

#         super().__init__(config)
#         # load the our trained model
#         self.model_train = model_train
#         self.main_input_name = self.model_train.main_input_name
#         # load another t5 model for summarization decoder
#         self.model_sum = T5ForConditionalGeneration.from_pretrained("t5-small")

#         # get the trained t5 encoder and the following auto-encoder
#         self.T5_encoder = self.model_train.model.encoder
#         self.encoder = self.model_train.encoder
#         self.decoder = self.model_train.decoder

#     def forward(self, input_ids, attention_mask, labels, **kwargs):
#         # Get the output of the T5 encoder
#         # with torch.no_grad():
#         output = self.T5_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

#         # the following auto-encoder
#         output = self.encoder(output)
#         output = self.decoder(output)

#         # no need to calculate the gradient above this point
#         output = output.detach()

#         # add the decoder for summarization
#         # note that in current preprocess_function, we just assign the input_ids of target to the
#         # labels of the dataset
#         output = self.model_sum.forward(
#             input_ids=input_ids, encoder_outputs=(output,), labels=labels, **kwargs)

#         return output

#     def generate(self, input_ids, attention_mask, **kwargs):
#         # t5 encoder
#         output = self.T5_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

#         # the following auto-encoder
#         output = self.encoder(output)
#         output = self.decoder(output)

#         # use another decoder to generate
#         output = self.model_sum.generate(input_ids=input_ids, encoder_outputs=(output,), **kwargs)

#         return output