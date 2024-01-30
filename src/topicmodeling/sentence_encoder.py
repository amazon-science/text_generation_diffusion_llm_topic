import numpy as np
import torch
 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flanT5_cnn_lighting import FlanT5NestCNNAutoencoder, T5AutoConfig
 
from sentence_transformers import SentenceTransformer
 
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer



class Encode_Sentence(FlanT5NestCNNAutoencoder):
     def __init__(self, config, pretrain_model_token, pretrain_model, *args, **kwargs):
         super().__init__(config)
         # model_name_or_path = kwargs.get('model_name_or_path', 'google/flan-t5-base')
         self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_token)
         self.load_state_dict(torch.load(pretrain_model))
         self.max_length = kwargs.get('max_length', 256)
         self.device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         self.to(self.device_name)
    
     def encode(self, sentences, batch_size=32, **kwargs):
        # Tokenize all sentences and create a TensorDataset
        sent_token = self.tokenizer(sentences, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        dataset = TensorDataset(sent_token.input_ids, sent_token.attention_mask)
        
        # Create a DataLoader for efficient batch loading
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        embeddings = []
        for input_ids, attention_mask in loader:
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

            with torch.no_grad():
                # Assuming your model returns embeddings as the last hidden state
                output = self.model.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
                output =  self.encoder(output) 
                #output = output.mean(dim=1)  # Example aggregation, e.g., mean pooling over token embeddings

            embeddings.append(output.cpu())

        # Concatenate all batches to get the final embeddings tensor
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return embeddings
        
    #  def encode(self, sentences, batch_size=1, **kwargs):
    #      """
    #      Returns a list of embeddings for the given sentences.
    #      Args:
    #          sentences (`List[str]`): List of sentences to encode
    #          batch_size (`int`): Batch size for the encoding
 
    #      Returns:
    #          `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
    #      """
 
    #      #device_encode = self.model.device
 
    #      sents_batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]
         
    #      output_ls = []
    #      attention_ls = []
         
    #      for sents_batch in sents_batches:
    #          # model tokenizor
    #          #print(sents_batch)
    #          sent_token = self.tokenizer(sents_batch, truncation=True, padding="max_length", 
    #                                      max_length=self.max_length, return_tensors="pt")
 
    #          # get the inputs and attention
    #          inputs = sent_token.input_ids.to(self.device_name)
    #          attention = sent_token.attention_mask.to(self.device_name)
 
    #          with torch.no_grad():
    #              # get the latent space embedding
    #              output = self.model.encoder(inputs, attention).last_hidden_state
    #              output =  self.encoder(output) 
 
    #          # Convert PyTorch tensor to NumPy array
    #          numpy_array = output.cpu().detach().numpy()
             
    #          output_ls.extend(numpy_array)
    #          attention_ls.extend(attention.cpu().detach().numpy())
         
    #      # Reshape NumPy array (num_sent, num_token, dim_token) to 2D matrix
    #      # numpy_array = np.concatenate(output_ls, axis=0)
    #      # embed = numpy_array.reshape((-1, numpy_array.shape[-1] * numpy_array.shape[-2]))
    #      # print(output_ls[0].shape)
    #      # embed_ls = [numpy_array.reshape((-1, numpy_array.shape[-1] * numpy_array.shape[-2])) for numpy_array in output_ls]
    #      embed_ls = [numpy_array.flatten() for numpy_array in output_ls]
    #      # print(embed_ls[0].shape)
    #      # print(len(embed_ls))
    #      # print(embed_ls[0])
 
    #      return embed_ls