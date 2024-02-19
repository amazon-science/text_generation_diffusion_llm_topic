"""
This module is used to train a nested FLAN-T5 and autoencoder structure for sentence reconstruction.

specifically, the module includes:
1. the precoess functions are updated for sentence construction only
2. training the PEFT of FLAN-T5, keep FLAN-T5 fixed
3. training the autoencoder
4. use the CNN data and slide windows
"""


from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
import logging
import sys
from transformers.models.t5.modeling_t5 import T5Block
import argparse
import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq, PreTrainedModel, T5Config, PretrainedConfig
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
import numpy as np
import datasets
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
import gc
import torch
import torch.nn.functional as F
from torch import nn
import nltk
import pathlib
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import boto3

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType

from tqdm import tqdm

from topicmodeling.CNN_Encoder import CNNEncoder, CNNDecoder, RNNEncoder

from torch.utils.data import DataLoader
import lightning as L
from transformers import default_data_collator, get_linear_schedule_with_warmup

metric = datasets.load_metric('sacrebleu')
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger

# Pick a logger and add it to Fabric
logger = TensorBoardLogger(root_dir="logs")
fabric = Fabric(loggers=logger)

# def preprocess_function(examples,padding="max_length"):
    
#     inputs = examples["input"] 
#     targets = examples["output"] 

#     inputs_encodings = tokenizer(inputs, truncation=True, padding="max_length", max_length=256)
#     targets_encodings = tokenizer(targets, truncation=True, padding="max_length", max_length=256)
#     labels = targets_encodings
#     if padding == "max_length":
#         labels = [
#             [(l if l != tokenizer.pad_token_id else 0) for l in label] for label in labels
#         ]

#     inputs_encodings["labels"] =  labels 
#     return inputs_encodings

# def preprocess_function(examples,padding="max_length"):
#     # print(examples)
#     inputs = examples["input"] 
#     targets = examples["output"]  

#     inputs_encodings = tokenizer(inputs, truncation=True, padding="max_length", max_length=256)
#     targets_encodings = tokenizer(targets, truncation=True, padding="max_length", max_length=256)
#     labels = targets_encodings
#     if padding == "max_length":
#         labels = [
#             [(l if l != tokenizer.pad_token_id else 0) for l in label] for label in labels
#         ]

#     inputs_encodings["labels"] =  labels 
#     return inputs_encodings

def preprocess_function(examples,padding="max_length"):
    
    inputs = [examples["input"]] 
    targets = [examples["output"] ]

    inputs_encodings = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    targets_encodings = tokenizer(targets, truncation=True, padding="max_length", max_length=512)
    labels = targets_encodings
    inputs_encodings["labels"] =  targets_encodings['input_ids']
    return inputs_encodings

# def preprocess_sliding_window_tokenize_cnn(examples, tokenizer, window_size, stride, 
#                                            padding="max_length", prompt=''):
#     """
#     window_size should be the same or smaller than the model input max_length
#     """

#     inputs = examples["article"]
#     # targets = examples["highlights"]

#     # get the labels first
#     # targets_encodings = tokenizer(targets, truncation=True, padding="max_length", max_length=window_size)
#     # labels = targets_encodings['input_ids']

#     # input tokens length
#     input_tokens = tokenizer.encode(inputs, add_special_tokens=False)
#     len_input_tokens = len(input_tokens)
#     # prompt tokens length
#     prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
#     len_prompt_tokens = len(prompt_tokens)

#     # adjust the windown size based on the token length of input and prompt
#     window_size -= len_prompt_tokens

#     # split the input text into overlapping windows with a fixed length
#     windows = [input_tokens[i:i+window_size] for i in range(0, len_input_tokens, stride)]

#     # add windown_size back
#     window_size += len_prompt_tokens

#     # generate outputs for each window
#     window_inputs_encodings = []
#     for windown_tokens in windows:
#         window_text = tokenizer.decode(windown_tokens, skip_special_tokens=True)
#         # add prompt
#         window_text = prompt + window_text
#         window_inputs_encoding = tokenizer(window_text, truncation=True, padding="max_length", max_length=window_size)
#         # window_inputs_encoding["labels"] = labels
#         window_inputs_encodings.append(window_inputs_encoding)

#     return {'window_inputs_encodings': window_inputs_encodings}


# def flatten_list_of_dict(batch):
#     return {
#         "input_ids": [dic['input_ids'] for ex_list_of_dict in batch['window_inputs_encodings'] for dic in ex_list_of_dict],
#         "attention_mask": [dic['attention_mask'] for ex_list_of_dict in batch['window_inputs_encodings'] for dic in ex_list_of_dict],
#         # "labels": [dic['input_ids'] for ex_list_of_dict in batch['window_inputs_encodings'] for dic in ex_list_of_dict]
#            }


def create_pretrain_dataset(dataset_type):
    """
    this fucntion is used to create a dataset for pretraining the model.
        1. dataset with similar sentence as input and output
        2. dataset with similar sentence and words as input and output
        3. dataset with same sentence as input and output
        4. dataset with same sentence and words as input and output
    @return:
    """
    ds = None
    
    if dataset_type == 1:
        # get the sentence datasets
        ds = load_dataset('xwjzds/pretrain_sts1')['train']
    elif dataset_type == 5:
        # get the sentence datasets
        ds = load_dataset('xwjzds/pretrain_sts_extend')['train']
    elif dataset_type == 6:
        # get the sentence datasets
        ds = load_dataset('xwjzds/pretrain_sts_similarity')['train']
    elif dataset_type == 7:
        # get the sentence datasets
        ds = load_dataset('xwjzds/pretrain_punctuation')['train']
    elif dataset_type == 8:
        # get the sentence datasets
        ds = load_dataset('xwjzds/pretrain_repeat_paraphrase')['train']
    elif dataset_type == 2:
        ds1 = load_dataset('whu9/sts_pretrain')['train']
        ds2 = load_dataset('whu9/phrase_similarity_pos')['train']
        ds3 = load_dataset('whu9/word_net_synset_lemma')['train']
        # concatenate
        ds = concatenate_datasets([ds1, ds2, ds3])
    elif dataset_type == 3:
        ds1 = load_dataset('whu9/sts_pretrain')['train']
        # get the same sentences from ds1
        ds = Dataset.from_dict(
            {'entity1': ds1['entity1'],
            'entity2': ds1['entity1']}
        )
    elif dataset_type == 4:
        ds1 = load_dataset('whu9/sts_pretrain')['train']
        ds2 = load_dataset('whu9/phrase_similarity_pos')['train']
        ds3 = load_dataset('whu9/word_net_synset_lemma')['train']
        # get the same sentences from ds1
        ds1_new = Dataset.from_dict(
            {'entity1': ds1['entity1'],
            'entity2': ds1['entity1']}
        )
        ds2_new = Dataset.from_dict(
            {'entity1': ds2['entity1'],
            'entity2': ds2['entity1']}
        )
        ds3_new = Dataset.from_dict(
            {'entity1': ds3['entity1'],
            'entity2': ds3['entity1']}
        )
        # concatenate
        ds = concatenate_datasets([ds1_new, ds2_new, ds3_new])
    else: 
        raise ValueError('no dataaset generated, out of scope')
        
    return ds


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# define Flan-T5 nest CNN autoencoder here
class T5AutoConfig(T5Config):

    def __init__(
        self,
        hidden_size1: int = 512,
        hidden_size3: int = 512,
        # output_size: int = 512 * 16,
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


class FlanT5NestCNNAutoencoder(PreTrainedModel):
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
        elif self.config_model == 'RNN':
            # self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
            self.encoder = RNNEncoder(
                               config)

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
        elif self.config_model == 'RNN':
            output = self.encoder(output) #batch size * seq length * embedding size, 1 * batch size * hidden_size
        #print(labels.shape, output.shape, input_ids.shape)
        #print(self.model.forward(input_ids=input_ids, labels=labels, **kwargs).shape)
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



def peak_memory(device='cuda'):
    """Track peak memory usage on the specified device"""
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6  # convert bytes to MB
    torch.cuda.reset_max_memory_allocated(device)  # reset peak memory
    return peak_mem

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--train_steps", type=int, default=-1)

    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--model_name", type=str, default='google/flan-t5-large')
    parser.add_argument("--hidden_size1", type=int, default=512)
    parser.add_argument("--hidden_size3", type=int, default=4)
    parser.add_argument("--hidden_size2", type=int, default=768)
    parser.add_argument("--output_size", type=int, default=4 * 768)
    parser.add_argument("--dataset_type", type=int, default=1)
    parser.add_argument("--test", type=int, default=0)
    # Data, model, and output directories
    # parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    # parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    # parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    # parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    args, _ = parser.parse_known_args()
    print(args.model_name + str(args.dataset_type))
    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load dataset
    dataset_load = create_pretrain_dataset(args.dataset_type)
    tokenized_data = dataset_load.train_test_split(test_size=0.01)
    if args.test == 1:
        tokenized_data = tokenized_data['test'].train_test_split(test_size=0.001)


    # load the tokenizor
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # # preprocess using the slide preprocess functions
    # cnn_dataset_aft = cnn_dataset.map(lambda examples: 
    #                                 preprocess_sliding_window_tokenize_cnn(examples, tokenizer,
    #                                                                         window_size=512, 
    #                                                                         stride=256,  
    #                                                                         prompt=''), 
    #                                 batched=False)
    # tokenized_data = cnn_dataset_aft.map(flatten_list_of_dict, batched=True, 
                                        # remove_columns=['window_inputs_encodings', 'article', 'highlights', 'id'])
    
    # get the model
    # creating model
    #using FSDP
    # startegy = FSDPStrategy(cpu_offload = True)
    #fabric = L.Fabric(accelerator="cuda", devices=8, strategy="DDP", precision = "16")
    #auto_wrap_policy = partial(transformer_auto_wrap_policy)
    # t5_auto_wrap_policy = partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls={
    #         T5Block,
    #     },
    # )
    # #sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
    # strategy = FSDPStrategy(
    #     auto_wrap_policy=t5_auto_wrap_policy,
    #     use_orig_params=True,
    #     #mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
    #     #activation_checkpointing=EncoderBlock,
    #     cpu_offload=True
    # )
    # equivalent to passing `"fsdp_cpu_offload"`
    fabric = L.Fabric(accelerator="auto", strategy="auto", devices="auto")
    fabric.launch()
    t5autoConfig = T5AutoConfig(
        hidden_size1=args.hidden_size1, hidden_size3=args.hidden_size3, 
        hidden_size2=args.hidden_size2, output_size=args.output_size, 
        model = args.model_name)
    #with fabric.init_module():
    with fabric.init_module():
        model = FlanT5NestCNNAutoencoder(t5autoConfig)
    
    # # only train decoder summarization models
    # # freeze the parameters of encoder
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    # for param in model.decoder.parameters():
    #     param.requires_grad = False

    # # check the freeze parameters of model
    # model_train_param_grads_ls = []
    # for param in model.encoder.parameters():
    #     model_train_param_grads_ls.append(param.requires_grad)
    # for param in model.decoder.parameters():
    #     model_train_param_grads_ls.append(param.requires_grad)
    # assert all(model_train_param_grads_ls) == False

    #print(model.print_trainable_parameters())

    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # training parameters
    #device = "cuda"
    
    checkpoint_name = "pretrain.pt"
    # max_length = 128
    lr = args.learning_rate
    num_epochs = args.epochs
    if args.test == 1:
        num_epochs = 10
    batch_size = args.train_batch_size
    train_dataset = tokenized_data["train"].map(preprocess_function)
    eval_dataset = tokenized_data["test"].map(preprocess_function)
    train_dataset = train_dataset.remove_columns(['input', 'output'])
    eval_dataset = eval_dataset.remove_columns(['input', 'output'])
    # create dataloader
    # train_dataset = tokenized_data["train"]
    # eval_dataset = tokenized_data["test"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, pin_memory=False
    )
    eval_dataloader = DataLoader(eval_dataset,  batch_size=batch_size, pin_memory=False)
    #print(train_dataloader)
    
    # set the number of the train steps
    if args.train_steps == -1:
        args.train_steps = len(train_dataloader) * num_epochs

    #optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        #Should we have a parameter to be number of training steps
        num_training_steps=args.train_steps,
    )

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # num_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=num_steps)

    # training and evaluation
    # do data parallel training
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    #model = model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # num_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=num_steps)
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    eval_dataloader = fabric.setup_dataloaders(eval_dataloader)
    results = []
    memory_tracker = []
    for epoch in range(num_epochs):
        gc.collect()
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            #print([(k, v) for k, v in batch.items()])
            # Accumulate gradient 8 batches at a time
            is_accumulating = step % 16 != 0
            batch = {k: torch.stack(v[0]).T for k, v in batch.items()}
            #skip the synchronization in .backward() during the accumulation phase
            #with fabric.no_backward_sync(model, enabled=is_accumulating):
            with fabric.autocast():
                outputs = model(**batch)
                loss = outputs.loss.mean()
            
            total_loss += loss.detach()
                #loss.backward()
                #print(f'Peak memory usage: {peak_memory()} MB')
                #memory_tracker.append(peak_memory())
            if step % 1 == 0:
                values = {"loss": loss}
                fabric.log_dict(values)


            #if not torch.isnan(loss):
            fabric.backward(loss)
            if not is_accumulating and not torch.isnan(loss):
                fabric.clip_gradients(model, optimizer, clip_val=0.25)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                memory_tracker.append(peak_memory())

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            #print([(k, v) for k, v in batch.items()])
            batch = {k: torch.stack(v[0]).T for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss.mean()
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits.mean(dim=0), -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(train_epoch_loss, eval_epoch_loss)
        results.append([train_epoch_loss, eval_epoch_loss ])
        print('flant5_nest_peft_PREFIX_TUNING_SEQ_2_SEQ_LM' + args.model_name  + str(args.dataset_type))
        #print(train_epoch_loss, train_ppl)
        peft_model_id = 'model/flant5_nest_peft_PREFIX_TUNING_SEQ_2_SEQ_LM' + args.model_name.split('/')[1]  + str(args.dataset_type)
        # saving model
        #print('flant5_nest_peft' + args.model_name + str(args.dataset_type))
        model = model.module if hasattr(model, 'module') else model
        model.save_pretrained(peft_model_id, from_pt=True) 
        torch.save(model.state_dict(), 'model/flant5_nest_peft' + args.model_name.split('/')[1] + str(args.dataset_type))
        # model.model.save_pretrained(peft_model_id)
        # model.save_pretrained(peft_model_id)
        tokenizer.save_pretrained(peft_model_id)
        inputs = tokenizer("He is hot", return_tensors="pt",  padding='max_length',  max_length = 512).input_ids.cuda()
        am = tokenizer("He is hot", return_tensors="pt",  padding='max_length',  max_length = 512).attention_mask.cuda()
        outputs = model.cuda().generate(inputs, am)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))        
        pd.DataFrame(results).to_csv('model/training_details_'+ args.model_name.split('/')[1] + str(args.dataset_type) + '.csv')
        #print(max(memory_tracker))
    