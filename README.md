

# DeTiME: Diffusion-Enhanced Topic Modeling using Encoder-decoder based LLM (Accepted by EMNLP 2023 as Findings)

This repository is the official implementation of [DeTiME: Diffusion-Enhanced Topic Modeling using Encoder-decoder based LLM](https://aclanthology.org/2023.findings-emnlp.606.pdf). 

vONTSS is a topic modeling method.


## Installation



To install requirements:

```setup
pip install -r requirements.txt
```


## Training and Evaluation

To train and evaluate the model, run this command:



Step 1: If the data is in the huggingface. specify --data_source as the repository of hugging face
If the data is a csv file specify where the data is and specify --data_source csv
Step 2: Define number of topics. if the number is 10 use --numb_embeddings 10
Step 3: Define the metric you want to evaluate, currently it supports diversity, c_v, c_uci, etc

Then you just have to run 
```train
python3 main.py --data_source xwjzds/ag_news --metric diversity --topk 20
```
It will output the diversity metric using data in xwjzds/ag_news 


## Interactive Code

Example of using dataset from OCTIS



```python
from octis.dataset.dataset import Dataset
import sys
sys.path.insert(0, '../src/topicmodeling')
from model import TopicModel
from datasets import load_dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence


dataset = Dataset()
dataset.fetch_dataset("20NewsGroup") #It can support 20NewsGroup, BBC_News, DBLP, DBPedia_IT
tm = TopicModel(numb_embeddings = 10)
texts = [' '.join(i) for i in dataset.get_corpus()]
model_output = tm.train_model(texts)
metric = TopicDiversity(topk=10)
topic_diversity_score = metric.score(model_output) # Compute score of diversity
cmetric = Coherence(texts =  tm.tp.lemmas,  measure='c_npmi')
coherence = cmetric.score(model_output) # Compute score of coherence
```

Example of using datasets from huggingface
```python
import sys
sys.path.insert(0, '../src/topicmodeling')
from model import TopicModel
from datasets import load_dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence


df = load_dataset('xwjzds/ag_news')
tm = TopicModel(numb_embeddings = 10)

model_output = tm.train_model(df['train']['text'])
metric = TopicDiversity(topk=10)
topic_diversity_score = metric.score(model_output) # Compute score of diversity
cmetric = Coherence(texts =  tm.tp.lemmas,  measure='c_npmi')
coherence = cmetric.score(model_output) # Compute score of coherence

```

## Arugument Explain

Arguments Explained:

--numb_embeddings: Number of embeddings (default is 10).

--epochs: Number of epochs for training (default is 20).

--batch_size: Batch size for training (default is 256).

--gpu_num: GPU number to use (default is 1).

--learning_rate: Learning rate (default is 0.002).

--weight_decay: Weight decay (default is 1.2e-6).

--penalty: Penalty term (default is 1).

--beta: Beta value (default is 1).

--temp: Temperature (default is 10).

--data_source: Data source type (default is 'huggingface'). Can be 'huggingface', 'csv', or 'txt'.

--data_path: Path to the data file for 'csv' or 'txt' (default is '').

--metrics: List of metrics to report (default is ['diversity', 'c_v', 'c_npmi', 'c_uci', 'u_mass']).

--topk: Top k words to report for diversity (default is 10).



## Results

Our model achieves the following performance on Ag News: 



| Model name         | Diversity       | C_v            | C_npmi         |
| ------------------ |---------------- | -------------- | -------------- |
| vONT               |     0.865       |      0.618     | 0.115          |
| DeTiME             |     0.93        |      0.645     | 0.113          |


we use existed embeddings in this code relase instead of using spherical embeddings. Training a spherical embeddings takes time. We noticed that this reported performance is better than the performance on our paper. 


## Citation
```
@inproceedings{xu-etal-2023-vontss,
    title = "v{ONTSS}: v{MF} based semi-supervised neural topic modeling with optimal transport",
    author = "Xu, Weijie  and
      Jiang, Xiaoyu  and
      Sengamedu Hanumantha Rao, Srinivasan  and
      Iannacci, Francis  and
      Zhao, Jinjin",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.271",
    doi = "10.18653/v1/2023.findings-acl.271",
    pages = "4433--4457",
    abstract = "Recently, Neural Topic Models (NTM), inspired by variational autoencoders, have attracted a lot of research interest; however, these methods have limited applications in the real world due to the challenge of incorporating human knowledge. This work presents a semi-supervised neural topic modeling method, vONTSS, which uses von Mises-Fisher (vMF) based variational autoencoders and optimal transport. When a few keywords per topic are provided, vONTSS in the semi-supervised setting generates potential topics and optimizes topic-keyword quality and topic classification. Experiments show that vONTSS outperforms existing semi-supervised topic modeling methods in classification accuracy and diversity. vONTSS also supports unsupervised topic modeling. Quantitative and qualitative experiments show that vONTSS in the unsupervised setting outperforms recent NTMs on multiple aspects: vONTSS discovers highly clustered and coherent topics on benchmark datasets. It is also much faster than the state-of-the-art weakly supervised text classification method while achieving similar classification performance. We further prove the equivalence of optimal transport loss and cross-entropy loss at the global minimum.",
}
@inproceedings{xu-etal-2023-detime,
    title = "{D}e{T}i{ME}: Diffusion-Enhanced Topic Modeling using Encoder-decoder based {LLM}",
    author = "Xu, Weijie  and
      Hu, Wenxiang  and
      Wu, Fanyou  and
      Sengamedu, Srinivasan",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.606",
    doi = "10.18653/v1/2023.findings-emnlp.606",
    pages = "9040--9057",
    abstract = "In the burgeoning field of natural language processing, Neural Topic Models (NTMs) and Large Language Models (LLMs) have emerged as areas of significant research interest. Despite this, NTMs primarily utilize contextual embeddings from LLMs, which are not optimal for clustering or capable for topic generation. Our study addresses this gap by introducing a novel framework named Diffusion-Enhanced Topic Modeling using Encoder-Decoder-based LLMs (DeTiME). DeTiME leverages Encoder-Decoder-based LLMs to produce highly clusterable embeddings that could generate topics that exhibit both superior clusterability and enhanced semantic coherence compared to existing methods. Additionally, by exploiting the power of diffusion, our framework also provides the capability to generate content relevant to the identified topics. This dual functionality allows users to efficiently produce highly clustered topics and related content simultaneously. DeTiME{'}s potential extends to generating clustered embeddings as well. Notably, our proposed framework proves to be efficient to train and exhibits high adaptability, demonstrating its potential for a wide array of applications.",
}
```