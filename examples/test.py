import sys
sys.path.insert(0, '../src/topicmodeling')
from model import TopicModel
from datasets import load_dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence


df = load_dataset('xwjzds/ag_news')
tm = TopicModel(numb_embeddings = 10)

model_output = tm.train_model(df['train']['text'], name = 'agnews')
metric = TopicDiversity(topk=10)
topic_diversity_score = metric.score(model_output) # Compute score of diversity
cmetric = Coherence(texts =  tm.tp.lemmas,  measure='c_npmi')
coherence1 = cmetric.score(model_output) # Compute score of coherence
cmetric = Coherence(texts =  tm.tp.lemmas,  measure='c_v')
coherence2 = cmetric.score(model_output) # Compute score of coherence
print(topic_diversity_score, coherence1, coherence2)