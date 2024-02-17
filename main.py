import argparse
import sys
sys.path.insert(0, 'src/topicmodeling')
from model import TopicModel
from datasets import load_dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
import pandas as pd
# Function to load data from different sources
def load_data(source, path=''):
    if source == 'huggingface':
        return load_dataset(path)['train']['text'][:1000]
    elif source == 'csv':
        df = pd.read_csv(path)
        return df['text'].tolist()
    elif source == 'txt':
        with open(path, 'r') as f:
            return f.readlines()
    else:
        raise ValueError("Invalid data source")

def evaluate_model(model_output, texts, metric, topk = 10):
    if metric != 'diversity': 
        metric = Coherence(texts=texts, measure=metric)
        metric_result = metric.score(model_output)
    else:
        metric = TopicDiversity(topk=topk)
        metric_result = metric.score(model_output)
    return metric_result

# Initialize argparse
parser = argparse.ArgumentParser(description='Topic Modeling Script')

# Add arguments
parser.add_argument('--numb_embeddings', type=int, default=10, help='Number of embeddings')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--gpu_num', type=int, default=1, help='GPU number')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1.2e-6, help='Weight decay')
parser.add_argument('--penalty', type=int, default=1, help='Penalty')
parser.add_argument('--beta', type=int, default=1, help='Beta')
parser.add_argument('--temp', type=int, default=10, help='Temperature')
parser.add_argument('--top_n_words', type=int, default=20, help='Top N words')
parser.add_argument('--num_representative_docs', type=int, default=5, help='Number of representative documents')
parser.add_argument('--top_n_topics', type=int, default=100, help='Top N topics')
parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension')
parser.add_argument('--data_source', type=str, default='huggingface', help='Data source type: huggingface, csv, txt')
parser.add_argument('--data_path', type=str, default='', help='Path to the data file for csv or txt')
parser.add_argument('--metrics', nargs='+', default=['diversity', 'c_v', 'c_npmi', 'c_uci', 'u_mass'], help='List of metrics to report')
parser.add_argument('--topk', type=int, default=10, help='top k words to report for diversity')

if __name__ == '__main__':
    # Parse arguments
    args = parser.parse_args()



    # Load dataset
    print(args.data_source)
    df_text = load_data(args.data_source, args.data_path)

    # Initialize and train the model
    tm = TopicModel(
        numb_embeddings=args.numb_embeddings,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gpu_num=args.gpu_num,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        penalty=args.penalty,
        beta=args.beta,
        temp=args.temp,
        top_n_words=args.top_n_words,
        num_representative_docs=args.num_representative_docs,
        top_n_topics=args.top_n_topics,
        embedding_dim=args.embedding_dim
    )

    model_output = tm.train_model(df_text, args.data_path.replace('/', '_'))

    scores = []
    #evaluation
    for metric in args.metrics:
        score = evaluate_model(model_output, tm.tp.lemmas, metric)
        scores.append(score)
        print(metric + ' is ' + str(score))


    print(scores)