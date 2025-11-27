from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# Initialize tokenizer and model for DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = TFAutoModel.from_pretrained("distilbert-base-multilingual-cased")

def compute_embedding(text):
    encoded_input = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(**encoded_input)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return embeddings.numpy()

# Load a subset of the wikipedia dataset (assuming structure and availability)
dataset = load_dataset("Cohere/wikipedia-22-12-en-embeddings",split="train", streaming=True)

#========Exercise 3.1 =========== 
def find_most_relevant_article(query_embedding, dataset, max_num_of_articles=None):
    most_relevant_article = None
    max_similarity = -1  # Start with lowest possible similarity
    count = 0

    for example in tqdm(dataset):
        if max_num_of_articles is not None and count >= max_num_of_articles:
            break

        article_text = example.get("text", "")
        if not article_text:
            continue

        try:
            article_embedding = compute_embedding(article_text)
            similarity = cosine_similarity(query_embedding, article_embedding)[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                most_relevant_article = article_text

            count += 1
        except Exception as e:
            print("Skipping article due to error:", e)
            continue

    return most_relevant_article, max_similarity

#========End Exercise 3.1 ===========


queries = ["Leonardo DiCaprio", "France", "Python", "Deep Learning"]

for query in queries:
    print("\n=== Query:", query, "===")
    input_embedding = compute_embedding(query)
    article, similarity = find_most_relevant_article(input_embedding, dataset, 1000)
    print("Most Relevant Article:", article[:500], "...")
    print("Similarity Score:", similarity)
