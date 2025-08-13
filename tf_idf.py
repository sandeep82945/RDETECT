from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os,json
import numpy as np
import torch
import random
import re

def log_odds_aggregate(prob_list):
    logits = [np.log(p / (1 - p)) for p in prob_list]
    mean_logit = np.mean(logits)
    aggregated_prob = 1 / (1 + np.exp(-mean_logit))
    return aggregated_prob


def sentence_position_change_attack(text, seed=None):
    """
    Perform a sentence position change attack by shuffling sentence order.

    Args:
        text (str): The original text.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        str: The text with sentences in a different order.
    """
    if seed is not None:
        random.seed(seed)

    # Split the text into sentences (basic punctuation-based splitting)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Remove any empty strings from list
    sentences = [s for s in sentences if s]

    # Shuffle the sentences
    shuffled = sentences[:]
    random.shuffle(shuffled)

    # Join them back together
    return " ".join(shuffled)

def extract_aspects(docs):
    aspects = ["Motivation/Impact", "Originality", "Soundness/Correctness", "Clarity"]
    asp = [{} for _ in range(len(aspects))]
    
    for index in range(len(asp)):
        aspect = aspects[index]
        for (id, revw) in docs.items():
            try:
                if type(revw) == dict:
                    asp[index][id] = " ".join(eval(revw['aspects'][aspect][7:-3]))
                elif type(revw) == list:
                    asp[index][id] = " ".join(eval(revw[0]['aspects'][aspect][7:-3]))
            except:
                if type(revw) == dict:
                    asp[index][id] = revw['aspects'][aspect]
                elif type(revw) == list:
                    asp[index][id] = revw[0]['aspects'][aspect]
                # print(f"Error at {id} at aspect {aspects[index]}")
    
    return asp

def process_aspects(reviews):
    for id, revw in reviews.items():
        yup = {}
        try:
            for asp_id, asp in reviews[id]['aspects'].items():
                try:
                    yup[asp_id] = " ".join(do_eval(asp))
                except:
                    yup[asp_id] = do_eval(asp)
            reviews[id]['aspects'] = yup
        except:
            for asp_id, asp in reviews[id][0]['aspects'].items():
                try:
                    yup[asp_id] = " ".join(do_eval(asp))
                except:
                    yup[asp_id] = do_eval(asp)

            reviews[id] = reviews[id][0]
            reviews[id]['aspects'] = yup

    return reviews

def do_eval(text):
    try:
        return eval(text[7:-3])
    except Exception as e:
        # print("Error")
        return text

def probability_finder(review, word_dict):
    total = 0
    used = set()

    for word in review.lower().split():
        if word not in used and word in word_dict:
            total += word_dict[word]
            used.add(word)

    return total

def convert_list_revw(revw):
    """Converts a list of sentences into one sentence"""
    if type(revw)==str:
        return revw
    else:
        revw_str = ''
        for asp,asp_revw in revw.items():
            for x in asp_revw:
                revw_str +=x 
    return revw_str

def tf_idf(documents,vocabulary):
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    tfidf_matrix = vectorizer.fit_transform(list(documents.values()))

    # Reduce the dimension using SVD Decomposition
    n_components = 1000  # Choose number of components (e.g., 100–300)
    svd = TruncatedSVD(n_components=n_components)

    # Fit and transform the sparse TF-IDF matrix to get reduced dense vectors
    reduced_vectors = svd.fit_transform(tfidf_matrix)

    tfidf_dict = {}
    for i, key in enumerate(documents.keys()):
        tfidf_scores = reduced_vectors[i]
        tfidf_dict[key] = tfidf_scores

    return {
            "revw_tf_idf":tfidf_dict,
            "tf_idf_matrix":reduced_vectors
        }

def get_aspect_tf_idf(aspect_data, vocabulary):
    aspects = ['motivation', 'originality','soundness', 'clarity']
    aspect_tfidf = {}

    for (index, asp) in enumerate(aspects):
        aspect_tfidf[asp] = tf_idf(aspect_data[index], vocabulary)

    return aspect_tfidf

def get_data(folders,path):
    data={}
    for f in folders:
        p = os.path.join(path, f)   
        for file in os.listdir(p):
            try:
                revw = json.load(open(os.path.join(p,file)))
                data[list(revw.keys())[0]] = list(revw.values())[0]
            except:
                pass
                # print(f"Error at {file} in folder {f}")

    return data

def combine_aspects(data):
    combined = {}
    aspects = ['motivation', 'originality','soundness', 'clarity']

    keys = []
    for a in aspects:
        keys = keys + list(data[a]['revw_tf_idf'].keys())

    keys = list(set(keys))

    # Average the aspects
    for id in keys:
        combined[id] = []
        for a in aspects:
            try:
                combined[id].append(data[a]['revw_tf_idf'][id])
            except:
                pass

        combined[id] = np.mean(combined[id], axis=0)

    return combined

def get_Xy_data(path, vocab_path):
    folders = ["ai_iclr","ai_neur","human_iclr","human_neur"]

    ai_data = get_data(folders[:2], path)
    human_data = get_data(folders[2:], path)

    ai_aspects = extract_aspects(ai_data)
    human_aspects = extract_aspects(human_data)

    vocabulary = json.load(open(vocab_path))

    ai_tf_idf = get_aspect_tf_idf(ai_aspects,vocabulary)
    human_tf_idf = get_aspect_tf_idf(human_aspects,vocabulary)

    ai = combine_aspects(ai_tf_idf)
    human = combine_aspects(human_tf_idf)

    X = torch.tensor(np.array(list(ai.values()) + list(human.values())), dtype=torch.float32)
    y = torch.tensor([[1] for _ in range(len(ai))] + [[0] for _ in range(len(human))], dtype=torch.float32)

    return X,y

if __name__=='__main__':
    path = "/Data/sandeep/Vardhan/Journal/Dataset/Aspects"
    vocab_path = "vocabulary.json"

    X,y = get_Xy_data(path, vocab_path)