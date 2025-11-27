#!/usr/bin/env python3
import os
from collections import Counter
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from tqdm import tqdm

# Download required NLTK data
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng']:
    try:
        if resource == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif resource == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab')
        elif resource in ['stopwords', 'wordnet']:
            nltk.data.find(f'corpora/{resource}')
        else:
            nltk.data.find(f'taggers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
pos_map = {'NN': 'NOUN',
           'VB': 'VERB', 
           'JJ': 'ADJ'}

def process_text(text):
    """Tokenize, lemmatize, remove punctuation/stopwords, and tag POS."""
    if not text.strip():
        return []
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]
    if not tokens:
        return []
    tagged = pos_tag(tokens)
    words = []
    for word, pos in tagged:
        if pos in pos_map:
            pos_type = pos_map[pos]
            wnl_pos = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a'}[pos_type]
            lemma = lemmatizer.lemmatize(word, pos=wnl_pos)
            words.append((lemma, pos_type))
    return words

# Process categories one by one
base_path = '/Users/marcolomele/Documents/Repos/bayesian-multisample/data_raw/twenty+newsgroups/twenty+newsgroups_original/20_newsgroups'
categories = ['sci.space', 'soc.religion.christian', 'alt.atheism', 'sci.med', 'talk.politics.guns', 'rec.motorcycles']
results = []

for cat_name in categories:
    print(f"\nProcessing {cat_name}...")
    cat_path = os.path.join(base_path, cat_name)
    
    # Initialize separate dictionaries per POS
    noun_counts = Counter()
    verb_counts = Counter()
    adj_counts = Counter()
    
    # Process each file in the category
    filenames = [f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))]
    for filename in tqdm(filenames, desc=f'Processing {cat_name}', leave=False):
        filepath = os.path.join(cat_path, filename)
        try:
            with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                text = f.read()
                words = process_text(text)
                for word, pos in words:
                    if pos == 'NOUN':
                        noun_counts[word] += 1
                    elif pos == 'VERB':
                        verb_counts[word] += 1
                    elif pos == 'ADJ':
                        adj_counts[word] += 1

        except Exception as e:
            continue
    
    # Get top X words per POS for this category
    X = 1000
    for pos_name, pos_counter in [('NOUN', noun_counts), ('VERB', verb_counts), ('ADJ', adj_counts)]:
        top_words = pos_counter.most_common(X)
        print(f"  Found {len(top_words)} {pos_name} words")
        for word, count in top_words:
            results.append({'category': cat_name, 'POS': pos_name, 'word': word, 'count': count})

# Create DataFrame and save
print(f"\nTotal results: {len(results)}")
if not results:
    df = pd.DataFrame(columns=['category', 'POS', 'word', 'count'])
else:
    df = pd.DataFrame(results)
    df = df.sort_values(['category', 'POS', 'count'], ascending=[True, True, False])

output_path = f'/Users/marcolomele/Documents/Repos/bayesian-multisample/data/twenty+newsgroups/newsgroup_words_{X}_6cats.csv'
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")
