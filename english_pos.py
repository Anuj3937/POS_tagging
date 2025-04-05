import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
import time

# Download NLTK tokenizer if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

print("Loading dataset...")
# Load the dataset from Hugging Face
dataset = load_dataset("batterydata/pos_tagging")

# Access the train and test splits
train_data = dataset["train"]
test_data = dataset["test"]

print(f"Dataset loaded. Train size: {len(train_data)}, Test size: {len(test_data)}")

# Function to extract features from words
def extract_features(word, i, sentence):
    """Extract features for a word in context."""
    features = {
        'word.lower': word.lower(),
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'word.prefix1': word[0] if len(word) > 0 else '',
        'word.prefix2': word[:2] if len(word) > 1 else word,
        'word.prefix3': word[:3] if len(word) > 2 else word,
        'word.suffix1': word[-1] if len(word) > 0 else '',
        'word.suffix2': word[-2:] if len(word) > 1 else word,
        'word.suffix3': word[-3:] if len(word) > 2 else word,
        'word.has_hyphen': '-' in word,
        'word.has_digit': any(char.isdigit() for char in word),
        'word.position': i,  # Position in sentence
        'word.length': len(word),
    }
    
    # Add context features (previous and next words)
    if i > 0:
        prev_word = sentence[i-1]
        features.update({
            'prev_word.lower': prev_word.lower(),
            'prev_word.istitle': prev_word.istitle(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence
        
    if i < len(sentence) - 1:
        next_word = sentence[i+1]
        features.update({
            'next_word.lower': next_word.lower(),
            'next_word.istitle': next_word.istitle(),
        })
    else:
        features['EOS'] = True  # End of sentence
        
    return features

print("Preparing training data...")
start_time = time.time()

# Prepare training data
X_train = []
y_train = []
tag_set = set()

# Corrected column names based on screenshots
for words, tags in zip(train_data['words'], train_data['labels']):
    for i, (word, tag) in enumerate(zip(words, tags)):
        X_train.append(extract_features(word, i, words))
        y_train.append(tag)
        tag_set.add(tag)

print(f"Training data prepared in {time.time() - start_time:.2f} seconds")
print(f"Number of unique POS tags: {len(tag_set)}")
print(f"Tags: {sorted(list(tag_set))}")

# Convert features to vectors
print("Vectorizing features...")
start_time = time.time()
vectorizer = DictVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
print(f"Feature vectorization completed in {time.time() - start_time:.2f} seconds")
print(f"Number of features: {X_train_vec.shape[1]}")

# Train model
print("Training model...")
start_time = time.time()
model = LogisticRegression(max_iter=1000, C=1.0, solver='saga', n_jobs=-1)
model.fit(X_train_vec, y_train)
print(f"Model training completed in {time.time() - start_time:.2f} seconds")

# Prepare test data
print("Preparing test data...")
start_time = time.time()
X_test = []
y_test = []

# Corrected column names based on screenshots
for words, tags in zip(test_data['words'], test_data['labels']):
    for i, (word, tag) in enumerate(zip(words, tags)):
        X_test.append(extract_features(word, i, words))
        y_test.append(tag)

X_test_vec = vectorizer.transform(X_test)
print(f"Test data prepared in {time.time() - start_time:.2f} seconds")

# Evaluate model
print("Evaluating model...")
y_pred = model.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred)
print(report)

# Create confusion matrix for the most common tags
print("Generating confusion matrix for top 10 tags...")
most_common_tags = pd.Series(y_test).value_counts().nlargest(10).index.tolist()
mask = np.isin(y_test, most_common_tags)
y_test_common = [y for i, y in enumerate(y_test) if mask[i]]
y_pred_common = [y for i, y in enumerate(y_pred) if mask[i]]

cm = confusion_matrix(y_test_common, y_pred_common, labels=most_common_tags)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=most_common_tags, yticklabels=most_common_tags)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Top 10 Tags)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Tag description dictionary
tag_descriptions = {
    'CC': 'Coordinating conjunction',
    'CD': 'Cardinal number',
    'DT': 'Determiner',
    'EX': 'Existential there',
    'FW': 'Foreign word',
    'IN': 'Preposition or subordinating conjunction',
    'JJ': 'Adjective',
    'JJR': 'Adjective, comparative',
    'JJS': 'Adjective, superlative',
    'LS': 'List item marker',
    'MD': 'Modal',
    'NN': 'Noun, singular or mass',
    'NNS': 'Noun, plural',
    'NNP': 'Proper noun, singular',
    'NNPS': 'Proper noun, plural',
    'PDT': 'Predeterminer',
    'POS': 'Possessive ending',
    'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'RB': 'Adverb',
    'RBR': 'Adverb, comparative',
    'RBS': 'Adverb, superlative',
    'RP': 'Particle',
    'SYM': 'Symbol',
    'TO': 'to',
    'UH': 'Interjection',
    'VB': 'Verb, base form',
    'VBD': 'Verb, past tense',
    'VBG': 'Verb, gerund or present participle',
    'VBN': 'Verb, past participle',
    'VBP': 'Verb, non-3rd person singular present',
    'VBZ': 'Verb, 3rd person singular present',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WP$': 'Possessive wh-pronoun',
    'WRB': 'Wh-adverb',
    '.': 'Punctuation',
    ',': 'Comma',
    ':': 'Colon or semicolon',
    '(': 'Left bracket',
    ')': 'Right bracket',
    '``': 'Opening quotation mark',
    "''": 'Closing quotation mark',
    '#': 'Pound sign',
    '$': 'Dollar sign',
    '-NONE-': 'Empty category'
}

# Function to tag a new sentence
def tag_sentence(sentence):
    words = word_tokenize(sentence)
    features = []
    
    for i, word in enumerate(words):
        features.append(extract_features(word, i, words))
    
    # Transform features
    X_vec = vectorizer.transform(features)
    
    # Predict tags
    predicted_tags = model.predict(X_vec)
    
    # Create a list of (word, tag, description) tuples
    tagged_words = []
    for word, tag in zip(words, predicted_tags):
        description = tag_descriptions.get(tag, "Unknown tag")
        tagged_words.append((word, tag, description))
    
    return tagged_words

# Interactive mode for user input
print("\n" + "="*80)
print("POS Tagger Interactive Mode")
print("Enter a sentence to get POS tags. Type 'exit' to quit.")
print("="*80)

while True:
    user_input = input("\nEnter a sentence: ")
    
    if user_input.lower() == 'exit':
        print("Exiting program. Goodbye!")
        break
    
    if not user_input.strip():
        print("Please enter a valid sentence.")
        continue
    
    tagged_words = tag_sentence(user_input)
    
    # Display the results in a formatted table
    print("\nWord\t\tTag\tDescription")
    print("-" * 60)
    for word, tag, description in tagged_words:
        # Adjust spacing based on word length
        spacing = "\t\t" if len(word) < 8 else "\t"
        print(f"{word}{spacing}{tag}\t{description}")
