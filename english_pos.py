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
import joblib
import os
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Enhanced feature extraction for better accuracy
def extract_features(word, i, sentence):
    """Extract rich features for a word in context."""
    features = {
        # Word form features
        'word.lower': word.lower(),
        'word': word,
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'word.length': len(word),
        
        # Character-level features
        'word.prefix1': word[0] if len(word) > 0 else '',
        'word.prefix2': word[:2] if len(word) > 1 else word,
        'word.prefix3': word[:3] if len(word) > 2 else word,
        'word.suffix1': word[-1] if len(word) > 0 else '',
        'word.suffix2': word[-2:] if len(word) > 1 else word,
        'word.suffix3': word[-3:] if len(word) > 2 else word,
        
        # Special character features
        'word.has_hyphen': '-' in word,
        'word.has_digit': any(char.isdigit() for char in word),
        'word.has_punctuation': any(not c.isalnum() for c in word),
        
        # Position features
        'word.position': i,
        'word.is_first': i == 0,
        'word.is_last': i == len(sentence) - 1,
    }
    
    # Context features (previous and next words)
    if i > 0:
        prev_word = sentence[i-1]
        features.update({
            'prev_word': prev_word,
            'prev_word.lower': prev_word.lower(),
            'prev_word.istitle': prev_word.istitle(),
            'prev_word.isupper': prev_word.isupper(),
        })
        
        if i > 1:
            prev_prev_word = sentence[i-2]
            features.update({
                'prev_prev_word.lower': prev_prev_word.lower(),
                'prev_bigram': f"{prev_prev_word.lower()}_{prev_word.lower()}"
            })
    else:
        features['BOS'] = True  # Beginning of sentence
        
    if i < len(sentence) - 1:
        next_word = sentence[i+1]
        features.update({
            'next_word': next_word,
            'next_word.lower': next_word.lower(),
            'next_word.istitle': next_word.istitle(),
            'next_word.isupper': next_word.isupper(),
        })
        
        if i < len(sentence) - 2:
            next_next_word = sentence[i+2]
            features.update({
                'next_next_word.lower': next_next_word.lower(),
                'next_bigram': f"{next_word.lower()}_{next_next_word.lower()}"
            })
    else:
        features['EOS'] = True  # End of sentence
    
    # Add bigram features
    if i > 0 and i < len(sentence) - 1:
        prev_word = sentence[i-1]
        next_word = sentence[i+1]
        features['prev_curr_bigram'] = f"{prev_word.lower()}_{word.lower()}"
        features['curr_next_bigram'] = f"{word.lower()}_{next_word.lower()}"
        
    return features

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
    '-NONE-': 'Empty category',
    '-LRB-': 'Left round bracket',
    '-RRB-': 'Right round bracket'
}

# Function to tag a new sentence
def tag_sentence(sentence, vectorizer, model, use_nltk_backup=True):
    """
    Tag a sentence with POS tags.
    
    Args:
        sentence: Input sentence as a string
        vectorizer: Fitted DictVectorizer
        model: Trained POS tagger model
        use_nltk_backup: Whether to use NLTK's tagger as backup for unknown words
        
    Returns:
        List of (word, tag, description) tuples
    """
    words = word_tokenize(sentence)
    features = []
    
    for i, word in enumerate(words):
        features.append(extract_features(word, i, words))
    
    # Transform features
    X_vec = vectorizer.transform(features)
    
    # Predict tags
    predicted_tags = model.predict(X_vec)
    
    # Use NLTK's tagger as backup for words the model is uncertain about
    if use_nltk_backup:
        nltk_tags = dict(nltk.pos_tag(words))
        
        # Get prediction probabilities
        tag_probs = model.predict_proba(X_vec)
        
        # For each word, if the model's confidence is low, use NLTK's tag
        for i, (word, tag) in enumerate(zip(words, predicted_tags)):
            max_prob = np.max(tag_probs[i])
            if max_prob < 0.8 and word in nltk_tags:  # Confidence threshold
                predicted_tags[i] = nltk_tags[word]
    
    # Create a list of (word, tag, description) tuples
    tagged_words = []
    for word, tag in zip(words, predicted_tags):
        description = tag_descriptions.get(tag, "Unknown tag")
        tagged_words.append((word, tag, description))
    
    return tagged_words

def main():
    # Check if model already exists
    MODEL_PATH = 'pos_tagger_model.joblib'
    VECTORIZER_PATH = 'pos_tagger_vectorizer.joblib'
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        print("Loading pre-trained model...")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        
        # Load tag set from file
        if os.path.exists('tag_set.txt'):
            with open('tag_set.txt', 'r') as f:
                tag_set = set(f.read().splitlines())
            print(f"Model loaded successfully with {len(tag_set)} tags.")
        else:
            print("Model loaded successfully.")
    else:
        print("Training new model...")
        print("Loading dataset...")
        # Load the dataset from Hugging Face
        dataset = load_dataset("batterydata/pos_tagging")

        # Access the train and test splits
        train_data = dataset["train"]
        test_data = dataset["test"]

        print(f"Dataset loaded. Train size: {len(train_data)}, Test size: {len(test_data)}")

        print("Preparing training data...")
        start_time = time.time()

        # Process data
        X_train = []
        y_train = []
        
        # Process in smaller batches to avoid memory issues
        batch_size = 500
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # The batch is a dictionary with keys 'words' and 'labels'
            words_list = batch['words']
            tags_list = batch['labels']
            
            # Process each sentence and its tags
            for j, (words, tags) in enumerate(zip(words_list, tags_list)):
                for k, (word, tag) in enumerate(zip(words, tags)):
                    X_train.append(extract_features(word, k, words))
                    y_train.append(tag)
        
        tag_set = set(y_train)
        
        print(f"Training data prepared in {time.time() - start_time:.2f} seconds")
        print(f"Number of unique POS tags: {len(tag_set)}")
        
        # Save tag set for future reference
        with open('tag_set.txt', 'w') as f:
            for tag in sorted(list(tag_set)):
                f.write(f"{tag}\n")

        # Vectorize features
        print("Vectorizing features...")
        start_time = time.time()
        vectorizer = DictVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        print(f"Feature vectorization completed in {time.time() - start_time:.2f} seconds")
        print(f"Number of features: {X_train_vec.shape[1]}")
        
        # Train model with optimized parameters
        print("Training model...")
        start_time = time.time()
        
        # Use LogisticRegression with optimized parameters for better accuracy
        model = LogisticRegression(
            C=1.0,  # Regularization strength
            solver='saga',  # Efficient for large datasets
            penalty='l2',  # Ridge regularization
            max_iter=200,  # Reduced from 1000
            tol=1e-4,  # Convergence tolerance
            n_jobs=-1,  # Use all cores
            random_state=42,  # For reproducibility
            class_weight='balanced',  # Handle class imbalance
            verbose=1  # Show progress
        )
        
        model.fit(X_train_vec, y_train)
        
        print(f"Model training completed in {time.time() - start_time:.2f} seconds")
        
        # Save the model and vectorizer
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print("Model and vectorizer saved to disk.")
        
        # Prepare test data
        print("Preparing test data...")
        start_time = time.time()
        
        X_test = []
        y_test = []
        
        # Process test data in batches
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            
            # Get words and tags from the batch
            words_list = batch['words']
            tags_list = batch['labels']
            
            # Process each sentence and its tags
            for j, (words, tags) in enumerate(zip(words_list, tags_list)):
                for k, (word, tag) in enumerate(zip(words, tags)):
                    X_test.append(extract_features(word, k, words))
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
        
        # Compare with NLTK's tagger
        nltk_tagged = nltk.pos_tag(word_tokenize(user_input))
        
        # Get our model's tags
        model_tagged = tag_sentence(user_input, vectorizer, model)
        
        # Display the results in a formatted table
        print("\nWord\t\tModel Tag\tNLTK Tag\tDescription")
        print("-" * 80)
        for i, (word, tag, description) in enumerate(model_tagged):
            # Adjust spacing based on word length
            word_spacing = "\t\t" if len(word) < 8 else "\t"
            
            # Get NLTK's tag for comparison
            nltk_tag = nltk_tagged[i][1] if i < len(nltk_tagged) else "N/A"
            
            # Highlight disagreements
            agreement = "✓" if tag == nltk_tag else "✗"
            
            print(f"{word}{word_spacing}{tag}\t\t{nltk_tag}\t\t{description} {agreement}")

if __name__ == "__main__":
    # This is required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
