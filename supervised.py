import streamlit as st
import codecs
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from collections import defaultdict, Counter

# Global Variables definition
tags = ['NN', 'NST', 'NNP', 'PRP', 'DEM', 'VM', 'VAUX', 'JJ', 'RB', 'PSP', 'RP', 'CC', 'WQ', 'QF', 'QC', 'QO', 'CL', 'INTF', 'INJ', 'NEG', 'UT', 'SYM', 'COMP', 'RDP', 'ECH', 'UNK', 'XC', 'START', 'END']

# Tag descriptions for better user experience
tag_descriptions = {
    'NN': 'Common Noun',
    'NST': 'Noun with spatial and temporal properties',
    'NNP': 'Proper Noun',
    'PRP': 'Pronoun',
    'DEM': 'Demonstrative',
    'VM': 'Main Verb',
    'VAUX': 'Auxiliary Verb',
    'JJ': 'Adjective',
    'RB': 'Adverb',
    'PSP': 'Postposition',
    'RP': 'Particle',
    'CC': 'Conjunction',
    'WQ': 'Question word',
    'QF': 'Quantifier',
    'QC': 'Cardinal',
    'QO': 'Ordinal',
    'CL': 'Classifier',
    'INTF': 'Intensifier',
    'INJ': 'Interjection',
    'NEG': 'Negative',
    'UT': 'Quotative',
    'SYM': 'Symbol',
    'COMP': 'Complementizer',
    'RDP': 'Reduplication',
    'ECH': 'Echo',
    'UNK': 'Unknown',
    'XC': 'Compound',
    'START': 'Sentence Start',
    'END': 'Sentence End'
}

class UnknownWordHandler:
    def __init__(self):
        # Common prefixes in Indian languages
        self.prefixes = {
            'अ': {'JJ': 0.6, 'NN': 0.3, 'RB': 0.1},  # Negative prefix
            'सु': {'JJ': 0.7, 'NN': 0.2, 'RB': 0.1},  # Good/well prefix
            'दु': {'JJ': 0.7, 'NN': 0.2, 'RB': 0.1},  # Bad/ill prefix
            'नि': {'JJ': 0.5, 'NN': 0.3, 'VM': 0.2},  # Without/free from
            'प्र': {'NN': 0.5, 'VM': 0.3, 'JJ': 0.2},  # Forward/pro prefix
            'वि': {'NN': 0.4, 'VM': 0.3, 'JJ': 0.3},  # Special/particular
            'स': {'NN': 0.5, 'JJ': 0.3, 'VM': 0.2},   # With/together
            'अन': {'JJ': 0.8, 'NN': 0.2},            # Not/without
            'उप': {'NN': 0.7, 'JJ': 0.2, 'VM': 0.1},  # Sub/under
            'अति': {'JJ': 0.7, 'RB': 0.2, 'NN': 0.1}, # Over/excessive
            'अधि': {'NN': 0.6, 'VM': 0.3, 'JJ': 0.1}  # Over/super
        }
        
        # Common suffixes in Indian languages
        self.suffixes = {
            # Verb suffixes
            'ना': {'VM': 0.9, 'VAUX': 0.1},
            'ता': {'VM': 0.8, 'VAUX': 0.2},
            'ते': {'VM': 0.8, 'VAUX': 0.2},
            'ती': {'VM': 0.8, 'VAUX': 0.2},
            'या': {'VM': 0.7, 'VAUX': 0.3},
            'गा': {'VM': 0.9, 'VAUX': 0.1},
            'गी': {'VM': 0.9, 'VAUX': 0.1},
            'गे': {'VM': 0.9, 'VAUX': 0.1},
            'कर': {'VM': 0.8, 'VAUX': 0.2},
            'रहा': {'VM': 0.8, 'VAUX': 0.2},
            'रही': {'VM': 0.8, 'VAUX': 0.2},
            'रहे': {'VM': 0.8, 'VAUX': 0.2},
            'ऊंगा': {'VM': 0.9, 'VAUX': 0.1},
            'ेगा': {'VM': 0.9, 'VAUX': 0.1},
            'ेगी': {'VM': 0.9, 'VAUX': 0.1},
            
            # Noun suffixes
            'ों': {'NN': 0.8, 'NST': 0.2},
            'ाएँ': {'NN': 0.9, 'NST': 0.1},
            'ियाँ': {'NN': 0.9, 'NST': 0.1},
            'ियों': {'NN': 0.9, 'NST': 0.1},
            'पन': {'NN': 0.8, 'JJ': 0.2},
            'त्व': {'NN': 0.9, 'NST': 0.1},
            'कार': {'NN': 0.9, 'NNP': 0.1},
            
            # Adjective suffixes
            'सा': {'JJ': 0.9, 'NN': 0.1},
            'सी': {'JJ': 0.9, 'NN': 0.1},
            'से': {'JJ': 0.7, 'PSP': 0.3},
            'वाला': {'JJ': 0.8, 'NN': 0.2},
            'वाली': {'JJ': 0.8, 'NN': 0.2},
            'वाले': {'JJ': 0.8, 'NN': 0.2},
            
            # Postposition suffixes
            'का': {'PSP': 0.9, 'NN': 0.1},
            'के': {'PSP': 0.9, 'NN': 0.1},
            'की': {'PSP': 0.9, 'NN': 0.1},
            'को': {'PSP': 0.9, 'NN': 0.1},
            'से': {'PSP': 0.8, 'NN': 0.2},
            'में': {'PSP': 0.9, 'NST': 0.1},
            'पर': {'PSP': 0.8, 'NST': 0.2},
            'तक': {'PSP': 0.9, 'NST': 0.1},
            'ने': {'PSP': 0.9, 'VM': 0.1},
        }
        
        # Character n-gram patterns for different POS tags
        self.char_ngrams = {
            'NN': ['ाव', 'त्व', 'पन', 'कार', 'गार'],
            'VM': ['ना', 'ता', 'ते', 'ती', 'गा', 'गी', 'गे'],
            'JJ': ['सा', 'सी', 'से', 'वाला', 'वाली', 'वाले'],
            'PSP': ['का', 'के', 'की', 'को', 'से', 'में', 'पर', 'तक', 'ने'],
            'RB': ['तः', 'दा', 'कर', 'रूप'],
            'VAUX': ['है', 'था', 'थी', 'थे', 'हूँ', 'हो', 'हैं']
        }
        
        # Special character patterns
        self.special_patterns = {
            'has_digits': {'QC': 0.8, 'QO': 0.1, 'NN': 0.1},
            'has_hyphen': {'JJ': 0.4, 'NN': 0.4, 'XC': 0.2},
            'has_uppercase': {'NNP': 0.9, 'NN': 0.1},
            'has_symbols': {'SYM': 0.9, 'NN': 0.1}
        }
        
        # Default tag distribution for completely unknown words
        self.default_distribution = {
            'NN': 0.5,    # Nouns are most common
            'VM': 0.2,    # Verbs
            'JJ': 0.1,    # Adjectives
            'NST': 0.05,  # Spatial/temporal nouns
            'PRP': 0.05,  # Pronouns
            'RB': 0.05,   # Adverbs
            'PSP': 0.05   # Postpositions
        }
        
        # Initialize n-gram cache
        self.ngram_cache = {}
    
    def has_digits(self, word):
        """Check if the word contains digits"""
        return any(char.isdigit() for char in word)
    
    def has_hyphen(self, word):
        """Check if the word contains hyphens"""
        return '-' in word
    
    def has_uppercase(self, word):
        """Check if the word contains uppercase letters (for languages that use Latin script)"""
        return any(char.isupper() for char in word)
    
    def has_symbols(self, word):
        """Check if the word contains symbols"""
        return any(not (char.isalpha() or char.isdigit() or char == '-' or char == '_') for char in word)
    
    def extract_char_ngrams(self, word, n=3):
        """Extract character n-grams from a word"""
        if len(word) < n:
            return [word]
        
        # Cache results for efficiency
        cache_key = f"{word}_{n}"
        if cache_key in self.ngram_cache:
            return self.ngram_cache[cache_key]
        
        ngrams = []
        for i in range(len(word) - n + 1):
            ngrams.append(word[i:i+n])
        
        self.ngram_cache[cache_key] = ngrams
        return ngrams
    
    def get_prefix_probabilities(self, word):
        """Get tag probabilities based on word prefix"""
        for prefix, tag_probs in self.prefixes.items():
            if word.startswith(prefix):
                return tag_probs
        return {}
    
    def get_suffix_probabilities(self, word):
        """Get tag probabilities based on word suffix"""
        # Try longer suffixes first
        for suffix_length in range(5, 0, -1):
            if len(word) <= suffix_length:
                continue
                
            suffix = word[-suffix_length:]
            if suffix in self.suffixes:
                return self.suffixes[suffix]
        
        return {}
    
    def get_ngram_probabilities(self, word):
        """Get tag probabilities based on character n-grams"""
        ngram_probs = {}
        
        # Extract 2-grams and 3-grams
        bigrams = self.extract_char_ngrams(word, 2)
        trigrams = self.extract_char_ngrams(word, 3)
        
        # Check if any n-grams match our patterns
        for tag, patterns in self.char_ngrams.items():
            matches = 0
            for pattern in patterns:
                if (pattern in bigrams) or (pattern in trigrams) or (pattern in word):
                    matches += 1
            
            if matches > 0:
                ngram_probs[tag] = matches / len(patterns)
        
        return ngram_probs
    
    def get_special_pattern_probabilities(self, word):
        """Get tag probabilities based on special patterns"""
        pattern_probs = {}
        
        if self.has_digits(word):
            pattern_probs.update(self.special_patterns['has_digits'])
        
        if self.has_hyphen(word):
            for tag, prob in self.special_patterns['has_hyphen'].items():
                pattern_probs[tag] = pattern_probs.get(tag, 0) + prob
        
        if self.has_uppercase(word):
            for tag, prob in self.special_patterns['has_uppercase'].items():
                pattern_probs[tag] = pattern_probs.get(tag, 0) + prob
        
        if self.has_symbols(word):
            for tag, prob in self.special_patterns['has_symbols'].items():
                pattern_probs[tag] = pattern_probs.get(tag, 0) + prob
        
        # Normalize probabilities if any patterns matched
        if pattern_probs:
            total = sum(pattern_probs.values())
            for tag in pattern_probs:
                pattern_probs[tag] /= total
        
        return pattern_probs
    
    def predict_tag_probabilities(self, word):
        """
        Predict tag probabilities for an unknown word using
        morphological analysis and character n-grams
        """
        # Get probabilities from different methods
        prefix_probs = self.get_prefix_probabilities(word)
        suffix_probs = self.get_suffix_probabilities(word)
        ngram_probs = self.get_ngram_probabilities(word)
        pattern_probs = self.get_special_pattern_probabilities(word)
        
        # Combine probabilities with weights
        # Suffix analysis is typically most reliable for Indian languages
        weights = {
            'suffix': 0.5,
            'prefix': 0.2,
            'ngram': 0.2,
            'pattern': 0.1
        }
        
        # Start with default distribution
        final_probs = self.default_distribution.copy()
        
        # Update with weighted probabilities from each method
        for tag, prob in prefix_probs.items():
            final_probs[tag] = final_probs.get(tag, 0) + (prob * weights['prefix'])
            
        for tag, prob in suffix_probs.items():
            final_probs[tag] = final_probs.get(tag, 0) + (prob * weights['suffix'])
            
        for tag, prob in ngram_probs.items():
            final_probs[tag] = final_probs.get(tag, 0) + (prob * weights['ngram'])
            
        for tag, prob in pattern_probs.items():
            final_probs[tag] = final_probs.get(tag, 0) + (prob * weights['pattern'])
        
        # Normalize probabilities
        total = sum(final_probs.values())
        for tag in final_probs:
            final_probs[tag] /= total
        
        return final_probs
    
    def get_emission_probability(self, word, tag_idx, tags):
        """
        Get emission probability for an unknown word given a tag index
        
        Args:
            word: The unknown word
            tag_idx: Index of the tag in the tags list
            tags: List of all possible tags
            
        Returns:
            Probability of the word given the tag
        """
        tag = tags[tag_idx]
        
        # Get probabilities for all tags
        tag_probs = self.predict_tag_probabilities(word)
        
        # Return probability for the requested tag, or a small value if not found
        return tag_probs.get(tag, 0.001)

def normalize_line(line):
    """
    Normalize line to handle inconsistent spacing in the training data
    """
    line = line.strip()
    if not line:
        return ""
    
    # Handle cases where there might be multiple spaces or tabs
    parts = [part for part in line.split() if part]
    
    # If we have exactly two parts, return word and tag
    if len(parts) == 2:
        return f"{parts[0]} {parts[1]}"
    # If we have more parts, assume the last one is the tag and the rest is the word
    elif len(parts) > 2:
        word = ' '.join(parts[:-1])
        tag = parts[-1]
        return f"{word} {tag}"
    
    return line

def split_corpus(filepath, train_ratio=0.8):
    """Split the corpus into training and testing sets"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_contents = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                file_contents = f.readlines()
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return [], []
    
    # Group lines into sentences
    sentences = []
    current_sentence = []
    
    for line in file_contents:
        line = normalize_line(line)
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 2:
            word, tag = parts[0], parts[-1]
            
            if word == "<s>":
                current_sentence = []
            elif word == "</s>":
                if current_sentence:
                    sentences.append(current_sentence)
            else:
                current_sentence.append((word, tag))
    
    # Ensure the last sentence is added
    if current_sentence:
        sentences.append(current_sentence)
    
    # Shuffle sentences
    random.shuffle(sentences)
    
    # Split into training and testing
    split_idx = int(len(sentences) * train_ratio)
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]
    
    return train_sentences, test_sentences

def max_connect(x, y, viterbi_matrix, emission, transmission_matrix):
    """
    Find the maximum probability path in the Viterbi algorithm
    """
    max_val = -99999
    path = -1
    
    for k in range(len(tags)):
        val = viterbi_matrix[k][x-1] * transmission_matrix[k][y]
        if val * emission > max_val:
            max_val = val
            path = k
    
    return max_val, path

def get_emission_probability(word, tag_idx, wordtypes, emission_matrix, tags):
    """Get emission probability with better unknown word handling"""
    if word in wordtypes:
        word_index = wordtypes.index(word)
        return emission_matrix[tag_idx][word_index]
    
    # For unknown words, use the UnknownWordHandler
    unknown_handler = UnknownWordHandler()
    return unknown_handler.get_emission_probability(word, tag_idx, tags)

def tag_sentence(sentence, wordtypes, emission_matrix, transmission_matrix):
    """
    Tag a single sentence using the Viterbi algorithm with improved unknown word handling
    
    Args:
        sentence: Input sentence as a string
        wordtypes: List of known words from training
        emission_matrix: Emission probabilities
        transmission_matrix: Transmission probabilities
        
    Returns:
        List of (word, tag) tuples
    """
    # Tokenize the sentence
    test_words = sentence.strip().split()
    
    # Initialize POS tags
    pos_tags = [-1] * len(test_words)
    
    # Initialize Viterbi matrix and path
    viterbi_matrix = []
    viterbi_path = []
    
    # Initialize viterbi matrix of size |tags| * |no of words in test sentence|
    for x in range(len(tags)):
        viterbi_matrix.append([])
        viterbi_path.append([])
        for y in range(len(test_words)):
            viterbi_matrix[x].append(0)
            viterbi_path[x].append(0)
    
    # Update viterbi matrix column wise
    for x in range(len(test_words)):
        for y in range(len(tags)):
            # Use improved emission probability function
            emission = get_emission_probability(test_words[x], y, wordtypes, emission_matrix, tags)
            
            if x > 0:
                max_val, viterbi_path[y][x] = max_connect(x, y, viterbi_matrix, emission, transmission_matrix)
            else:
                max_val = 1
            
            viterbi_matrix[y][x] = emission * max_val
    
    # Identify the max probability in last column
    maxval = -999999
    maxs = -1
    for x in range(len(tags)):
        if viterbi_matrix[x][len(test_words)-1] > maxval:
            maxval = viterbi_matrix[x][len(test_words)-1]
            maxs = x
    
    # Backtrack and identify best tags for each word
    for x in range(len(test_words)-1, -1, -1):
        pos_tags[x] = maxs
        maxs = viterbi_path[maxs][x]
    
    # Create a list of (word, tag) tuples
    tagged_words = []
    for i, tag_idx in enumerate(pos_tags):
        tagged_words.append((test_words[i], tags[tag_idx], tag_descriptions.get(tags[tag_idx], "Unknown tag")))
    
    return tagged_words

def train_model_on_sentences(sentences):
    """
    Train the POS tagging model using the given sentences
    
    Args:
        sentences: List of sentences, where each sentence is a list of (word, tag) tuples
        
    Returns:
        wordtypes: List of known words
        emission_matrix: Emission probabilities
        transmission_matrix: Transmission probabilities
    """
    exclude = ["", " ", "<s>", "</s>", "START", "END"]
    wordtypes = []
    tagscount = []
    
    print("Processing training sentences...")
    
    # Initialize count of each tag to Zero's
    for x in range(len(tags)):
        tagscount.append(0)
    
    # Calculate count of each tag in the training corpus and also the wordtypes in the corpus
    for sentence in sentences:
        for word, tag in sentence:
            # Skip sentence markers
            if word in ["<s>", "</s>"]:
                continue
                
            if word not in wordtypes and word not in exclude:
                wordtypes.append(word)
                
            if tag in tags and tag not in exclude:
                tagscount[tags.index(tag)] += 1
    
    print(f"Found {len(wordtypes)} unique words and {sum(tagscount)} tagged tokens in training data")
    
    # If no words or tags found, return empty model
    if len(wordtypes) == 0 or sum(tagscount) == 0:
        print("Warning: No valid data found in training sentences")
        # Initialize with small default values to avoid division by zero
        for i in range(len(tags)):
            tagscount[i] = 1  # Add a small count to each tag
        
        # Add a dummy word if needed
        if len(wordtypes) == 0:
            wordtypes.append("DUMMY")
    
    # Declare variables for emission and transmission matrix
    emission_matrix = []
    transmission_matrix = []
    
    # Initialize emission matrix
    for x in range(len(tags)):
        emission_matrix.append([])
        for y in range(len(wordtypes)):
            emission_matrix[x].append(0)
    
    # Initialize transmission matrix
    for x in range(len(tags)):
        transmission_matrix.append([])
        for y in range(len(tags)):
            transmission_matrix[x].append(0)
    
    print("Building emission and transmission matrices...")
    
    # Update emission and transmission matrix with appropriate counts
    for sentence in sentences:
        row_id = -1
        for word, tag in sentence:
            # Skip sentence markers
            if word in ["<s>", "</s>"]:
                if word == "<s>":
                    row_id = -1  # Reset for new sentence
                continue
                
            if tag in tags and word in wordtypes:
                col_id = wordtypes.index(word)
                prev_row_id = row_id
                row_id = tags.index(tag)
                emission_matrix[row_id][col_id] += 1
                if prev_row_id != -1:
                    transmission_matrix[prev_row_id][row_id] += 1
    
    # Add smoothing to avoid zero probabilities
    smoothing_value = 0.1  # Increased from 0.001 to 0.1 for better regularization
    
    # Divide each entry in emission matrix by appropriate tag count to store probabilities
    for x in range(len(tags)):
        for y in range(len(wordtypes)):
            emission_matrix[x][y] = (emission_matrix[x][y] + smoothing_value) / (tagscount[x] + smoothing_value * len(wordtypes))
    
    # Divide each entry in transmission matrix by appropriate tag count to store probabilities
    for x in range(len(tags)):
        for y in range(len(tags)):
            transmission_matrix[x][y] = (transmission_matrix[x][y] + smoothing_value) / (tagscount[x] + smoothing_value * len(tags))
    
    return wordtypes, emission_matrix, transmission_matrix

def train_model(filepath):
    """
    Train the POS tagging model using the given training file
    
    Args:
        filepath: Path to the training file
        
    Returns:
        wordtypes: List of known words
        emission_matrix: Emission probabilities
        transmission_matrix: Transmission probabilities
    """
    print("Loading training data...")
    # Split corpus into training and validation sets (80/20 split)
    train_sentences, _ = split_corpus(filepath, train_ratio=0.8)
    
    # Train model on training sentences
    return train_model_on_sentences(train_sentences)

def evaluate_model(wordtypes, emission_matrix, transmission_matrix, test_sentences):
    """Evaluate the model on test sentences"""
    correct_tags = 0
    total_tags = 0
    
    # Confusion matrix
    confusion_matrix = {}
    
    for sentence in test_sentences:
        # Extract words and gold tags
        words = [word for word, _ in sentence]
        gold_tags = [tag for _, tag in sentence]
        
        # Tag the sentence
        sentence_text = ' '.join(words)
        tagged_words = tag_sentence(sentence_text, wordtypes, emission_matrix, transmission_matrix)
        
        # Compare with gold tags
        for i, (word, predicted_tag, _) in enumerate(tagged_words):
            if i < len(gold_tags):
                total_tags += 1
                gold_tag = gold_tags[i]
                
                # Update confusion matrix
                if gold_tag not in confusion_matrix:
                    confusion_matrix[gold_tag] = {}
                if predicted_tag not in confusion_matrix[gold_tag]:
                    confusion_matrix[gold_tag][predicted_tag] = 0
                confusion_matrix[gold_tag][predicted_tag] += 1
                
                if predicted_tag == gold_tag:
                    correct_tags += 1
    
    # Calculate accuracy
    accuracy = correct_tags / total_tags if total_tags > 0 else 0
    
    # Print confusion matrix for most common tags
    print("\nConfusion Matrix (top tags):")
    common_tags = ['NN', 'VM', 'JJ', 'PRP', 'PSP', 'VAUX', 'RB']
    for gold_tag in common_tags:
        if gold_tag in confusion_matrix:
            print(f"{gold_tag}: ", end="")
            for pred_tag in common_tags:
                count = confusion_matrix[gold_tag].get(pred_tag, 0)
                print(f"{pred_tag}:{count} ", end="")
            print()
    
    return accuracy

def k_fold_cross_validation(filepath, k=5):
    """Perform k-fold cross-validation"""
    # Read and prepare data
    train_sentences, _ = split_corpus(filepath, train_ratio=1.0)
    
    # Shuffle data
    random.shuffle(train_sentences)
    
    # Split into k folds
    fold_size = len(train_sentences) // k
    folds = []
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else len(train_sentences)
        folds.append(train_sentences[start_idx:end_idx])
    
    # Perform k-fold cross-validation
    accuracies = []
    for i in range(k):
        # Use fold i as test set, rest as training set
        test_fold = folds[i]
        train_folds = [fold for j, fold in enumerate(folds) if j != i]
        train_data = [item for fold in train_folds for item in fold]
        
        # Train model
        wordtypes, emission_matrix, transmission_matrix = train_model_on_sentences(train_data)
        
        # Evaluate on test fold
        accuracy = evaluate_model(wordtypes, emission_matrix, transmission_matrix, test_fold)
        accuracies.append(accuracy)
        print(f"Fold {i+1} accuracy: {accuracy:.4f}")
    
    # Return average accuracy
    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average accuracy: {avg_accuracy:.4f}")
    return avg_accuracy

def estimate_accuracy(filepath):
    """
    Estimate model accuracy using proper train/test split
    
    Args:
        filepath: Path to the training file
        
    Returns:
        accuracy: Estimated accuracy of the model
    """
    print("Estimating model accuracy using train/test split...")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return 0.0
    
    # Split corpus into training and testing sets
    train_sentences, test_sentences = split_corpus(filepath, train_ratio=0.8)
    
    # If either split is empty, return 0 accuracy
    if not train_sentences or not test_sentences:
        print(f"Warning: No valid sentences found in {filepath}")
        return 0.0
    
    # Train model on training split
    wordtypes, emission_matrix, transmission_matrix = train_model_on_sentences(train_sentences)
    
    # Evaluate on test split
    accuracy = evaluate_model(wordtypes, emission_matrix, transmission_matrix, test_sentences)
    
    print(f"Accuracy on test set: {accuracy:.4f}")
    return accuracy

def combine_training_data(filepaths):
    """Combine multiple training files"""
    combined_sentences = []
    
    for filepath in filepaths:
        train_sentences, _ = split_corpus(filepath)
        combined_sentences.extend(train_sentences)
    
    return combined_sentences

def build_lexical_context_model(sentences):
    """Build a model of word bigrams to help with ambiguous words"""
    bigram_counts = defaultdict(Counter)
    
    for sentence in sentences:
        for i in range(1, len(sentence)):
            prev_word, prev_tag = sentence[i-1]
            curr_word, curr_tag = sentence[i]
            bigram_counts[prev_word][curr_word] += 1
    
    return bigram_counts

def get_contextual_probability(prev_word, curr_word, bigram_counts):
    """Get probability of curr_word given prev_word"""
    if prev_word in bigram_counts and sum(bigram_counts[prev_word].values()) > 0:
        return bigram_counts[prev_word][curr_word] / sum(bigram_counts[prev_word].values())
    return 0.001

# Main function for command-line usage
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='POS Tagger with improved unknown word handling')
    parser.add_argument('--train', type=str, help='Path to training file')
    parser.add_argument('--test', type=str, help='Path to test file')
    parser.add_argument('--cv', action='store_true', help='Perform cross-validation')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--tag', type=str, help='Sentence to tag')
    
    args = parser.parse_args()
    
    if args.train:
        # Train the model
        wordtypes, emission_matrix, transmission_matrix = train_model(args.train)
        
        # Evaluate if test file is provided
        if args.test:
            _, test_sentences = split_corpus(args.test, train_ratio=0)
            accuracy = evaluate_model(wordtypes, emission_matrix, transmission_matrix, test_sentences)
            print(f"Accuracy on test set: {accuracy:.4f}")
        
        # Perform cross-validation if requested
        if args.cv:
            k_fold_cross_validation(args.train, k=args.folds)
        
        # Tag a sentence if provided
        if args.tag:
            tagged_words = tag_sentence(args.tag, wordtypes, emission_matrix, transmission_matrix)
            for word, tag, description in tagged_words:
                print(f"{word}\t{tag}\t{description}")
    else:
        print("Please provide a training file with --train")

# Streamlit UI
def streamlit_ui():
    st.set_page_config(
        page_title="POS Tagger App",
        page_icon="🏷️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #26A69A;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .tag {
        font-weight: bold;
        color: #1E88E5;
    }
    .tag-description {
        color: #616161;
        font-style: italic;
    }
    .word-container {
        display: inline-block;
        margin: 0.3rem;
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: #E3F2FD;
        border: 1px solid #90CAF9;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #9E9E9E;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1 class='main-header'>Parts of Speech (POS) Tagger</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>An interactive tool for analyzing the grammatical structure of sentences</p>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/grammar.png", width=100)
    st.sidebar.title("POS Tagger Settings")
    
    # Language selection
    languages = {
        "Hindi": "./data/hindi_training.txt",
        "Kannada": "./data/kannada_training.txt",
        "Tamil": "./data/tamil_training.txt",
        "Telugu": "./data/telugu_training.txt",
        "Marwari": "./data/marwari_training.txt",
        "Marathi": "./data/marathi_training.txt",
        "Punjabi": "./data/punjabi_training.txt",
        "Gujarati": "./data/gujarati_training.txt",
        "Malayalam": "./data/malayalam_training.txt",
        "Bengali": "./data/bengali_training.txt"
    }
    
    selected_language = st.sidebar.selectbox("Select Language", list(languages.keys()))
    
    # Upload custom training data option
    st.sidebar.markdown("---")
    st.sidebar.subheader("Custom Training Data")
    uploaded_file = st.sidebar.file_uploader("Upload your own training data", type=["txt"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        with open("uploaded_training.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        training_file = "uploaded_training.txt"
        st.sidebar.success("Custom training data uploaded successfully!")
    else:
        training_file = languages[selected_language]
    
    # Model training section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Training")
    
    # Advanced options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced Options")
    use_cross_validation = st.sidebar.checkbox("Use Cross-Validation", value=False)
    num_folds = st.sidebar.slider("Number of Folds", min_value=2, max_value=10, value=5, disabled=not use_cross_validation)
    
    # Train model button
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model..."):
            if use_cross_validation:
                accuracy = k_fold_cross_validation(training_file, k=num_folds)
                st.session_state.accuracy = accuracy
                st.sidebar.success(f"Cross-validation complete! Average accuracy: {accuracy:.2%}")
            
            wordtypes, emission_matrix, transmission_matrix = train_model(training_file)
            
            # Store model in session state
            st.session_state.wordtypes = wordtypes
            st.session_state.emission_matrix = emission_matrix
            st.session_state.transmission_matrix = transmission_matrix
            
            # Estimate accuracy if not using cross-validation
            if not use_cross_validation:
                accuracy = estimate_accuracy(training_file)
                st.session_state.accuracy = accuracy
                st.sidebar.success(f"Model trained successfully! Estimated accuracy: {accuracy:.2%}")
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This app uses a Hidden Markov Model with the Viterbi algorithm to perform Parts of Speech (POS) tagging. "
        "It includes improved unknown word handling using morphological analysis, which is particularly effective "
        "for morphologically rich languages like Hindi, Marathi, and other Indian languages."
    )
    
    # Main content area
    tabs = st.tabs(["Tagger", "Tag Information", "Model Information"])
    
    with tabs[0]:
        st.markdown("<h2 class='sub-header'>Text Analysis</h2>", unsafe_allow_html=True)
        
        # Example sentences based on selected language
        example_sentences = {
            "Hindi": "मैं आज बाजार जाऊंगा",
            "Kannada": "ನಾನು ಇಂದು ಮಾರುಕಟ್ಟೆಗೆ ಹೋಗುತ್ತೇನೆ",
            "Tamil": "நான் இன்று சந்தைக்குச் செல்வேன்",
            "Telugu": "నేను ఈరోజు మార్కెట్కు వెళతాను",
            "Marwari": "म्हें आज बजार जावुंला",
            "Marathi": "मी आज बाजारात जाईन",
            "Punjabi": "ਮੈਂ ਅੱਜ ਬਾਜ਼ਾਰ ਜਾਵਾਂਗਾ",
            "Gujarati": "હું આજે બજારમાં જઈશ",
            "Malayalam": "ഞാൻ ഇന്ന് മാർക്കറ്റിൽ പോകും",
            "Bengali": "আমি আজ বাজারে যাব"
        }
        
        default_example = example_sentences.get(selected_language, "Type your sentence here")
        
        # User input
        user_input = st.text_area("Enter text for POS tagging:", value=default_example, height=100)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            tag_button = st.button("Tag Text", key="tag_button", use_container_width=True)
        
        with col2:
            clear_button = st.button("Clear Results", key="clear_button", use_container_width=True)
            
        if clear_button:
            if 'tagged_results' in st.session_state:
                del st.session_state.tagged_results
        
        # Process the text when the button is clicked
        if tag_button and user_input:
            if 'wordtypes' not in st.session_state or 'emission_matrix' not in st.session_state or 'transmission_matrix' not in st.session_state:
                st.warning("Please train the model first!")
            else:
                with st.spinner("Analyzing text..."):
                    tagged_words = tag_sentence(
                        user_input, 
                        st.session_state.wordtypes, 
                        st.session_state.emission_matrix, 
                        st.session_state.transmission_matrix
                    )
                    st.session_state.tagged_results = tagged_words
        
        # Display results
        if 'tagged_results' in st.session_state and st.session_state.tagged_results:
            st.markdown("<h3>Tagged Results:</h3>", unsafe_allow_html=True)
            
            # Visual representation of tagged words
            html_output = "<div style='line-height: 2.5;'>"
            for word, tag, description in st.session_state.tagged_results:
                html_output += f"<div class='word-container'>{word} <span class='tag'>({tag})</span></div> "
            html_output += "</div>"
            
            st.markdown(html_output, unsafe_allow_html=True)
            
            # Detailed table view
            st.markdown("<h3>Detailed Analysis:</h3>", unsafe_allow_html=True)
            
            # Create a DataFrame for better display
            df = pd.DataFrame(
                [(word, tag, description) for word, tag, description in st.session_state.tagged_results],
                columns=["Word", "POS Tag", "Description"]
            )
            
            st.dataframe(df, use_container_width=True)
            
            # Visualization of tag distribution
            st.markdown("<h3>Tag Distribution:</h3>", unsafe_allow_html=True)
            
            # Count tags
            tag_counts = {}
            for _, tag, _ in st.session_state.tagged_results:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Create DataFrame for visualization
            tag_df = pd.DataFrame(list(tag_counts.items()), columns=["Tag", "Count"])
            tag_df = tag_df.sort_values("Count", ascending=False)
            
            # Create bar chart
            chart = alt.Chart(tag_df).mark_bar().encode(
                x=alt.X('Tag', sort=None),
                y='Count',
                color=alt.Color('Tag', legend=None),
                tooltip=['Tag', 'Count']
            ).properties(
                width=600,
                height=400,
                title="Distribution of POS Tags"
            )
            
            st.altair_chart(chart, use_container_width=True)
    
    with tabs[1]:
        st.markdown("<h2 class='sub-header'>POS Tag Information</h2>", unsafe_allow_html=True)
        
        # Display tag information in a table
        tag_info = pd.DataFrame(
            [(tag, tag_descriptions[tag]) for tag in tags],
            columns=["Tag", "Description"]
        )
        
        st.dataframe(tag_info, use_container_width=True)
        
        # Add some explanatory text
        st.markdown("""
        <div class='highlight'>
        <p>Parts of Speech (POS) tagging is the process of marking up words in text according to their grammatical categories. 
        This helps in understanding the syntactic structure of sentences and is a fundamental step in many natural language processing tasks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("<h2 class='sub-header'>Model Information</h2>", unsafe_allow_html=True)
        
        if 'wordtypes' in st.session_state and 'accuracy' in st.session_state:
            # Display model statistics
            st.markdown(f"""
            <div class='highlight'>
            <p><strong>Model Statistics:</strong></p>
            <ul>
                <li>Language: {selected_language}</li>
                <li>Vocabulary Size: {len(st.session_state.wordtypes)} words</li>
                <li>Estimated Accuracy: {st.session_state.accuracy:.2%}</li>
                <li>Number of POS Tags: {len(tags)}</li>
                <li>Unknown Word Handling: Improved (morphological analysis)</li>
                <li>Evaluation Method: {'Cross-validation' if use_cross_validation else 'Train/Test Split'}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Add explanation of the algorithm
            st.markdown("""
            <p><strong>Algorithm:</strong> This POS tagger uses a Hidden Markov Model (HMM) with the Viterbi algorithm. 
            The model calculates two main probability matrices:</p>
            <ul>
                <li><strong>Emission Probabilities:</strong> The probability of a word given a specific tag</li>
                <li><strong>Transmission Probabilities:</strong> The probability of transitioning from one tag to another</li>
            </ul>
            <p>The Viterbi algorithm finds the most likely sequence of tags for a given sentence by calculating the maximum probability path through these matrices.</p>
            
            <p><strong>Improved Unknown Word Handling:</strong> The model uses morphological analysis to better predict tags for unknown words by examining:</p>
            <ul>
                <li>Word prefixes and suffixes common in Indian languages</li>
                <li>Character n-grams that are characteristic of specific parts of speech</li>
                <li>Special patterns like digits, hyphens, and symbols</li>
            </ul>
            <p>This approach is particularly effective for morphologically rich languages like Hindi, Marathi, and other Indian languages.</p>
            
            <p><strong>Preventing Overtraining:</strong> The model implements several techniques to prevent overtraining:</p>
            <ul>
                <li>Proper train/test split (80/20)</li>
                <li>Add-k smoothing (k=0.1) for regularization</li>
                <li>Cross-validation option for more robust evaluation</li>
                <li>Sophisticated unknown word handling</li>
            </ul>
            """, unsafe_allow_html=True)
        else:
            st.info("Train the model to see statistics and information.")
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p>POS Tagger App with Improved Unknown Word Handling</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command-line mode
        main()
    else:
        # Streamlit UI mode
        try:
            import streamlit as st
            streamlit_ui()
        except ImportError:
            print("Streamlit not installed. Running in command-line mode.")
            print("Use --train to specify a training file.")
            print("Use --tag to specify a sentence to tag.")
