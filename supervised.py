import codecs
import os
import sys
import time
import random
import numpy as np

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

def tag_sentence(sentence, wordtypes, emission_matrix, transmission_matrix):
    """
    Tag a single sentence using the Viterbi algorithm
    
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
            if test_words[x] in wordtypes:
                word_index = wordtypes.index(test_words[x])
                emission = emission_matrix[y][word_index]
            else:
                emission = 0.001  # Smoothing for unknown words
            
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
    exclude = ["", " ", "<s>", "</s>", "START", "END"]
    wordtypes = []
    tagscount = []
    
    print("Loading training data...")
    # Open training file to read the contents
    f = codecs.open(filepath, 'r', encoding='utf-8')
    file_contents = f.readlines()
    
    # Initialize count of each tag to Zero's
    for x in range(len(tags)):
        tagscount.append(0)
    
    # Calculate count of each tag in the training corpus and also the wordtypes in the corpus
    for line in file_contents:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) == 2:
            word, tag = parts
            
            # Skip sentence markers
            if word in ["<s>", "</s>"]:
                continue
                
            if word not in wordtypes and word not in exclude:
                wordtypes.append(word)
                
            if tag in tags and tag not in exclude:
                tagscount[tags.index(tag)] += 1
    
    f.close()
    
    print(f"Found {len(wordtypes)} unique words and {sum(tagscount)} tagged tokens in training data")
    
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
    # Open training file to update emission and transmission matrix
    f = codecs.open(filepath, 'r', encoding='utf-8')
    file_contents = f.readlines()
    
    # Update emission and transmission matrix with appropriate counts
    row_id = -1
    for line in file_contents:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) == 2:
            word, tag = parts
            
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
    
    # Divide each entry in emission matrix by appropriate tag count to store probabilities
    for x in range(len(tags)):
        for y in range(len(wordtypes)):
            if tagscount[x] != 0:
                emission_matrix[x][y] = float(emission_matrix[x][y]) / tagscount[x]
    
    # Divide each entry in transmission matrix by appropriate tag count to store probabilities
    for x in range(len(tags)):
        for y in range(len(tags)):
            if tagscount[x] != 0:
                transmission_matrix[x][y] = float(transmission_matrix[x][y]) / tagscount[x]
    
    return wordtypes, emission_matrix, transmission_matrix

def estimate_accuracy(filepath):
    """
    Estimate model accuracy using cross-validation on training data
    
    Args:
        filepath: Path to the training file
        
    Returns:
        accuracy: Estimated accuracy of the model
    """
    print("Estimating model accuracy using cross-validation...")
    
    # Read training data
    f = codecs.open(filepath, 'r', encoding='utf-8')
    file_contents = f.readlines()
    f.close()
    
    # Group lines into sentences
    sentences = []
    current_sentence = []
    
    for line in file_contents:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) == 2:
            word, tag = parts
            
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
    
    # Split into 80% training, 20% validation
    split_idx = int(len(sentences) * 0.8)
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]
    
    # Create training data file
    train_data = []
    for sentence in train_sentences:
        train_data.append("<s> START")
        for word, tag in sentence:
            train_data.append(f"{word} {tag}")
        train_data.append("</s> END")
    
    # Write to temporary file
    temp_train_file = "temp_train.txt"
    with codecs.open(temp_train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_data))
    
    # Train model on training split
    wordtypes, emission_matrix, transmission_matrix = train_model(temp_train_file)
    
    # Evaluate on validation split
    total_words = 0
    correct_tags = 0
    
    for sentence in val_sentences:
        # Extract words and gold tags
        words = [word for word, _ in sentence]
        gold_tags = [tag for _, tag in sentence]
        
        # Tag the sentence
        sentence_text = ' '.join(words)
        tagged_words = tag_sentence(sentence_text, wordtypes, emission_matrix, transmission_matrix)
        
        # Compare with gold tags
        for i, (_, predicted_tag, _) in enumerate(tagged_words):
            if i < len(gold_tags):
                total_words += 1
                if predicted_tag == gold_tags[i]:
                    correct_tags += 1
    
    # Clean up temporary file
    os.remove(temp_train_file)
    
    # Calculate accuracy
    accuracy = correct_tags / total_words if total_words > 0 else 0
    return accuracy

def main():
    start_time = time.time()
    
    # Language options
    languages = {
        "1": {"name": "Hindi", "train": "./data/hindi_training.txt", "test": "./data/hindi_testing.txt"},
        "2": {"name": "Kannada", "train": "./data/kannada_training.txt", "test": "./data/kannada_testing.txt"},
        "3": {"name": "Tamil", "train": "./data/tamil_training.txt", "test": "./data/tamil_testing.txt"},
        "4": {"name": "Telugu", "train": "./data/telugu_training.txt", "test": "./data/telugu_testing.txt"}
    }
    
    # Display language options
    print("Select a language for POS tagging:")
    for key, lang in languages.items():
        print(f"{key}. {lang['name']}")
    
    # Get user choice
    choice = input("Enter your choice (1-4): ")
    
    if choice not in languages:
        print("Invalid choice. Defaulting to Hindi.")
        choice = "1"
    
    selected_language = languages[choice]
    print(f"\nSelected language: {selected_language['name']}")
    
    # Train the model
    train_start_time = time.time()
    wordtypes, emission_matrix, transmission_matrix = train_model(selected_language['train'])
    training_time = time.time() - train_start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Estimate model accuracy
    accuracy = estimate_accuracy(selected_language['train'])
    print(f"Estimated model accuracy: {accuracy:.4f}")
    
    # Process test data
    print("\nProcessing test data...")
    test_start_time = time.time()
    
    # Open the testing file to read test sentences
    try:
        file_test = codecs.open(selected_language['test'], 'r', encoding='utf-8')
        test_input = file_test.readlines()
        file_test.close()
        
        # Count number of test sentences
        test_count = len([line for line in test_input if line.strip()])
        print(f"Found {test_count} test sentences")
        
        testing_time = time.time() - test_start_time
        print(f"Testing completed in {testing_time:.2f} seconds")
    except FileNotFoundError:
        print(f"Warning: Test file {selected_language['test']} not found")
    
    # Interactive mode for user input
    print("\n" + "="*80)
    print(f"{selected_language['name']} POS Tagger Interactive Mode")
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
        
        # Get our model's tags
        model_tagged = tag_sentence(user_input, wordtypes, emission_matrix, transmission_matrix)
        
        # Display the results in a formatted table
        print("\nWord\t\tTag\t\tDescription")
        print("-" * 60)
        for word, tag, description in model_tagged:
            # Adjust spacing based on word length
            word_spacing = "\t\t" if len(word) < 8 else "\t"
            tag_spacing = "\t\t" if len(tag) < 8 else "\t"
            
            print(f"{word}{word_spacing}{tag}{tag_spacing}{description}")

if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        print(f"Couldn't find the module - {error}, kindly install before proceeding.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please make sure all required files exist in the data directory.")
