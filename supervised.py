import streamlit as st
import codecs
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

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
    
    st.text("Loading training data...")
    # Open training file to read the contents
    try:
        if not os.path.exists(filepath):
            st.warning(f"Warning: File not found: {filepath}")
            # Return empty model if file doesn't exist
            return [], [[]], [[]]
            
        with open(filepath, 'r', encoding='utf-8') as f:
            file_contents = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                file_contents = f.readlines()
        except Exception as e:
            st.error(f"Error reading file {filepath}: {e}")
            return [], [[]], [[]]
    
    # Initialize count of each tag to Zero's
    for x in range(len(tags)):
        tagscount.append(0)
    
    # Calculate count of each tag in the training corpus and also the wordtypes in the corpus
    for line in file_contents:
        line = normalize_line(line)
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 2:
            word, tag = parts[0], parts[-1]
            
            # Skip sentence markers
            if word in ["<s>", "</s>"]:
                continue
                
            if word not in wordtypes and word not in exclude:
                wordtypes.append(word)
                
            if tag in tags and tag not in exclude:
                tagscount[tags.index(tag)] += 1
    
    st.text(f"Found {len(wordtypes)} unique words and {sum(tagscount)} tagged tokens in training data")
    
    # If no words or tags found, return empty model
    if len(wordtypes) == 0 or sum(tagscount) == 0:
        st.warning(f"Warning: No valid data found in {filepath}")
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
    
    st.text("Building emission and transmission matrices...")
    # Process file contents again to update emission and transmission matrix
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                file_contents = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                file_contents = f.readlines()
        except Exception as e:
            st.error(f"Error reading file {filepath}: {e}")
            # Continue with empty file_contents
            file_contents = []
    
    # Update emission and transmission matrix with appropriate counts
    row_id = -1
    for line in file_contents:
        line = normalize_line(line)
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 2:
            word, tag = parts[0], parts[-1]
            
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
    smoothing_value = 0.001
    
    # Divide each entry in emission matrix by appropriate tag count to store probabilities
    for x in range(len(tags)):
        for y in range(len(wordtypes)):
            emission_matrix[x][y] = (emission_matrix[x][y] + smoothing_value) / (tagscount[x] + smoothing_value * len(wordtypes))
    
    # Divide each entry in transmission matrix by appropriate tag count to store probabilities
    for x in range(len(tags)):
        for y in range(len(tags)):
            transmission_matrix[x][y] = (transmission_matrix[x][y] + smoothing_value) / (tagscount[x] + smoothing_value * len(tags))
    
    return wordtypes, emission_matrix, transmission_matrix

def estimate_accuracy(filepath):
    """
    Estimate model accuracy using cross-validation on training data
    
    Args:
        filepath: Path to the training file
        
    Returns:
        accuracy: Estimated accuracy of the model
    """
    st.text("Estimating model accuracy using cross-validation...")
    
    # Check if file exists
    if not os.path.exists(filepath):
        st.warning(f"Warning: File not found: {filepath}")
        return 0.0
    
    # Read training data
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_contents = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                file_contents = f.readlines()
        except Exception as e:
            st.error(f"Error reading file {filepath}: {e}")
            return 0.0
    
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
    
    # If no sentences found, return 0 accuracy
    if not sentences:
        st.warning(f"Warning: No valid sentences found in {filepath}")
        return 0.0
    
    # Shuffle sentences
    random.shuffle(sentences)
    
    # Split into 80% training, 20% validation
    split_idx = int(len(sentences) * 0.8)
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]
    
    # If either split is empty, use all data for both
    if not train_sentences or not val_sentences:
        train_sentences = sentences
        val_sentences = sentences
    
    # Create training data file
    train_data = []
    for sentence in train_sentences:
        train_data.append("<s> START")
        for word, tag in sentence:
            train_data.append(f"{word} {tag}")
        train_data.append("</s> END")
    
    # Write to temporary file
    temp_train_file = "temp_train.txt"
    with open(temp_train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_data))
    
    # Train model on training split
    wordtypes, emission_matrix, transmission_matrix = train_model(temp_train_file)
    
    # If model training failed, return 0 accuracy
    if not wordtypes:
        if os.path.exists(temp_train_file):
            os.remove(temp_train_file)
        return 0.0
    
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
    if os.path.exists(temp_train_file):
        os.remove(temp_train_file)
    
    # Calculate accuracy
    accuracy = correct_tags / total_words if total_words > 0 else 0
    return accuracy

# Streamlit UI
def main():
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
    # Train model button
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model..."):
            wordtypes, emission_matrix, transmission_matrix = train_model(training_file)
            
            # Store model in session state
            st.session_state.wordtypes = wordtypes
            st.session_state.emission_matrix = emission_matrix
            st.session_state.transmission_matrix = transmission_matrix
            
            # Estimate accuracy
            accuracy = estimate_accuracy(training_file)
            st.session_state.accuracy = accuracy
            
            st.sidebar.success(f"Model trained successfully! Estimated accuracy: {accuracy:.2%}")
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This app uses a Hidden Markov Model with the Viterbi algorithm to perform Parts of Speech (POS) tagging. "
        "It can analyze text in multiple Indian languages to identify grammatical components."
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
            """, unsafe_allow_html=True)
        else:
            st.info("Train the model to see statistics and information.")
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p>POS Tagger App</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state variables if they don't exist
    if 'wordtypes' not in st.session_state:
        st.session_state.wordtypes = []
    if 'emission_matrix' not in st.session_state:
        st.session_state.emission_matrix = [[]]
    if 'transmission_matrix' not in st.session_state:
        st.session_state.transmission_matrix = [[]]
    if 'accuracy' not in st.session_state:
        st.session_state.accuracy = 0.0
    main()
