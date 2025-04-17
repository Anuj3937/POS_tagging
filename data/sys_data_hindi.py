import random
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import os

class HindiPOSDatasetGenerator:
    def __init__(self):
        # POS tags based on the IIIT Hyderabad tagset
        self.pos_tags = [
            'NN', 'NST', 'NNP', 'PRP', 'DEM', 'VM', 'VAUX', 'JJ', 'RB', 
            'PSP', 'RP', 'QF', 'QC', 'CC', 'WQ', 'QO', 'INTF', 'INJ', 
            'NEG', 'SYM', 'XC', 'RDP', 'ECH', 'UNK'
        ]
        
        # Tag distribution probabilities (based on common distributions in Hindi)
        self.tag_distribution = {
            'NN': 0.26,    # Nouns are very common
            'VM': 0.14,    # Verbs
            'JJ': 0.09,    # Adjectives
            'PRP': 0.08,   # Pronouns
            'PSP': 0.08,   # Postpositions
            'VAUX': 0.07,  # Auxiliary verbs
            'CC': 0.05,    # Conjunctions
            'DEM': 0.04,   # Demonstratives
            'RP': 0.03,    # Particles
            'NST': 0.03,   # Spatial/temporal expressions
            'NNP': 0.03,   # Proper nouns
            'QF': 0.02,    # Quantifiers
            'RB': 0.02,    # Adverbs
            'QC': 0.02,    # Cardinals
            'XC': 0.01,    # Compounds
            'WQ': 0.01,    # Question words
            'INTF': 0.01,  # Intensifiers
            'QO': 0.005,   # Ordinals
            'NEG': 0.005,  # Negation
            'SYM': 0.005,  # Symbols
            'INJ': 0.005,  # Interjections
            'RDP': 0.005,  # Reduplication
            'ECH': 0.005,  # Echo words
            'UNK': 0.005   # Unknown
        }
        
        # Base vocabulary with common Hindi words for each POS tag
        self.base_vocabulary = self.initialize_base_vocabulary()
        
        # Transition probabilities between POS tags (simplified bigram model)
        self.transitions = self.initialize_transitions()
        
        # For generating word variations
        self.prefixes = ['अ', 'सु', 'दु', 'नि', 'प्र', 'वि', 'स', 'अन', 'उप', 'अति', 'महा', 'कु']
        self.suffixes = ['ों', 'ाएँ', 'ियाँ', 'ियों', 'ता', 'त्व', 'पन', 'वाला', 'वाली', 'वाले', 'कर', 'कार']
        
        # Common sentence patterns in Hindi
        self.sentence_patterns = [
            ['PRP', 'NN', 'PSP', 'VM', 'VAUX'],
            ['DEM', 'NN', 'JJ', 'VM'],
            ['NN', 'PSP', 'NN', 'VM', 'VAUX'],
            ['PRP', 'NN', 'VM'],
            ['NNP', 'NN', 'PSP', 'VM', 'VAUX'],
            ['PRP', 'JJ', 'NN', 'VM'],
            ['NN', 'CC', 'NN', 'VM', 'VAUX'],
            ['PRP', 'QF', 'NN', 'VM'],
            ['DEM', 'NN', 'PSP', 'NN', 'VM'],
            ['NN', 'PSP', 'PRP', 'VM', 'VAUX'],
            ['PRP', 'NN', 'PSP', 'VM', 'NEG'],
            ['NN', 'JJ', 'VM', 'VAUX'],
            ['PRP', 'NNP', 'PSP', 'VM', 'VAUX'],
            ['NN', 'NST', 'PSP', 'VM'],
            ['WQ', 'NN', 'VM', 'VAUX'],
            # Adding patterns with adverbs (RB)
            ['PRP', 'RB', 'NN', 'VM', 'VAUX'],
            ['PRP', 'RB', 'VM', 'VAUX'],
            ['RB', 'NN', 'PSP', 'VM'],
            ['NN', 'RB', 'VM', 'VAUX']
        ]
        
        # Generated vocabulary will be expanded from base
        self.generated_vocabulary = defaultdict(list)
        
        # Statistics tracking
        self.unique_words = set()
        self.token_count = 0
        
    def initialize_base_vocabulary(self):
        """Initialize a base vocabulary of common Hindi words with their POS tags"""
        vocab = {
            'NN': ['आदमी', 'लड़का', 'लड़की', 'घर', 'किताब', 'गाँव', 'शहर', 'देश', 'पानी', 'खाना', 'विद्यार्थी', 'स्कूल', 
                  'काम', 'समय', 'पैसा', 'आँख', 'हाथ', 'पैर', 'सिर', 'मन', 'विचार', 'प्यार', 'माँ', 
                  'पिता', 'भाई', 'बहन', 'दोस्त', 'खेत', 'फल', 'फूल', 'पेड़', 'पत्ता', 'हवा', 'बारिश', 'दिन', 'रात'],
            
            'NST': ['ऊपर', 'नीचे', 'अंदर', 'बाहर', 'आगे', 'पीछे', 'सामने', 'बगल', 'पास', 'दूर', 'बीच', 'तरफ'],
            
            'NNP': ['दिल्ली', 'मुंबई', 'कोलकाता', 'भारत', 'गंगा', 'हिमालय', 'राम', 'सीता', 
                   'मोहन', 'राधा', 'कृष्ण', 'सचिन', 'अमिताभ', 'नरेंद्र', 'सुनील', 'अनिल', 'सुनीता', 'अनिता'],
            
            'PRP': ['मैं', 'तुम', 'वह', 'यह', 'हम', 'आप', 'वे', 'ये', 'मेरा', 'तुम्हारा', 'उसका', 'इसका', 'हमारा', 'आपका', 'उनका', 'इनका', 'स्वयं', 'अपना'],
            
            'DEM': ['यह', 'वह', 'ये', 'वे', 'इस', 'उस', 'इन', 'उन', 'ऐसा', 'वैसा', 'ऐसे', 'वैसे'],
            
            'VM': ['जा', 'आ', 'कर', 'बोल', 'खा', 'पी', 'देख', 'सुन', 'लिख', 'पढ़', 'धो', 'सो', 'उठ', 
                  'बैठ', 'चल', 'दौड़', 'नाच', 'हँस', 'रो', 'सोच', 'समझ', 'सीख', 'सिखा', 'दे', 'ले'],
            
            'VAUX': ['है', 'था', 'थी', 'थे', 'हूँ', 'हो', 'हैं', 'थीं', 'रहा', 'रही', 'रहे', 'सकता', 'सकती', 'सकते', 'चुका', 'चुकी', 'चुके', 'पाता', 'पाती', 'पाते'],
            
            'JJ': ['अच्छा', 'बुरा', 'बड़ा', 'छोटा', 'लंबा', 'नाटा', 'मोटा', 'पतला', 'सुंदर', 'कुरूप', 
                  'होशियार', 'मूर्ख', 'अमीर', 'गरीब', 'नया', 'पुराना', 'ताजा', 'सूखा', 'गीला', 'लाल', 'काला', 'सफेद', 'नीला', 'हरा', 'पीला'],
            
            # Fixed RB list with proper formatting and additional adverbs
            'RB': ['जल्दी', 'देर', 'अब', 'तब', 'कब', 'कभी', 'हमेशा', 'शायद', 'तेज', 'धीरे', 'आज', 
                  'जोर', 'अचानक', 'धीरे-धीरे', 'एकदम', 'फिर', 'दोबारा', 'यहाँ', 'वहाँ', 'कहाँ', 
                  'कल', 'परसों', 'सुबह', 'शाम', 'रात', 'प्रतिदिन', 'नियमित', 'अक्सर', 'बहुधा', 'सदैव'],
            
            'PSP': ['का', 'के', 'की', 'ने', 'को', 'से', 'में', 'पर', 'तक', 'लिए', 'वाला', 'वाली', 'वाले'],
            
            'RP': ['भी', 'ही', 'तो', 'तक', 'सिर्फ', 'केवल', 'मात्र', 'बस', 'ही'],
            
            'QF': ['सब', 'सारे', 'कुछ', 'थोड़ा', 'बहुत', 'कई', 'अनेक', 'ज्यादा', 'कम', 'अधिक'],
            
            'QC': ['एक', 'दो', 'तीन', 'चार', 'पाँच', 'छह', 'सात', 'आठ', 'नौ', 'दस', 'बीस', 'सौ', 'हजार', 'लाख', 'करोड़'],
            
            'CC': ['और', 'या', 'एवं', 'तथा', 'किंतु', 'परंतु', 'लेकिन', 'मगर', 'क्योंकि', 'इसलिए', 'अगर', 'तो'],
            
            'WQ': ['क्या', 'कौन', 'कौनसा', 'कौनसी', 'कौनसे', 'कैसा', 'कैसी', 'कैसे', 'क्यों', 'कब', 'कहाँ', 'कितना', 'कितनी', 'कितने'],
            
            'QO': ['पहला', 'दूसरा', 'तीसरा', 'चौथा', 'पाँचवा', 'छठा', 'सातवाँ', 'आठवाँ', 'नौवाँ', 'दसवाँ'],
            
            'INTF': ['बहुत', 'अति', 'बेहद', 'अत्यंत', 'बिलकुल', 'एकदम', 'साफ', 'बिल्कुल', 'जरा'],
            
            'INJ': ['अरे', 'ओह', 'वाह', 'शाबाश', 'अफसोस', 'हाय', 'बाप रे', 'अच्छा', 'ओहो'],
            
            'NEG': ['नहीं', 'ना', 'न', 'मत', 'नही', 'नहि'],
            
            'SYM': ['.', ',', '!', '?', ':', ';', '-', '(', ')', '"', "'"],
            
            'XC': ['धड़', 'भर', 'पट', 'खुश', 'नाराज', 'आनंद', 'दुःख', 'गुस्सा', 'प्रेम', 'नफरत'],
            
            'RDP': ['धीरे-धीरे', 'जल्दी-जल्दी', 'कभी-कभी', 'थोड़ा-थोड़ा', 'अलग-अलग', 'देखते-देखते', 'सोचते-सोचते'],
            
            'ECH': ['चाय-वाय', 'पानी-वानी', 'खाना-वाना', 'कपड़े-वपड़े', 'रोटी-वोटी', 'पैसा-वैसा']
        }
        return vocab
    
    def initialize_transitions(self):
        """Initialize transition probabilities between POS tags"""
        # This is a simplified model - in a real scenario, these would be learned from data
        transitions = {
            'START': {'NN': 0.3, 'PRP': 0.2, 'DEM': 0.15, 'NNP': 0.1, 'JJ': 0.1, 'QF': 0.05, 'QC': 0.05, 'WQ': 0.05},
            'NN': {'VM': 0.3, 'PSP': 0.25, 'CC': 0.1, 'JJ': 0.1, 'NN': 0.1, 'VAUX': 0.05, 'RP': 0.05, 'SYM': 0.05},
            'VM': {'VAUX': 0.4, 'NN': 0.2, 'SYM': 0.2, 'CC': 0.1, 'RP': 0.05, 'PSP': 0.05},
            'JJ': {'NN': 0.7, 'VM': 0.1, 'CC': 0.1, 'RP': 0.05, 'INTF': 0.05},
            'PRP': {'VM': 0.3, 'NN': 0.2, 'PSP': 0.2, 'JJ': 0.1, 'VAUX': 0.1, 'RP': 0.1},
            'PSP': {'NN': 0.4, 'PRP': 0.2, 'VM': 0.2, 'JJ': 0.1, 'NST': 0.1},
            'VAUX': {'SYM': 0.4, 'CC': 0.2, 'NN': 0.1, 'VAUX': 0.1, 'RP': 0.1, 'PSP': 0.1},
            'CC': {'NN': 0.3, 'PRP': 0.2, 'VM': 0.2, 'JJ': 0.1, 'DEM': 0.1, 'QF': 0.05, 'QC': 0.05},
            'DEM': {'NN': 0.6, 'JJ': 0.2, 'NST': 0.1, 'VM': 0.1},
            'QF': {'NN': 0.8, 'JJ': 0.1, 'VM': 0.1},
            'QC': {'NN': 0.8, 'QO': 0.1, 'VM': 0.1},
            'NNP': {'NN': 0.3, 'VM': 0.3, 'PSP': 0.2, 'CC': 0.1, 'JJ': 0.1},
            'NST': {'PSP': 0.4, 'NN': 0.3, 'VM': 0.2, 'JJ': 0.1},
            'RP': {'NN': 0.3, 'VM': 0.3, 'JJ': 0.2, 'PRP': 0.1, 'PSP': 0.1},
            'RB': {'VM': 0.5, 'JJ': 0.2, 'NN': 0.2, 'VAUX': 0.1},
            'WQ': {'VM': 0.4, 'NN': 0.3, 'PRP': 0.2, 'JJ': 0.1},
            'INTF': {'JJ': 0.6, 'NN': 0.2, 'VM': 0.1, 'RB': 0.1},
            'NEG': {'VM': 0.4, 'VAUX': 0.3, 'NN': 0.2, 'JJ': 0.1},
            'SYM': {'START': 1.0},  # After a symbol, we can start a new sentence
            'INJ': {'NN': 0.3, 'PRP': 0.3, 'SYM': 0.2, 'VM': 0.1, 'JJ': 0.1},
            'QO': {'NN': 0.8, 'VM': 0.1, 'JJ': 0.1},
            'XC': {'NN': 0.4, 'VM': 0.3, 'JJ': 0.2, 'VAUX': 0.1},
            'RDP': {'NN': 0.3, 'VM': 0.3, 'RB': 0.2, 'JJ': 0.2},
            'ECH': {'VM': 0.4, 'NN': 0.3, 'JJ': 0.2, 'VAUX': 0.1}
        }
        return transitions
    
    def expand_vocabulary(self, target_size=1500):
        """Expand the vocabulary to reach the target size"""
        # First copy the base vocabulary
        for tag, words in self.base_vocabulary.items():
            self.generated_vocabulary[tag] = words.copy()
            for word in words:
                self.unique_words.add(word)
        
        # Continue generating words until we reach the target size
        while len(self.unique_words) < target_size:
            # Pick a random POS tag based on distribution
            tag = random.choices(list(self.tag_distribution.keys()), 
                                weights=list(self.tag_distribution.values()))[0]
            
            if tag in self.base_vocabulary and self.base_vocabulary[tag]:
                # Generate a new word by modifying an existing one
                base_word = random.choice(self.base_vocabulary[tag])
                new_word = self.generate_word_variation(base_word, tag)
                
                if new_word not in self.unique_words:
                    self.generated_vocabulary[tag].append(new_word)
                    self.unique_words.add(new_word)
        
        print(f"Expanded vocabulary to {len(self.unique_words)} unique words")
        return self.generated_vocabulary
    
    def generate_word_variation(self, word, tag):
        """Generate a variation of a word based on its POS tag"""
        # Different strategies based on POS tag
        if tag in ['NN', 'NNP', 'NST']:
            # For nouns, we can add prefixes/suffixes
            if random.random() < 0.5 and len(word) > 3:
                # Add a suffix
                return word + random.choice(self.suffixes)
            else:
                # Add a prefix
                return random.choice(self.prefixes) + word
        
        elif tag in ['VM', 'VAUX']:
            # For verbs, modify the ending
            verb_endings = ['ता', 'ती', 'ते', 'ना', 'नी', 'ने', 'रहा', 'रही', 'रहे', 'कर', 'या', 'ई', 'ए']
            if len(word) > 3:
                return word[:-1] + random.choice(verb_endings)
            else:
                return word + random.choice(verb_endings)
        
        elif tag in ['JJ', 'RB']:
            # For adjectives and adverbs
            adj_endings = ['सा', 'सी', 'से', 'पन', 'ता', 'त्व', 'वाला', 'वाली', 'वाले']
            if random.random() < 0.6:
                return word + random.choice(adj_endings)
            else:
                return random.choice(self.prefixes) + word
        
        elif tag == 'PRP':
            # For pronouns, add case markers
            case_markers = ['को', 'का', 'की', 'के', 'से', 'ने', 'पर', 'में']
            return word + random.choice(case_markers)
        
        else:
            # For other tags, make small modifications
            if len(word) > 3:
                # Change a character
                pos = random.randint(1, len(word) - 2)
                chars = list(word)
                hindi_chars = 'अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह'
                chars[pos] = random.choice(hindi_chars)
                return ''.join(chars)
            else:
                # Just add a character
                hindi_chars = 'अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह'
                return word + random.choice(hindi_chars)
    
    def generate_sentence(self):
        """Generate a random Hindi sentence with POS tags"""
        if random.random() < 0.7:
            # Use a predefined pattern 70% of the time
            pattern = random.choice(self.sentence_patterns)
            sentence = []
            
            for tag in pattern:
                if tag in self.generated_vocabulary and self.generated_vocabulary[tag]:
                    word = random.choice(self.generated_vocabulary[tag])
                    sentence.append((word, tag))
                    self.token_count += 1
        else:
            # Generate using transition probabilities 30% of the time
            sentence = []
            current_tag = 'START'
            
            # Generate a sentence of random length (3-15 words)
            sentence_length = random.randint(3, 15)
            
            for _ in range(sentence_length):
                if current_tag == 'START' or current_tag not in self.transitions:
                    # If we're at the start or have an unknown tag, pick based on START transitions
                    next_tag = random.choices(
                        list(self.transitions['START'].keys()),
                        weights=list(self.transitions['START'].values())
                    )[0]
                else:
                    # Otherwise use the transition probabilities
                    next_tag = random.choices(
                        list(self.transitions[current_tag].keys()),
                        weights=list(self.transitions[current_tag].values())
                    )[0]
                
                # If we end with a symbol, we're done
                if next_tag == 'SYM':
                    if self.generated_vocabulary[next_tag]:
                        word = random.choice(self.generated_vocabulary[next_tag])
                        sentence.append((word, next_tag))
                        self.token_count += 1
                    break
                
                # Add the word with its tag
                if next_tag in self.generated_vocabulary and self.generated_vocabulary[next_tag]:
                    word = random.choice(self.generated_vocabulary[next_tag])
                    sentence.append((word, next_tag))
                    self.token_count += 1
                    current_tag = next_tag
        
        return sentence
    
    def generate_dataset(self, num_sentences=2000, min_tokens=20000):
        """Generate a dataset with the specified number of sentences and minimum tokens"""
        # First expand the vocabulary
        self.expand_vocabulary()
        
        # Generate sentences until we have enough tokens
        sentences = []
        while self.token_count < min_tokens or len(sentences) < num_sentences:
            sentence = self.generate_sentence()
            sentences.append(sentence)
            
            # Print progress
            if len(sentences) % 100 == 0:
                print(f"Generated {len(sentences)} sentences, {self.token_count} tokens so far")
        
        print(f"Dataset generation complete: {len(sentences)} sentences, {self.token_count} tokens, {len(self.unique_words)} unique words")
        return sentences
    
    def save_dataset(self, sentences, filename="hindi_pos_dataset.txt"):
        """Save the dataset in the same format as the Hindi training data"""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(sentences):
                f.write("<s> START\n")
                for word, tag in sentence:
                    f.write(f"{word} {tag}\n")
                f.write("</s> END\n")
        
        print(f"Dataset saved to {filename}")
    
    def generate_and_save(self, num_sentences=2000, min_tokens=20000, filename="hindi_pos_dataset.txt"):
        """Generate and save a dataset"""
        sentences = self.generate_dataset(num_sentences, min_tokens)
        self.save_dataset(sentences, filename)
        return sentences

# Usage
generator = HindiPOSDatasetGenerator()
sentences = generator.generate_and_save(num_sentences=2500, min_tokens=25000)

# Print some statistics
print(f"Total unique words: {len(generator.unique_words)}")
print(f"Total tokens: {generator.token_count}")

# Print a few sample sentences
print("\nSample sentences:")
for i in range(5):
    idx = random.randint(0, len(sentences) - 1)
    print(f"Sentence {idx}:")
    for word, tag in sentences[idx]:
        print(f"{word} {tag}", end=" | ")
    print("\n")
