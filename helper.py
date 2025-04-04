def main(language_index, test_file_path):
    import codecs
    import os
    import sys
    import time

    start_time = time.time()

    # Path of training files
    filepath = ["./data/hindi_training_unsupervised.txt", "./data/telugu_training_unsupervised.txt", "./data/kannada_training_unsupervised.txt", "./data/tamil_training_unsupervised.txt"]
    languages = ["hindi", "telugu", "kannada", "tamil"]
    exclude = ["", "", "START", "END"]
    wordtypes = []
    tagscount = []

    # Open training file to read the contents
    f = codecs.open(filepath[language_index], 'r', encoding='utf-8')
    file_contents = f.readlines()

    # Get all possible tags from the training file
    tags = []
    for x in range(len(file_contents)):
        line = file_contents[x].strip().split(' ')
        if len(line) > 1 and line[1] not in tags and line[1] not in exclude:
            tags.append(line[1])

    # Initialize count of each tag to Zero's
    for x in range(len(tags)):
        tagscount.append(0)

    # Calculate count of each tag in the training corpus and also the wordtypes in the corpus
    for x in range(len(file_contents)):
        line = file_contents[x].strip().split(' ')
        for i, word in enumerate(line):
            if i == 0:
                if word not in wordtypes and word not in exclude:
                    wordtypes.append(word)
            else:
                if word in tags and word not in exclude:
                    tagscount[tags.index(word)] += 1

    f.close()

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

    # Open training file to update emission and transmission matrix
    f = codecs.open(filepath[language_index], 'r', encoding='utf-8')
    file_contents = f.readlines()

    # Update emission and transmission matrix with appropriate counts
    row_id = -1
    for x in range(len(file_contents)):
        line = file_contents[x].strip().split(' ')
        if line[0] not in exclude:
            col_id = wordtypes.index(line[0])
            prev_row_id = row_id
            row_id = tags.index(line[1])
            emission_matrix[row_id][col_id] += 1
            if prev_row_id != -1:
                transmission_matrix[prev_row_id][row_id] += 1
        else:
            row_id = -1

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

    print(time.time() - start_time, "seconds for training")

    # Start of Testing Phase
    start_time = time.time()

    # Open the testing file to read test sentences
    file_test = codecs.open(test_file_path, 'r', encoding='utf-8')
    test_input = file_test.readlines()

    # Declare variables for test words and pos tags
    test_words = []
    pos_tags = []

    # Create an output file to write the output tags for each sentences
    file_output = codecs.open("./output/" + languages[language_index] + "_tags_unsupervised.txt", 'w', 'utf-8')
    file_output.close()

    # For each line POS tags are computed
    for j in range(len(test_input)):
        test_words = []
        pos_tags = []
        line = test_input[j].strip().split(' ')

        for word in line:
            test_words.append(word)
            pos_tags.append(-1)

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
                    tag_index = tags.index(tags[y])
                    emission = emission_matrix[tag_index][word_index]
                else:
                    emission = 0.001

                if x > 0:
                    max_val, viterbi_path[y][x] = max_connect(x, y, viterbi_matrix, emission, transmission_matrix)
                else:
                    max_val = 1

                viterbi_matrix[y][x] = emission * max_val

        # Identify the max probability in last column i.e. best tag for last word in test sentence
        maxval = -999999
        maxs = -1
        for x in range(len(tags)):
            if viterbi_matrix[x][len(test_words)-1] > maxval:
                maxval = viterbi_matrix[x][len(test_words)-1]
                maxs = x

        # Backtrack and identify best tags for each words
        for x in range(len(test_words)-1, -1, -1):
            pos_tags[x] = maxs
            maxs = viterbi_path[maxs][x]

        # Print output to the file.
        file_output = codecs.open("./output/" + languages[language_index] + "_tags_unsupervised.txt", 'a', 'utf-8')
        for i, x in enumerate(pos_tags):
            file_output.write(test_words[i] + "_" + tags[x] + " ")
        file_output.write(" ._.\n")

    f.close()
    file_output.close()
    file_test.close()

    print(time.time() - start_time, "seconds for testing 100 Sentences")
    print("\nKindly check ./output/" + languages[language_index] + "_tags_unsupervised.txt for POS tags")

def max_connect(x, y, viterbi_matrix, emission, transmission_matrix):
    max_val = -99999
    path = -1

    for k in range(len(viterbi_matrix)):
        val = viterbi_matrix[k][x-1] * transmission_matrix[k][y]
        if val * emission > max_val:
            max_val = val
            path = k

    return max_val, path

if __name__ == "__main__":
    print("This is a helper file and should not be run directly.")
    print("Please use unsupervised.py or supervised.py instead.")
