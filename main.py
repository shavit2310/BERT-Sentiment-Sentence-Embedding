#!/usr/bin/python3

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from simcse import SimCSE

import argparse
import logging as log
import json
import sys
import os

# Import the model and tokenizer. The package will download the model automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

# Declare constants
# For full run results
INPUT_SENTENCES_FILE = 'full_sentences.json'  # Optional files are available in data directory
SENTENCES_INPUT_FILE_PATH = 'data\\full_sentences.json'

# For specific similarity check of two sentences
OUTPUT_SENTENCES_FILE = 'Sentences_similarity_log.csv'
SENTENCES_OUTPUT_FILE_PATH = 'output\\Sentences_similarity_log.csv'
HEADER_ALL = '\nSentence similarity log'

OUTPUT_SPECIFIC_SIMILARITY_FILE = 'Specific_sentences_similarity_grade.csv'
SPECIFIC_SIMILARITY_SENTENCES_OUTPUT_FILE_PATH = 'output\\Specific_sentences_similarity_grade.csv'
HEADER_SPECIFIC = '\nCheck specific sentences similarity score'

# Similarity evaluation score
THRESHOLDS = 0.7620


def parse_args(sentences=None):
    # Handle standard input parameters

    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", help="Prints the supplied argument")

    parsed_args = parser.parse_args(sys.argv[1:])

    return parsed_args


def validate_command_line_input_parameters(input_value):
    # Handle new given json file

    if os.path.exists(input_value):
        # Verify specified path contains the existing file
        file_name, file_extension = os.path.splitext(input_value)
        if file_extension == '.json':
            return input_value

    return None


def input_source(argv):
    # Update input file source from standard input

    input_sentences = None

    # Input file from standard input
    if argv.sentences:
        input_sentences = validate_command_line_input_parameters(argv.sentences)

    # When no failure on receiving json file from standard error:
    # no arguments given or path problem or file is not json type
    if not input_sentences:
        # Set test file
        log.warning('File name is not found, or not json file type, the template will be used')

        main_dir = os.path.dirname(os.path.abspath(INPUT_SENTENCES_FILE))
        input_sentences = os.path.join(main_dir, SENTENCES_INPUT_FILE_PATH)

    return input_sentences


def load_sentences(input_sentences):
    # Initiate a list with the input sentences
    sentences = []
    with open(input_sentences) as file:
        for line in json.load(file):
            if len(line.strip()) > 0:
                sentences.append(line.strip())

    return sentences


def build_sentences_corpus(sentences):
    # build index for a group of sentences

    model.build_index(sentences)


def user_sentence():
    # Input a sentence from user

    user_preferences = input("Enter a phrase: ")

    if not user_preferences:
        log.warning('No sentence received, can not perform search')

    return user_preferences


def encode_user_phrase(user_preference):
    # Encode specific sentence

    return model.encode(user_preference)


def embedding_sentences(inputs):
    # encoding sentences into embeddings

    model1 = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model1(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    return embeddings


def evaluate_sentences_similarities():
    # Grade two sentence similarity

    model1 = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    texts = []
    final_list = []

    texts.append(user_sentence())
    texts.append(user_sentence())

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    embeddings = embedding_sentences(inputs)

    for i in range(1, len(embeddings)):
        # Cosine similarities are in [-1, 1]. Higher means more similar
        most_similar = 1 - cosine(embeddings[i], embeddings[i - 1])

        final_list.append(texts[i] + " : " + texts[i - 1] + ' scored: ' + str(most_similar))

    # Print and save to file the specific similarity check
    main_dir = os.path.dirname(os.path.abspath(OUTPUT_SPECIFIC_SIMILARITY_FILE))
    input_specific_sentences = os.path.join(main_dir, SPECIFIC_SIMILARITY_SENTENCES_OUTPUT_FILE_PATH)
    review_outputs(final_list, input_specific_sentences, HEADER_SPECIFIC)


def find_similarity(user_preference):
    # search a sentence among the encoded group of sentences

    result = model.search(user_preference)

    return result


def score_to_boolean_rate(input_text, output_text):
    # Rate scored similarity to: 1 - pass THRESHOLD, 0 - not pass THRESHOLD

    pairs_similarity = []

    if input_text is None:
        log.warning('No sentence received, can not rate the similarity')

    for current_sentence in output_text:
        if list(current_sentence)[1] >= THRESHOLDS:
            # similarity found, Optional multiple similarities sentences
            pairs_similarity.append("\'{input}\', is similar to \'{output}\', rated score is: 1".
                                    format(input=input_text, output=list(current_sentence)[0]))
        else:
            # The context founded, was not pass in the corpus
            pairs_similarity.append("A \'{input}\' sentence was not found, in the thresholds bounders, "
                                    "rated score is: 0".format(input=input_text))

    return pairs_similarity


def review_outputs(final_texts_by_score, archive, header=None):
    # Save output to a file

    archive = open(archive, "a")
    archive.write(header)
    for match in final_texts_by_score:
        archive.write('\n')
        if isinstance(match, int) or isinstance(match, float):
            archive.write(str(match))
        else:
            archive.write(''.join(match))
        print(match)

    archive.close()


def main(argv):
    # Handle standard input parameters
    args = parse_args(argv)

    # Update input file source
    input_sentences = input_source(args)

    # Load sentences from device
    sentences = load_sentences(input_sentences)

    # Build index for the input sentences - encode the group of sentences
    build_sentences_corpus(sentences)

    # User sentence input
    user_preference = user_sentence()

    # Find most similar sentence in the encoded sentences from the loaded file
    most_similar = find_similarity(user_preference)

    # Rate similarity pair sentences to pass (=1) not pass(=0) thresholds
    final_texts_by_score = score_to_boolean_rate(user_preference, most_similar)

    # Calculate cosine similarities
    # Turn to Note due it was not required in the assignment
    # evaluate_sentences_similarities()

    # Printout and archives final results
    main_dir = os.path.dirname(os.path.abspath(OUTPUT_SENTENCES_FILE))
    input_sentences = os.path.join(main_dir, SENTENCES_OUTPUT_FILE_PATH)
    review_outputs(final_texts_by_score, input_sentences, HEADER_ALL)


if __name__ == '__main__':
    main(sys.argv[1:])
