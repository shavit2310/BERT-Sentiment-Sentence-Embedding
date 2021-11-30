#!/usr/bin/python3

import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

import argparse
import logging as log
import json
import sys
import os
import numpy

# Declare constants
# Input sentences as corpus
INPUT_SENTENCES_FILE = 'full_sentences.json'  # Optional files are available in data directory
SENTENCES_INPUT_FILE_PATH = 'data\\full_sentences.json'

# Output for each check done
OUTPUT_SENTENCES_FILE = 'Sentences_similarity_log.csv'
SENTENCES_OUTPUT_FILE_PATH = 'output\\Sentences_similarity_log.csv'
HEADER = '\nSentence similarity log'

# Similarity evaluation score
THRESHOLDS = 0.7620

# Import the model and tokenizer. The package will download the model automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model1 = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")


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


def tokenize_input(sentence):
    # Convert text to transformer-readable token IDs.

    return tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")


def encoding(sentences):
    # Replace the build and encode sentence
    batch_size = 64
    single_sentence = False

    # For Usr input sentence
    if isinstance(sentences, str):
        sentences = [sentences]
        single_sentence = True

    # start embedding
    embedding_list = []
    with torch.no_grad():
        total_batch = len(sentences) // batch_size + (1 if len(sentences) % batch_size > 0 else 0)
        for batch_id in range(total_batch):
            inputs = tokenize_input(sentences[batch_id * batch_size:(batch_id + 1) * batch_size])
            # truncate to tuple
            inputs = {k: v for k, v in inputs.items()}

            # Start embedding
            embeddings = model1(**inputs, return_dict=True).pooler_output
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            embedding_list.append(embeddings)
            
    embeddings = torch.cat(embedding_list, 0)

    # For User input
    if single_sentence:
        embeddings = embeddings[0]

    # Convert data type to numpy
    if not isinstance(embeddings, numpy.ndarray):
        embeddings = embeddings.numpy()

    # Save the encoded sentences to sentences corpus
    if not single_sentence:
        log.info("Building index...")
        model1.index = {"sentences": sentences}
        model1.index["index"] = embeddings

    return embeddings


def user_sentence():
    # User input sentence

    user_preferences = input("Enter a phrase: ")

    if not user_preferences:
        log.warning('No sentence received, can not perform search')

    return user_preferences


def evaluate_cousin_score(query_vectors, corpus_vectors):
    # returns an N*M similarity array: format of: index in list: score

    return cosine_similarity(query_vectors, corpus_vectors).tolist()


def pull_passed_thresholds(similarities):
    # Return the index and score of sentences who their score pass the THRESHOLDS

    id_and_score_pass = []
    for sentence_index, sentence_score in enumerate(similarities):
        if sentence_score >= THRESHOLDS:
            # Verify score upper the the set THRESHOLDS
            id_and_score_pass.append((sentence_index, sentence_score))

    return id_and_score_pass


def generated_text_of_passed_thresholds(id_and_score_pass):
    # From the index, retrieve sentence text, format of: text: score

    return [(model1.index["sentences"][idx], score) for idx, score in id_and_score_pass]


def find_similarity(to_search_embeddings, corpus_embeddings):
    # search a sentence among the encoded group of sentences

    pass_results = []
    id_and_score_pass = []
    query_vectors = to_search_embeddings
    corpus_vectors = corpus_embeddings

    # Handle batch of sentences or single sentence
    single_query, single_corpus = len(query_vectors.shape) == 1, len(corpus_vectors.shape) == 1
    if single_query:
        query_vectors = query_vectors.reshape(1, -1)
    if single_corpus:
        corpus_vectors = corpus_vectors.reshape(1, -1)

    # Retrieve the similarity score of each corpus's sentence to the searched one
    similarities = evaluate_cousin_score(query_vectors, corpus_vectors)

    if single_query:
        id_and_score_pass = pull_passed_thresholds(similarities[0])
        if single_corpus:
            id_and_score_pass = pull_passed_thresholds(float(similarities[0]))

    # Retrieve similarities found texts from vectors
    if id_and_score_pass:
        pass_results = generated_text_of_passed_thresholds(id_and_score_pass)

    return pass_results


def score_to_boolean_rate(user_preference, pass_similarity):
    # Rate scored similarity to: 1 - pass THRESHOLD, 0 - not pass THRESHOLD

    pairs_similarity = []

    if not len(pass_similarity):
        # The context founded, was not pass in the corpus
        pairs_similarity.append("\'{input}\' sentence was not found, in the thresholds bounders, "
                                "rated score is: 0".format(input=user_preference))
    else:
        for current_sentence in pass_similarity:
            # Optional more then one match
            # similarity found, Optional multiple similarities sentences
            pairs_similarity.append("\'{input}\', is similar to \'{output}\', rated score is: 1".
                                    format(input=user_preference, output=list(current_sentence)[0]))

    return pairs_similarity


def review_outputs(final_texts_by_score):
    # Save output to a file

    main_dir = os.path.dirname(os.path.abspath(OUTPUT_SENTENCES_FILE))
    input_sentences = os.path.join(main_dir, SENTENCES_OUTPUT_FILE_PATH)

    archive = open(input_sentences, "a")
    archive.write(HEADER)
    for match in final_texts_by_score:
        archive.write('\n')
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
    corpus_embeddings = encoding(sentences)

    # User sentence input
    user_preference = user_sentence()

    # Build index for the input sentences - encode the User input
    to_search_embeddings = encoding(user_preference)

    # Find most similar sentence in the encoded sentences from the loaded file
    pass_similarity = find_similarity(to_search_embeddings, corpus_embeddings)

    # Rate similarity pair sentences to pass (=1) not pass(=0) thresholds
    final_texts_by_score = score_to_boolean_rate(user_preference, pass_similarity)

    # Printout and archives final results
    review_outputs(final_texts_by_score)


if __name__ == '__main__':
    main(sys.argv[1:])
