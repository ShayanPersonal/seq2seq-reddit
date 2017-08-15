import numpy as np
import random

def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
    
    sequence_lengths = [min(50, len(seq))for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            if j == 50:
                break
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]

import pickle
import praw

def batch_up(in_out_pairs, batch_size):
    while True:
        yield zip(*random.sample(in_out_pairs, batch_size))

def preprocess_pairs_character(pickle_file):
    def flatten(submission, tree):
        result = []
        for comment, _ in tree:
            result.append((submission.title.encode('utf8'), comment.body.encode('utf8')))
        return result

    reddit_tree = pickle.load(open(pickle_file, 'rb'))
    input_output_pairs = []
    for submission, tree in reddit_tree:
        input_output_pairs.extend(flatten(submission, tree))

    '''
    #We have our input output pairs. Encode them to integers
    char_id = 2
    word_lookup = {}

    for thread, reply in input_output_pairs:
        for char in thread:
            if char not in word_lookup:
                word_lookup[char_id] = char
                word_lookup[char] = char_id
                char_id += 1
        for char in reply:
            if char not in word_lookup:
                word_lookup[char_id] = char
                word_lookup[char] = char_id
                char_id += 1

    in_out_pairs_to_int = []
    for thread, reply in input_output_pairs:
        for char in thread:
            print(char)
        thread_int = [word_lookup[char] for char in thread]
        reply_int = [word_lookup[reply] for char in reply]
        in_out_pairs_to_int.append((thread_int, reply_int))
    '''
    in_out_pairs_to_int = []
    for thread, reply in input_output_pairs:
        thread_int = [char for char in thread]
        reply_int = [char for char in reply]
        in_out_pairs_to_int.append((thread_int, reply_int))

    return in_out_pairs_to_int #, word_lookup

def preprocess_pairs_word(pickle_file):
    def flatten(submission, tree):
        result = []
        for comment, _ in tree:
            result.append((submission.title.encode('utf8'), comment.body.encode('utf8')))
        return result

    reddit_tree = pickle.load(open(pickle_file, 'rb'))
    input_output_pairs = []
    for submission, tree in reddit_tree:
        input_output_pairs.extend(flatten(submission, tree))

    #We have our input output pairs. Encode them to integers
    from collections import Counter
    import re
    import string
    word_id = 2
    word_lookup = {}
    keepchars = string.ascii_letters + ' ' + '\n'

    for thread, reply in input_output_pairs:
        thread = ''.join([chr(c) for c in thread if chr(c) in keepchars])
        reply = ''.join([chr(c) for c in reply if chr(c) in keepchars])
        for word in thread.split():
            if word not in word_lookup:
                word_lookup[word_id] = word
                word_lookup[word] = word_id
                word_id += 1
        for word in reply.split():
            if word not in word_lookup:
                word_lookup[word_id] = word
                word_lookup[word] = word_id
                word_id += 1

    in_out_pairs_to_int = []
    for thread, reply in input_output_pairs:
        thread = ''.join([chr(c) for c in thread if chr(c) in keepchars])
        reply = ''.join([chr(c) for c in reply if chr(c) in keepchars])
        
        thread_int = [word_lookup[word] for word in thread.split()]
        reply_int = [word_lookup[word] for word in reply.split()]
        in_out_pairs_to_int.append((thread_int, reply_int))

    return in_out_pairs_to_int, word_lookup, word_id