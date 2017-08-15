import pickle
import praw

def preprocess_and_return_pairs_and_table():
    def flatten(submission, tree):
        result = []
        for comment, _ in tree:
            result.append((submission.title.encode('utf8'), comment.body.encode('utf8')))
        return result

    reddit_tree = pickle.load(open('top1000.p', 'rb'))
    input_output_pairs = []
    for submission, tree in reddit_tree:
        input_output_pairs.extend(flatten(submission, tree))

    #We have our input output pairs. Encode them to integers
    char_id = 2
    word_lookup = {}

    for thread, reply in input_output_pairs:
        for char in thread:
            if char not in word_lookup:
                word_lookup[char_id] = char
                word_lookup[char] = char_id
        for char in reply:
            if char not in word_lookup:
                word_lookup[char_id] = char
                word_lookup[char] = char_id

    return input_output_pairs, word_lookup