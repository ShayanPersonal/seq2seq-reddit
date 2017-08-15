#!/usr/bin/env python
from gensim.models import KeyedVectors
import numpy as np
import tensorflow as tf
import helpers
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import sys

print("Loading dataset")
word_vectors = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
in_out_pairs, word_lookup, vocab_size = helpers.preprocess_pairs_word('top1000.p')
vector_matrix = np.zeros((vocab_size, 300), dtype=np.float32)
vector_matrix[1] = np.array([0.5]*300)
print("Creating vector matrix")
for i in range(2, vocab_size):
    if word_lookup[i].lower() in word_vectors:
        vector_matrix[i] = word_vectors[word_lookup[i].lower()]
    else:
        vector_matrix[i] = np.array([0.5]*300)
del word_vectors
print("Loaded dataset")

tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

vocab_size = len(word_lookup)
input_embedding_size = 300

encoder_hidden_units = 50
decoder_hidden_units = encoder_hidden_units * 2

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_targets_length')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')


#Embeddings
print(vector_matrix.shape)
embeddings = tf.Variable(vector_matrix)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

#Encoder
encoder_cell_forward = LSTMCell(encoder_hidden_units)
encoder_cell_backwards = LSTMCell(encoder_hidden_units)

((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_forward,
                                    cell_bw=encoder_cell_backwards,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
    )

encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)

#Decoder
decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

decoder_lengths = decoder_targets_length

#Output projection
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input
    
    elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended

    finished = tf.reduce_all(elements_finished) # -> boolean scalar
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished, 
            input,
            state,
            output,
            loop_state)

def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

decoder_prediction = tf.argmax(decoder_logits, 2)

#Optimizer
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())


#Train on toy task
batch_size = 48
batches = helpers.batch_up(in_out_pairs, batch_size)

def next_feed():
    in_batch, out_batch = next(batches)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(in_batch)
    decoder_targets_, decoder_targets_lengths_ = helpers.batch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in out_batch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
        decoder_targets_length: decoder_targets_lengths_,
    }

loss_track = []

max_batches = 5000
batches_in_epoch = 100

saver = tf.train.Saver()

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(' '.join([word_lookup[word_id] for word_id in inp if word_id in word_lookup])))
                print('    predicted > {}'.format(' '.join([word_lookup[word_id] for word_id in pred if word_id in word_lookup])))
                if i >= 2:
                    save_path = saver.save(sess, "./saved/model.ckpt")
                    break
            print()

except KeyboardInterrupt:
    print('training interrupted')

import matplotlib.pyplot as plt
plt.plot(loss_track)
plt.show()
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))