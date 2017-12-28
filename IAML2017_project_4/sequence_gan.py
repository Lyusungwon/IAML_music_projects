import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops.rnn_cell import BasicRNNCell, BasicLSTMCell, GRUCell
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from abc_io import ABC_READER
from tensorflow.python.platform import gfile
from time import strftime, localtime, time
import pickle
#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32  # embedding dimension
HIDDEN_DIM = 32  # hidden state dimension of lstm cell
SEQ_LENGTH = 80  # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 0  # supervise (maximum likelihood estimation) epochs
PRE_EPOCH_NUM_DIS = 0
SEED = 88
BATCH_SIZE = 5

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_hidden_dim = 32
dis_num_layers = 2
dis_dropout_keep_prob = 0.5
dis_batch_size = 5

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 5
rollout_num = 4
generated_num = BATCH_SIZE
ckpt_path = 'ckpt/-239'
accuracies = []


def generate_samples(sess, trainable_model, batch_size, generated_num):
    # Generate Samples
    tmp_negative_songs = []
    for _ in range(int(generated_num / batch_size)):
        tmp_negative_songs.extend(trainable_model.generate(sess))

    return tmp_negative_songs


def generate_samples_with_midi(sess, trainable_model, ABC_READER, rollout, batch_size, generated_num, iter, start_time):
    # Generate Samples
    full_samples = []
    for _ in range(int(generated_num / batch_size)):
        full_samples.extend(trainable_model.generate(sess))

    # Save Generated Samples As Midi
    # print generated_samples
    songs = ABC_READER.trans_trans_songs_to_raw(full_samples)
    entire_song = ''
    for idx, song in enumerate(songs):
        song = ''.join(song)
        song = "X:" + str(idx) + "\n" + song + "\n\n"
        entire_song += song

    # print entire_song

    dir = 'midi' + '/run-%02d%02d-%02d%02d' % tuple(start_time)[1:5] + '/rollout_iter%03d' % iter
    if not gfile.Exists(dir):
        gfile.MakeDirs(dir)

    abc_filename = dir + '/rollout_iter%03d_song.abc' % iter
    f = open(abc_filename, 'w')
    f.write(entire_song)
    f.close()
    # rollout.abc_to_midi(abc_filename)
    return full_samples


def target_loss(sess, target_lstm, data_loader):
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    abc_reader = ABC_READER(SEQ_LENGTH, 'abc_all_parsed.txt')
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE)  # For testing
    testset_data_loader = Gen_Data_loader(BATCH_SIZE)
    vocab_size = len(abc_reader.note_info_dict)
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, hidden_dim=dis_hidden_dim, num_layers=dis_num_layers, embedding_size=dis_embedding_dim)  # ilter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    gen_data_loader.create_batches(abc_reader.trans_abc_train)

    log = open('save/experiment-log.txt', 'w')
    saver = tf.train.Saver(max_to_keep=1000)
    saver.restore(sess, ckpt_path)

    #  pre-train generator
    print('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            likelihood_data_loader.create_batches(generate_samples(sess, generator, BATCH_SIZE, generated_num))
            val_loss = target_loss(sess, generator, likelihood_data_loader)
            print('pre-train epoch ', epoch, 'test_loss ', val_loss)
            buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(val_loss) + '\n'
            log.write(buffer)

    print('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for _ in range(PRE_EPOCH_NUM_DIS):

        dis_data_loader.load_train_data(abc_reader.trans_abc_train, generate_samples(sess, generator, BATCH_SIZE, generated_num))
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _, acc = sess.run([discriminator.train_op, discriminator.accuracy], feed)
                accuracies.append(acc)
    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    start_time = localtime(time())

    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, rollout_num, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:

            likelihood_data_loader.create_batches(generate_samples_with_midi(sess, generator, abc_reader, rollout, BATCH_SIZE, generated_num, total_batch, start_time))
            test_loss = target_loss(sess, generator, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print('total_batch: ', total_batch, 'test_loss: ', test_loss)
            log.write(buffer)
            generator.save_variables(sess, ckpt_path, total_batch, saver)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):
            dis_data_loader.load_train_data(abc_reader.trans_abc_train, generate_samples(sess, generator, BATCH_SIZE, generated_num))

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, acc = sess.run([discriminator.train_op, discriminator.accuracy], feed)
                    accuracies.append(acc)

    testset_data_loader.create_batches(abc_reader.trans_abc_test)
    test_loss = target_loss(sess, generator, testset_data_loader)
    print('test data test_loss ', test_loss)
    print(accuracies)
    buffer = '\tnll:\t' + str(val_loss) + '\n'
    log.write(buffer)
    log.close()

if __name__ == '__main__':
    main()
