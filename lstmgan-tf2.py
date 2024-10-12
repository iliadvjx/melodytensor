import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
import utils
import midi_statistics
import argparse
import os
import time
import datetime
import shutil
import mmd

# تعریف ثوابت
REG_CONSTANT = 0.1
DISABLE_FEED_PREVIOUS = True
GAUSSIAN_NOISE = False
UNIDIRECTIONAL_D = True
BIDIRECTIONAL_G = False
RANDOM_INPUT_SCALE = 10.0
ATTENTION_LENGTH = 0
FEED_COND_D = True
DROPOUT_KEEP_PROB = 0.9
D_LR_FACTOR = 0.3
LEARNING_RATE = 0.1
PRETRAINING_D = False
LR_DECAY = 0.98
DATA_MATRIX = "data/processed_dataset_matrices/full_data_matrix.npy"
TRAIN_DATA_MATRIX = "data/processed_dataset_matrices/train_data_matrix.npy"
VALIDATE_DATA_MATRIX = "data/processed_dataset_matrices/valid_data_matrix.npy"
TEST_DATA_MATRIX = "data/processed_dataset_matrices/test_data_matrix.npy"
LOSS_WRONG_D = False
INPUT_VECTOR = "input_vector.npy"
MULTI = True
CONDITION = True
SONGLENGTH = 20
PRETRAINING_EPOCHS = 1
NUM_MIDI_FEATURES = 3
NUM_SYLLABLE_FEATURES = 20
NUM_SONGS = 5000000
BATCH_SIZE = 512
REG_SCALE = 1.0
TRAIN_RATE = 0.8
VALIDATION_RATE = 0.1
HIDDEN_SIZE_G = 400
HIDDEN_SIZE_D = 400
NUM_LAYERS_G = 2
NUM_LAYERS_D = 2
ADAM = False
DISABLE_L2_REG = True
MAX_GRAD_NORM = 5.0
FEATURE_MATCHING = False
MAX_EPOCH = 400
EPOCHS_BEFORE_DECAY = 30
SONGLENGTH_CEILING = 20

class RNNGAN(tf.keras.Model):
    def __init__(self, num_song_features, num_meta_features, songlength, conditioning='multi'):
        super(RNNGAN, self).__init__()
        self.songlength = songlength
        self.num_song_features = num_song_features
        self.num_meta_features = num_meta_features
        self.conditioning = conditioning

        # لایه‌های Generator
        self.generator_cells = [tf.keras.layers.LSTMCell(HIDDEN_SIZE_G) for _ in range(NUM_LAYERS_G)]
        self.generator_rnn = tf.keras.layers.RNN(self.generator_cells, return_sequences=True, return_state=True)
        self.generator_dense = tf.keras.layers.Dense(num_song_features)

        # لایه‌های Discriminator
        self.discriminator_cells = [tf.keras.layers.LSTMCell(HIDDEN_SIZE_D) for _ in range(NUM_LAYERS_D)]
        self.discriminator_rnn = tf.keras.layers.RNN(self.discriminator_cells, return_sequences=True, return_state=True)
        self.discriminator_dense = tf.keras.layers.Dense(1, activation='sigmoid')

        # بهینه‌سازها
        if ADAM:
            self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE * D_LR_FACTOR)
        else:
            self.generator_optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
            self.discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE * D_LR_FACTOR)

        # تابع هزینه
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def generate(self, batch_size, conditioning_data=None, training=True):
        random_input = tf.random.uniform(
            shape=(batch_size, self.songlength, int(RANDOM_INPUT_SCALE * self.num_song_features)),
            minval=0.0, maxval=1.0, dtype=tf.float32)

        # افزودن شرط در صورت نیاز
        if self.conditioning == 'multi' and conditioning_data is not None:
            generator_input = tf.concat([random_input, conditioning_data], axis=-1)
        else:
            generator_input = random_input

        if DROPOUT_KEEP_PROB < 1.0:
            generator_input = tf.nn.dropout(generator_input, rate=1 - DROPOUT_KEEP_PROB)

        gen_output, *gen_states = self.generator_rnn(generator_input, training=training)
        generated_features = self.generator_dense(gen_output)
        return generated_features

    def discriminate(self, song_data, conditioning_data=None, training=True):
        # افزودن شرط در صورت نیاز
        if self.conditioning == 'multi' and conditioning_data is not None and FEED_COND_D:
            discriminator_input = tf.concat([song_data, conditioning_data], axis=-1)
        else:
            discriminator_input = song_data

        if DROPOUT_KEEP_PROB < 1.0:
            discriminator_input = tf.nn.dropout(discriminator_input, rate=1 - DROPOUT_KEEP_PROB)

        disc_output, *disc_states = self.discriminator_rnn(discriminator_input, training=training)
        decision = self.discriminator_dense(disc_output)
        # میانگین‌گیری بر روی زمان
        decision = tf.reduce_mean(decision, axis=1)
        return decision

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output, wrong_output=None):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        if wrong_output is not None and LOSS_WRONG_D:
            wrong_loss = self.cross_entropy(tf.zeros_like(wrong_output), wrong_output)
            total_loss = real_loss + fake_loss + wrong_loss
        else:
            total_loss = real_loss + fake_loss
        return total_loss

# تابع آموزش یک گام
def train_step(model, song_data, conditioning_data, wrong_conditioning_data, batch_size, pretraining=False):
    with tf.GradientTape(persistent=True) as tape:
        generated_songs = model.generate(batch_size, conditioning_data)

        real_output = model.discriminate(song_data, conditioning_data)
        fake_output = model.discriminate(generated_songs, conditioning_data)
        if wrong_conditioning_data is not None and LOSS_WRONG_D:
            wrong_output = model.discriminate(song_data, wrong_conditioning_data)
        else:
            wrong_output = None

        if pretraining:
            gen_loss = tf.reduce_mean(tf.square(song_data - generated_songs))
            disc_loss = None
        else:
            gen_loss = model.generator_loss(fake_output)
            disc_loss = model.discriminator_loss(real_output, fake_output, wrong_output)

        # افزودن جریمه L2 در صورت نیاز
        if not DISABLE_L2_REG:
            reg_losses = model.losses  # این شامل ضرایب L2 از لایه‌ها می‌شود
            reg_loss = REG_CONSTANT * tf.add_n(reg_losses)
            gen_loss += reg_loss
            if disc_loss is not None:
                disc_loss += reg_loss

    # به‌روزرسانی گرادیان‌ها
    if pretraining:
        gradients_of_generator = tape.gradient(gen_loss, model.generator_rnn.trainable_variables + model.generator_dense.trainable_variables)
        model.generator_optimizer.apply_gradients(zip(gradients_of_generator, model.generator_rnn.trainable_variables + model.generator_dense.trainable_variables))
    else:
        gradients_of_generator = tape.gradient(gen_loss, model.generator_rnn.trainable_variables + model.generator_dense.trainable_variables)
        gradients_of_discriminator = tape.gradient(disc_loss, model.discriminator_rnn.trainable_variables + model.discriminator_dense.trainable_variables)

        # اعمال کلیپینگ گرادیان
        gradients_of_generator, _ = tf.clip_by_global_norm(gradients_of_generator, MAX_GRAD_NORM)
        gradients_of_discriminator, _ = tf.clip_by_global_norm(gradients_of_discriminator, MAX_GRAD_NORM)

        model.generator_optimizer.apply_gradients(zip(gradients_of_generator, model.generator_rnn.trainable_variables + model.generator_dense.trainable_variables))
        model.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, model.discriminator_rnn.trainable_variables + model.discriminator_dense.trainable_variables))

    return gen_loss, disc_loss

# حلقه اصلی آموزش
def train_model(dataset, model, epochs, train_data, validate_data, test_data):
    best_mmd_overall = np.inf
    best_epoch = 0

    # ایجاد مسیر برای ذخیره مدل‌ها
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=model.generator_optimizer,
                                     discriminator_optimizer=model.discriminator_optimizer,
                                     model=model)

    for epoch in range(epochs):
        start = time.time()

        # تنظیم نرخ یادگیری
        if epoch >= EPOCHS_BEFORE_DECAY:
            lr_decay = LR_DECAY ** (epoch - EPOCHS_BEFORE_DECAY)
            model.generator_optimizer.learning_rate = LEARNING_RATE * lr_decay
            model.discriminator_optimizer.learning_rate = LEARNING_RATE * D_LR_FACTOR * lr_decay

        print('شروع دوره {} با نرخ یادگیری: {:.5f}'.format(epoch + 1, model.generator_optimizer.learning_rate.numpy()))

        gen_losses = []
        disc_losses = []

        for batch_data in dataset:
            # استخراج داده‌های ورودی
            if CONDITION:
                song_data, conditioning_data, wrong_conditioning_data = batch_data
            else:
                song_data = batch_data
                conditioning_data = None
                wrong_conditioning_data = None

            batch_size = song_data.shape[0]

            # پیش‌آموزش
            if epoch < PRETRAINING_EPOCHS:
                gen_loss, _ = train_step(model, song_data, conditioning_data, wrong_conditioning_data, batch_size, pretraining=True)
                gen_losses.append(gen_loss.numpy())
            else:
                gen_loss, disc_loss = train_step(model, song_data, conditioning_data, wrong_conditioning_data, batch_size)
                gen_losses.append(gen_loss.numpy())
                disc_losses.append(disc_loss.numpy())

        end = time.time()
        print('زمان برای دوره {}: {:.2f} ثانیه'.format(epoch + 1, end - start))
        print('میانگین هزینه Generator: {:.5f}'.format(np.mean(gen_losses)))
        if disc_losses:
            print('میانگین هزینه Discriminator: {:.5f}'.format(np.mean(disc_losses)))

        # ذخیره مدل هر 15 دوره
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print("مدل در دوره {} ذخیره شد.".format(epoch + 1))

        # ارزیابی مدل با استفاده از MMD
        # تولید نمونه‌ها برای مجموعه اعتبارسنجی
        validation_songs = []
        for i in range(len(validate_data)):
            song_data = np.random.uniform(size=(1, SONGLENGTH, NUM_MIDI_FEATURES)).astype(np.float32)
            if CONDITION:
                conditioning_data = validate_data[i, NUM_MIDI_FEATURES * SONGLENGTH:].reshape(1, SONGLENGTH, NUM_SYLLABLE_FEATURES)
            else:
                conditioning_data = None

            generated_features = model.generate(1, conditioning_data, training=False)
            sample = generated_features.numpy().squeeze(axis=0)
            discretized_sample = utils.discretize(sample)
            discretized_sample = midi_statistics.tune_song(discretized_sample)
            validation_songs.append(discretized_sample)

        # محاسبه MMD
        val_gen_pitches = np.array([song[:, 0] for song in validation_songs])
        val_dat_pitches = validate_data[:, :NUM_MIDI_FEATURES * SONGLENGTH:NUM_MIDI_FEATURES]
        MMD_pitch = mmd.Compute_MMD(val_gen_pitches, val_dat_pitches)
        print("MMD pitch:", MMD_pitch)

        val_gen_duration = np.array([song[:, 1] for song in validation_songs])
        val_dat_duration = validate_data[:, 1:NUM_MIDI_FEATURES * SONGLENGTH:NUM_MIDI_FEATURES]
        MMD_duration = mmd.Compute_MMD(val_gen_duration, val_dat_duration)
        print("MMD duration:", MMD_duration)

        val_gen_rests = np.array([song[:, 2] for song in validation_songs])
        val_dat_rests = validate_data[:, 2:NUM_MIDI_FEATURES * SONGLENGTH:NUM_MIDI_FEATURES]
        MMD_rest = mmd.Compute_MMD(val_gen_rests, val_dat_rests)
        print("MMD rest:", MMD_rest)

        MMD_overall = MMD_pitch + MMD_duration + MMD_rest
        print("MMD overall:", MMD_overall)
        # ... [کدهای قبلی شما]

# مقداردهی اولیه لیست‌ها برای ذخیره معیارها
        midi_numbers_span_list = []
        repetitions_3_list = []
        repetitions_2_list = []
        unique_midi_numbers_list = []
        notes_without_rest_list = []
        average_rest_value_list = []
        song_length_list = []

# حلقه ارزیابی
        print("EVAL..")
        validation_songs = []
        for i in range(len(validate_data)):
            song_data = np.random.uniform(size=(1, SONGLENGTH, NUM_MIDI_FEATURES)).astype(np.float32)
            if CONDITION:
                conditioning_data = validate_data[i, SONGLENGTH * NUM_MIDI_FEATURES:].reshape(1, SONGLENGTH, NUM_SYLLABLE_FEATURES)
                conditioning_data = tf.convert_to_tensor(conditioning_data, dtype=tf.float32)
            else:
                conditioning_data = None

            generated_features = model.generate(1, conditioning_data, training=False)
            sample = generated_features.numpy().squeeze(axis=0)  # شکل: [SONGLENGTH, NUM_MIDI_FEATURES]
            discretized_sample = utils.discretize(sample)
            discretized_sample = midi_statistics.tune_song(discretized_sample)
            discretized_sample = np.array(discretized_sample)
            validation_songs.append(discretized_sample)

            # محاسبه معیارها برای ملودی تولیدشده
            midi_numbers = discretized_sample[:, 0]  # فرض بر این است که ستون اول شماره نت‌های MIDI است
            rest_values = discretized_sample[:, 2]   # فرض بر این است که ستون سوم مقادیر سکوت است

            # بازه شماره نت‌های MIDI
            midi_span = midi_numbers.max() - midi_numbers.min()
            midi_numbers_span_list.append(midi_span)

            # تعداد تکرارهای ۳ نتی
            repetitions_3 = midi_statistics.count_repetitions(midi_numbers, n=3)
            repetitions_3_list.append(repetitions_3)

            # تعداد تکرارهای ۲ نتی
            repetitions_2 = midi_statistics.count_repetitions(midi_numbers, n=2)
            repetitions_2_list.append(repetitions_2)

            # تعداد نت‌های MIDI منحصربه‌فرد
            unique_midi_numbers = len(np.unique(midi_numbers))
            unique_midi_numbers_list.append(unique_midi_numbers)

            # تعداد نت‌های بدون سکوت
            notes_without_rest = np.sum(rest_values == 0)
            notes_without_rest_list.append(notes_without_rest)

            # میانگین مقدار سکوت در ملودی
            average_rest = np.mean(rest_values)
            average_rest_value_list.append(average_rest)

            # طول ملودی
            song_length = len(midi_numbers)
            song_length_list.append(song_length)

        # پس از پردازش همه ملودی‌ها، میانگین معیارها را محاسبه کنید
        avg_midi_span = np.mean(midi_numbers_span_list)
        avg_repetitions_3 = np.mean(repetitions_3_list)
        avg_repetitions_2 = np.mean(repetitions_2_list)
        avg_unique_midi_numbers = np.mean(unique_midi_numbers_list)
        avg_notes_without_rest = np.mean(notes_without_rest_list)
        avg_average_rest_value = np.mean(average_rest_value_list)
        avg_song_length = np.mean(song_length_list)

        # نمایش معیارها
        print(f"Average MIDI Numbers Span: {avg_midi_span:.1f}")
        print(f"Average 3-MIDI Numbers Repetitions: {avg_repetitions_3:.1f}")
        print(f"Average 2-MIDI Numbers Repetitions: {avg_repetitions_2:.1f}")
        print(f"Average Number of Unique MIDI: {avg_unique_midi_numbers:.1f}")
        print(f"Average Number of Notes Without Rest: {avg_notes_without_rest:.1f}")
        print(f"Average Rest Value Within Song: {avg_average_rest_value:.1f}")
        print(f"Average Song Length: {avg_song_length:.1f}")

        # ... [ادامه کدهای شما]

        # ذخیره بهترین مدل بر اساس MMD overall
        if MMD_overall < best_mmd_overall:
            print("بهترین مدل در دوره {} با MMD overall: {:.5f}".format(epoch + 1, MMD_overall))
            best_mmd_overall = MMD_overall
            best_epoch = epoch + 1
            model.save('./saved_gan_models/best_model')

    print("بهترین مدل در دوره {} با MMD overall: {:.5f}".format(best_epoch, best_mmd_overall))

def main():
    # بارگذاری داده‌ها
    train = np.load(TRAIN_DATA_MATRIX)
    validate = np.load(VALIDATE_DATA_MATRIX)
    test = np.load(TEST_DATA_MATRIX)

    if not CONDITION:
        train = train[:, :SONGLENGTH * NUM_MIDI_FEATURES]
        validate = validate[:, :SONGLENGTH * NUM_MIDI_FEATURES]
        test = test[:, :SONGLENGTH * NUM_MIDI_FEATURES]

    print("Training set: ", train.shape[0], " songs")
    print("Validation set: ", validate.shape[0], " songs")
    print("Test set: ", test.shape[0], " songs")

    # ایجاد dataset با استفاده از tf.data
    if CONDITION:
        def generator():
            for i in range(len(train)):
                song_data = train[i, :SONGLENGTH * NUM_MIDI_FEATURES].reshape(SONGLENGTH, NUM_MIDI_FEATURES)
                conditioning_data = train[i, SONGLENGTH * NUM_MIDI_FEATURES:].reshape(SONGLENGTH, NUM_SYLLABLE_FEATURES)
                # ایجاد داده‌های شرطی نادرست
                wrong_index = np.random.randint(len(train))
                wrong_conditioning_data = train[wrong_index, SONGLENGTH * NUM_MIDI_FEATURES:].reshape(SONGLENGTH, NUM_SYLLABLE_FEATURES)
                yield song_data.astype(np.float32), conditioning_data.astype(np.float32), wrong_conditioning_data.astype(np.float32)
        output_types = (tf.float32, tf.float32, tf.float32)
        dataset = tf.data.Dataset.from_generator(generator, output_types=output_types)
    else:
        def generator():
            for i in range(len(train)):
                song_data = train[i, :SONGLENGTH * NUM_MIDI_FEATURES].reshape(SONGLENGTH, NUM_MIDI_FEATURES)
                yield song_data.astype(np.float32)
        output_types = tf.float32
        dataset = tf.data.Dataset.from_generator(generator, output_types=output_types)

    dataset = dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE)

    # مقداردهی اولیه مدل
    model = RNNGAN(num_song_features=NUM_MIDI_FEATURES, num_meta_features=NUM_SYLLABLE_FEATURES, songlength=SONGLENGTH, conditioning='multi')

    # بارگذاری چک‌پوینت در صورت وجود
    checkpoint_dir = './training_checkpoints'
    checkpoint = tf.train.Checkpoint(generator_optimizer=model.generator_optimizer,
                                     discriminator_optimizer=model.discriminator_optimizer,
                                     model=model)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print("مدل از چک‌پوینت {} بازیابی شد.".format(latest_checkpoint))

    # آموزش مدل
    train_model(dataset, model, epochs=MAX_EPOCH, train_data=train, validate_data=validate, test_data=test)

if __name__ == '__main__':
    main()
