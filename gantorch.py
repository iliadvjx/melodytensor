import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
BATCH_SIZE = 256
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

class RNNGAN(nn.Module):
    def __init__(self, num_song_features, num_meta_features, songlength, conditioning='multi'):
        super(RNNGAN, self).__init__()
        self.songlength = songlength
        self.num_song_features = num_song_features
        self.num_meta_features = num_meta_features
        self.conditioning = conditioning

        # لایه‌های Generator
        input_size_generator = int(RANDOM_INPUT_SCALE * num_song_features) + (num_meta_features if conditioning == 'multi' else 0)
        self.generator_lstm = nn.LSTM(
            input_size=input_size_generator,
            hidden_size=HIDDEN_SIZE_G,
            num_layers=NUM_LAYERS_G,
            batch_first=True
        )
        self.generator_dropout = nn.Dropout(p=1 - DROPOUT_KEEP_PROB)
        self.generator_dense = nn.Linear(HIDDEN_SIZE_G, num_song_features)

        # لایه‌های Discriminator
        input_size_discriminator = num_song_features + (num_meta_features if conditioning == 'multi' and FEED_COND_D else 0)
        self.discriminator_lstm = nn.LSTM(
            input_size=input_size_discriminator,
            hidden_size=HIDDEN_SIZE_D,
            num_layers=NUM_LAYERS_D,
            batch_first=True
        )
        self.discriminator_dropout = nn.Dropout(p=1 - DROPOUT_KEEP_PROB)
        self.discriminator_dense = nn.Linear(HIDDEN_SIZE_D, 1)
        self.discriminator_sigmoid = nn.Sigmoid()

        # توابع هزینه
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def generate(self, batch_size, conditioning_data=None, training=True):
        device = next(self.parameters()).device
        random_input = torch.rand(
            (batch_size, self.songlength, int(RANDOM_INPUT_SCALE * self.num_song_features)),
            device=device
        )

        if self.conditioning == 'multi' and conditioning_data is not None:
            generator_input = torch.cat([random_input, conditioning_data], dim=-1)
        else:
            generator_input = random_input

        if training and DROPOUT_KEEP_PROB < 1.0:
            generator_input = self.generator_dropout(generator_input)

        gen_output, _ = self.generator_lstm(generator_input)
        generated_features = self.generator_dense(gen_output)
        return generated_features

    def discriminate(self, song_data, conditioning_data=None, training=True):
        device = next(self.parameters()).device

        if self.conditioning == 'multi' and conditioning_data is not None and FEED_COND_D:
            discriminator_input = torch.cat([song_data, conditioning_data], dim=-1)
        else:
            discriminator_input = song_data

        if training and DROPOUT_KEEP_PROB < 1.0:
            discriminator_input = self.discriminator_dropout(discriminator_input)

        disc_output, _ = self.discriminator_lstm(discriminator_input)
        decision = self.discriminator_dense(disc_output)
        decision = self.discriminator_sigmoid(decision)
        # میانگین‌گیری بر روی زمان
        decision = decision.mean(dim=1).squeeze()
        return decision

    def generator_loss(self, fake_output):
        target = torch.ones_like(fake_output)
        return self.bce_loss(fake_output, target)

    def discriminator_loss(self, real_output, fake_output, wrong_output=None):
        real_target = torch.ones_like(real_output)
        fake_target = torch.zeros_like(fake_output)

        real_loss = self.bce_loss(real_output, real_target)
        fake_loss = self.bce_loss(fake_output, fake_target)

        if wrong_output is not None and LOSS_WRONG_D:
            wrong_target = torch.zeros_like(wrong_output)
            wrong_loss = self.bce_loss(wrong_output, wrong_target)
            total_loss = real_loss + fake_loss + wrong_loss
        else:
            total_loss = real_loss + fake_loss

        return total_loss

def train_step(model, song_data, conditioning_data, wrong_conditioning_data, batch_size, pretraining, generator_optimizer, discriminator_optimizer):
    device = next(model.parameters()).device
    model.train()
    song_data = song_data.to(device)
    if conditioning_data is not None:
        conditioning_data = conditioning_data.to(device)
    if wrong_conditioning_data is not None:
        wrong_conditioning_data = wrong_conditioning_data.to(device)

    # صفر کردن گرادیان‌ها
    generator_optimizer.zero_grad()
    if not pretraining:
        discriminator_optimizer.zero_grad()

    generated_songs = model.generate(batch_size, conditioning_data, training=True)

    if pretraining:
        gen_loss = model.mse_loss(generated_songs, song_data)
        gen_loss.backward()
        # کلیپینگ گرادیان
        torch.nn.utils.clip_grad_norm_(model.generator_lstm.parameters(), MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(model.generator_dense.parameters(), MAX_GRAD_NORM)
        generator_optimizer.step()
        disc_loss = None
    else:
        real_output = model.discriminate(song_data, conditioning_data)
        fake_output = model.discriminate(generated_songs.detach(), conditioning_data)
        if wrong_conditioning_data is not None and LOSS_WRONG_D:
            wrong_output = model.discriminate(song_data, wrong_conditioning_data)
        else:
            wrong_output = None

        disc_loss = model.discriminator_loss(real_output, fake_output, wrong_output)
        disc_loss.backward()
        # کلیپینگ گرادیان
        torch.nn.utils.clip_grad_norm_(model.discriminator_lstm.parameters(), MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(model.discriminator_dense.parameters(), MAX_GRAD_NORM)
        discriminator_optimizer.step()

        # به‌روزرسانی Generator
        generator_optimizer.zero_grad()
        generated_songs = model.generate(batch_size, conditioning_data, training=True)
        fake_output = model.discriminate(generated_songs, conditioning_data)
        gen_loss = model.generator_loss(fake_output)
        gen_loss.backward()
        # کلیپینگ گرادیان
        torch.nn.utils.clip_grad_norm_(model.generator_lstm.parameters(), MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(model.generator_dense.parameters(), MAX_GRAD_NORM)
        generator_optimizer.step()

    return gen_loss.item(), disc_loss.item() if disc_loss is not None else None

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

    # ایجاد dataset با استفاده از torch.utils.data.Dataset و DataLoader
    if CONDITION:
        class MusicDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.song_data = data[:, :SONGLENGTH * NUM_MIDI_FEATURES].reshape(-1, SONGLENGTH, NUM_MIDI_FEATURES)
                self.conditioning_data = data[:, SONGLENGTH * NUM_MIDI_FEATURES:].reshape(-1, SONGLENGTH, NUM_SYLLABLE_FEATURES)

            def __len__(self):
                return len(self.song_data)

            def __getitem__(self, idx):
                song = self.song_data[idx].astype(np.float32)
                condition = self.conditioning_data[idx].astype(np.float32)
                # ایجاد داده‌های شرطی نادرست
                wrong_idx = np.random.randint(len(self.song_data))
                wrong_condition = self.conditioning_data[wrong_idx].astype(np.float32)
                return song, condition, wrong_condition
        dataset = MusicDataset(train)
    else:
        class MusicDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.song_data = data[:, :SONGLENGTH * NUM_MIDI_FEATURES].reshape(-1, SONGLENGTH, NUM_MIDI_FEATURES)

            def __len__(self):
                return len(self.song_data)

            def __getitem__(self, idx):
                song = self.song_data[idx].astype(np.float32)
                return song
        dataset = MusicDataset(train)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # مقداردهی اولیه مدل
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("RUN ON ", device)
    model = RNNGAN(num_song_features=NUM_MIDI_FEATURES, num_meta_features=NUM_SYLLABLE_FEATURES, songlength=SONGLENGTH, conditioning='multi')
    model.to(device)

    # تنظیم بهینه‌سازها
    if ADAM:
        generator_optimizer = optim.Adam(list(model.generator_lstm.parameters()) + list(model.generator_dense.parameters()), lr=LEARNING_RATE)
        discriminator_optimizer = optim.Adam(list(model.discriminator_lstm.parameters()) + list(model.discriminator_dense.parameters()), lr=LEARNING_RATE * D_LR_FACTOR)
    else:
        generator_optimizer = optim.SGD(list(model.generator_lstm.parameters()) + list(model.generator_dense.parameters()), lr=LEARNING_RATE)
        discriminator_optimizer = optim.SGD(list(model.discriminator_lstm.parameters()) + list(model.discriminator_dense.parameters()), lr=LEARNING_RATE * D_LR_FACTOR)

    # تنظیم برنامه‌ریزی نرخ یادگیری
    def lr_lambda(epoch):
        if epoch < EPOCHS_BEFORE_DECAY:
            return 1.0
        else:
            return LR_DECAY ** (epoch - EPOCHS_BEFORE_DECAY + 1)

    generator_scheduler = optim.lr_scheduler.LambdaLR(generator_optimizer, lr_lambda=lr_lambda)
    discriminator_scheduler = optim.lr_scheduler.LambdaLR(discriminator_optimizer, lr_lambda=lr_lambda)

    # بارگذاری چک‌پوینت در صورت وجود
    checkpoint_dir = './training_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
    start_epoch = 0
    best_mmd_overall = np.inf
    best_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_mmd_overall = checkpoint['best_mmd_overall']
        best_epoch = checkpoint['best_epoch']
        print(f"مدل از چک‌پوینت دوره {start_epoch} بارگذاری شد.")

    # حلقه اصلی آموزش
    for epoch in range(start_epoch, MAX_EPOCH):
        start_time = time.time()

        # تنظیم نرخ یادگیری
        if epoch >= EPOCHS_BEFORE_DECAY:
            generator_scheduler.step()
            discriminator_scheduler.step()

        print(f"دوره {epoch + 1}/{MAX_EPOCH}, نرخ یادگیری: {generator_optimizer.param_groups[0]['lr']:.5f}")

        gen_losses = []
        disc_losses = []

        for batch_data in dataloader:
            if CONDITION:
                song_data, conditioning_data, wrong_conditioning_data = batch_data
            else:
                song_data = batch_data
                conditioning_data = None
                wrong_conditioning_data = None

            batch_size = song_data.size(0)

            gen_loss, disc_loss = train_step(
                model,
                song_data,
                conditioning_data,
                wrong_conditioning_data,
                batch_size,
                pretraining=(epoch < PRETRAINING_EPOCHS),
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer
            )
            gen_losses.append(gen_loss)
            if disc_loss is not None:
                disc_losses.append(disc_loss)

        end_time = time.time()
        print(f"زمان برای دوره {epoch + 1}: {end_time - start_time:.2f} ثانیه")
        print(f"میانگین هزینه Generator: {np.mean(gen_losses):.5f}")
        if disc_losses:
            print(f"میانگین هزینه Discriminator: {np.mean(disc_losses):.5f}")

        # ذخیره مدل هر ۱۵ دوره
        if (epoch + 1) % 15 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                'best_mmd_overall': best_mmd_overall,
                'best_epoch': best_epoch
            }, checkpoint_path)
            print(f"مدل در دوره {epoch + 1} ذخیره شد.")

        # ارزیابی مدل با استفاده از MMD
        model.eval()
        validation_songs = []
        with torch.no_grad():
            for i in range(len(validate)):
                song_data = torch.rand(1, SONGLENGTH, NUM_MIDI_FEATURES).to(device)
                if CONDITION:
                    conditioning_data = validate[i, SONGLENGTH * NUM_MIDI_FEATURES:].reshape(1, SONGLENGTH, NUM_SYLLABLE_FEATURES)
                    conditioning_data = torch.tensor(conditioning_data, dtype=torch.float32).to(device)
                else:
                    conditioning_data = None

                generated_features = model.generate(1, conditioning_data, training=False)
                sample = generated_features.cpu().numpy().squeeze(0)
                discretized_sample = utils.discretize(sample)
                discretized_sample = np.array(discretized_sample)  # تبدیل به آرایه NumPy
                validation_songs.append(discretized_sample)

        # محاسبه MMD
        val_gen_pitches = np.array([song[:, 0] for song in validation_songs])
        val_dat_pitches = validate[:, :NUM_MIDI_FEATURES * SONGLENGTH:NUM_MIDI_FEATURES]
        MMD_pitch = mmd.Compute_MMD(val_gen_pitches, val_dat_pitches)
        print("MMD pitch:", MMD_pitch)

        val_gen_duration = np.array([song[:, 1] for song in validation_songs])
        val_dat_duration = validate[:, 1:NUM_MIDI_FEATURES * SONGLENGTH:NUM_MIDI_FEATURES]
        MMD_duration = mmd.Compute_MMD(val_gen_duration, val_dat_duration)
        print("MMD duration:", MMD_duration)

        val_gen_rests = np.array([song[:, 2] for song in validation_songs])
        val_dat_rests = validate[:, 2:NUM_MIDI_FEATURES * SONGLENGTH:NUM_MIDI_FEATURES]
        MMD_rest = mmd.Compute_MMD(val_gen_rests, val_dat_rests)
        print("MMD rest:", MMD_rest)

        MMD_overall = MMD_pitch + MMD_duration + MMD_rest
        print("MMD overall:", MMD_overall)

        # ذخیره بهترین مدل بر اساس MMD overall
        if MMD_overall < best_mmd_overall:
            print(f"بهترین مدل در دوره {epoch + 1} با MMD overall: {MMD_overall:.5f}")
            best_mmd_overall = MMD_overall
            best_epoch = epoch + 1
            torch.save(model.state_dict(), './saved_gan_models/best_model.pth')
        
            # ... [Existing code above]

# Initialize lists to store metrics
        midi_numbers_span_list = []
        repetitions_3_list = []
        repetitions_2_list = []
        unique_midi_numbers_list = []
        notes_without_rest_list = []
        average_rest_value_list = []
        song_length_list = []

        # Validation loop
        model.eval()
        validation_songs = []
        with torch.no_grad():
            for i in range(len(validate)):
                song_data = torch.rand(1, SONGLENGTH, NUM_MIDI_FEATURES).to(device)
                if CONDITION:
                    conditioning_data = validate[i, SONGLENGTH * NUM_MIDI_FEATURES:].reshape(1, SONGLENGTH, NUM_SYLLABLE_FEATURES)
                    conditioning_data = torch.tensor(conditioning_data, dtype=torch.float32).to(device)
                else:
                    conditioning_data = None

                generated_features = model.generate(1, conditioning_data, training=False)
                sample = generated_features.cpu().numpy().squeeze(0)
                discretized_sample = utils.discretize(sample)
                discretized_sample = midi_statistics.tune_song(discretized_sample)
                discretized_sample = np.array(discretized_sample)
                validation_songs.append(discretized_sample)

                # Compute metrics for the current song
                midi_numbers = discretized_sample[:, 0]  # Assuming the first column is MIDI note numbers
                rest_values = discretized_sample[:, 2]   # Assuming the third column is rest durations

                # MIDI Numbers Span
                midi_span = midi_numbers.max() - midi_numbers.min()
                midi_numbers_span_list.append(midi_span)

                # Repetitions of 3-MIDI numbers
                repetitions_3 = midi_statistics.count_repetitions(midi_numbers, n=3)
                repetitions_3_list.append(repetitions_3)

                # Repetitions of 2-MIDI numbers
                repetitions_2 = midi_statistics.count_repetitions(midi_numbers, n=2)
                repetitions_2_list.append(repetitions_2)

                # Number of Unique MIDI numbers
                unique_midi_numbers = len(np.unique(midi_numbers))
                unique_midi_numbers_list.append(unique_midi_numbers)

                # Number of Notes Without Rest
                notes_without_rest = np.sum(rest_values == 0)
                notes_without_rest_list.append(notes_without_rest)

                # Average Rest Value Within Song
                average_rest = np.mean(rest_values)
                average_rest_value_list.append(average_rest)

                # Song Length
                song_length = len(midi_numbers)
                song_length_list.append(song_length)

        # After processing all songs, compute the average metrics
        avg_midi_span = np.mean(midi_numbers_span_list)
        avg_repetitions_3 = np.mean(repetitions_3_list)
        avg_repetitions_2 = np.mean(repetitions_2_list)
        avg_unique_midi_numbers = np.mean(unique_midi_numbers_list)
        avg_notes_without_rest = np.mean(notes_without_rest_list)
        avg_average_rest_value = np.mean(average_rest_value_list)
        avg_song_length = np.mean(song_length_list)

        # Print the metrics
        print(f"Average MIDI Numbers Span: {avg_midi_span:.1f}")
        print(f"Average 3-MIDI Numbers Repetitions: {avg_repetitions_3:.1f}")
        print(f"Average 2-MIDI Numbers Repetitions: {avg_repetitions_2:.1f}")
        print(f"Average Number of Unique MIDI: {avg_unique_midi_numbers:.1f}")
        print(f"Average Number of Notes Without Rest: {avg_notes_without_rest:.1f}")
        print(f"Average Rest Value Within Song: {avg_average_rest_value:.1f}")
        print(f"Average Song Length: {avg_song_length:.1f}")

        # ... [Rest of your code]


    print(f"بهترین مدل در دوره {best_epoch} با MMD overall: {best_mmd_overall:.5f}")

if __name__ == '__main__':
    main()
