import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import midi_statistics
import os
import time
import mmd
import math
import torch.nn.functional as F

# Define constants
REG_CONSTANT = 0.1
DISABLE_FEED_PREVIOUS = True
GAUSSIAN_NOISE = False
UNIDIRECTIONAL_D = True
BIDIRECTIONAL_G = False
RANDOM_INPUT_SCALE = 44
ATTENTION_LENGTH = 0
FEED_COND_D = True
RANDOM_INPUT_DIM = 100  # Adjusted to make the input dimension divisible by NUM_HEADS_G
DROPOUT_KEEP_PROB = 0.1
D_LR_FACTOR = 0.1
LEARNING_RATE = 0.0001
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
BATCH_SIZE = 128
REG_SCALE = 1.0
TRAIN_RATE = 0.8
VALIDATION_RATE = 0.1
HIDDEN_SIZE_G = 512
HIDDEN_SIZE_D = 256
NUM_LAYERS_G = 6
NUM_LAYERS_D = 4
ADAM = True
DISABLE_L2_REG = True
MAX_GRAD_NORM = 5.0
FEATURE_MATCHING = True
MAX_EPOCH = 500
EPOCHS_BEFORE_DECAY = 30
SONGLENGTH_CEILING = 20

# Set number of heads
NUM_HEADS_G = 8
NUM_HEADS_D = 8

# Positional Encoding class remains the same
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of [max_len, d_model] representing the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the positional encodings once in log space.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [sequence_length, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return x

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, num_song_features, num_meta_features, songlength, conditioning='multi'):
        super(Generator, self).__init__()
        self.songlength = songlength
        self.num_song_features = num_song_features
        self.num_meta_features = num_meta_features
        self.conditioning = conditioning

        # Transformer parameters
        self.num_heads = NUM_HEADS_G
        self.dropout = DROPOUT_KEEP_PROB
        self.generator_dropout = nn.Dropout(p=DROPOUT_KEEP_PROB)
        # Generator layers
        input_size_generator = RANDOM_INPUT_DIM + (num_meta_features if conditioning == 'multi' else 0)
        self.generator_input_linear = nn.Linear(input_size_generator, HIDDEN_SIZE_G)
        self.generator_pos_encoder = PositionalEncoding(HIDDEN_SIZE_G, max_len=songlength)
        encoder_layer_generator = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE_G,
            nhead=self.num_heads,
            dim_feedforward=HIDDEN_SIZE_G * 4,
            dropout=self.dropout,
            activation='gelu'
        )
        self.generator_transformer = nn.TransformerEncoder(
            encoder_layer_generator,
            num_layers=NUM_LAYERS_G
        )
        self.generator_output_linear = nn.Linear(HIDDEN_SIZE_G, num_song_features)

    def forward(self, batch_size, conditioning_data=None, training=True):
        device = next(self.parameters()).device
        random_input = torch.randn(
            (batch_size, self.songlength, RANDOM_INPUT_DIM),
            device=device
        )

        if self.conditioning == 'multi' and conditioning_data is not None:
            generator_input = torch.cat([random_input, conditioning_data], dim=-1)
        else:
            generator_input = random_input

        if training and DROPOUT_KEEP_PROB < 1.0:
            generator_input = self.generator_dropout(generator_input)
        # Project input to model dimension
        generator_input = self.generator_input_linear(generator_input)
        generator_input = generator_input.transpose(0, 1)  # [seq_len, batch_size, model_dim]

        # Add positional encoding
        generator_input = self.generator_pos_encoder(generator_input)

        if training and DROPOUT_KEEP_PROB < 1.0:
            generator_input = F.dropout(generator_input, p=self.dropout, training=training)

        # Transformer expects [sequence_length, batch_size, model_dim]
        gen_output = self.generator_transformer(generator_input)

        # Transform back to [batch_size, sequence_length, model_dim]
        gen_output = gen_output.transpose(0, 1)
        generated_features = self.generator_output_linear(gen_output)
        return generated_features

# Define the Discriminator class
class Discriminator(nn.Module):
    def __init__(self, num_song_features, num_meta_features, songlength, conditioning='multi'):
        super(Discriminator, self).__init__()
        self.songlength = songlength
        self.num_song_features = num_song_features
        self.num_meta_features = num_meta_features
        self.conditioning = conditioning

        # Transformer parameters
        self.num_heads = NUM_HEADS_D
        self.dropout = DROPOUT_KEEP_PROB
        self.discriminator_dropout = nn.Dropout(p=DROPOUT_KEEP_PROB)
        # Discriminator layers
        input_size_discriminator = num_song_features + (num_meta_features if conditioning == 'multi' and FEED_COND_D else 0)
        self.discriminator_input_linear = nn.Linear(input_size_discriminator, HIDDEN_SIZE_D)
        self.discriminator_pos_encoder = PositionalEncoding(HIDDEN_SIZE_D, max_len=songlength)
        encoder_layer_discriminator = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE_D,
            nhead=self.num_heads,
            dim_feedforward=HIDDEN_SIZE_D * 4,
            dropout=self.dropout,
            activation='gelu'
        )
        self.discriminator_transformer = nn.TransformerEncoder(
            encoder_layer_discriminator,
            num_layers=NUM_LAYERS_D
        )
        self.discriminator_output_linear = nn.Linear(HIDDEN_SIZE_D, 1)
        self.discriminator_sigmoid = nn.Sigmoid()

    def forward(self, song_data, conditioning_data=None, training=True):
        device = next(self.parameters()).device

        if self.conditioning == 'multi' and conditioning_data is not None and FEED_COND_D:
            discriminator_input = torch.cat([song_data, conditioning_data], dim=-1)
        else:
            discriminator_input = song_data

        # Project input to model dimension
        discriminator_input = self.discriminator_input_linear(discriminator_input)
        discriminator_input = discriminator_input.transpose(0, 1)  # [seq_len, batch_size, model_dim]

        # Add positional encoding
        discriminator_input = self.discriminator_pos_encoder(discriminator_input)
        if training:
            noise = torch.randn_like(discriminator_input) * 0.1  # Adjust noise level as needed
            discriminator_input += noise
        if training and DROPOUT_KEEP_PROB < 1.0:
            discriminator_input = F.dropout(discriminator_input, p=self.dropout, training=training)

        # Transformer expects [sequence_length, batch_size, model_dim]
        disc_output = self.discriminator_transformer(discriminator_input)

        # Transform back to [batch_size, sequence_length, model_dim]
        disc_output = disc_output.transpose(0, 1)

        # Pass through output layer
        decision = self.discriminator_output_linear(disc_output)
        decision = self.discriminator_sigmoid(decision)

        # Mean over time dimension
        decision = decision.mean(dim=1).squeeze()
        return decision

# Loss functions outside the models
def generator_loss(fake_output):
    target = torch.ones_like(fake_output)
    return nn.BCELoss()(fake_output, target)

def discriminator_loss(real_output, fake_output, wrong_output=None):
    real_target = torch.full_like(real_output, 0.9)  # Label Smoothing
    fake_target = torch.zeros_like(fake_output)

    real_loss = nn.BCELoss()(real_output, real_target)
    fake_loss = nn.BCELoss()(fake_output, fake_target)

    if wrong_output is not None and LOSS_WRONG_D:
        wrong_target = torch.zeros_like(wrong_output)
        wrong_loss = nn.BCELoss()(wrong_output, wrong_target)
        total_loss = real_loss + fake_loss + wrong_loss
    else:
        total_loss = real_loss + fake_loss

    return total_loss

# Update train_step function
def train_step(generator, discriminator, song_data, conditioning_data, wrong_conditioning_data, batch_size, pretraining, generator_optimizer, discriminator_optimizer):
    device = next(generator.parameters()).device
    generator.train()
    discriminator.train()
    song_data = song_data.to(device)
    if conditioning_data is not None:
        conditioning_data = conditioning_data.to(device)
    if wrong_conditioning_data is not None:
        wrong_conditioning_data = wrong_conditioning_data.to(device)

    # Zero gradients
    generator_optimizer.zero_grad()
    if not pretraining:
        discriminator_optimizer.zero_grad()

    generated_songs = generator(batch_size, conditioning_data, training=True)

    if pretraining:
        gen_loss = nn.MSELoss()(generated_songs, song_data)
        gen_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(generator.parameters(), MAX_GRAD_NORM)
        generator_optimizer.step()
        disc_loss = None
    else:
        real_output = discriminator(song_data, conditioning_data)
        fake_output = discriminator(generated_songs.detach(), conditioning_data)
        if wrong_conditioning_data is not None and LOSS_WRONG_D:
            wrong_output = discriminator(song_data, wrong_conditioning_data)
        else:
            wrong_output = None

        disc_loss = discriminator_loss(real_output, fake_output, wrong_output)
        disc_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), MAX_GRAD_NORM)
        discriminator_optimizer.step()

        # Update Generator
        generator_optimizer.zero_grad()
        generated_songs = generator(batch_size, conditioning_data, training=True)
        fake_output = discriminator(generated_songs, conditioning_data)
        gen_loss = generator_loss(fake_output)
        gen_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(generator.parameters(), MAX_GRAD_NORM)
        generator_optimizer.step()

    return gen_loss.item(), disc_loss.item() if disc_loss is not None else None

def main():
    # Load data
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

    # Create dataset using torch.utils.data.Dataset and DataLoader
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
                # Create wrong conditioning data
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

    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("RUN ON ", device)
    generator = Generator(
        num_song_features=NUM_MIDI_FEATURES,
        num_meta_features=NUM_SYLLABLE_FEATURES,
        songlength=SONGLENGTH,
        conditioning='multi',
    ).to(device)

    discriminator = Discriminator(
        num_song_features=NUM_MIDI_FEATURES,
        num_meta_features=NUM_SYLLABLE_FEATURES,
        songlength=SONGLENGTH,
        conditioning='multi',
    ).to(device)

    # Set up optimizers
    if ADAM:
        generator_optimizer = optim.Adam(
            generator.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )
        discriminator_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=LEARNING_RATE * D_LR_FACTOR,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )
    else:
        generator_optimizer = optim.SGD(generator.parameters(), lr=LEARNING_RATE)
        discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=LEARNING_RATE * D_LR_FACTOR)

    # Learning rate schedulers
    def lr_lambda(epoch):
        if epoch < EPOCHS_BEFORE_DECAY:
            return 1.0
        else:
            return LR_DECAY ** (epoch - EPOCHS_BEFORE_DECAY + 1)

    generator_scheduler = optim.lr_scheduler.LambdaLR(generator_optimizer, lr_lambda=lr_lambda)
    discriminator_scheduler = optim.lr_scheduler.LambdaLR(discriminator_optimizer, lr_lambda=lr_lambda)

    # Load checkpoint if exists
    checkpoint_dir = './training_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
    start_epoch = 0
    best_mmd_overall = np.inf
    best_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_mmd_overall = checkpoint['best_mmd_overall']
        best_epoch = checkpoint['best_epoch']
        print(f"Loaded model from checkpoint at epoch {start_epoch}")

    # Main training loop
    for epoch in range(start_epoch, MAX_EPOCH):
        start_time = time.time()

        # Adjust learning rates
        if epoch >= EPOCHS_BEFORE_DECAY:
            generator_scheduler.step()
            discriminator_scheduler.step()

        print(f"Epoch {epoch + 1}/{MAX_EPOCH}, Learning Rate: {generator_optimizer.param_groups[0]['lr']:.5f}")

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
                generator,
                discriminator,
                song_data,
                conditioning_data,
                wrong_conditioning_data,
                batch_size,
                pretraining=(epoch < PRETRAINING_EPOCHS),
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
            )
            gen_losses.append(gen_loss)
            if disc_loss is not None:
                disc_losses.append(disc_loss)


        print(f"Average Generator Loss: {np.mean(gen_losses):.5f}")
        if disc_losses:
            print(f"Average Discriminator Loss: {np.mean(disc_losses):.5f}")

        # Save model every 15 epochs
        if (epoch + 1) % 15 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                    'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                    'best_mmd_overall': best_mmd_overall,
                    'best_epoch': best_epoch,
                },
                checkpoint_path,
            )
            print(f"Model saved at epoch {epoch + 1}")

        # Evaluation and MMD calculation
        generator.eval()
        validation_songs = []
        
             # Initialize lists to store metrics
        midi_numbers_span_list = []
        repetitions_3_list = []
        repetitions_2_list = []
        unique_midi_numbers_list = []
        notes_without_rest_list = []
        average_rest_value_list = []
        song_length_list = []
        
        
        with torch.no_grad():
            for i in range(len(validate)):
                # Prepare conditioning data
                if CONDITION:
                    conditioning_data = validate[i, SONGLENGTH * NUM_MIDI_FEATURES:].reshape(
                        1, SONGLENGTH, NUM_SYLLABLE_FEATURES
                    )
                    conditioning_data = torch.tensor(conditioning_data, dtype=torch.float32).to(device)
                else:
                    conditioning_data = None

                # Generate song
                generated_features = generator(1, conditioning_data, training=False)
                sample = generated_features.cpu().numpy().squeeze(0)
                discretized_sample = utils.discretize(sample)
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

            # Compute MMD
            val_gen_pitches = np.array([song[:, 0] for song in validation_songs])
            val_dat_pitches = validate[:, : NUM_MIDI_FEATURES * SONGLENGTH : NUM_MIDI_FEATURES]
            MMD_pitch = mmd.Compute_MMD(val_gen_pitches, val_dat_pitches)
            print("MMD pitch:", MMD_pitch)

            val_gen_duration = np.array([song[:, 1] for song in validation_songs])
            val_dat_duration = validate[:, 1 : NUM_MIDI_FEATURES * SONGLENGTH : NUM_MIDI_FEATURES]
            MMD_duration = mmd.Compute_MMD(val_gen_duration, val_dat_duration)
            print("MMD duration:", MMD_duration)

            val_gen_rests = np.array([song[:, 2] for song in validation_songs])
            val_dat_rests = validate[:, 2 : NUM_MIDI_FEATURES * SONGLENGTH : NUM_MIDI_FEATURES]
            MMD_rest = mmd.Compute_MMD(val_gen_rests, val_dat_rests)
            print("MMD rest:", MMD_rest)

            MMD_overall = MMD_pitch + MMD_duration + MMD_rest
            print("MMD overall:", MMD_overall)

# After processing all songs, compute the average metrics
            avg_midi_span = np.mean(midi_numbers_span_list)
            avg_repetitions_3 = np.mean(repetitions_3_list)
            avg_repetitions_2 = np.mean(repetitions_2_list)
            avg_unique_midi_numbers = np.mean(unique_midi_numbers_list)
            avg_notes_without_rest = np.mean(notes_without_rest_list)
            avg_average_rest_value = np.mean(average_rest_value_list)
            avg_song_length = np.mean(song_length_list)
            print("=====Midi stats-----")
            # Print the metrics
            print(f"Average MIDI Numbers Span: {avg_midi_span:.1f}")
            print(f"Average 3-MIDI Numbers Repetitions: {avg_repetitions_3:.1f}")
            print(f"Average 2-MIDI Numbers Repetitions: {avg_repetitions_2:.1f}")
            print(f"Average Number of Unique MIDI: {avg_unique_midi_numbers:.1f}")
            print(f"Average Number of Notes Without Rest: {avg_notes_without_rest:.1f}")
            print(f"Average Rest Value Within Song: {avg_average_rest_value:.1f}")
            print(f"Average Song Length: {avg_song_length:.1f}")
            end_time = time.time()
            print(f"Time for epoch {epoch + 1}: {end_time - start_time:.2f} seconds")
            print("==========================@@@@===========================")
            # Save best model based on MMD overall
            if MMD_overall < best_mmd_overall:
                print(f"Best model at epoch {epoch + 1} with MMD overall: {MMD_overall:.5f}")
                best_mmd_overall = MMD_overall
                best_epoch = epoch + 1
                torch.save({
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                }, './saved_gan_models/best_model.pth')


    print(f"Best model at epoch {best_epoch} with MMD overall: {best_mmd_overall:.5f}")

if __name__ == '__main__':
    main()
