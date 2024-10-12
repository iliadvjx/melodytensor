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

# Define constants
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
BATCH_SIZE = 300
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

# Set number of heads to 5 to ensure divisibility
NUM_HEADS_G = 5
NUM_HEADS_D = 5

class TransformerGAN(nn.Module):
    def __init__(self, num_song_features, num_meta_features, songlength, conditioning='multi'):
        super(TransformerGAN, self).__init__()
        self.songlength = songlength
        self.num_song_features = num_song_features
        self.num_meta_features = num_meta_features
        self.conditioning = conditioning

        # Generator
        input_size_generator = int(RANDOM_INPUT_SCALE * num_song_features) + (num_meta_features if conditioning == 'multi' else 0)
        self.generator_input_dim = input_size_generator  # Should be divisible by NUM_HEADS_G
        self.generator_output_dim = num_song_features

        self.generator_positional_encoding = PositionalEncoding(self.generator_input_dim, dropout=1 - DROPOUT_KEEP_PROB, max_len=songlength)
        self.generator_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.generator_input_dim,
                nhead=NUM_HEADS_G,
                dim_feedforward=HIDDEN_SIZE_G,
                dropout=1 - DROPOUT_KEEP_PROB,
                activation='relu'
            ),
            num_layers=NUM_LAYERS_G
        )
        self.generator_dense = nn.Linear(self.generator_input_dim, self.generator_output_dim)

        # Discriminator
        input_size_discriminator = num_song_features + (num_meta_features if conditioning == 'multi' and FEED_COND_D else 0)
        self.discriminator_input_dim = input_size_discriminator  # Should be divisible by NUM_HEADS_D
        self.discriminator_output_dim = 1

        self.discriminator_positional_encoding = PositionalEncoding(self.discriminator_input_dim, dropout=1 - DROPOUT_KEEP_PROB, max_len=songlength)
        self.discriminator_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.discriminator_input_dim,
                nhead=NUM_HEADS_D,
                dim_feedforward=HIDDEN_SIZE_D,
                dropout=1 - DROPOUT_KEEP_PROB,
                activation='relu'
            ),
            num_layers=NUM_LAYERS_D
        )
        self.discriminator_dense = nn.Linear(self.discriminator_input_dim, self.discriminator_output_dim)
        self.discriminator_sigmoid = nn.Sigmoid()

        # Loss functions
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

        # Apply positional encoding
        generator_input = self.generator_positional_encoding(generator_input.permute(1, 0, 2))  # Shape: [sequence_length, batch_size, input_dim]

        gen_output = self.generator_transformer(generator_input)  # Shape: [sequence_length, batch_size, input_dim]

        gen_output = gen_output.permute(1, 0, 2)  # Shape: [batch_size, sequence_length, input_dim]
        generated_features = self.generator_dense(gen_output)  # Shape: [batch_size, sequence_length, num_song_features]

        return generated_features

    def discriminate(self, song_data, conditioning_data=None, training=True):
        device = next(self.parameters()).device

        if self.conditioning == 'multi' and conditioning_data is not None and FEED_COND_D:
            discriminator_input = torch.cat([song_data, conditioning_data], dim=-1)
        else:
            discriminator_input = song_data

        # Apply positional encoding
        discriminator_input = self.discriminator_positional_encoding(discriminator_input.permute(1, 0, 2))  # Shape: [sequence_length, batch_size, input_dim]

        disc_output = self.discriminator_transformer(discriminator_input)  # Shape: [sequence_length, batch_size, input_dim]

        disc_output = disc_output.permute(1, 0, 2)  # Shape: [batch_size, sequence_length, input_dim]
        decision = self.discriminator_dense(disc_output)  # Shape: [batch_size, sequence_length, 1]
        decision = self.discriminator_sigmoid(decision)

        # Average over time dimension
        decision = decision.mean(dim=1).squeeze()  # Shape: [batch_size]
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(1)  # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [sequence_length, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# The rest of the code remains largely the same, except that you should replace instances of RNNGAN with TransformerGAN

def train_step(model, song_data, conditioning_data, wrong_conditioning_data, batch_size, pretraining, generator_optimizer, discriminator_optimizer):
    device = next(model.parameters()).device
    model.train()
    song_data = song_data.to(device)
    if conditioning_data is not None:
        conditioning_data = conditioning_data.to(device)
    if wrong_conditioning_data is not None:
        wrong_conditioning_data = wrong_conditioning_data.to(device)

    # Zero gradients
    generator_optimizer.zero_grad()
    if not pretraining:
        discriminator_optimizer.zero_grad()

    generated_songs = model.generate(batch_size, conditioning_data, training=True)

    if pretraining:
        gen_loss = model.mse_loss(generated_songs, song_data)
        gen_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
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
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        discriminator_optimizer.step()

        # Update Generator
        generator_optimizer.zero_grad()
        generated_songs = model.generate(batch_size, conditioning_data, training=True)
        fake_output = model.discriminate(generated_songs, conditioning_data)
        gen_loss = model.generator_loss(fake_output)
        gen_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
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

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("RUN ON ", device)
    model = TransformerGAN(num_song_features=NUM_MIDI_FEATURES, num_meta_features=NUM_SYLLABLE_FEATURES, songlength=SONGLENGTH, conditioning='multi')
    model.to(device)

    # Set up optimizers
    if ADAM:
        generator_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        discriminator_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * D_LR_FACTOR)
    else:
        generator_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        discriminator_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE * D_LR_FACTOR)

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
        model.load_state_dict(checkpoint['model_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_mmd_overall = checkpoint['best_mmd_overall']
        best_epoch = checkpoint['best_epoch']
        print(f"Loaded model from checkpoint at epoch {start_epoch}")

    # Training loop remains the same
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
        print(f"Time for epoch {epoch + 1}: {end_time - start_time:.2f} seconds")
        print(f"Average Generator Loss: {np.mean(gen_losses):.5f}")
        if disc_losses:
            print(f"Average Discriminator Loss: {np.mean(disc_losses):.5f}")

        # Save model every 15 epochs
        if (epoch + 1) % 15 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                'best_mmd_overall': best_mmd_overall,
                'best_epoch': best_epoch
            }, checkpoint_path)
            print(f"Model saved at epoch {epoch + 1}")

        # Evaluation and MMD calculation
        # ... [Include your evaluation code here, same as before]

    print(f"Best model at epoch {best_epoch} with MMD overall: {best_mmd_overall:.5f}")

if __name__ == '__main__':
    main()
