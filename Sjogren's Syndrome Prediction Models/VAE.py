import os
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

import logging
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from tensorboard import program
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class RadiomicsDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, scaler):
        self.scaler = scaler
        self.features = torch.FloatTensor(self.scaler.fit_transform(features))
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def rescaler(self, features):
        return self.scaler.inverse_transform(features)


class VAE(nn.Module):

    def __init__(self, in_features, latent_size, y_size=0):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        self.encoder_forward = nn.Sequential(
            nn.Linear(in_features + y_size, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, self.latent_size * 2)
        )

        self.decoder_forward = nn.Sequential(
            nn.Linear(self.latent_size + y_size, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )

    def encoder(self, X):
        out = self.encoder_forward(X)
        mu = out[:, :self.latent_size]
        log_var = out[:, self.latent_size:]
        return mu, log_var

    def decoder(self, z):
        mu_prime = self.decoder_forward(z)
        return mu_prime

    def reparameterization(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        z = mu + epsilon * torch.sqrt(log_var.exp())
        return z
    

    def forward(self, X, *args, **kwargs):
        mu, log_var = self.encoder(X)
        z = self.reparameterization(mu, log_var)
        mu_prime = self.decoder(z)
        return mu_prime, mu, log_var
    
class KL_loss(nn.Module):
    def __init__(self):
        super(KL_loss, self).__init__()

    def forward(self, mu, log_var):
        return torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var - 1).sum(dim=1))
    
class reconstruction_loss(nn.Module):
    def __init__(self):
        super(reconstruction_loss, self).__init__()

    def forward(self, X, mu_prime):
        return torch.mean(torch.square(X - mu_prime).sum(dim=1))

class CVAE(VAE):

    def __init__(self, config):
        super(CVAE, self).__init__(
            config['model']['in_features'], 
            config['model']['latent_size'], 
            config['model']['y_size']
        )
        
        # 构建编码器层
        encoder_layers = []
        in_dim = config['model']['in_features'] + config['model']['y_size']
        
        for hidden_dim in config['network']['encoder']['hidden_layers']:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if config['network']['encoder']['use_batch_norm'] else nn.Identity(),
                getattr(nn, config['network']['encoder']['activation'])(),
                nn.Dropout(config['network']['encoder']['dropout_rate'])
            ])
            in_dim = hidden_dim
            
        encoder_layers.append(nn.Linear(in_dim, self.latent_size * 2))
        self.encoder_forward = nn.Sequential(*encoder_layers)
        
        # 构建解码器层
        decoder_layers = []
        in_dim = self.latent_size + config['model']['y_size']
        
        for hidden_dim in config['network']['decoder']['hidden_layers']:
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if config['network']['decoder']['use_batch_norm'] else nn.Identity(),
                getattr(nn, config['network']['decoder']['activation'])(),
            ])
            in_dim = hidden_dim
            
        decoder_layers.extend([
            nn.Linear(in_dim, config['model']['in_features']),
            getattr(nn, config['network']['decoder']['final_activation'])()
        ])
        self.decoder_forward = nn.Sequential(*decoder_layers)
        
    def forward(self, X, y=None, *args, **kwargs):
        y = y.to(next(self.parameters()).device)
        X_given_Y = torch.cat((X, y.unsqueeze(1)), dim=1)

        mu, log_var = self.encoder(X_given_Y)
        z = self.reparameterization(mu, log_var)
        z_given_Y = torch.cat((z, y.unsqueeze(1)), dim=1)

        mu_prime_given_Y = self.decoder(z_given_Y)
        return mu_prime_given_Y, mu, log_var


def setup_logger(log_dir='logs'):
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging format
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理程序
    logger.handlers.clear()
    
    # 创建文件处理程序
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 创建控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 添加处理程序到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def train(model, optimizer, data_loader, device, writer, epoch, name='VAE'):
    model.train()
    
    epoch_metrics = {
        'total_loss': 0,
        'reconstruction_loss': 0,
        'latent_loss': 0
    }
    
    pbar = tqdm(data_loader)
    for batch_idx, (X, y) in enumerate(pbar):
        batch_size = X.shape[0]
        X = X.view(batch_size, -1).to(device)
        model.zero_grad()

        if name == 'VAE':
            mu_prime, mu, log_var = model(X)
        else:
            mu_prime, mu, log_var = model(X, y)

        # Calculate losses
        reconstruction_loss = F.mse_loss(mu_prime, X, reduction='mean')
        # reconstruction_loss = torch.mean(torch.square(X - mu_prime).sum(dim=1))
        latent_loss = torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var - 1).sum(dim=1))
        loss = reconstruction_loss + latent_loss

        loss.backward()
        optimizer.step()

        # Update metrics
        
        epoch_metrics['total_loss'] += loss.item()
        epoch_metrics['reconstruction_loss'] += reconstruction_loss.item()
        epoch_metrics['latent_loss'] += latent_loss.item()
        
        # Log batch-level metrics
        global_step = epoch * len(data_loader) + batch_idx
        writer.add_scalar('Loss/batch/total', loss.item(), global_step)
        writer.add_scalar('Loss/batch/reconstruction', reconstruction_loss.item(), global_step)
        writer.add_scalar('Loss/batch/latent', latent_loss.item(), global_step)
        
        pbar.set_description(f'Epoch: {epoch} Loss: {loss.item():.4f}')

    # Calculate averages
    for key in epoch_metrics:
        epoch_metrics[key] /= len(data_loader)
        
    return epoch_metrics

def validate(model, data_loader, device, writer, epoch, name='VAE'):
    model.eval()
    val_metrics = {
        'total_loss': 0,
        'reconstruction_loss': 0,
        'latent_loss': 0
    }
    
    with torch.no_grad():
        for X, y in data_loader:
            batch_size = X.shape[0]
            X = X.view(batch_size, -1).to(device)

            if name == 'VAE':
                mu_prime, mu, log_var = model(X)
            else:
                mu_prime, mu, log_var = model(X, y)
            # reconstruction_loss = torch.mean(torch.square(X - mu_prime).sum(dim=1)) 
            reconstruction_loss = F.mse_loss(mu_prime, X, reduction='mean')
            latent_loss = torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var - 1).sum(dim=1))
            loss = reconstruction_loss + latent_loss

            val_metrics['total_loss'] += loss.item()
            val_metrics['reconstruction_loss'] += reconstruction_loss.item()
            val_metrics['latent_loss'] += latent_loss.item()

    # Calculate averages
    for key in val_metrics:
        val_metrics[key] /= len(data_loader)
        
    return val_metrics

def main(train_features, train_labels, val_features, val_labels, config_path, save_folder):
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # Set up experiment name and paths
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'CVAE_experiment_{timestamp}'
    log_dir = os.path.join('runs', experiment_name)
    
    os.makedirs(log_dir, exist_ok=True)
    config_save_path = os.path.join(log_dir, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    
    # Initialize logger and tensorboard
    logger = setup_logger(log_dir)
    writer = SummaryWriter(log_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Update configuration with input features dimension
    config['model']['in_features'] = train_features.shape[1]
    
    logger.info(f'Feature dimension: {config["model"]["in_features"]}')
    logger.info(f'Batch size: {config["training"]["batch_size"]}')
    logger.info(f'Latent dimension: {config["model"]["latent_size"]}')
    logger.info(f'Learning rate: {config["training"]["learning_rate"]}')
    
    # Create train dataset
    scaler = MinMaxScaler()
    scaler.fit(train_features)
    train_dataset = RadiomicsDataset(train_features, train_labels, scaler)
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    # Create validation dataset
    val_dataset = RadiomicsDataset(val_features, val_labels, scaler)
    
    # Create data loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Initialize model and optimizer
    cvae = CVAE(config).to(device)
    optimizer = getattr(torch.optim, config['training']['optimizer'])(
        cvae.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Add learning rate scheduler
    if config['training']['scheduler']['type']:
        scheduler = getattr(torch.optim.lr_scheduler, config['training']['scheduler']['type'])(
            optimizer,
            patience=config['training']['scheduler']['patience'],
            factor=config['training']['scheduler']['factor']
        )
    
    best_val_loss = float('inf')
    
    logger.info('Starting CVAE training...')
    for epoch in range(1, config['training']['epochs'] + 1):
        # Training
        train_metrics = train(cvae, optimizer, train_loader, device, writer, epoch, name='CVAE')
        
        # Validation
        val_metrics = validate(cvae, val_loader, device, writer, epoch, name='CVAE')
        
        # Log metrics
        writer.add_scalar('Loss/epoch/train_total', train_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/epoch/train_reconstruction', train_metrics['reconstruction_loss'], epoch)
        writer.add_scalar('Loss/epoch/train_latent', train_metrics['latent_loss'], epoch)
        
        writer.add_scalar('Loss/epoch/val_total', val_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/epoch/val_reconstruction', val_metrics['reconstruction_loss'], epoch)
        writer.add_scalar('Loss/epoch/val_latent', val_metrics['latent_loss'], epoch)
        
        logger.info(
            f"Epoch: {epoch}, "
            f"Train Loss: {train_metrics['total_loss']:.4f}, "
            f"Val Loss: {val_metrics['total_loss']:.4f}"
        )
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            model_save_path = os.path.join(log_dir, 'best_cvae_model.pth')
            torch.save(cvae.state_dict(), model_save_path)
            logger.info(f'New best model saved with validation loss: {best_val_loss:.4f}')

    # Save model
    model_save_path = os.path.join(log_dir, 'cvae_model.pth')
    torch.save(cvae.state_dict(), model_save_path)
    logger.info(f'Model saved to: {model_save_path}')
    
    writer.close()
    
    # Load the best model for generation
    best_model_path = os.path.join(log_dir, 'best_cvae_model.pth')
    generation_model = CVAE(config).to(device)
    generation_model.load_state_dict(torch.load(best_model_path))
    generation_model.eval()
    logger.info(f'Loaded best model from {best_model_path} for feature generation')
    
    # Generate augmented features using the best model
    @torch.no_grad()
    def generate_features(model, n_samples, label, device):
        z = torch.randn(n_samples, config['model']['latent_size']).to(device)
        y = torch.full((n_samples,), label).to(device)
        z_given_Y = torch.cat((z, y.unsqueeze(1)), dim=1)
        generated = model.decoder(z_given_Y).cpu().numpy()
        return train_dataset.rescaler(generated)
    
    # Generate new samples for each class using the best model
    for label in np.unique(train_labels):
        new_features = generate_features(
            generation_model,  # Using the loaded best model
            config['sampling']['n_samples_per_class'], 
            label, 
            device
        )
        logger.info(f"Generated {config['sampling']['n_samples_per_class']} new feature samples for label {label}")
        features_df = pd.DataFrame(new_features)
        features_df['label'] = label
        features_df.to_csv(os.path.join(save_folder, f'generated_features_{label}.csv'), index=False)
        logger.info(f"Saved generated features for label {label} in {save_folder}")

if __name__ == '__main__':
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'runs'])
    url = tb.launch()
    print(f"TensorBoard is on: {url}")
    
    args = argparse.ArgumentParser()
    args.add_argument('--features_path', type=str, default=r'examples\MLMath\feature\label_1.csv')
    args.add_argument('--config', type=str, default=r'examples\MLMath\config\config.yaml')
    args = args.parse_args()
    main(args.features_path, args.config)