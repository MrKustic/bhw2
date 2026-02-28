import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import EncoderDecoderRNN
import sacrebleu

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})

def plot_losses(train_losses: List[float], val_losses: List[float], bleu_scores: List[float]):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    """
    Calculate train and validation perplexities given lists of losses
    """
    # train_perplexities = np.exp(np.array(train_losses))
    # val_perplexities = np.exp(np.array(val_losses))

    # axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    # axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    # axs[1].set_ylabel('perplexity')

    axs[1].plot(range(1, len(bleu_scores) + 1), bleu_scores, label='val')
    axs[1].set_ylabel('bleu4 score')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model: EncoderDecoderRNN, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0

    model.train()
    for (indices, lengths), (target, target_lengths) in tqdm(loader, desc=tqdm_desc):
        """
        Process one training step: calculate loss,
        call backward and make one optimizer step.
        Accumulate sum of losses for different batches in train_loss
        """
        optimizer.zero_grad()

        logits = model(indices.to(device), lengths, target[:, :-1].to(device), target_lengths - 1)
        new_length = max(logits.shape[1], target.shape[1] - 1)
        logits = torch.nn.functional.pad(logits, (0, 0, 0, new_length - logits.shape[1])) # (B, L, V)
        pad_target = torch.nn.functional.pad(target[:, 1:], (0, new_length - target.shape[1] + 1, 0, 0))
        loss = criterion(logits.transpose(1, 2), pad_target.to(device))
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * indices.shape[0]   

    train_loss /= len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model: EncoderDecoderRNN, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = next(model.parameters()).device
    val_loss = 0.0

    model_translation = []

    model.eval()
    for (indices, lengths), (target, target_lengths) in tqdm(loader, desc=tqdm_desc):
        """
        Process one validation step: calculate loss.
        Accumulate sum of losses for different batches in val_loss
        """
        logits = model(indices.to(device), lengths, target.to(device)[:, :-1], target_lengths - 1)
        new_length = max(logits.shape[1], target.shape[1] - 1)
        logits = torch.nn.functional.pad(logits, (0, 0, 0, new_length - logits.shape[1])) # (B, L, V)
        pad_target = torch.nn.functional.pad(target[:, 1:], (0, new_length - target.shape[1] + 1, 0, 0))
        loss = criterion(logits.transpose(1, 2), pad_target.to(device))
        val_loss += loss.item() * indices.shape[0]
        model_translation += model.inference(indices.to(device), lengths)

    val_loss /= len(loader.dataset)
    print(model_translation[:10], loader.dataset.texts[1][:10])

    bleu_score = sacrebleu.corpus_bleu(model_translation, [loader.dataset.texts[1]], tokenize='none').score

    return val_loss, bleu_score


def train(model: EncoderDecoderRNN, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=5):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    """
    train_losses, val_losses, bleu_scores = [], [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss, bleu_score = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
        bleu_scores += [bleu_score]
        plot_losses(train_losses, val_losses, bleu_scores)

        for (indices, lengths), (_, _) in val_loader:
            print(model.inference(indices[:num_examples, :], lengths[:num_examples]))
