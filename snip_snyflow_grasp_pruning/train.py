import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    correct1 = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        _, pred = output.topk(1, dim=1)
        correct1 += pred.eq(target.view(-1, 1)).sum().item()
        if verbose and (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    return average_loss, accuracy1

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(1, dim=1)
            correct1 += pred.eq(target.view(-1, 1)).sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1

def plot_metrics(df, epoch, result_dir):
    plt.figure(figsize=(12, 8))

    # Plot train and test loss
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['train_loss'], label='Train Loss')
    plt.plot(df.index, df['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss Over Epochs')
    plt.legend()

    # Plot top-1 accuracy
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['train_accuracy1'], label='Train Top-1 Accuracy')
    plt.plot(df.index, df['test_accuracy1'], label='Test Top-1 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Top-1 Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(result_dir, 'training_plots.png'))
    plt.show()

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose, result_dir):
    os.makedirs('plots', exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    best_accuracy = 0.0
    test_loss, test_accuracy1 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, np.nan, test_accuracy1]]
    columns = ['train_loss', 'test_loss', 'train_accuracy1', 'test_accuracy1']
    df = pd.DataFrame(rows, columns=columns)
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy1 = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        test_loss, test_accuracy1 = eval(model, loss, test_loader, device, verbose)
        row = [train_loss, test_loss, train_accuracy1, test_accuracy1]
        scheduler.step()
        rows.append(row)
        df = pd.DataFrame(rows, columns=columns)
        plot_metrics(df, epoch, result_dir)  # Plot and save after each epoch

        # Save the best model based on highest validation accuracy
        if test_accuracy1 > best_accuracy:
            best_accuracy = test_accuracy1
            torch.save(model.state_dict(), os.path.join(result_dir, 'best_model.pth'))
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
    
    return df

# Example usage:
# model, loss, optimizer, scheduler, train_loader, test_loader, device = ... (your setup here)
# df = train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs=20, verbose=True, result_dir='results')
