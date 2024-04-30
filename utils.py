# 1. imports
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import pandas as pd
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from IPython import get_ipython
    if 'IPKernelApp' not in get_ipython().config:  # Not in a notebook
        from tqdm import tqdm
    else:  # In a notebook
        from tqdm.notebook import tqdm
except ImportError:  # IPython is not installed
    from tqdm import tqdm
    
def get_accuracy(logits, targets):
        """
        Calculate the accuracy of predictions made by a model using nn.CrossEntropyLoss.
        
        Args:
            logits: A tensor of shape (batch_size, num_classes) containing the raw output scores from the model.
            targets: A tensor of shape (batch_size,) containing the actual class labels.
        
        Returns:
            A float representing the accuracy of the predictions.
        """
        # Get the indices of the maximum value of all elements in the input tensor,
        # which are the predicted class labels.
        _, predicted_labels = torch.max(logits, 1)
        
        # Calculate the number of correctly predicted labels.
        correct_predictions = (predicted_labels == targets).sum().item()
        
        # Calculate the accuracy.
        accuracy = correct_predictions / targets.size(0)
        
        return accuracy
    
def run_training_and_validation(model, device, initial_lr, num_epochs, criterion, optimiser, train_dataloader, val_dataloader, metrics = False, manual_lr_schedule = False, scheduler_func=None, plot = False):
    train_epoch_losses = []
    train_epoch_accuracy = []
    val_epoch_losses = []
    val_epoch_accuracy = []
    
    for epoch in range(num_epochs):
        train_running_batch_losses = []
        train_running_batch_accuracy = []
        
        if epoch == num_epochs-1:
            train_all_preds = []
            train_all_labels = []
            val_all_preds = []
            val_all_labels = []
        
        if manual_lr_schedule:
            optimiser = scheduler_func(optimiser, epoch, initial_lr)

        
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            
            accuracy = get_accuracy(outputs, labels)
            
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            train_running_batch_losses.append(loss.item())
            train_running_batch_accuracy.append(accuracy)
            # if i % 50 == 0:
            #   training_progress_bar.set_description(f'Training Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Acc: {accuracy:.4f}')
            
            if epoch == num_epochs-1:
                _, preds = torch.max(outputs, 1)
                train_all_preds.extend(preds.cpu().numpy())  # Move predictions to CPU and convert to numpy for sklearn
                train_all_labels.extend(labels.cpu().numpy())  # Move labels to CPU and convert to numpy

        train_epoch_losses.append(sum(train_running_batch_losses)/len(train_running_batch_losses))
        train_epoch_accuracy.append(sum(train_running_batch_accuracy)/len(train_running_batch_accuracy))
        model.eval()
        with torch.no_grad():
            val_running_batch_losses = []
            val_running_batch_accuracy = []

            for i, (images, labels) in enumerate(val_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                accuracy = get_accuracy(outputs, labels)

                val_running_batch_losses.append(loss.item())
                val_running_batch_accuracy.append(accuracy)
                # if i % 20 == 0:
                #   val_progress_bar.set_description(f'Validation Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(val_dataloader)}], Loss: {loss.item():.4f}, Acc: {accuracy:.4f}')
                
                if epoch == num_epochs-1:
                    _, preds = torch.max(outputs, 1)
                    val_all_preds.extend(preds.cpu().numpy())  # Move predictions to CPU and convert to numpy for sklearn
                    val_all_labels.extend(labels.cpu().numpy())  # Move labels to CPU and convert to numpy

            val_epoch_losses.append(sum(val_running_batch_losses)/len(val_running_batch_losses))
            val_epoch_accuracy.append(sum(val_running_batch_accuracy)/len(val_running_batch_accuracy))
            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_epoch_losses[epoch]:.4f}, Acc: {train_epoch_accuracy[epoch]:.4f} | Val Loss: {val_epoch_losses[epoch]:.4f}, Acc: {val_epoch_accuracy[epoch]:.4f}')
            class_names = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            
    
    if plot:
        plot_single_train_val_smoothed(train_epoch_losses, val_epoch_losses, train_epoch_accuracy, val_epoch_accuracy, num_epochs, smoothing_window=10, title=f'single run lr={initial_lr}, decay={manual_lr_schedule}')
    

            
    if metrics:
        train_report = classification_report(train_all_labels, train_all_preds, target_names=(class_names))
        val_report = classification_report(val_all_labels, val_all_preds, target_names=(class_names))
        # print('FINAL EPOCH TRAINING SUMMARY:')
        # print(train_report)
        # print('FINAL EPOCH VALIDATION SUMMARY:')
        # print(val_report)
        
        return (model,train_epoch_losses, train_epoch_accuracy, val_epoch_losses, val_epoch_accuracy, train_report,val_report)
    else:
        return (model, train_epoch_losses, train_epoch_accuracy, val_epoch_losses, val_epoch_accuracy, 0,0)

def run_testing(model, device, criterion, test_dataloader):
    model.eval()
    with torch.no_grad():
        test_running_batch_losses = []
        test_running_batch_accuracy = []
        test_all_preds = []
        test_all_labels = []

        for i, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            accuracy = get_accuracy(outputs, labels)

            test_running_batch_losses.append(loss.item())
            test_running_batch_accuracy.append(accuracy)
            # test_progress_bar.set_description(f'testidation Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(test_dataloader)}], Loss: {loss.item():.4f}, Acc: {accuracy:.4f}')
            _, preds = torch.max(outputs, 1)
            test_all_preds.extend(preds.cpu().numpy())  # Move predictions to CPU and convert to numpy for sklearn
            test_all_labels.extend(labels.cpu().numpy())  # Move labels to CPU and convert to numpy

    test_loss = sum(test_running_batch_losses)/len(test_running_batch_losses)
    test_accuracy = sum(test_running_batch_accuracy)/len(test_running_batch_accuracy)

    print('TESTING COMPLETE!!')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
    report = classification_report(test_all_labels, test_all_preds, target_names=(['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']))
    print(report)
    return test_loss, test_accuracy, report

class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DropoutNet(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer after the first FC layer
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Applying dropout after activation
        x = self.fc2(x)
        return x

class BatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x

def plot_adjusting_lr(initial_lr=0.1, decay_rate=0.01, num_epochs=50):
    print('ADJUSTING!')
    learning_rates = []
    epochs = list(range(num_epochs))

    for epoch in epochs:
        lr = initial_lr / (1 + decay_rate * epoch)
        learning_rates.append(lr)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, learning_rates, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'Adjusting Learning Rate\nInitial LR: {initial_lr}, Decay Rate: {decay_rate}')
    plt.grid(True)
    plt.xticks(range(0, num_epochs, 5))
    plt.show()
    


def display_accuracy_heatmap(path_to_load):
    print('hi')
    with open(path_to_load, 'r') as file:
        results = json.load(file)
    
    rates = []
    av_test_losses = []
    av_test_accuracy = []
    for rate, value_dict in results.items():
        rates.append(rate)
        av_test_losses.append(value_dict['av_test_loss'])
        av_test_accuracy.append(value_dict['av_test_accuracy'])
    
    # Creating the DataFrame
    df = pd.DataFrame({
        'Average Test Loss': av_test_losses,
        'Average Test Accuracy': av_test_accuracy
    }, index=rates)
    
    # Applying conditional formatting to highlight the best value in each column
    def highlight_best(column):
        if column.name == 'Average Test Loss':
            is_best = column == column.min()
        else:
            is_best = column == column.max()
        return ['background: green' if v else '' for v in is_best]
    
    styled_df = df.style.apply(highlight_best, axis=0)
    
    return styled_df

def plot_single_model_performance(single_var_multi_run_data, title=None, enforce_axis=False):
    epochs = range(1, len(single_var_multi_run_data['av_train_losses']) + 1)
    n_runs = len(single_var_multi_run_data['all_train_losses'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if title:
        title += f' across {n_runs} runs'
        fig.suptitle(title, fontsize=12)

    # Plot losses
    for train_loss, val_loss in zip(single_var_multi_run_data['all_train_losses'], single_var_multi_run_data['all_val_losses']):
        ax1.plot(epochs, train_loss, color='blue', alpha=0.3, linewidth=0.5, label='Individual Run Training Losses')
        ax1.plot(epochs, val_loss, color='orange', alpha=0.3, linewidth=0.5, label='Individual Run Validation Losses')
    ax1.plot(epochs, single_var_multi_run_data['av_train_losses'], color='blue', linewidth=1.2, label='Average Training Loss')
    ax1.plot(epochs, single_var_multi_run_data['av_val_losses'], color='orange', linewidth=1.2, label='Average Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Losses')
    
    # Remove duplicate labels in the legend
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = ["Average Training Loss", "Average Validation Loss", "Individual Run Training Losses", "Individual Run Validation Losses"]
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax1.legend(unique_handles, unique_labels)

    # Plot accuracies
    for train_acc, val_acc in zip(single_var_multi_run_data['all_train_accuracies'], single_var_multi_run_data['all_val_accuracies']):
        ax2.plot(epochs, train_acc, color='blue', alpha=0.3, linewidth=0.5, label='Individual Run Training Accuracies')
        ax2.plot(epochs, val_acc, color='orange', alpha=0.3, linewidth=0.5, label='Individual Run Validation Accuracies')
    ax2.plot(epochs, single_var_multi_run_data['av_train_acc'], color='blue', linewidth=1.2, label='Average Training Accuracy')
    ax2.plot(epochs, single_var_multi_run_data['av_val_acc'], color='orange', linewidth=1.2, label='Average Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracies')
    
    # Remove duplicate labels in the legend
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = ["Average Training Accuracy", "Average Validation Accuracy", "Individual Run Training Accuracies", "Individual Run Validation Accuracies"]
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax2.legend(unique_handles, unique_labels)
    
    if enforce_axis:
        ax1.set_ylim(0, 5)
        ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()    

def plot_all_models_performance_from_disk(path_to_load, variable_name=None, enforce_axis=False):
    with open(path_to_load, 'r') as file:
        averaged_results = json.load(file)
        
    for variable_val, data in averaged_results.items():
        plot_single_model_performance(data, title=f'Training/Validation Losses and Accuracy for {variable_name} = {variable_val} across', enforce_axis=enforce_axis)

def plot_performance_comparison_from_file(path_to_load, enforce_axis=False):
    with open(path_to_load, 'r') as file:
        results = json.load(file)
    learning_rates = list(results.keys())
    num_epochs = len(results[learning_rates[0]]['av_train_losses'])

    # Set the figure size
    fig_size = (12, 8)

    # Create a single figure with a 2x2 grid of subplots
    fig, ((ax_train_loss, ax_train_acc), (ax_val_loss, ax_val_acc)) = plt.subplots(2, 2, figsize=fig_size)

    # Add main titles above each pair of plots
    fig.text(0.5, 0.95, 'Performance During Training (averages)', ha='center', fontsize=14)
    fig.text(0.5, 0.48, 'Performance During Validation (averages)', ha='center', fontsize=14)

    # Plot average training loss
    for lr in learning_rates:
        ax_train_loss.plot(range(1, num_epochs + 1), results[lr]['av_train_losses'], label=str(lr))
    ax_train_loss.set_xlabel('Epoch')
    ax_train_loss.set_ylabel('Average Training Loss')
    ax_train_loss.set_title('Losses')
    ax_train_loss.legend(title='Learning Rates', loc='upper right')

    # Plot average training accuracy
    for lr in learning_rates:
        ax_train_acc.plot(range(1, num_epochs + 1), results[lr]['av_train_acc'], label=str(lr))
    ax_train_acc.set_xlabel('Epoch')
    ax_train_acc.set_ylabel('Average Training Accuracy')
    ax_train_acc.set_title('Accuracies')
    ax_train_acc.legend(title='Learning Rates', loc='upper right')

    # Plot average validation loss
    for lr in learning_rates:
        ax_val_loss.plot(range(1, num_epochs + 1), results[lr]['av_val_losses'], label=str(lr))
    ax_val_loss.set_xlabel('Epoch')
    ax_val_loss.set_ylabel('Average Validation Loss')
    ax_val_loss.set_title('Losses')
    ax_val_loss.legend(title='Learning Rates', loc='upper right')

    # Plot average validation accuracy
    for lr in learning_rates:
        ax_val_acc.plot(range(1, num_epochs + 1), results[lr]['av_val_acc'], label=str(lr))
    ax_val_acc.set_xlabel('Epoch')
    ax_val_acc.set_ylabel('Average Validation Accuracy')
    ax_val_acc.set_title('Accuracies')
    ax_val_acc.legend(title='Learning Rates', loc='upper right')

    if enforce_axis:
        ax_val_acc.set_ylim(0, 1)
        ax_val_loss.set_ylim(0, 5)
        ax_train_acc.set_ylim(0, 1)
        ax_train_loss.set_ylim(0, 5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the spacing and positioning of subplots
    plt.show()
    # Create an additional figure based on the number of items being compared
    if len(learning_rates) > 2:
        fig_acc, (ax_train_acc_new, ax_val_acc_new) = plt.subplots(1, 2, figsize=(12, 4))
        fig_acc.suptitle('Comparative Accuracies', fontsize=12)
        # Plot training accuracy
        for lr in learning_rates:
            ax_train_acc_new.plot(range(1, num_epochs + 1), results[lr]['av_train_acc'], label=str(lr))
        ax_train_acc_new.set_xlabel('Epoch')
        ax_train_acc_new.set_ylabel('Average Training Accuracy')
        ax_train_acc_new.set_title('Training Accuracy')
        ax_train_acc_new.legend(title='Learning Rates', loc='upper right')
        
        # Plot validation accuracy
        for lr in learning_rates:
            ax_val_acc_new.plot(range(1, num_epochs + 1), results[lr]['av_val_acc'], label=str(lr))
        ax_val_acc_new.set_xlabel('Epoch')
        ax_val_acc_new.set_ylabel('Average Validation Accuracy')
        ax_val_acc_new.set_title('Validation Accuracy')
        ax_val_acc_new.legend(title='Learning Rates', loc='upper right')
        if enforce_axis:
            ax_val_acc.set_ylim(0, 1)
            ax_train_acc.set_ylim(0, 1)
        plt.tight_layout()
        plt.show()
    
    elif len(learning_rates) == 2:
        fig_acc_two, ax_acc_two = plt.subplots(figsize=(6, 4))
        fig_acc_two.suptitle('Comparative Accuracies', fontsize=12)

        for lr in learning_rates:
            ax_acc_two.plot(range(1, num_epochs + 1), results[lr]['av_val_acc'], label=f"Validation ({lr})", linestyle='-')
            ax_acc_two.plot(range(1, num_epochs + 1), results[lr]['av_train_acc'], label=f"Training ({lr})", linestyle='--')
        
        ax_acc_two.set_xlabel('Epoch')
        ax_acc_two.set_ylabel('Accuracy')
        ax_acc_two.set_title('Accuracy Comparison')
        ax_acc_two.legend(loc='upper right')
        
        if enforce_axis:
            ax_acc_two.set_ylim(0, 1)
            
        plt.tight_layout()
        plt.show()


def plot_single_train_val_smoothed(train_epoch_losses, val_epoch_losses, train_epoch_accuracy, val_epoch_accuracy, num_epochs, smoothing_window=5, title=None):
    # Convert lists to pandas Series
    train_epoch_losses_series = pd.Series(train_epoch_losses)
    val_epoch_losses_series = pd.Series(val_epoch_losses)
    train_epoch_accuracy_series = pd.Series(train_epoch_accuracy)
    val_epoch_accuracy_series = pd.Series(val_epoch_accuracy)

    # Calculate moving averages using the provided smoothing window
    smooth_train_epoch_losses = train_epoch_losses_series.rolling(window=smoothing_window).mean()
    smooth_val_epoch_losses = val_epoch_losses_series.rolling(window=smoothing_window).mean()
    smooth_train_epoch_accuracy = train_epoch_accuracy_series.rolling(window=smoothing_window).mean()
    smooth_val_epoch_accuracy = val_epoch_accuracy_series.rolling(window=smoothing_window).mean()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Plot training and validation loss with moving averages
    ax[0].plot(train_epoch_losses, label='Training Loss', alpha=0.3)
    ax[0].plot(val_epoch_losses, label='Validation Loss', alpha=0.3)
    ax[0].plot(smooth_train_epoch_losses, label='Smoothed Training Loss', color='blue')
    ax[0].plot(smooth_val_epoch_losses, label='Smoothed Validation Loss', color='orange')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].legend()

    # Set x-axis to show each epoch as a tick
    ax[1].set_xticks(range(0, num_epochs + 1, 10))

    # Plot training and validation accuracy with moving averages
    ax[1].plot(train_epoch_accuracy, label='Training Accuracy', alpha=0.3)
    ax[1].plot(val_epoch_accuracy, label='Validation Accuracy', alpha=0.3)
    ax[1].plot(smooth_train_epoch_accuracy, label='Smoothed Training Accuracy', color='blue')
    ax[1].plot(smooth_val_epoch_accuracy, label='Smoothed Validation Accuracy', color='orange')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training and Validation Accuracy')
    ax[1].legend()

    # Set x-axis to show each epoch as a tick
    ax[1].set_xticks(range(0, num_epochs + 1, 10))

    # Set y-axis for accuracy to range from 0 to 1 with ticks at intervals of 0.1
    ax[1].set_ylim(0, 1)
    ax[1].set_yticks([i * 0.1 for i in range(11)])
    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()

'''
old/unused:


def print_all_averaged_runs_together_from_file(path_to_load, num_epochs, title=None):
        # Reading from the file
    with open(path_to_load, 'r') as file:
        averaged_results = json.load(file)
    # Assuming num_epochs was defined somewhere in your previous code.
    epochs = range(num_epochs)
    
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with 1x2 subplots array
    if title:
        fig.suptitle(title, fontsize=16)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Color for each learning rate (add more if needed)
    color_index = 0

    for learning_rate, metrics in averaged_results.items():
        # Extract averages from the dictionary
        avg_train_losses = metrics['av_train_losses']
        avg_val_losses = metrics['av_val_losses']
        avg_train_accuracies = metrics['av_train_acc']
        avg_val_accuracies = metrics['av_val_acc']
        
        # Select color
        color = colors[color_index % len(colors)]
        color_index += 1

        # Subplot for loss
        axs[0].plot(epochs, avg_train_losses, linestyle='--', color=color, label=f'Train LR={learning_rate}')
        axs[0].plot(epochs, avg_val_losses, linestyle='-', color=color, label=f'Val LR={learning_rate}')
        axs[0].set_title('Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_ylim(0, 3)  # Set y-axis limits for loss
        axs[0].set_yticks([i * 0.2 for i in range(16)])  # Set ticks from 0 to 3 with interval 0.2
        axs[0].legend()

        # Subplot for accuracy
        axs[1].plot(epochs, avg_train_accuracies, linestyle='--', color=color, label=f'Train LR={learning_rate}')
        axs[1].plot(epochs, avg_val_accuracies, linestyle='-', color=color, label=f'Val LR={learning_rate}')
        axs[1].set_title('Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_ylim(0, 1)  # Set y-axis limits for accuracy
        axs[1].set_yticks([i * 0.1 for i in range(11)])  # Set ticks from 0 to 1 with interval 0.1
        axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplots to fit into figure area.
    plt.show()
    
def print_sep_runs_from_file(path_to_load, num_epochs, manual_ticks = False, ticks = [0,0,0,0]):
                # Reading from the file
    with open(path_to_load, 'r') as file:
        averaged_results = json.load(file)
        # Assuming num_epochs was defined somewhere in your previous code.
    accuracy_y_limit, accuracy_y_ticks, accuracy_x_limit, accuracy_x_ticks = ticks
    
    epochs = np.arange(num_epochs)  # Easier to work with NumPy array for indexing

    # Setup the colors for individual runs and for the average
    run_colors = ['lightblue', 'lightgreen', 'salmon', 'lightgrey', 'lavender']
    avg_color = 'black'  # Color for averaged data

    for learning_rate in averaged_results:
        data = averaged_results[learning_rate]
        
        # Create figures
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1x2 subplot array for each learning rate
        fig.suptitle(f'Training and Validation Performance for Learning Rate = {learning_rate}')
        
        # Plotting Losses
        # Individual runs
        for i in range(len(averaged_results.keys())):
            axs[0].plot(epochs, data['all_train_losses'][i], color=run_colors[i], alpha=0.5, label=f'Train Run {i+1}' if i == 0 else "")
            axs[0].plot(epochs, data['all_val_losses'][i], color=run_colors[i], alpha=0.5, label=f'Val Run {i+1}' if i == 0 else "")
        
        # Averaged data
        axs[0].plot(epochs, data['av_train_losses'], color=avg_color, linewidth=2, label='Avg Train Loss')
        axs[0].plot(epochs, data['av_val_losses'], color=avg_color, linewidth=2, label='Avg Val Loss')
        axs[0].set_title('Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        if manual_ticks:
            axs[0].set_ylim(0, accuracy_x_limit)  # Set the y-axis range for loss
            axs[0].set_yticks(np.arange(0, accuracy_x_limit, accuracy_x_ticks))  # Set y-tick
            
        axs[0].legend()

        # Plotting Accuracies
        # Individual runs
        for i in range(len(averaged_results.keys())):
            axs[1].plot(epochs, data['all_train_accuracies'][i], color=run_colors[i], alpha=0.5, label=f'Train Acc Run {i+1}' if i == 0 else "")
            axs[1].plot(epochs, data['all_val_accuracies'][i], color=run_colors[i], alpha=0.5, label=f'Val Acc Run {i+1}' if i == 0 else "")
        
        # Averaged data
        axs[1].plot(epochs, data['av_train_acc'], color=avg_color, linewidth=2, label='Avg Train Accuracy')
        axs[1].plot(epochs, data['av_val_acc'], color=avg_color, linewidth=2, label='Avg Val Accuracy')
        axs[1].set_title('Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        if manual_ticks:
            axs[1].set_ylim(0, accuracy_y_limit)  # Set the y-axis range for accuracy
            axs[1].set_yticks(np.arange(0, 1, accuracy_y_ticks))  # Set y-ticks at intervals of 0.8
        axs[1].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplots to fit into figure area.
        plt.show()
        
'''