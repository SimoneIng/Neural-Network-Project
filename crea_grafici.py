import json
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tabulate import tabulate
import seaborn as sns

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create and save plots from neural network training data.')
    parser.add_argument('params_file', type=str, help='Path to the JSON file containing hyperparameters.')
    parser.add_argument('metrics_file', type=str, help='Path to the JSON file containing training metrics.')
    parser.add_argument('--class_metrics_file', type=str, help='Path to the JSON file containing classification metrics.', default=None)
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save the plots to.')
    return parser.parse_args()

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_statistics(params, metrics):
    """Calculate various statistics for each trial."""
    statistics = {}
    
    for trial_id in metrics.keys():
        trial_stats = {}
        trial_metrics = metrics[trial_id]
        trial_params = params[trial_id]
        
        # Basic statistics
        trial_stats['max_val_acc'] = max(trial_metrics['val_acc'])
        trial_stats['max_val_acc_epoch'] = trial_metrics['val_acc'].index(trial_stats['max_val_acc']) + 1
        trial_stats['final_val_acc'] = trial_metrics['val_acc'][-1]
        trial_stats['final_train_acc'] = trial_metrics['train_acc'][-1]
        trial_stats['final_val_loss'] = trial_metrics['val_loss'][-1]
        trial_stats['final_train_loss'] = trial_metrics['train_loss'][-1]
        
        # Convergence speed (epochs to reach 90% of max validation accuracy)
        target_acc = 0.9 * trial_stats['max_val_acc']
        for epoch, acc in enumerate(trial_metrics['val_acc']):
            if acc >= target_acc:
                trial_stats['epochs_to_90pct'] = epoch + 1
                break
        else:
            trial_stats['epochs_to_90pct'] = None
        
        # Overfitting metrics
        trial_stats['overfitting'] = trial_stats['final_train_acc'] - trial_stats['final_val_acc']
        
        # Learning stability (std dev of last 5 epochs or fewer if not available)
        last_n = min(5, len(trial_metrics['val_acc']))
        trial_stats['stability'] = np.std(trial_metrics['val_acc'][-last_n:])
        
        # Early learning rate (average improvement in first 5 epochs or fewer if not available)
        first_n = min(5, len(trial_metrics['val_acc']) - 1)
        if first_n > 0:
            improvements = [trial_metrics['val_acc'][i+1] - trial_metrics['val_acc'][i] for i in range(first_n)]
            trial_stats['early_learning_rate'] = np.mean(improvements) if improvements else 0
        else:
            trial_stats['early_learning_rate'] = 0
            
        # Architecture complexity
        if 'num_conv_layers' in trial_params and 'hidden_layers' in trial_params:
            trial_stats['total_layers'] = trial_params['num_conv_layers'] + trial_params['hidden_layers']
        
        # Store all parameters for reference
        trial_stats['params'] = trial_params
        
        statistics[trial_id] = trial_stats
    
    return statistics

def find_best_trial(statistics, criteria='final_val_acc'):
    """Find the best trial based on given criteria."""
    best_trial_id = max(statistics.keys(), key=lambda k: statistics[k][criteria])
    return best_trial_id, statistics[best_trial_id]

def plot_classification_metrics(class_metrics, output_dir, trial_id="best"):
    """Create plots for classification metrics such as precision, recall, and F1-score."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract per-class metrics
    class_labels = [k for k in class_metrics.keys() if k.isdigit()]
    
    metrics_data = {
        'precision': [class_metrics[c]['precision'] for c in class_labels],
        'recall': [class_metrics[c]['recall'] for c in class_labels],
        'f1-score': [class_metrics[c]['f1-score'] for c in class_labels],
        'support': [class_metrics[c]['support'] for c in class_labels]
    }
    
    # Plot precision, recall, and F1-score for each class
    plt.figure(figsize=(14, 8))
    bar_width = 0.25
    index = np.arange(len(class_labels))
    
    plt.bar(index, metrics_data['precision'], bar_width, label='Precision', color='blue', alpha=0.7)
    plt.bar(index + bar_width, metrics_data['recall'], bar_width, label='Recall', color='green', alpha=0.7)
    plt.bar(index + 2*bar_width, metrics_data['f1-score'], bar_width, label='F1-Score', color='red', alpha=0.7)
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title(f'Classification Metrics by Class (Trial {trial_id})')
    plt.xticks(index + bar_width, class_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trial_{trial_id}_class_metrics.png'))
    plt.close()
    
    # Plot support (number of samples) for each class
    plt.figure(figsize=(12, 6))
    plt.bar(class_labels, metrics_data['support'], color='purple', alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title(f'Class Distribution in Test Set (Trial {trial_id})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trial_{trial_id}_class_distribution.png'))
    plt.close()
    
    # Create a heatmap-style visualization for precision, recall, f1-score
    plt.figure(figsize=(10, 8))
    metrics_df = pd.DataFrame({
        'Class': class_labels,
        'Precision': metrics_data['precision'],
        'Recall': metrics_data['recall'],
        'F1-Score': metrics_data['f1-score']
    }).set_index('Class')
    
    # Create heatmap
    sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
    plt.title(f'Classification Performance Heatmap (Trial {trial_id})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trial_{trial_id}_metrics_heatmap.png'))
    plt.close()
    
    # Create a radar chart for each class
    if len(class_labels) <= 10:  # For digit classification, we know we have 10 classes
        # Set up the radar chart
        fig = plt.figure(figsize=(14, 10))
        
        # Define metrics to compare
        metrics_names = ['Precision', 'Recall', 'F1-Score']
        
        # Calculate the number of rows and columns for subplots
        n_rows = int(np.ceil(len(class_labels) / 5))
        n_cols = min(5, len(class_labels))
        
        # Create a radar chart for each class
        for i, class_label in enumerate(class_labels):
            ax = fig.add_subplot(n_rows, n_cols, i+1, polar=True)
            
            # Set the angles for the metrics
            angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Set the values for this class
            values = [
                class_metrics[class_label]['precision'],
                class_metrics[class_label]['recall'],
                class_metrics[class_label]['f1-score']
            ]
            values += values[:1]  # Close the loop
            
            # Plot the values
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            
            # Set the labels
            ax.set_thetagrids(np.degrees(angles[:-1]), metrics_names)
            
            # Set the y-ticks
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_ylim(0, 1)
            
            # Set the title
            ax.set_title(f'Class {class_label}', size=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'trial_{trial_id}_class_radar.png'))
        plt.close()
    
    # Create a single plot with all metrics sorted by F1-score
    plt.figure(figsize=(14, 8))
    
    # Sort classes by F1-score
    sorted_indices = np.argsort(metrics_data['f1-score'])[::-1]  # Sort in descending order
    sorted_classes = [class_labels[i] for i in sorted_indices]
    sorted_precision = [metrics_data['precision'][i] for i in sorted_indices]
    sorted_recall = [metrics_data['recall'][i] for i in sorted_indices]
    sorted_f1 = [metrics_data['f1-score'][i] for i in sorted_indices]
    
    x = np.arange(len(sorted_classes))
    
    plt.bar(x - bar_width, sorted_precision, bar_width, label='Precision', color='blue', alpha=0.7)
    plt.bar(x, sorted_recall, bar_width, label='Recall', color='green', alpha=0.7)
    plt.bar(x + bar_width, sorted_f1, bar_width, label='F1-Score', color='red', alpha=0.7)
    
    # Add text annotations for F1-scores
    for i, v in enumerate(sorted_f1):
        plt.text(i + bar_width, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.xlabel('Class (Sorted by F1-Score)')
    plt.ylabel('Score')
    plt.title(f'Classification Performance Metrics (Trial {trial_id})')
    plt.xticks(x, sorted_classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trial_{trial_id}_sorted_metrics.png'))
    plt.close()
    
    return class_metrics['accuracy']

def generate_summary_report(statistics, best_trial_id, class_metrics=None, output_dir='plots'):
    """Generate a summary report of all trials and save it as a text file."""
    # Create a DataFrame for better formatting
    report_data = []
    columns = ['Trial ID', 'Final Val Acc', 'Max Val Acc', 'Epoch of Max', 'Overfitting', 'Stability', 
               'Early Learning Rate', 'Optimizer', 'Batch Size', 'Conv Layers', 'Hidden Layers']
    
    for trial_id, stats in statistics.items():
        row = [
            trial_id,
            f"{stats['final_val_acc']}",
            f"{stats['max_val_acc']}",
            stats['max_val_acc_epoch'],
            f"{stats['overfitting']}",
            f"{stats['stability']}",
            f"{stats['early_learning_rate']}",
            stats['params']['optimizer'],
            stats['params']['batch_size'],
            stats['params'].get('num_conv_layers', 'num_conv_layers N/A'),
            stats['params']['hidden_layers']
        ]
        report_data.append(row)
    
    # Sort by final validation accuracy
    report_data.sort(key=lambda x: float(x[1]), reverse=True)
    
    # Create the report
    report = "# Neural Network Training Results Summary\n\n"
    
    # Add best trial information
    best_stats = statistics[best_trial_id]
    report += f"## Best Trial: {best_trial_id}\n"
    report += f"- Final Validation Accuracy: {best_stats['final_val_acc']}\n"
    report += f"- Maximum Validation Accuracy: {best_stats['max_val_acc']} (Epoch {best_stats['max_val_acc_epoch']})\n"
    report += f"- Optimizer: {best_stats['params']['optimizer']}\n"
    report += f"- Batch Size: {best_stats['params']['batch_size']}\n"

    if 'num_conv_layers' in best_stats['params']:
        report += f"- Architecture: {best_stats['params']['num_conv_layers']} conv layers, {best_stats['params']['hidden_layers']} hidden layers\n"

    report += f"- Learning Rate: {best_stats['params']['learning_rate']}\n"
    report += f"- Activation Function: {best_stats['params']['activation_fn']}\n"
    
    # Add classification metrics if available
    if class_metrics:
        report += f"- Overall Test Accuracy: {class_metrics['accuracy']}\n"
        report += f"- Macro Average F1-Score: {class_metrics['macro avg']['f1-score']}\n"
        report += f"- Weighted Average F1-Score: {class_metrics['weighted avg']['f1-score']}\n"
        
        # Add per-class F1 scores
        report += "\n### Per-Class Performance (Best Trial)\n"
        report += "| Class | Precision | Recall | F1-Score | Support |\n"
        report += "|-------|-----------|--------|----------|--------|\n"
        
        for cls in sorted([k for k in class_metrics.keys() if k.isdigit()], key=int):
            cls_metrics = class_metrics[cls]
            report += f"| {cls} | {cls_metrics['precision']} | {cls_metrics['recall']} | {cls_metrics['f1-score']} | {int(cls_metrics['support'])} |\n"
        
        # Add class with best and worst F1 scores
        digit_classes = [k for k in class_metrics.keys() if k.isdigit()]
        best_f1_class = max(digit_classes, key=lambda c: class_metrics[c]['f1-score'])
        worst_f1_class = min(digit_classes, key=lambda c: class_metrics[c]['f1-score'])
        
        report += f"\n- Best Recognized Class: {best_f1_class} (F1-Score: {class_metrics[best_f1_class]['f1-score']})\n"
        report += f"- Worst Recognized Class: {worst_f1_class} (F1-Score: {class_metrics[worst_f1_class]['f1-score']})\n"
    
    report += "\n## All Trials Comparison\n"
    report += tabulate(report_data, headers=columns, tablefmt="pipe") + "\n\n"
    
    # Add explanations
    report += "## Metrics Explanation\n"
    report += "- **Final Val Acc**: Validation accuracy at the final epoch\n"
    report += "- **Max Val Acc**: Maximum validation accuracy achieved\n"
    report += "- **Epoch of Max**: Epoch where maximum validation accuracy was achieved\n"
    report += "- **Overfitting**: Difference between final training and validation accuracy (higher means more overfitting)\n"
    report += "- **Stability**: Standard deviation of validation accuracy in last 5 epochs (lower means more stable)\n"
    report += "- **Early Learning Rate**: Average improvement in validation accuracy during first 5 epochs (higher means faster initial learning)\n"
    
    if class_metrics:
        report += "- **Precision**: The ratio of true positives to all predicted positives (TP / (TP + FP))\n"
        report += "- **Recall**: The ratio of true positives to all actual positives (TP / (TP + FN))\n"
        report += "- **F1-Score**: The harmonic mean of precision and recall (2 * Precision * Recall / (Precision + Recall))\n"
        report += "- **Support**: The number of samples in each class\n"
    
    # Save the report
    with open(os.path.join(output_dir, 'summary_report.md'), 'w') as f:
        f.write(report)
    
    return report

def create_and_save_plots(params, metrics, statistics, best_trial_id, output_dir):
    """Create and save various plots."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # For each trial
    for trial_id in metrics.keys():
        trial_params = params[trial_id]
        trial_metrics = metrics[trial_id]
        
        epochs = list(range(1, len(trial_metrics['train_loss']) + 1))
        
        # Create a figure for loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, trial_metrics['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, trial_metrics['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        title_suffix = " (BEST TRIAL)" if trial_id == best_trial_id else ""
        plt.title(f'Trial {trial_id} - Loss vs Epoch{title_suffix}\n'
                  f'Batch Size: {trial_params["batch_size"]}, '
                  f'Optimizer: {trial_params["optimizer"]}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'trial_{trial_id}_loss.png'))
        plt.close()
        
        # Create a figure for accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, trial_metrics['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(epochs, trial_metrics['val_acc'], 'r-', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Trial {trial_id} - Accuracy vs Epoch{title_suffix}\n'
                  f'Batch Size: {trial_params["batch_size"]}, '
                  f'Optimizer: {trial_params["optimizer"]}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'trial_{trial_id}_accuracy.png'))
        plt.close()
        
        # Plot the difference between training and validation accuracy (overfitting visualization)
        plt.figure(figsize=(12, 6))
        overfitting = [train - val for train, val in zip(trial_metrics['train_acc'], trial_metrics['val_acc'])]
        plt.plot(epochs, overfitting, 'g-')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Training - Validation Accuracy')
        plt.title(f'Trial {trial_id} - Overfitting Analysis{title_suffix}\n'
                  f'Final Overfitting: {overfitting[-1]}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'trial_{trial_id}_overfitting.png'))
        plt.close()
    
    # Create comparison plots
    if len(metrics.keys()) > 1:
        # Find the maximum number of epochs across all trials
        max_epochs = max([len(metrics[trial_id]['train_loss']) for trial_id in metrics.keys()])
        
        # Compare training loss
        plt.figure(figsize=(12, 6))
        for trial_id in metrics.keys():
            trial_epochs = list(range(1, len(metrics[trial_id]['train_loss']) + 1))
            line_style = '-' if trial_id == best_trial_id else '--'
            line_width = 2 if trial_id == best_trial_id else 1.5
            plt.plot(trial_epochs, metrics[trial_id]['train_loss'], 
                     linestyle=line_style, linewidth=line_width,
                     label=f'Trial {trial_id} - {params[trial_id]["optimizer"]}' + (" (BEST)" if trial_id == best_trial_id else ""))
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True)
        plt.xlim(1, max_epochs)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_train_loss.png'))
        plt.close()
        
        # Compare validation loss
        plt.figure(figsize=(12, 6))
        for trial_id in metrics.keys():
            trial_epochs = list(range(1, len(metrics[trial_id]['val_loss']) + 1))
            line_style = '-' if trial_id == best_trial_id else '--'
            line_width = 2 if trial_id == best_trial_id else 1.5
            plt.plot(trial_epochs, metrics[trial_id]['val_loss'], 
                     linestyle=line_style, linewidth=line_width,
                     label=f'Trial {trial_id} - {params[trial_id]["optimizer"]}' + (" (BEST)" if trial_id == best_trial_id else ""))
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Comparison')
        plt.legend()
        plt.grid(True)
        plt.xlim(1, max_epochs)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_val_loss.png'))
        plt.close()
        
        # Compare training accuracy
        plt.figure(figsize=(12, 6))
        for trial_id in metrics.keys():
            trial_epochs = list(range(1, len(metrics[trial_id]['train_acc']) + 1))
            line_style = '-' if trial_id == best_trial_id else '--'
            line_width = 2 if trial_id == best_trial_id else 1.5
            plt.plot(trial_epochs, metrics[trial_id]['train_acc'], 
                     linestyle=line_style, linewidth=line_width,
                     label=f'Trial {trial_id} - {params[trial_id]["optimizer"]}' + (" (BEST)" if trial_id == best_trial_id else ""))
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy Comparison')
        plt.legend()
        plt.grid(True)
        plt.xlim(1, max_epochs)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_train_accuracy.png'))
        plt.close()
        
        # Compare validation accuracy
        plt.figure(figsize=(12, 6))
        for trial_id in metrics.keys():
            trial_epochs = list(range(1, len(metrics[trial_id]['val_acc']) + 1))
            line_style = '-' if trial_id == best_trial_id else '--'
            line_width = 2 if trial_id == best_trial_id else 1.5
            plt.plot(trial_epochs, metrics[trial_id]['val_acc'], 
                     linestyle=line_style, linewidth=line_width,
                     label=f'Trial {trial_id} - {params[trial_id]["optimizer"]}' + (" (BEST)" if trial_id == best_trial_id else ""))
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy Comparison')
        plt.legend()
        plt.grid(True)
        plt.xlim(1, max_epochs)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_val_accuracy.png'))
        plt.close()
        
        # Create a plot comparing final validation accuracy for each trial
        plt.figure(figsize=(10, 6))
        trial_ids = sorted(metrics.keys(), key=lambda x: int(x) if x.isdigit() else x)
        final_val_accs = [metrics[trial_id]['val_acc'][-1] for trial_id in trial_ids]
        optimizer_types = [params[trial_id]['optimizer'] for trial_id in trial_ids]
        
        colors = ['green' if tid == best_trial_id else 'blue' for tid in trial_ids]
        bars = plt.bar(trial_ids, final_val_accs, color=colors)
        plt.xlabel('Trial ID')
        plt.ylabel('Final Validation Accuracy')
        plt.title('Final Validation Accuracy Comparison')
        plt.ylim(0, 1)  # Accuracy is between 0 and 1
        
        # Add optimizer type and final accuracy as text on the bars
        for i, (bar, acc, opt, tid) in enumerate(zip(bars, final_val_accs, optimizer_types, trial_ids)):
            label_suffix = " (BEST)" if tid == best_trial_id else ""
            plt.text(i, acc + 0.02, f'{acc}{label_suffix}', ha='center')
            plt.text(i, acc / 2, opt, ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_final_val_accuracy.png'))
        plt.close()
        
        # Create a radar chart for multi-metric comparison
        if len(metrics.keys()) <= 5:  # Radar charts work best with few categories
            # Select metrics to compare
            metrics_to_compare = ['final_val_acc', 'max_val_acc', 'early_learning_rate', 'stability']
            # Invert stability (lower is better)
            inverted_metrics = ['stability']
            
            # Normalize the metrics for better visualization
            normalized_stats = {}
            for metric in metrics_to_compare:
                values = [statistics[tid][metric] for tid in statistics]
                min_val = min(values)
                max_val = max(values)
                if max_val == min_val:
                    normalized_stats[metric] = {tid: 0.5 for tid in statistics}
                else:
                    normalized_stats[metric] = {}
                    for tid in statistics:
                        if metric in inverted_metrics:
                            # Invert: 1 - normalized value (lower is better)
                            normalized_stats[metric][tid] = 1 - (statistics[tid][metric] - min_val) / (max_val - min_val)
                        else:
                            normalized_stats[metric][tid] = (statistics[tid][metric] - min_val) / (max_val - min_val)
            
            # Set up the radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Set the labels for the chart
            angles = np.linspace(0, 2*np.pi, len(metrics_to_compare), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            plt.xticks(angles[:-1], [m.replace('_', ' ').title() for m in metrics_to_compare])
            
            # Plot each trial
            for trial_id in statistics:
                values = [normalized_stats[metric][trial_id] for metric in metrics_to_compare]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, 
                        linewidth=2 if trial_id == best_trial_id else 1.5,
                        label=f'Trial {trial_id}' + (" (BEST)" if trial_id == best_trial_id else ""))
                ax.fill(angles, values, alpha=0.1)
            
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Multi-Metric Performance Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'radar_comparison.png'))
            plt.close()
            
        # Create a correlation plot between hyperparameters and performance metrics
        # Let's focus on batch_size and learning_rate as these are numerical
        if len(metrics.keys()) >= 3:  # Need at least 3 points for a meaningful plot
            try:
                # Extract hyperparameters and metrics
                hyperparams = {
                    'batch_size': [params[tid]['batch_size'] for tid in params],
                    'learning_rate': [params[tid]['learning_rate'] for tid in params],
                    'num_conv_layers': [params[tid]['num_conv_layers'] for tid in params]
                }
                
                performance = {
                    'final_val_acc': [statistics[tid]['final_val_acc'] for tid in statistics]
                }
                
                # Create plots for each hyperparam vs performance
                for hyperparam in hyperparams:
                    plt.figure(figsize=(10, 6))
                    x = hyperparams[hyperparam]
                    y = performance['final_val_acc']
                    
                    # Highlight the best trial
                    sizes = [100 if tid == best_trial_id else 50 for tid in params]
                    colors = ['green' if tid == best_trial_id else 'blue' for tid in params]
                    
                    plt.scatter(x, y, s=sizes, c=colors)
                    
                    # Add trial labels
                    for i, tid in enumerate(params):
                        plt.annotate(f'Trial {tid}', (x[i], y[i]), 
                                    textcoords="offset points", 
                                    xytext=(0,10), 
                                    ha='center')
                    
                    plt.xlabel(hyperparam.replace('_', ' ').title())
                    plt.ylabel('Final Validation Accuracy')
                    plt.title(f'Impact of {hyperparam.replace("_", " ").title()} on Performance')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'correlation_{hyperparam}.png'))
                    plt.close()
            except Exception as e:
                print(f"Could not create correlation plots: {e}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Load JSON data
    params = load_json(args.params_file)
    metrics = load_json(args.metrics_file)
    
    # Calculate statistics
    statistics = calculate_statistics(params, metrics)
    
    # Find best trial
    best_trial_id, best_stats = find_best_trial(statistics)
    
    # Load classification metrics if provided
    class_metrics = None
    if args.class_metrics_file:
        class_metrics = load_json(args.class_metrics_file)
        # Create plots for classification metrics
        plot_classification_metrics(class_metrics, args.output_dir, best_trial_id)
    
    # Create and save plots
    create_and_save_plots(params, metrics, statistics, best_trial_id, args.output_dir)
    
    # Generate summary report
    report = generate_summary_report(statistics, best_trial_id, class_metrics, args.output_dir)
    
    print(f'Best trial: {best_trial_id} with validation accuracy: {best_stats["final_val_acc"]}')
    if class_metrics:
        print(f'Test accuracy: {class_metrics["accuracy"]}')
    print(f'Plots and report saved to {args.output_dir}/')
    
    # Print summary to console
    print("\nSummary Report:")
    print(report)

if __name__ == '__main__':
    main()