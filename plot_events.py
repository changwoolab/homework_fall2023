import argparse
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tensorboard_data(logdir):
    """Read data from a TensorBoard event file."""
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    
    # Get all scalar tags
    tags = event_acc.Tags()['scalars']
    
    data = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': np.array(steps), 'values': np.array(values)}
    
    return data


def plot_metrics(data, save_path=None, title=None):
    """Plot all metrics from the data dictionary."""
    num_metrics = len(data)
    
    if num_metrics == 0:
        print("No metrics found!")
        return
    
    # Create subplots
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for idx, (metric_name, metric_data) in enumerate(sorted(data.items())):
        ax = axes[idx]
        steps = metric_data['steps']
        values = metric_data['values']
        
        ax.plot(steps, values, linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, y=1.00)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig


def plot_specific_metrics(data, metrics_to_plot, save_path=None, title=None):
    """Plot specific metrics from the data."""
    num_metrics = len(metrics_to_plot)
    
    if num_metrics == 0:
        print("No metrics to plot!")
        return
    
    # Create subplots
    cols = min(2, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 5*rows))
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for idx, metric_name in enumerate(metrics_to_plot):
        if metric_name not in data:
            print(f"Warning: Metric '{metric_name}' not found in data")
            continue
            
        ax = axes[idx]
        metric_data = data[metric_name]
        steps = metric_data['steps']
        values = metric_data['values']
        
        ax.plot(steps, values, linewidth=2)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, y=1.00)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig


def compare_runs(logdirs, labels=None, metrics_to_plot=None, save_path=None):
    """Compare multiple runs by plotting them together."""
    all_data = []
    
    for logdir in logdirs:
        if os.path.isdir(logdir):
            # Find event file in directory
            event_files = [f for f in os.listdir(logdir) if f.startswith('events.out.tfevents')]
            if event_files:
                logdir = os.path.join(logdir, event_files[0])
        
        data = read_tensorboard_data(logdir)
        all_data.append(data)
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(logdirs))]
    
    # Get common metrics
    if metrics_to_plot is None:
        all_metrics = set()
        for data in all_data:
            all_metrics.update(data.keys())
        metrics_to_plot = sorted(all_metrics)
    
    # Plot comparison
    num_metrics = len(metrics_to_plot)
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for data, label in zip(all_data, labels):
            if metric_name in data:
                metric_data = data[metric_name]
                steps = metric_data['steps']
                values = metric_data['values']
                ax.plot(steps, values, linewidth=1.5, label=label, alpha=0.8)
        
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot TensorBoard event files')
    parser.add_argument('logdir', type=str, help='Path to event file or directory containing event file')
    parser.add_argument('--metrics', type=str, nargs='+', help='Specific metrics to plot')
    parser.add_argument('--save', type=str, help='Path to save the plot')
    parser.add_argument('--title', type=str, help='Title for the plot')
    parser.add_argument('--list-metrics', action='store_true', help='List all available metrics')
    
    args = parser.parse_args()
    
    # Handle directory vs file
    logdir = args.logdir
    if os.path.isdir(logdir):
        # Find event file in directory
        event_files = [f for f in os.listdir(logdir) if f.startswith('events.out.tfevents')]
        if not event_files:
            print(f"No event files found in {logdir}")
            return
        logdir = os.path.join(logdir, event_files[0])
        print(f"Reading from: {logdir}")
    
    # Read data
    print("Reading TensorBoard data...")
    data = read_tensorboard_data(logdir)
    
    print(f"Found {len(data)} metrics")
    
    # List metrics if requested
    if args.list_metrics:
        print("\nAvailable metrics:")
        for metric in sorted(data.keys()):
            print(f"  - {metric}")
        return
    
    # Plot
    if args.metrics:
        plot_specific_metrics(data, args.metrics, save_path=args.save, title=args.title)
    else:
        plot_metrics(data, save_path=args.save, title=args.title)


if __name__ == '__main__':
    main()

