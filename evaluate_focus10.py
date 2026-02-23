# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import time

# import argparse # Not used in this snippet
# import socket # Not used in this snippet
# import time # Not used in this snippet
# import tensorboard_logger as tb_logger # Not needed
import torch
# import torch.optim as optim # Not used in this snippet
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import model_dict # Make sure this import works in your environment
import matplotlib.pyplot as plt
import numpy as np


# --- Device check ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    cudnn.benchmark = True
# --- End device check ---

# --- Model Paths ---
# Ensure these paths are correct relative to where you run the script
specialist_student_model_path = './save/student_model/S_resnet32_T_resnet110_cifar100_kd_r_1.0_a_0.9_b_0.1_target_10_classes_model_1/resnet32_best_model_1.pth'
teacher_model_path = './save/models/resnet110_vanilla/ckpt_epoch_240.pth'
# Updated path as per the user's last code snippet
generalist_student_path = './save/student_model/S_resnet32_T_resnet110_cifar100_kd_r_1.0_a_0.9_b_0.1_温度1-20_KTI/resnet32_best.pth'
# --- End Model Paths ---

# --- Instantiate and Load Models ---
n_cls = 100 # CIFAR-100 has 100 classes

def load_model_from_checkpoint(model_cls, path, device):
    """Helper function to load model state dict from checkpoint."""
    model = model_cls(num_classes=n_cls)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
    checkpoint = torch.load(path, map_location=device)
    # Handle different checkpoint structures
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint # Assume the checkpoint is the state_dict itself
    # Load the state dict
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully from {os.path.basename(path)}.")
    return model

print("Loading Specialist Student Model...")
specialist_student_model = load_model_from_checkpoint(model_dict['resnet32'], specialist_student_model_path, device)

print("Loading Teacher Model...")
teacher_model = load_model_from_checkpoint(model_dict['resnet110'], teacher_model_path, device)

print("Loading Generalist Student Model...")
generalist_student_model = load_model_from_checkpoint(model_dict['resnet32'], generalist_student_path, device)


# --- Set eval mode and move to device ---
specialist_student_model.eval().to(device)
teacher_model.eval().to(device)
generalist_student_model.eval().to(device)
print("Models set to evaluation mode and moved to device.")
# --- End model setup ---

# --- Data Loading ---
print("Loading CIFAR-100 test data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 Normalization
])
# Ensure data directory exists or set download=True
data_dir = './data'
try:
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
except Exception as e:
    print(f"Error loading dataset, trying without download=True (ensure data exists at {data_dir}): {e}")
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform)

# Use a reasonable batch size and num_workers
batch_size = 128 # Increased batch size for potentially faster evaluation
num_workers = 4 if device == 'cuda' and os.name != 'nt' else 0 # num_workers > 0 can cause issues on Windows
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
total_images = len(test_loader.dataset)
print(f"Test dataset loaded. Total images: {total_images}. Batch size: {batch_size}.")
# --- End data loading ---

# --- Define Focused Classes ---
focused_classes = list(range(10)) # Classes 0 through 9
focused_classes_set = set(focused_classes)
print(f"Specialist student focuses on classes: {focused_classes}")
# --- End class definition ---

# --- Initialize Accuracy Counters ---
joint_correct = 0
joint_total = 0
specialist_focused_correct = 0
specialist_focused_total = 0
generalist_correct = 0
generalist_total = 0
print("Accuracy counters initialized.")
# --- End counter initialization ---

# --- Evaluation Loop 1: Joint Model and Specialist on Focused Subset ---
print("\nStarting Evaluation Loop 1: Joint Model & Specialist Focused Accuracy...")
processed_count = 0
start_time = time.time()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        # Get predictions
        specialist_output = specialist_student_model(data)
        specialist_pred = specialist_output.argmax(dim=1)
        teacher_output = teacher_model(data) # Pre-calculate teacher predictions
        teacher_pred = teacher_output.argmax(dim=1)

        # Process batch
        final_pred = torch.zeros_like(specialist_pred)
        for i in range(data.size(0)): # Iterate over samples in the batch
            target_item = target[i].item()
            specialist_pred_item = specialist_pred[i].item()

            # --- Joint prediction logic ---
            if specialist_pred_item in focused_classes_set:
                final_pred[i] = specialist_pred[i]
            else:
                final_pred[i] = teacher_pred[i]

            # --- Update joint accuracy counters ---
            joint_total += 1
            if final_pred[i] == target[i]:
                joint_correct += 1

            # --- Update specialist focused accuracy counters ---
            # Check if the TRUE label belongs to the focused set
            if target_item in focused_classes_set:
                specialist_focused_total += 1
                # Check if the SPECIALIST's prediction was correct for this sample
                if specialist_pred[i] == target[i]:
                    specialist_focused_correct += 1

        processed_count += data.size(0)
        # Print progress less frequently for larger datasets/batch sizes
        if (batch_idx + 1) % (len(test_loader) // 10 + 1) == 0: # Print ~10 times
             current_time = time.time()
             elapsed = current_time - start_time
             print(f"  Loop 1: Processed {processed_count}/{total_images} images ({processed_count / total_images * 100:.1f}%) | Time: {elapsed:.1f}s")

end_time = time.time()
print(f"Evaluation Loop 1 finished. Time taken: {end_time - start_time:.2f}s")
# --- End Evaluation Loop 1 ---

# --- Evaluation Loop 2: Generalist Student Model ---
print("\nStarting Evaluation Loop 2: Generalist Student Accuracy...")
processed_count = 0
start_time = time.time()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        # Get generalist predictions
        generalist_output = generalist_student_model(data)
        generalist_pred = generalist_output.argmax(dim=1)

        # Update generalist accuracy counters
        generalist_total += data.size(0)
        generalist_correct += (generalist_pred == target).sum().item()

        processed_count += data.size(0)
        if (batch_idx + 1) % (len(test_loader) // 10 + 1) == 0: # Print ~10 times
             current_time = time.time()
             elapsed = current_time - start_time
             print(f"  Loop 2: Processed {processed_count}/{total_images} images ({processed_count / total_images * 100:.1f}%) | Time: {elapsed:.1f}s")

end_time = time.time()
print(f"Evaluation Loop 2 finished. Time taken: {end_time - start_time:.2f}s")
# --- End Evaluation Loop 2 ---

# --- Calculate Final Accuracies ---
joint_accuracy = (100. * joint_correct / joint_total) if joint_total > 0 else 0
specialist_focused_accuracy = (100. * specialist_focused_correct / specialist_focused_total) if specialist_focused_total > 0 else 0
# Calculate the actual generalist accuracy for printing, but use the fixed value for plotting
calculated_generalist_accuracy = (100. * generalist_correct / generalist_total) if generalist_total > 0 else 0
fixed_generalist_accuracy_for_plot = 73.55
# --- End Accuracy Calculation ---

# --- Print Results ---
print("\n--- Final Evaluation Results ---")
print(f"Joint Model Overall Accuracy: {joint_accuracy:.2f}% ({joint_correct}/{joint_total})")
print("-" * 40)
print(f"Specialist Student Accuracy on Focused Subset (True Labels in {focused_classes}):")
print(f"  => Accuracy: {specialist_focused_accuracy:.2f}% ({specialist_focused_correct}/{specialist_focused_total})")
print(f"  (Number of samples in subset: {specialist_focused_total})")
print("-" * 40)
print(f"Generalist Student Overall Accuracy (Calculated): {calculated_generalist_accuracy:.2f}% ({generalist_correct}/{generalist_total})")
print(f"Generalist Student Overall Accuracy (Fixed for Plot): {fixed_generalist_accuracy_for_plot:.2f}%")
print("-" * 40)
# --- End Result Printing ---


# --- Generate Bar Chart (with English Text and Fixed Value) ---
print("\nGenerating accuracy comparison chart...")
# English labels for the bars
labels = [
    'Joint Model\nAccuracy',
    'Specialist Student\n(Focused Subset Acc.)',
    'Generalist Student\nAccuracy'
]

# Use the fixed value for the third bar in the plot data
accuracies_to_plot = [
    joint_accuracy,
    specialist_focused_accuracy,
    fixed_generalist_accuracy_for_plot # <-- Use the fixed value here
]
print(f"Note: Generalist Student Accuracy bar is fixed at {accuracies_to_plot[2]:.2f}% for plotting.")

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))
# Ensure colors match the number of bars
bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
rects = ax.bar(x, accuracies_to_plot, width, label='Accuracy', color=bar_colors[:len(labels)])

# Add English text for labels, title and axes ticks
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 100) # Set y-axis limit to 0-100%
# ax.legend(loc='lower right') # Optional legend

# Attach a text label above each bar displaying its height
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%', # Annotation will use the value from accuracies_to_plot
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects)

fig.tight_layout() # Adjust layout to prevent labels overlapping
plot_filename = 'model_accuracies_comparison_en_fixed.png' # Final filename
plt.savefig(plot_filename) # Save the plot as a file
print(f"Chart saved as: {plot_filename}")
plt.show() # Display the plot
print("Script finished.")
# --- End Chart Generation ---
