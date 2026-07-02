import os
import subprocess
import sys
import multiprocessing
import time

# Define datasets and their paths
# Assuming datasets are in ../dataset relative to this script

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

base_dataset_dir = os.path.join(project_root, "dataset")
base_output_dir = os.path.join(project_root, "saved_models")  # New directory for reproducible models

# Train only the requested datasets.
datasets = [
    "random",
    "EC_split_N2",
]

# Check only the requested datasets, preserving this order.
available_datasets = []
for dataset_name in datasets:
    dataset_path = os.path.join(base_dataset_dir, dataset_name)
    if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "train.csv")):
        available_datasets.append(dataset_name)
    else:
        print(f"Skipping missing dataset: {dataset_name}")

print(f"Found datasets: {available_datasets}")

training_script = os.path.join(current_dir, "amnet_train.py")

# System configuration
TOTAL_CPU_CORES = multiprocessing.cpu_count()
TARGET_CPU_USAGE = 0.9  # Use more CPU resources
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6] 
NUM_EXPERIMENTS = len(available_datasets)

# Calculate resources per experiment
total_workers_target = int(TOTAL_CPU_CORES * TARGET_CPU_USAGE)
# This will be used for dataset preprocessing n_jobs
n_jobs_per_experiment = max(1, total_workers_target // NUM_EXPERIMENTS)

print(f"Total CPU Cores: {TOTAL_CPU_CORES}")
print(f"Target Workers Total: {total_workers_target}")
print(f"Jobs per Experiment: {n_jobs_per_experiment}")
print(f"Available GPUs: {AVAILABLE_GPUS}")

def run_experiment(dataset_name, gpu_id, seed=42):
    print(f"\n{'='*50}")
    print(f"Starting training for dataset: {dataset_name} on GPU {gpu_id} with seed {seed}")
    print(f"{'='*50}")
    
    dataset_path = os.path.join(base_dataset_dir, dataset_name)
    output_path = os.path.join(base_output_dir, dataset_name)
    
    # Environment variables for this process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Construct command
    # Using python executable from current environment
    cmd = [
        sys.executable,
        training_script,
        "--dataset_path", dataset_path,
        "--output_path", output_path,
        "--n_epochs", "200", 
        "--batch_size", "2048", # Increased batch size for A100
        "--num_workers", "0", # Use 0 workers for DataLoader as data is cached in memory
        "--n_jobs", str(n_jobs_per_experiment),
        "--seed", str(seed)  # Add seed parameter
    ]
    
    print(f"Running command for {dataset_name}: {' '.join(cmd)}")
    
    try:
        # Use subprocess.Popen to run in parallel, but here we are inside a multiprocessing pool worker
        # so subprocess.run is fine as it blocks this worker
        subprocess.run(cmd, check=True, env=env)
        print(f"Successfully finished {dataset_name} on GPU {gpu_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error training {dataset_name} on GPU {gpu_id}: {e}")
        return False

if __name__ == "__main__":
    processes = []
    
    # Option 1: Use same seed for all datasets (reproducibility)
    USE_SAME_SEED = True
    BASE_SEED = 42
    
    # Assign experiments to GPUs
    # If we have more experiments than GPUs, we'd need a queue, but here 5 experiments < 7 GPUs
    for i, dataset_name in enumerate(available_datasets):
        gpu_id = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]
        
        # Option 1: Same seed for all (default)
        # Option 2: Different seed per dataset (BASE_SEED + i)
        seed = BASE_SEED if USE_SAME_SEED else BASE_SEED + i
        
        p = multiprocessing.Process(target=run_experiment, args=(dataset_name, gpu_id, seed))
        processes.append(p)
        p.start()
        print(f"Launched process for {dataset_name} on GPU {gpu_id} with seed {seed}")

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\nAll experiments completed.")
