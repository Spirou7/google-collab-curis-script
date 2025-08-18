# Remote Machine Setup and Execution Guide

## Prerequisites
- SSH access to a remote machine (Linux/Ubuntu preferred)
- Python 3.8+ on the remote machine
- Git installed on the remote machine
- Sufficient disk space (~10GB recommended)

## Step 1: Connect to Remote Machine
```bash
ssh username@remote-machine-address
```

## Step 2: Clone or Transfer the Repository

### Option A: Clone from Git (if repository is on GitHub/GitLab)
```bash
git clone https://github.com/yourusername/curis_version_script_4.git
cd curis_version_script_4
```

### Option B: Transfer from Local Machine
On your local machine:
```bash
# Create a tar archive (excluding large files)
tar -czf curis_project.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='*.png' \
    --exclude='*.npz' \
    --exclude='parallel_optimizer_results_*' \
    --exclude='optimizer_comparison_results_*' \
    curis_version_script_4/

# Transfer to remote machine
scp curis_project.tar.gz username@remote-machine-address:~/
```

On the remote machine:
```bash
tar -xzf curis_project.tar.gz
cd curis_version_script_4
```

## Step 3: Setup Python Environment

### Create Virtual Environment
```bash
# Install python3-venv if not available
sudo apt-get update
sudo apt-get install python3-venv python3-pip

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

## Step 4: Install Dependencies

### Create requirements.txt (if not exists)
```bash
cat > requirements.txt << 'EOF'
tensorflow==2.13.0
numpy==1.24.3
matplotlib==3.7.1
scikit-learn
pandas
Pillow
EOF
```

### Install packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### For GPU Support (if available)
```bash
# Check if GPU is available
nvidia-smi

# If GPU is available, install GPU version
pip install tensorflow[and-cuda]==2.13.0
```

## Step 5: Verify Installation
```bash
# Test TensorFlow installation
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"

# Test if the project structure is correct
ls -la fault_injection/scripts/
ls -la fault_injection/models/
ls -la fault_injection/core/
```

## Step 6: Prepare Data (if needed)
```bash
# The script should automatically download CIFAR-10 data
# If you need to pre-download:
python3 -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"
```

## Step 7: Run the Experiment

### Quick Test Run
```bash
# Minimal test to verify everything works
python3 fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 1 \
    --steps-after-injection 10 \
    --optimizers adam sgd
```

### Production Runs

#### Small Experiment (Good for initial testing)
```bash
python3 fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 5 \
    --steps-after-injection 50 \
    --optimizers adam sgd rmsprop
```

#### Medium Experiment
```bash
python3 fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 20 \
    --steps-after-injection 100 \
    --optimizers adam sgd rmsprop adamw
```

#### Large Experiment (Recommended for meaningful results)
```bash
python3 fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 50 \
    --steps-after-injection 200 \
    --optimizers adam sgd rmsprop adamw nadam adadelta
```

## Step 8: Run in Background (Important for Long Experiments)

### Using nohup
```bash
nohup python3 fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 50 \
    --steps-after-injection 200 \
    --optimizers adam sgd rmsprop adamw \
    > experiment_log.txt 2>&1 &

# Check the process
ps aux | grep python3

# Monitor the log
tail -f experiment_log.txt
```

### Using screen (Recommended)
```bash
# Install screen if not available
sudo apt-get install screen

# Create a new screen session
screen -S optimizer_experiment

# Run the experiment
python3 fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 50 \
    --steps-after-injection 200 \
    --optimizers adam sgd rmsprop adamw

# Detach from screen: Press Ctrl+A then D

# List screen sessions
screen -ls

# Reattach to session
screen -r optimizer_experiment

# Kill screen session when done
# Press Ctrl+A then K (while attached)
```

### Using tmux (Alternative to screen)
```bash
# Install tmux if not available
sudo apt-get install tmux

# Create new tmux session
tmux new -s optimizer_exp

# Run the experiment
python3 fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 50 \
    --steps-after-injection 200 \
    --optimizers adam sgd rmsprop adamw

# Detach: Press Ctrl+B then D

# List sessions
tmux ls

# Reattach
tmux attach -t optimizer_exp
```

## Step 9: Monitor Progress

### Check Output Directory
```bash
# Find the latest results directory
ls -lt fault_injection/ | grep parallel_optimizer_results

# Monitor the latest experiment
LATEST_DIR=$(ls -td fault_injection/parallel_optimizer_results_* | head -1)
echo "Monitoring: $LATEST_DIR"

# Count completed experiments
ls -la $LATEST_DIR/experiment_*/results.json | wc -l

# Check intermediate summary
cat $LATEST_DIR/intermediate_summary.json | python3 -m json.tool
```

### Monitor System Resources
```bash
# CPU and Memory usage
htop  # or top if htop not available

# GPU usage (if available)
watch -n 1 nvidia-smi

# Disk usage
df -h
```

## Step 10: Retrieve Results

### On the remote machine, compress results:
```bash
# Find your results directory
RESULTS_DIR=$(ls -td fault_injection/parallel_optimizer_results_* | head -1)

# Create archive
tar -czf optimizer_results.tar.gz $RESULTS_DIR
```

### On your local machine, download results:
```bash
scp username@remote-machine-address:~/curis_version_script_4/optimizer_results.tar.gz ./

# Extract
tar -xzf optimizer_results.tar.gz

# View the final report
cat parallel_optimizer_results_*/final_report.md
```

## Performance Tips for Remote Machines

### For GPU-enabled Machines
```bash
# Verify GPU is being used
python3 -c "
import tensorflow as tf
print('GPUs Available:', tf.config.list_physical_devices('GPU'))
"

# Remove CPU-only configuration in the script
# Comment out these lines in test_optimizer_mitigation_v3.py:
# tf.config.set_visible_devices([], 'GPU')
# tf.config.set_soft_device_placement(True)
```

### For Multi-CPU Machines
```bash
# Set number of threads for TensorFlow
export TF_NUM_INTEROP_THREADS=4
export TF_NUM_INTRAOP_THREADS=8

# Run experiment
python3 fault_injection/scripts/test_optimizer_mitigation_v3.py ...
```

## Troubleshooting

### Issue: Out of Memory
```bash
# Check memory
free -h

# Reduce batch size by editing fault_injection/core/config.py
# Change BATCH_SIZE from 1024 to 512 or 256
```

### Issue: Permission Denied
```bash
# Ensure script is executable
chmod +x fault_injection/scripts/test_optimizer_mitigation_v3.py

# Check file ownership
ls -la fault_injection/scripts/
```

### Issue: Module Not Found
```bash
# Ensure you're in the correct directory
pwd  # Should show .../curis_version_script_4

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from the project root
cd ~/curis_version_script_4
python3 fault_injection/scripts/test_optimizer_mitigation_v3.py ...
```

### Issue: TensorFlow Version Conflicts
```bash
# Uninstall and reinstall
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow==2.13.0
```

## Estimated Runtimes

### CPU-only Machine (8-16 cores)
- 1 experiment, 2 optimizers, 10 steps: ~5-10 minutes
- 10 experiments, 4 optimizers, 100 steps: ~2-4 hours
- 50 experiments, 4 optimizers, 200 steps: ~12-20 hours

### GPU Machine (single GPU)
- 1 experiment, 2 optimizers, 10 steps: ~1-2 minutes
- 10 experiments, 4 optimizers, 100 steps: ~30-60 minutes
- 50 experiments, 4 optimizers, 200 steps: ~3-5 hours

### Multi-GPU Machine
- Performance scales roughly linearly with proper configuration
- May require code modifications for multi-GPU support

## Script for Automated Setup

Save this as `setup_remote.sh`:
```bash
#!/bin/bash

# Update system
sudo apt-get update

# Install dependencies
sudo apt-get install -y python3-venv python3-pip git screen htop

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install tensorflow==2.13.0 numpy matplotlib scikit-learn pandas Pillow

# Verify installation
python3 -c "import tensorflow as tf; print('TF Version:', tf.__version__)"

# Create results directory
mkdir -p fault_injection/results

echo "Setup complete! Ready to run experiments."
echo "Activate environment with: source venv/bin/activate"
```

Run with:
```bash
chmod +x setup_remote.sh
./setup_remote.sh
```

## Quick Start Commands Summary
```bash
# 1. SSH to remote
ssh username@remote-machine

# 2. Navigate to project
cd curis_version_script_4

# 3. Activate environment
source venv/bin/activate

# 4. Run in screen
screen -S exp
python3 fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 20 \
    --steps-after-injection 100 \
    --optimizers adam sgd rmsprop adamw

# 5. Detach (Ctrl+A, D) and logout
```

## Best Practices
1. Always use `screen` or `tmux` for long-running experiments
2. Start with a small test run to verify setup
3. Monitor disk space - results can be large
4. Save intermediate results (done automatically every 5 experiments)
5. Keep a log of your experiments with different parameters
6. Compress results before transferring to save bandwidth