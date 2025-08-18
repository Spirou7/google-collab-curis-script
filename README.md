# ISCA50_AE_Extend - Fault Injection Framework

## 🚀 Complete Docker Workflow Guide

This guide provides a complete workflow for running fault injection experiments using Docker with named volumes (no host mounting permissions required).

### Prerequisites
- Docker installed and running
- Access to run Docker commands
- ~2GB disk space for Docker images and results

---

## 📋 Quick Start (5 minutes)

### 1. Build the Docker Image
```bash
./shell_scripts/docker_run.sh build
```

### 2. Run a Test Experiment (2-3 minutes)
```bash
./shell_scripts/docker_run.sh optimizer \
    --baseline adam \
    --test-optimizers sgd \
    --num-experiments 1 \
    --steps-after-injection 20
```

### 3. Check Results in Docker Volumes
```bash
./shell_scripts/docker_run.sh list-results
```

### 4. Extract Results to Local Machine
```bash
./shell_scripts/docker_run.sh extract-safe
```

### 5. View Your Results
```bash
# Results are in ./extracted_results/ (NOT ./results/)
ls ./extracted_results/optimizer/run_*/

# View the report
cat ./extracted_results/optimizer/run_*/final_report.md

# Check visualizations (open in image viewer)
ls ./extracted_results/optimizer/run_*/*.png
```

### 6. View Results from Remote Server
If you're on a remote SSH server, use one of these methods:

**Option A: VS Code Remote SSH (Recommended)**
- Install VS Code with Remote-SSH extension
- **Important**: Increase timeout to 60 seconds:
  - Open VS Code settings (Cmd/Ctrl + ,)
  - Search for "remote.SSH.connectTimeout"
  - Change from default (15) to 60 seconds
- Connect to your server and browse images directly

**Option B: Create HTML Report**
```bash
python create_html_report.py ./extracted_results/optimizer
# Download the report.html file and open locally
```

**Option C: Web Server**
```bash
cd extracted_results && python -m http.server 8888
# Visit http://YOUR_SERVER_IP:8888 in browser
```

**Option D: Interactive Helper**
```bash
./shell_scripts/view_results.sh  # Guides you through all viewing options
```

---

## 🔬 Full Experiment Workflow

### Step 1: Build Docker Image
```bash
./shell_scripts/docker_run.sh build
```
This creates the `fault-injection-experiment:latest` image with all dependencies.

### Step 2: Run Optimizer Comparison Experiment

#### Quick Test (5 minutes, 3 experiments)
```bash
./shell_scripts/docker_run.sh optimizer \
    --baseline adam \
    --test-optimizers sgd rmsprop \
    --num-experiments 3 \
    --steps-after-injection 50
```

#### Standard Run (30-60 minutes, 10 experiments)
```bash
./shell_scripts/docker_run.sh optimizer \
    --baseline adam \
    --test-optimizers sgd rmsprop adagrad \
    --num-experiments 15 \
    --steps-after-injection 100
```

#### Full Study (Several hours, 100 experiments)
```bash
./shell_scripts/docker_run.sh optimizer \
    --baseline adam \
    --test-optimizers sgd rmsprop adagrad adamax nadam \
    --num-experiments 100 \
    --steps-after-injection 500
```

### Step 3: Monitor Progress
```bash
# Check if experiment is still running
docker ps | grep fault_injection

# View live logs
docker logs -f fault_injection_runner

# Check what's been saved so far
./shell_scripts/docker_run.sh list-results
```

### Step 4: Extract Results
```bash
# Primary extraction method
./shell_scripts/docker_run.sh extract-safe

# Alternative if above fails
./shell_scripts/extract_volumes.sh auto

# Results will be in ./extracted_results/optimizer/run_TIMESTAMP/
ls -la ./extracted_results/optimizer/
```

### Step 5: Analyze Results

Your results directory will contain:
```
./extracted_results/optimizer/run_20250814_XXXXXX/
├── final_report.md                    # Human-readable summary
├── summary_visualizations.png         # Overall comparison charts
├── all_injection_configs.json        # Injection parameters used
├── experiment_000/
│   ├── results.json                  # Detailed metrics
│   ├── recovery_comparison.png       # Recovery curves
│   ├── degradation_rates.png        # Degradation analysis
│   ├── recovery_adam.csv            # Adam optimizer data
│   ├── recovery_sgd.csv             # SGD optimizer data
│   └── injection_config.json        # Specific injection details
├── experiment_001/
│   └── ...
└── experiment_002/
    └── ...
```

---

## 🛠️ Available Commands

### Docker Run Script Commands
```bash
./shell_scripts/docker_run.sh build              # Build Docker image
./shell_scripts/docker_run.sh interactive        # Start interactive shell
./shell_scripts/docker_run.sh optimizer [args]   # Run optimizer experiment
./shell_scripts/docker_run.sh list-results       # List files in volumes
./shell_scripts/docker_run.sh extract-safe       # Extract results (safest method)
./shell_scripts/docker_run.sh volume-info        # Show volume information
./shell_scripts/docker_run.sh clean-volumes      # Delete all volumes (WARNING!)
./shell_scripts/docker_run.sh backup            # Create backup of all volumes
```

### Experiment Parameters
```bash
--baseline OPTIMIZER           # Base optimizer (adam, sgd, rmsprop, etc.)
--test-optimizers OPT1 OPT2   # Optimizers to compare
--num-experiments N            # Number of experiments to run
--steps-after-injection N      # Training steps after fault injection
--learning-rate LR            # Learning rate (default: 0.001)
--seed N                      # Random seed for reproducibility
```

---

## 🔍 Troubleshooting

### Issue: "Permission denied" when extracting
```bash
# Use the alternative extraction method
./shell_scripts/extract_volumes.sh auto

# Or manually extract specific files
docker run --rm -v fault_injection_optimizer:/data:ro alpine \
    tar cf - -C /data . | tar xf - -C ./manual_extract/
```

### Issue: Can't find results after extraction
Results are in `./extracted_results/` NOT `./results/`:
```bash
find ./extracted_results -name "*.json" -o -name "*.png"
```

### Issue: Experiment seems to hang
```bash
# Check container status
docker ps | grep fault

# View recent logs
docker logs --tail 50 fault_injection_runner

# Stop if needed
docker stop fault_injection_runner
```

### Issue: Out of disk space
```bash
# Check Docker space usage
docker system df

# Clean up old containers and images
docker system prune -a

# Remove experiment volumes (WARNING: deletes data!)
./shell_scripts/docker_run.sh clean-volumes
```

---

## 📊 Understanding Results

### final_report.md
Contains:
- Summary statistics for each optimizer
- Recovery success rates
- Average performance metrics
- Comparative analysis

### summary_visualizations.png
Shows:
- Box plots comparing optimizer resilience
- Recovery time distributions
- Performance degradation patterns

### Individual Experiment Files
Each `experiment_XXX/` folder contains:
- Detailed fault injection configuration
- Step-by-step recovery data
- Optimizer-specific CSV files with training metrics
- Visualization plots

---

## 🧪 Testing the Setup

Before running long experiments, verify your setup:

```bash
# 1. Run the volume test
./shell_scripts/test_volumes.sh

# 2. Run a minimal experiment
./shell_scripts/docker_run.sh optimizer \
    --baseline adam \
    --test-optimizers sgd \
    --num-experiments 1 \
    --steps-after-injection 10

# 3. Check and extract results
./shell_scripts/check_optimizer_results.sh
./shell_scripts/docker_run.sh extract-safe

# 4. Verify files exist
ls ./extracted_results/optimizer/run_*/
```

---

## 🏗️ Architecture

### Docker Volumes Used
- `fault_injection_results` - General results
- `fault_injection_optimizer` - Optimizer comparison results  
- `fault_injection_output` - Output files
- `fault_injection_checkpoints` - Model checkpoints

### How It Works
1. Docker containers run experiments
2. Results save to Docker-managed named volumes
3. Volumes persist after container stops
4. Extraction copies from volumes to local filesystem
5. No host mounting required - avoids permission issues

---

## 📚 Original Documentation

### Fault Injection Methodology
We provide our fault injection framework for various workloads. In each experiment, we:
1. Pick a random training epoch
2. Select a random training step
3. Choose a random layer (forward or backward pass)
4. Apply a random software fault model
5. Continue training to observe outcomes

### System Requirements (Original TPU Setup)
- **Hardware**: Google Cloud TPU VMs
- **Software**: TensorFlow 2.6.0, NumPy 1.19.5

### Experiment Outcomes
Three main outcomes observed:
1. **Masked** - Fault has no significant impact
2. **Immediate INFs/NaNs** - Catastrophic failure
3. **SlowDegrade** - Gradual performance degradation

For more details on the research methodology, see the original paper.

---

## 📧 Support

For issues or questions:
- Check the troubleshooting section above
- Review `./shell_scripts/docker_run.sh help` for command options
- Examine logs with `docker logs fault_injection_runner`

---

*Last updated: August 2024*
