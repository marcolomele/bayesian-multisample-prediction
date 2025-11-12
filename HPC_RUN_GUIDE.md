# Step-by-Step Guide: Running Experiments on Bocconi HPC

This guide will walk you through running your three experiments on the university's HPC system.

## Prerequisites
- VPN connection to Bocconi (if required): https://bocconi.sharepoint.com/sites/BocconiStudentsHPC/SitePages/SSH-Login.aspx
- SSH access credentials (username: 3152128)

---

## Step 1: Connect to the HPC Login Node

From your local terminal, SSH into the HPC:

```bash
ssh 3152128@slnode-da.sm.unibocconi.it
```

Enter your password when prompted.

---

## Step 2: Set Up Your Working Directory

Once connected, create a directory structure for your project:

```bash
# Create main project directory
mkdir -p ~/bayesian_multisample/{scripts,data,logs,outputs}

# Navigate to the project directory
cd ~/bayesian_multisample
```

---

## Step 3: Set Up Python Environment

Create and activate a conda environment with the required packages:

```bash
# Load conda module
module load modules/miniconda3

# Create a new environment (or use existing General_Env if preferred)
conda create --name bayesian_env python=3.9 -y

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate bayesian_env

# Install required packages
conda install -y numpy scipy pandas tqdm
# OR use pip if conda doesn't have them:
# pip install numpy scipy pandas tqdm
```

**Note:** Keep the terminal session open, or remember to activate the environment each time you log in:
```bash
module load modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate bayesian_env
```

---

## Step 4: Upload Your Files to HPC

**From your LOCAL machine** (open a NEW terminal window, stay in your local project directory):

### 4a. Upload Python scripts
```bash
# Navigate to your local project directory
cd /Users/marcolomele/Documents/Repos/bayesian-multisample/scripts

# Upload all Python scripts
scp *.py 3152128@slnode-da.sm.unibocconi.it:~/bayesian_multisample/scripts/
```

### 4b. Upload configuration files
```bash
# Upload all config JSON files
scp config_*.json 3152128@slnode-da.sm.unibocconi.it:~/bayesian_multisample/scripts/
```

### 4c. Upload data files
```bash
# Upload data directory (this may take a while depending on data size)
# From the project root directory:
cd /Users/marcolomele/Documents/Repos/bayesian-multisample

# Upload the entire data directory
scp -r data/ 3152128@slnode-da.sm.unibocconi.it:~/bayesian_multisample/
```

**Note:** The `scp` command syntax is: `scp <local_file> <username>@<host>:<remote_path>`

---

## Step 5: Verify Files Are Uploaded

**Back on the HPC** (in your SSH session), verify everything is in place:

```bash
cd ~/bayesian_multisample

# Check scripts
ls -la scripts/

# Check data (verify the paths match your config files)
ls -la data/twenty+newsgroups/
ls -la data/namesbystate/
ls -la data/wilderness/
```

**Important:** Check that the data paths in your config files are correct. The config files use relative paths like `../data/...`, so make sure the structure matches.

---

## Step 6: Create a SLURM Script

Create a SLURM batch script to run your experiments:

```bash
nano ~/bayesian_multisample/scripts/run_experiments.sh
```

Paste the following content (adjust resources as needed):

```bash
#!/bin/bash
#SBATCH --job-name=bayesian_experiments
#SBATCH --account=3176145
#SBATCH --partition=dsba
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err

# Load conda module
module load modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate bayesian_env

# Navigate to scripts directory
cd ~/bayesian_multisample/scripts

# Run all three experiments sequentially
echo "Starting experiment 1: news..."
python experiment.py --config config_news.json

echo "Starting experiment 2: names..."
python experiment.py --config config_names.json

echo "Starting experiment 3: wilderness..."
python experiment.py --config config_wilderness.json

echo "All experiments completed!"

# Deactivate environment
conda deactivate
module unload modules/miniconda3
```

Save and exit:
- Press `Ctrl+X`
- Press `Y` to confirm
- Press `Enter` to save

Make the script executable:
```bash
chmod +x ~/bayesian_multisample/scripts/run_experiments.sh
```

---

## Step 7: Submit the Job to SLURM

Submit your job to the SLURM scheduler:

```bash
cd ~/bayesian_multisample/scripts
sbatch run_experiments.sh
```

You should see output like:
```
Submitted batch job 12345
```

**Note the job ID** - you'll use it to check the status.

---

## Step 8: Monitor Your Job

### Check job status:
```bash
squeue -u 3152128
```

### Check job details:
```bash
squeue -j <JOB_ID>
```

### View output in real-time (if job is running):
```bash
tail -f ~/bayesian_multisample/logs/bayesian_experiments_<JOB_ID>.out
```

### View errors:
```bash
tail -f ~/bayesian_multisample/logs/bayesian_experiments_<JOB_ID>.err
```

---

## Step 9: Check Results

Once the job completes, check the results:

```bash
cd ~/bayesian_multisample/scripts

# Check output directories
ls -la results_news_final_V2/
ls -la results_namesbystate_final/
ls -la results_wilderness_final/
```

---

## Step 10: Download Results (Optional)

**From your LOCAL machine**, download the results:

```bash
# Download results directories
scp -r 3152128@slnode-da.sm.unibocconi.it:~/bayesian_multisample/scripts/results_* /Users/marcolomele/Documents/Repos/bayesian-multisample/scripts/

# Or download specific result directories:
# scp -r 3152128@slnode-da.sm.unibocconi.it:~/bayesian_multisample/scripts/results_news_final_V2 /Users/marcolomele/Documents/Repos/bayesian-multisample/scripts/
# scp -r 3152128@slnode-da.sm.unibocconi.it:~/bayesian_multisample/scripts/results_namesbystate_final /Users/marcolomele/Documents/Repos/bayesian-multisample/scripts/
# scp -r 3152128@slnode-da.sm.unibocconi.it:~/bayesian_multisample/scripts/results_wilderness_final /Users/marcolomele/Documents/Repos/bayesian-multisample/scripts/

# Or download specific files
scp 3152128@slnode-da.sm.unibocconi.it:~/bayesian_multisample/logs/*.out /Users/marcolomele/Documents/Repos/bayesian-multisample/scripts/
```

---

## Troubleshooting

### Job is pending/not starting:
- Check available resources: `sinfo`
- Check your account limits: `sacct -u 3152128`
- Try reducing requested resources (CPU, memory, time)

### Job fails immediately:
- Check the error log: `cat ~/bayesian_multisample/logs/bayesian_experiments_<JOB_ID>.err`
- Verify Python environment is activated correctly
- Check that all files are uploaded and paths are correct

### Import errors:
- Make sure all Python scripts are in the same directory
- Verify all dependencies are installed: `conda list` or `pip list`

### Path errors:
- Verify data paths in config files match the actual data location on HPC
- Remember: config files use relative paths like `../data/...`

### Need to cancel a job:
```bash
scancel <JOB_ID>
```

---

## Quick Reference Commands

```bash
# Connect to HPC
ssh 3152128@slnode-da.sm.unibocconi.it

# Activate environment
module load modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate bayesian_env

# Submit job
cd ~/bayesian_multisample/scripts
sbatch run_experiments.sh

# Check status
squeue -u 3152128

# View output
tail -f ~/bayesian_multisample/logs/bayesian_experiments_<JOB_ID>.out

# Exit HPC
exit
```

---

## Notes

- **Login node limitations**: You can only save files on the login node, not on compute nodes
- **Long-running jobs**: Use SLURM for anything that takes more than a few minutes
- **Resource requests**: Adjust `--cpus-per-task`, `--mem`, and `--time` based on your needs
- **Account number**: Verify your account number (3176145) matches what's in the SLURM script
- **Partition**: The script uses `dsba` partition - verify this is correct for your account

