# Distributed fine tuning using PyTorch and Hugging Face with Slurm

[Slurm](https://slurm.schedmd.com/overview.html) is a workload manager that is able to schedule and deploy distributed
training jobs with minimal overhead. Slurm clusters consist of a "controller node" which serves as the orchestrator,
and "worker nodes" that are assigned to jobs. This example assumes that you already have a
[Slurm cluster](https://slurm.schedmd.com/quickstart_admin.html) with Intel® Xeon® Scalable Processors worker nodes.

## Setup

Prerequisites:

* Slurm cluster with Intel® Xeon® Scalable Processors

On the cluster nodes, you will need:
* The Slurm batch script (only on necessary on the controller node)
* Python envrionment with dependencies installed (including MPI)
* Fine tuning scripts
* Intel MPI

For this example, we will be running [Llama2 fine tuning scripts](/docker/hf_k8s/scripts). The instructions below will
create a conda environment with the dependencies needed to run these scripts. This environment needs to exist on all of
the cluster nodes. If you are running a different script, your dependencies to install will be different.

1. Create and activate a conda or virtual environment:
   ```
   conda create --name=slurm_env python=3.9
   conda activate slurm_env
   ```
2. Install dependencies:
   ```
   ONECCL_VER=2.0.0
   TORCH_VER=2.0.1+cpu
   IPEX_VER=2.0.100
   INC_VER=2.3

   python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
          https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/cpu/intel_extension_for_pytorch-${IPEX_VER}%2Bcpu-cp39-cp39-linux_x86_64.whl \
          https://intel-extension-for-pytorch.s3.amazonaws.com/torch_ccl/cpu/oneccl_bind_pt-${ONECCL_VER}%2Bcpu-cp39-cp39-linux_x86_64.whl \
          'mkl-include==2023.2.0' \
          'mkl==2023.2.0' \
          'onnx==1.13.0' \
          'protobuf==3.20.3' \
          neural-compressor==${INC_VER} \
          torch==${TORCH_VER} \
          SentencePiece \
          accelerate \
          datasets \
          einops \
          evaluate \
          nltk \
          onnxruntime \
          onnxruntime-extensions \
          peft \
          psutil \
          py-cpuinfo \
          rouge_score \
          tokenizers
   ```


## Deploying the job

1. Configure the job, by exporting environment variables that are used by the batch script. If these environment
   variables are not set, default values will be used.

   | Environment Variable Name | Default value | Description |
   |---------------------------|---------------|-------------|
   | `INSTANCES_PER_NODE`      | `1`           | The number of instances to run per node from mpirun. |
   | `MASTER_PORT`             | `25679`       | A free port to use for communication. Note that `MASTER_ADDR` is assigned in the Slurm batch script. |

1. Create a directory for output logs (on all nodes):
   ```
   LOG_DIR=/tmp/$USER/logs

   mkdir -p LOG_DIR
   ```
   This directory is passed to the sbatch `--output` arg in the next step.

1. From the controller node, source the Intel oneCCL Bindings for PyTorch vars file to access Intel MPI in your
   environment, then submit the batch script to Slurm:
   ```
   # Activate your conda or virtual environment
   conda activate slurm_env

   # Submit the batch script and specify the number of nodes, output log file name, along with your python script and args
   sbatch --nodes=2 --output=${LOG_DIR}/torch_%A.log slurm_mpirun.sh <python script>
   ```
   For example, to do a demo run the [Llama2 fine tuning scripts](docker/hf_k8s/scripts) (limited to 10 steps) on 2
   nodes, the command would look like:
   ```
   sbatch --nodes=2 --output=${LOG_DIR}/llama2_%A.log slurm_mpirun.sh finetune.py \
       --model_name_or_path "meta-llama/Llama-2-7b-hf" \
       --dataset_name "medalpaca/medical_meadow_medical_flashcards" \
       --dataset_concatenation "True" \
       --per_device_train_batch_size "8" \
       --per_device_eval_batch_size "8" \
       --gradient_accumulation_steps "1" \
       --learning_rate 2e-5 \
       --num_train_epochs 1 \
       --max_steps 10 \
       --output_dir "/tmp/${USER}" \
       --overwrite_output_dir \
       --validation_split_percentage .2 \
       --use_fast_tokenizer "False" \
       --use_lora "True" \
       --lora_rank "8" \
       --lora_alpha "16" \
       --lora_dropout "0.1" \
       --lora_target_modules "q_proj vproj" \
       --use_cpu "True" \
       --do_train "True" \
       --use_ipex "True" \
       --ddp_backend "ccl"
   ```
1. Check the Slurm queue using `squeue` to see the status of your job:
   ```
   $ squeue
   JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
     347     debug   llama2 username  R       0:24      2 clx[4898-4899]
   ```
1. On the primary worker node, tail the log file (your log file name will vary based on your sbatch `--output` parameter
   and the Slurm job id):
   ```
   $ tail -f ${LOG_DIR}/llama2_347.log
   ```

## Troubleshooting

Below are some things to check if your job does not appear to be running properly:
* On the worker nodes, view the Slurm logs by running: `sudo tail -f /var/log/slurm/slurmd.log`. If there were errors
  when submitting the job, that will be shown here.
* Ensure that your Slurm log directory exists on all of the worker nodes before running the Slurm job.
* Ensure that your virtual environment exists on all of the nodes in the same location.
* Ensure that your python scripts exist on all of the nodes.
* If you are using a gated model (like Llama 2), ensure that you are logged in to Hugging Face on all of the nodes.
