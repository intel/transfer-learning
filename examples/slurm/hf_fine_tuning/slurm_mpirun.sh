#!/bin/bash -l
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#SBATCH --job-name=hf_pytorch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=output/torch_%A.log

CMD=$@
if [ -z "${CMD}" ]; then
  echo "No command parameters were passed to the script. This script expects the python script with args to be passed as a parameter."
  exit 1
fi

INSTANCES_PER_NODE="${INSTANCES_PER_NODE:-1}"

if [[ $SLURM_NNODES == 1 ]] && [[ $INSTANCES_PER_NODE == 1 ]]; then
  export CCL_WORKER_COUNT=0
  MPI_CMD=""
else
  # Setup env variables for distributed jobs
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  export MASTER_PORT="${MASTER_PORT:-25679}"
  export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE * $INSTANCES_PER_NODE))
  export CCL_WORKER_COUNT="${CCL_WORKER_COUNT:-2}"
  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "WORLD_SIZE=${WORLD_SIZE}"
  echo "CCL_WORKER_COUNT=${CCL_WORKER_COUNT}"

  # Write hostfile
  HOSTFILE_PATH=hostfile
  scontrol show hostname $SLURM_JOB_NODELIST | perl -ne 'chomb; print "$_"x1'> ${HOSTFILE_PATH}

  # Create mpirun command
  BIND_TO="${BIND_TO:-socket}"
  MPI_CMD="mpirun --hostfile ${HOSTFILE_PATH} \
    -n $((${SLURM_NNODES} * ${INSTANCES_PER_NODE} )) \
    --bind-to ${BIND_TO} \
    -x MASTER_ADDR=${MASTER_ADDR} \
    -x MASTER_PORT=${MASTER_PORT} \
    -x WORLD_SIZE=${WORLD_SIZE} \
    -x CCL_WORKER_COUNT=${CCL_WORKER_COUNT}"

  if [[ ! -z "${MPI_ARGS}" ]]; then
    MPI_CMD="${MPI_CMD} ${MPI_ARGS}"
  fi
fi

# Generate the full command to run
FULL_CMD="${MPI_CMD} python ${CMD}"

# Print the command
echo $FULL_CMD
echo ""

# Run the command
eval $FULL_CMD
