#################################################################

# Edit the following paths to match your setup
qwen_ckpt_path=$1
RUNS=${2:-10}
CUSTOM_SUFFIX=${3:-""}
CUSTOM_NAME=${4:-""}
ABLATION_GRAY=${5:-""}
shift 5 2>/dev/null || true

SERVER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dp-size)
            SERVER_ARGS+=(--dp-size "$2")
            shift 2
            ;;
        --batch-size-one-gpu)
            SERVER_ARGS+=(--batch-size-one-gpu "$2")
            shift 2
            ;;
        --server-args)
            IFS=' ' read -r -a extra <<< "$2"
            SERVER_ARGS+=("${extra[@]}")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

BASE_DIR='/mnt/pub/wyf/workspace/neuroncap'
NUSCENES_PATH='/nas_pub_data/nuScenes/nuScenes_all'
# Model related stuff
MODEL_NAME='Impromptu'
MODEL_FOLDER=$BASE_DIR/$MODEL_NAME

# Rendering related stuff
RENDERING_FOLDER=$BASE_DIR/'neurad-studio'
RENDERING_CHECKPOITNS_PATH=$BASE_DIR/'neurad-studio/checkpoints'

# NCAP related stuff
NCAP_FOLDER=$BASE_DIR/'neuro-ncap'

# server port file
PORT_FILE='/mnt/pub/wyf/workspace/neuroncap/neuro-ncap/port.txt'

#################################################################

# SLURM related stuff
CLEAN_PATH=$(echo "$qwen_ckpt_path" | sed 's|/$||')
LAST_DIR=$(basename "$CLEAN_PATH")
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
if [ -n "$CUSTOM_NAME" ]; then
    NAME="$CUSTOM_NAME"
else
    NAME="${LAST_DIR}_${CUSTOM_SUFFIX}_${TIMESTAMP}"
fi
PORT_FILE="workdir/${NAME}/port.txt"
PAST_POS_PATH="workdir/${NAME}/past_pos.npy"
EGO_STATUS_PATH="workdir/${NAME}/ego_status.txt"

if [ -z "$qwen_ckpt_path" ]; then
    echo "Usage: $0 <qwen_ckpt_path> [runs] [custom_suffix] [custom_name] [ablation_gray]"
    echo "       [--dp-size <int>] [--batch-size-one-gpu <int>] [--server-args \"<args>\"]"
    echo "  qwen_ckpt_path: Path to Qwen checkpoint"
    echo "  runs: Number of runs (default: 1)"
    echo "  custom_suffix: Custom suffix for NAME (default: empty)"
    echo "  custom_name: Custom NAME to use (default: auto-generated)"
    echo "  ablation_gray: Enable ablation study with solid gray images (default: empty)"
    echo "  --dp-size: Data parallel size for launch_server.py"
    echo "  --batch-size-one-gpu: Per-GPU batch size for launch_server.py"
    echo "  --server-args: Quoted string of extra args for launch_server.py"
    exit 1
fi

echo "Using RUNS=$RUNS"
echo "Using CUSTOM_SUFFIX='$CUSTOM_SUFFIX'"
echo "Generated NAME=$NAME"

# assert we are standing in the right folder, which is NCAP folder
if [ $PWD != $NCAP_FOLDER ]; then
    echo "Please run this script from the NCAP folder"
    exit 1
fi

# assert all the other folders are present
if [ ! -d $MODEL_FOLDER ]; then
    echo "Model folder not found"
    exit 1
fi

if [ ! -d $RENDERING_FOLDER ]; then
    echo "Rendering folder not found"
    exit 1
fi

mkdir -p "workdir/${NAME}"

# --- ensure NVRTC is on LD_LIBRARY_PATH (auto-detect from pip wheel) ---
#NVRTC_LIBDIR=$( /home/wyf/miniconda3/envs/sglang/bin/python - <<'PY'
#import os, nvidia.cuda_nvrtc
#print(os.path.join(os.path.dirname(nvidia.cuda_nvrtc.__file__), "lib"))
#PY
#)
#export LD_LIBRARY_PATH="${NVRTC_LIBDIR}:${LD_LIBRARY_PATH}"
#echo "[run.sh] Using NVRTC from: ${NVRTC_LIBDIR}"
# --- end NVRTC path ---

/home/wyf/miniconda3/envs/sglang/bin/python /mnt/pub/wyf/workspace/neuroncap/Impromptu/inference/launch_server.py \
  -qwen_ckpt_path "$qwen_ckpt_path" \
  --port_file "$PORT_FILE" \
  "${SERVER_ARGS[@]}"

qwen_infer_port=$(cat $PORT_FILE)
echo "--------------qwen_infer_port is $qwen_infer_port---------------------"

for SCENARIO in "stationary" "frontal" "side"; do
    array_file=ncap_slurm_array_$SCENARIO
    id_to_seq=scripts/arrays/${array_file}.txt
#stationary_num = 10; frontal = 5; side_num = 5
    if [ $SCENARIO == "stationary" ]; then
        num_scenarios=10
    elif [ $SCENARIO == "frontal" ]; then
        num_scenarios=5
    elif [ $SCENARIO == "side" ]; then
        num_scenarios=5
    fi
    for i in $(seq 1 $num_scenarios); do
        sequence=$(awk -v ArrayTaskID=$i '$1==ArrayTaskID {print $2}' $id_to_seq)
        if [ -z $sequence ]; then
            echo "undefined sequence"
            exit 0
        fi
        output_dir="output/$NAME/$SCENARIO-$sequence"
        completed_runs=0
        if [ -d "$output_dir" ]; then
            for run_num in $(seq 0 $((RUNS-1))); do
                if [ -d "$output_dir/run_$run_num" ]; then
                    completed_runs=$((completed_runs + 1))
                fi
            done
        fi
        
        if [ $completed_runs -eq $RUNS ]; then
            echo "Skipping scenario $SCENARIO with sequence $sequence - all $RUNS runs already completed"
            continue
        fi
        if [ $completed_runs -gt 0 ] && [ $completed_runs -lt $RUNS ]; then
            echo "Found incomplete runs in: $output_dir"
            echo "Current runs: $completed_runs/$RUNS"
            echo "Do you want to clear all existing run_* directories in this path? (Y/N)"
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                echo "Clearing existing run directories in $output_dir"
                rm -rf "$output_dir"/run_*
                completed_runs=0
            fi
        fi
        echo "Running scenario $SCENARIO with sequence $sequence (completed: $completed_runs/$RUNS)"
        BASE_DIR=$BASE_DIR\
         NUSCENES_PATH=$NUSCENES_PATH\
         MODEL_NAME=$MODEL_NAME\
         MODEL_FOLDER=$MODEL_FOLDER\
         RENDERING_FOLDER=$RENDERING_FOLDER\
         RENDERING_CHECKPOITNS_PATH=$RENDERING_CHECKPOITNS_PATH\
         NCAP_FOLDER=$NCAP_FOLDER\
         NAME=$NAME\
         qwen_infer_port=$qwen_infer_port\
         PAST_POS_PATH=$PAST_POS_PATH\
         EGO_STATUS_PATH=$EGO_STATUS_PATH\
         qwen_ckpt_path=$qwen_ckpt_path\
         ABLATION_GRAY=$ABLATION_GRAY\
         scripts/run_local_render.sh $sequence $SCENARIO --scenario-category=$SCENARIO --runs $RUNS
        #exit 0
    done
done
kill $MODEL_PID