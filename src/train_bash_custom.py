import os, subprocess

TRAIN_ARGS = os.getenv('TRAIN_ARGS')
TRAIN_ARGS_ADD = os.getenv('TRAIN_ARGS_ADD')
run_com = os.getenv('RUN_CMD','accelerate') # accelerate or deepspeed
NUM_GPUS = os.getenv('NUM_GPUS','1')

print(os.listdir('/opt/ml/input/data/training'), flush=True)

if run_com=='deepspeed':
    print("running deepspeed", flush=True)
    subprocess.run(
        ["deepspeed", "--num_gpus", NUM_GPUS, "/app/src/train.py"]+
        ["--deepspeed", "/app/examples/deepspeed/ds_z3_config.json"]+
        TRAIN_ARGS.split()+
        TRAIN_ARGS_ADD.split(),
        check=True
        )
elif run_com=='accelerate':
    print("running accelerate", flush=True)
    config_file = "/app/examples/accelerate/single_config_8.yaml"
    if 'gemma' in TRAIN_ARGS or 'gemma' in TRAIN_ARGS_ADD:
        config_file = "/app/examples/accelerate/fsdp_config_gemma3_8.yaml"

    subprocess.run(
        ["accelerate", "launch", "--config_file", config_file,
         "/app/src/train.py"]+
        TRAIN_ARGS.split()+
        TRAIN_ARGS_ADD.split(),
        check=True
        )
else:
    subprocess.run(
        ["python", "src/train.py"]+
        TRAIN_ARGS.split()+
        TRAIN_ARGS_ADD.split(),
        check=True
        )
