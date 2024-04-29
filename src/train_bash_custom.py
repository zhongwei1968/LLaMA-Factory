import os, subprocess

TRAIN_ARGS = os.getenv('TRAIN_ARGS')
TRAIN_ARGS_ADD = os.getenv('TRAIN_ARGS_ADD')
run_com = os.getenv('RUN_CMD','accelerate') # accelerate or deepspeed
NUM_GPUS = os.getenv('NUM_GPUS','1')

print(os.listdir('/opt/ml/input/data/training'))

if run_com=='deepspeed':
    subprocess.run(
        ["deepspeed", "--num_gpus", NUM_GPUS, "/app/src/train_bash.py"]+
        ["--deepspeed", "/app/examples/deepspeed/ds_z3_config.json"]+
        TRAIN_ARGS.split()+
        TRAIN_ARGS_ADD.split(),
        check=True
        )
elif run_com=='accelerate':
    subprocess.run(
        ["accelerate", "launch", "--config_file", "/app/examples/accelerate/single_config.yaml",
         "/app/src/train_bash.py"]+
        TRAIN_ARGS.split()+
        TRAIN_ARGS_ADD.split(),
        check=True
        )
else:
    subprocess.run(
        ["python", "src/train_bash.py"]+
        TRAIN_ARGS.split()+
        TRAIN_ARGS_ADD.split(),
        check=True
        )
