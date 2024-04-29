import os, subprocess

TRAIN_ARGS = os.getenv('TRAIN_ARGS')
TRAIN_ARGS_ADD = os.getenv('TRAIN_ARGS_ADD')
DEEPSPEED=os.getenv('DEEPSPEED','true')
NUM_GPUS = os.getenv('NUM_GPUS','1')

print(os.listdir('/opt/ml/input/data/training'))

if DEEPSPEED=='true':
    subprocess.run(
        ["deepspeed", "--num_gpus", NUM_GPUS, "/app/src/train_bash.py"]+
        ["--deepspeed", "/app/examples/deepspeed/ds_z3_config.json"]+
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
