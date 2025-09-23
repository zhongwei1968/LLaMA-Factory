import os
import subprocess
import tarfile

import boto3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


aws_region = os.environ["aws_region"]
sagemaker = boto3.client('sagemaker', region_name=aws_region)
s3 = boto3.client("s3", region_name=aws_region)


def get_trainingjob_id(pipeline_arn, TrainStepName):
    """get sagemaker training job id.
    this is used for finding the model dir in S3 which uses the same name
    format e.g., pipelines-0r1qgmbno9li-merge-base-WkjtazkNfy"""

    response = sagemaker.list_pipeline_execution_steps(
        PipelineExecutionArn=pipeline_arn
    )
    pipeline = response["PipelineExecutionSteps"]
    for step in pipeline:
        if step["StepName"] == TrainStepName:
            arn = step["Metadata"]["ProcessingJob"]["Arn"]
            # arn = step["Metadata"]["TrainingJob"]["Arn"]
            job_name = arn.split("/")[-1]
            return job_name


def unzip_tar(local_file_path, extracted_dir):
    """unzip a tar.gz model file"""
    with tarfile.open(local_file_path, 'r:gz') as tar:
        tar.extractall(path=extracted_dir)


def zip_tar(dir_path, output_filename):
    """compress merged model dir to tar.gz
    output at script execution point"""
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(dir_path, arcname=os.path.basename(dir_path))


def merge_model(
    model_path,
    adapter_path,
    export_dir,
    torch_dtype=torch.bfloat16
):
    """merge adaptor to base model"""

    # place the model on GPU
    device_map = {"": "cuda"}

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2"
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    # NOTE: merge LoRA weights
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    print(f"[INFO] Model saved to {export_dir}")


def run_llamafactory_cli(
        model_path,
        adapter_checkpt_path,
        template,
        export_dir,
        export_size=5,
        export_device="cpu"
):
    """merge adaptor to base model using llamafactory cli"""
    command = [
        'llamafactory-cli', 'export',
            '--model_name_or_path', model_path,
            '--adapter_name_or_path', adapter_checkpt_path,
            '--template', template,  # src/llamafactory/data/template.py
            '--finetuning_type', 'lora',
            '--export_dir', export_dir,
            '--export_size', export_size,
            '--export_device', export_device
        ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"[INFO] Command output: {result.stdout}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR]: {e.stderr}")


def merge(
        pipeline_arn,
        TrainStepName,
        training_job_path,
        base_model,
        checkpoint
    ):
    """main func to download adaptors and base model, merge them,
    and upload again to S3"""

    # variables
    s3_path = training_job_path.replace("s3://", "")
    bucket, key_prefix = s3_path.split("/", 1)
    sagemaker_dir = "/opt/ml/processing/input"
    base_model_path = f"{sagemaker_dir}"
    local_adaptor_path = f"{sagemaker_dir}/model.tar.gz"
    unzipped_dir = f"{sagemaker_dir}/unzipped/"
    merged_dir = f"{sagemaker_dir}/merged/"
    # merged_dir = unzipped_dir
    compressed_filename = "merged_model.tar.gz"
    template = "llama3-legal"

    # remove later ------------
    job_name = "pipelines-3xwniybdkmai-FinetuneLLM-g09VTB5uVP"

    # download adaptors and unzip ---
    # job_name = get_trainingjob_id(pipeline_arn, TrainStepName)
    print("[INFO] Downloading adaptor checkpoints from S3...", flush=True)
    s3_key = f"{key_prefix}/{job_name}/output/model.tar.gz"
    s3.download_file(bucket, s3_key, local_adaptor_path)
    print("[INFO] Adaptor checkpoints downloaded", flush=True)
    unzip_tar(local_adaptor_path, unzipped_dir)
    print("[INFO] model.tar.gz is unzipped", flush=True)

    # merge adaptor to base model ---
    print("[INFO] merging adaptor to base model")
    merge_model(base_model_path, f"{unzipped_dir}/{checkpoint}", merged_dir)
    # run_llamafactory_cli(
    #     base_model_path, f"{unzipped_dir}/{checkpoint}", template, merged_dir
    # )

    # compress & upload merged model ---
    print("[INFO] Compressing merged model...", flush=True)
    zip_tar(merged_dir, compressed_filename)
    print("[INFO] Uploading the merged model to S3...", flush=True)
    s3_key = f"{key_prefix}/{job_name}/output/{compressed_filename}"
    s3.upload_file(f"./{compressed_filename}", bucket, s3_key)
    print(f"[INFO] Merged model uploaded to s3://{bucket}/{s3_key}", flush=True)


if __name__ == "__main__":

    pipeline_arn = os.environ["pipeline_arn"]
    TrainStepName = os.environ["TrainStepName"]
    training_job_path = os.environ["training_job_path"]
    base_model = os.environ["base_model"]
    checkpoint = os.environ["checkpoint"]

    merge(pipeline_arn, TrainStepName, training_job_path, base_model, checkpoint)

