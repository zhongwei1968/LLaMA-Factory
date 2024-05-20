FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

COPY requirements.txt /app/

# Update and upgrade the package lists
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN pip install --upgrade pip

ENV NVM_DIR /usr/local/nvm

# Copy the current directory contents into the container at /app
COPY . /app

# Remove the /opt/pytorch/ directory
RUN rm -rf /opt/pytorch/
RUN rm -rf /usr/local/nvm/
RUN rm -rf /workspace/
RUN rm -rf /opt/hpcx/

RUN pip install -r requirements.txt

# RUN pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton --index-url https://download.pytorch.org/whl/cu121
RUN pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"

RUN pip install -e .[deepspeed,accelerate,metrics,bitsandbytes,peft,qwen]
RUN pip install flash-attn==2.3.3 --no-build-isolation
RUN pip uninstall transformer_engine -y

VOLUME [ "/root/.cache/huggingface/", "/app/data", "/app/output" ]
EXPOSE 7860

CMD [ "llamafactory-cli", "webui" ]
