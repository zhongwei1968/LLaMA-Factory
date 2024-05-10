FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

RUN pip install -e .[deepspeed,metrics,bitsandbytes,qwen]
RUN pip install flash-attn==2.3.3 --no-build-isolation

VOLUME [ "/root/.cache/huggingface/", "/app/data", "/app/output" ]
EXPOSE 7860

CMD [ "llamafactory-cli", "webui" ]
