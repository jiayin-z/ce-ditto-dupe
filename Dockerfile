FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest


# set working directory
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

# Install required packages
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_lg
RUN python -m nltk.downloader stopwords
RUN conda install -c conda-forge nvidia-apex

# Copies the trainer code to the docker image.
COPY ./blocking /app/blocking
COPY ./data /app/data
COPY ./ditto_light /app/ditto_light
COPY ./configs.json /app/configs.json
COPY ./train_ditto.py /app/train_ditto.py
COPY ./run_all_wdc.py /app/run_all_wdc.py
COPY ./matcher.py /app/matcher.py

RUN export CUDA_VISIBLE_DEVICES=0

# # Set up the entry point to invoke the trainer.
# ENTRYPOINT [ "python", "train_ditto.py", \
#     "--task", "Structured/Beer", \
#     "--batch_size", "64", \
#     "--max_len", "64", \
#     "--lr", "3e-5", \
#     "--n_epochs", "40", \
#     "--lm", "distilbert", \
#     "--fp16", \
#     "--da", "del" \
#     "--dk", "product" \
#     "--summarize" ]

ENTRYPOINT [ "bash" ]