FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:latest


# set working directory
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

# Install required packages
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_lg
RUN python -m nltk.downloader stopwords
RUN conda install -c conda-forge nvidia-apex

# patch faulty apex
WORKDIR /
RUN git clone https://github.com/NVIDIA/apex
WORKDIR apex
# if GPU available
# RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# if NO GPU available
RUN pip install -v --disable-pip-version-check --no-cache-dir ./
WORKDIR /app

# Copies the trainer code to the docker image.
COPY ./blocking /app/blocking
COPY ./data /app/data
COPY ./ditto_light /app/ditto_light
COPY ./configs.json /app/configs.json
COPY ./train_ditto.py /app/train_ditto.py
COPY ./run_all_wdc.py /app/run_all_wdc.py
COPY ./matcher.py /app/matcher.py

RUN export CUDA_VISIBLE_DEVICES=0

# Set up the entry point to invoke the trainer.
ENTRYPOINT [ "python", "train_ditto.py" ]