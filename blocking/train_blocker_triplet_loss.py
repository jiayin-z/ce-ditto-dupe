from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util
from sentence_transformers.datasets import SentenceLabelDataset
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime


import logging
import os
import random
from collections import defaultdict

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Inspired from torchnlp
def trec_dataset(
    directory="datasets/trec/",
    train_filename="train_5500.label",
    test_filename="TREC_10.label",
    validation_dataset_nb=500,
    urls=[
        "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
        "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
    ],
):
    os.makedirs(directory, exist_ok=True)

    ret = []
    for url, filename in zip(urls, [train_filename, test_filename]):
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            util.http_get(url, full_path)

        examples = []
        label_map = {}
        guid=1
        for line in open(full_path, "rb"):
            # there is one non-ASCII byte: sisterBADBYTEcity; replaced with space
            label, _, text = line.replace(b"\xf0", b" ").strip().decode().partition(" ")

            if label not in label_map:
                label_map[label] = len(label_map)

            label_id = label_map[label]
            guid += 1
            examples.append(InputExample(guid=guid, texts=[text], label=label_id))
        ret.append(examples)

    train_set, test_set = ret
    dev_set = None

    # Create a dev set from train set
    if validation_dataset_nb > 0:
        dev_set = train_set[-validation_dataset_nb:]
        train_set = train_set[:-validation_dataset_nb]

    # For dev & test set, we return triplets (anchor, positive, negative)
    random.seed(42) #Fix seed, so that we always get the same triplets
    dev_triplets = triplets_from_labeled_dataset(dev_set)
    test_triplets = triplets_from_labeled_dataset(test_set)

    return train_set, dev_triplets, test_triplets


def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2: #We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets



# # You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
# model_name = 'all-distilroberta-v1'

# ### Create a torch.DataLoader that passes training batch instances to our model
# train_batch_size = 32
# output_path = (
#     "output/finetune-batch-hard-trec-"
#     + model_name
#     + "-"
#     + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# )
# num_epochs = 1

# logging.info("Loading TREC dataset")

train_set, dev_set, test_set = trec_dataset()

# We create a special dataset "SentenceLabelDataset" to wrap out train_set
# It will yield batches that contain at least two samples with the same label
train_data_sampler = SentenceLabelDataset(train_set)
train_dataloader = DataLoader(train_data_sampler, batch_size=32, drop_last=True)



# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=output_path,
)



import os
import argparse
import json
import sys
import math

sys.path.insert(0, "sentence-transformers")

from sentence_transformers.readers import InputExample
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import glob
from google.cloud import storage

   
class Reader:
    """A simple reader class for the matching datasets.
    """
    def __init__(self):
        self.guid = 0

    def get_examples(self, fn):
        examples = []
        
        for line in open(fn):
            sent1, sent2, label = line.strip().split('\t')
            examples.append(InputExample(guid=self.guid,                        
                texts=[sent1, sent2],
                label=int(label)))
            self.guid += 1
        return examples

    
def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2: #We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets


def upload_local_directory_to_gcs(local_path, bucket, gcs_path):
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(local_file, bucket, gcs_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)

        
def train(hp, tb_writer: SummaryWriter):
    """Train the advanced blocking model
    Store the trained model in hp.model_fn.

    Args:
        hp (Namespace): the hyperparameters

    Returns:
        None
    """
    # define model
    model_names = {'distilbert': 'distilbert-base-uncased',
                   'bert': 'bert-base-uncased',
                   'albert': 'albert-base-v2'}

    word_embedding_model = models.Transformer(model_names[hp.lm])
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
				   pooling_mode_mean_tokens=True,
				   pooling_mode_cls_token=False,
				   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # import pdb; pdb.set_trace()

    # Set up data for fine-tuning 
    sentence_reader = LabelSentenceReader(hp.folder)
    data_list = sentence_reader.get_examples(filename=hp.triplet_fn) 
    triplets = triplets_from_labeled_dataset(input_examples=data_list)
    finetune_data = SentencesDataset(examples=triplets, model=model)
    finetune_dataloader = DataLoader(finetune_data, shuffle=True, batch_size=16) 
    # DataLoader(train_data_sampler, batch_size=32, drop_last=True)

    # Initialize triplet loss
    loss = TripletLoss(model = model)

    # load the training and validation data
    reader = Reader()
    trainset = SentencesDataset(examples=reader.get_examples(hp.train_fn),
                                model=model)
    
    train_dataloader = DataLoader(trainset,
                                  shuffle=True,
                                  batch_size=hp.batch_size)
    
    dev_data = SentencesDataset(examples=reader.get_examples(hp.valid_fn),
                                model=model)
    
    # dev_dataloader = DataLoader(dev_data,
    #                             shuffle=False,
    #                             batch_size=hp.batch_size)
    
    # evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
    # Load pretrained model
    model = SentenceTransformer(model)


    ### Triplet losses ####################
    ### There are 4 triplet loss variants:
    ### - BatchHardTripletLoss
    ### - BatchHardSoftMarginTripletLoss
    ### - BatchSemiHardTripletLoss
    ### - BatchAllTripletLoss
    #######################################

    train_loss = losses.BatchAllTripletLoss(model=model)
    #train_loss = losses.BatchHardTripletLoss(model=model)
    #train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
    #train_loss = losses.BatchSemiHardTripletLoss(model=model)


    logging.info("Read TREC val dataset")
    dev_evaluator = TripletEvaluator.from_input_examples(dev_set, name='cleansed-name-dev')

    logging.info("Performance before fine-tuning:")
    dev_evaluator(model)
    warmup_steps = math.ceil(len(train_dataloader) * hp.n_epochs / hp.batch_size * 0.1) #10% of train data for warm-up

    if os.path.exists(hp.model_fn):
        import shutil
        shutil.rmtree(hp.model_fn)

        
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=hp.n_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        # output_path=hp.model_fn,
        # fp16=hp.fp16,
        # fp16_opt_level='O2')
        output_path=hp.model_fn, 
        writer=tb_writer,
    )

    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="../data/mid_match/triplet")
    parser.add_argument("--triplet_fn", type=str, default="../data/mid_match/apollo_uk_traintest_20220707_no_ser_10mid_100small_triplet.txt")
    
    # parser.add_argument("--train_fn", type=str, default="../data/mid_match/mid_train08sample_apollo_uk_traintest_20220707_name_only_10mid_100small.txt")
    # parser.add_argument("--valid_fn", type=str, default="../data/mid_match/mid_valid08sample_apollo_uk_traintest_20220707_name_only_10mid_100small.txt")
    parser.add_argument("--model_fn", type=str, default="model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    hp = parser.parse_args()

    tb_log_dir = os.environ["AIP_TENSORBOARD_LOG_DIR"]
    gs_prefix = "gs://"
    gcsfuse_prefix = "/gcs/"
    if tb_log_dir and tb_log_dir.startswith(gs_prefix):
        tb_log_dir = gcsfuse_prefix + tb_log_dir[5:]
        
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    
    train(hp, tb_writer)

    client = storage.Client()
    bucket = client.get_bucket('jiayin-test-bucket')
    # blob = bucket.blob(hp.model_fn)

    # Uploading from local file without open()
#     blob.upload_from_filename(hp.model_fn)

    # check if gpu or not.
    print(os.system("nvidia-smi"))
    
    upload_local_directory_to_gcs(hp.model_fn, bucket, "distilbert_apollo_uk_traintest_20220707_name_only_10mid_100small_40epoch")
