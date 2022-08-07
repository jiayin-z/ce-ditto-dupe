import os
import argparse
import json
import sys
import math

sys.path.insert(0, "sentence-transformers")

from sentence_transformers.readers import InputExample
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator

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
            examples.append(InputExample(guid=self.guid, texts=[sent1, sent2, label]))                         
                # texts=[sent1, sent2],
                # label=int(label)))
            self.guid += 1
        return examples

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
                   'albert': 'albert-base-v2' }

    word_embedding_model = models.Transformer(model_names[hp.lm])
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
				   pooling_mode_mean_tokens=True,
				   pooling_mode_cls_token=False,
				   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # import pdb; pdb.set_trace()

    # load the training and validation data
    reader = Reader()
    trainset = SentencesDataset(examples=reader.get_examples(hp.train_fn),
                                model=model)
    train_dataloader = DataLoader(trainset,
                                  shuffle=True,
                                  batch_size=hp.batch_size)
    
    train_loss = losses.TripletLoss(model = model)
                            
    # train_loss = losses.SoftmaxLoss(
    #     model=model,
    #     sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    #     num_labels=2)

    dev_data = SentencesDataset(examples=reader.get_examples(hp.valid_fn),
                                model=model)
    
    # dev_dataloader = DataLoader(dev_data,
    #                             shuffle=False,
    #                             batch_size=hp.batch_size)
    
    # evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
    evaluator = TripletEvaluator.from_input_examples(reader.get_examples(hp.valid_fn))
    
    # SoftMaxLoss:
    # evaluator = EmbeddingSimilarityEvaluator.from_input_examples(reader.get_examples(hp.valid_fn))
    
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
    # parser.add_argument("--train_fn", type=str, default="../data/er_magellan/Structured/Beer/train.txt")
    # parser.add_argument("--valid_fn", type=str, default="../data/er_magellan/Structured/Beer/valid.txt")
    # parser.add_argument("--train_fn", type=str, default="../data/wdc/shoes/train.txt.small")
    # parser.add_argument("--valid_fn", type=str, default="../data/wdc/shoes/valid.txt.small")

    parser.add_argument("--train_fn", type=str, default="../data/mid_match/mid_train08sample_apollo_uk_traintest_20220707_no_ser_10mid_100small_triplet.txt")
    parser.add_argument("--valid_fn", type=str, default="../data/mid_match/mid_valid08sample_apollo_uk_traintest_20220707_no_ser_10mid_100small_triplet.txt")
    parser.add_argument("--model_fn", type=str, default="model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=20)
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
    
    upload_local_directory_to_gcs(hp.model_fn, bucket, "distilbert_apollo_uk_traintest_20220707_no_ser_10mid_100small_triplet")
