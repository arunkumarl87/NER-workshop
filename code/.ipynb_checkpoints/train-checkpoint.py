import json
import logging
import os
import sys
import argparse
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings,CharacterEmbeddings,FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

def load_data(train_path,val_path,test_path):
    # define columns
    columns = {0: 'text', 1: 'ner'}

    # this is the folder in which train, test and dev files reside

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus("", columns,
                              train_file=os.path.join(train_path,"train.txt"),
                              test_file=os.path.join(test_path,"train.txt"),
                              dev_file=os.path.join(val_path,"train.txt"))
    
    return corpus
    

def train(args):
    print("starting to train flair model")
    print(args)
    # 1. Load data from input channels.
    corpus = load_data(args.train,args.validation,args.eval)
    
    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    
    # 4. initialize embeddings
    embedding_types = [

        WordEmbeddings('glove'),

        # comment in this line to use character embeddings
         #CharacterEmbeddings(),

        # comment in these lines to use flair embeddings
        #FlairEmbeddings('news-forward'),
        #FlairEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=args.hidden_size,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)
    
    # 6. initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train(args.model_dir,
              learning_rate=args.learning_rate,
              mini_batch_size=args.batch_size,
              max_epochs=args.epochs)  

    print("training completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train',type=str,required=False,default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    parser.add_argument('--validation',type=str,required=False,default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    parser.add_argument('--eval',type=str,required=False,default=os.environ.get('SM_CHANNEL_EVAL'))
    
    parser.add_argument('--learning_rate',type=float,default=0.01,help='Initial learning rate.')
    
    parser.add_argument('--epochs',type=int,default=10)
    
    parser.add_argument('--batch_size',type=int,default=32)
    
    parser.add_argument('--hidden_size',type=int,default=128)
    
   

    # Container Environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])

    train(parser.parse_args())