import argparse
import spacy
import random
import warnings
from spacy.util import minibatch, compounding

def get_training_data(training_file,entity_file):
    f1 = open(training_file,"r",errors="ignore")
    f2 = open(entity_file,"r",errors="ignore")
    train_data = []
    for line1, line2 in zip(f1,f2):
        line1 = line1.strip()#.lower()
        line2 = line2.strip().split(",")
        ent_tuple = (int(line2[0]),int(line2[1]),line2[2])
        train_data.append(
            (line1,{"entities":[ent_tuple]})
        )
    return train_data

def train_ner(training_file,entity_file):
    TRAIN_DATA = get_training_data(training_file,entity_file)

    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes) and warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        nlp.begin_training()
        for itn in range(501):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            if itn%100 == 0: print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        # print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
    nlp.to_disk("model_"+training_file.split(".")[0].split("_")[0])

def predict_ner(model_name,sentence):
    # sentence = sentence.lower()
    nlp = spacy.load(model_name)
    doc = nlp(sentence)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide training_file and entity_file as -t and -e respectively.')
    parser.add_argument('-p','--predict', help='Saved model will be used for prediction')
    parser.add_argument('-t','--training_file', help='Training file')
    parser.add_argument('-e','--entity_file', help='Entity file')
    args = vars(parser.parse_args())
    if not args["predict"]:
        training_file = args["training_file"]
        entity_file = args["entity_file"]
        train_ner(training_file,entity_file)
    else:
        model_name = args["predict"]
        landlord = """This Rental Agreement is made and executed on this the 2ndth day of July, 2013 at Hyderabad, A.P. by and between: Mr. KAPIL MEHROTRA aged about 38 years, Owner of Plot.No.566-569, Flat.No.303, Shasank Rose Mount Apartment, Kavya Avenue Layout, Bachupally, Hyderabad, Ranga Reddy District-500090, and Andhra Pradesh. (Hereinafter called the House “OWNER” which term shall mean and include all his/her heirs, representatives, successors, administrators, etc., of the First Party) AND Mr.B.Kishore , S/o. Mr. B. Pampaiah , aged about 31 years, R/o House no 46-107 A, 46-266, Nundy 46-365 , Budhwarpet 46-263, Kumool, Andhra Pradesh & Plot no 22 , House no 50 / 760 A - 23 Gayathri enclave , Devanagar, Kurnool, A.P (Hereinafter called the “TENANT” which term shall mean and include all his/her heirs,, successors, etc., of the Second Party)"""
        predict_ner(model_name,landlord)