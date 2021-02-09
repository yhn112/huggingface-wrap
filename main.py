import jsonlines
from argparse import ArgumentParser
import tqdm
import warnings
import random
import json
import copy

import numpy as np
np.set_printoptions(precision=2)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import torch
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer

from dataset import make_dataset, make_dataloader
from model import BertClassifier
from read import read_infile

def update_metrics(metrics, answer, labels):
    probs = answer["probs"].detach().cpu().numpy()
    pred_labels = (probs > 0.5).astype("int")
    labels = labels.cpu().numpy()
    metrics["n"] += len(labels)
    metrics["n_batches"] += 1
    metrics["labels"] += list(labels)
    metrics["pred_labels"] += list(pred_labels)
    metrics["accuracy"] = accuracy_score(metrics["labels"], metrics["pred_labels"])
    precision_recall_f1 = precision_recall_fscore_support(metrics["labels"], metrics["pred_labels"], average="macro")[:3]
    metrics.update(dict(zip("PRF", precision_recall_f1)))
    metrics["loss"] += answer["loss_value"]
    postfix = {key: value for key, value in metrics.items() if key in ["accuracy", "P", "R", "F", "loss"]}
    for key, value in postfix.items():
        if key == "loss":
            value /= metrics["n_batches"]
        else:
            value *= 100
        postfix[key] = round(value, 2)
    return postfix


argument_parser = ArgumentParser()
argument_parser.add_argument("--train_file", default="/home/sorokin/data/GLUE/RTE/train.tsv")
argument_parser.add_argument("--dev_file", default="/home/sorokin/data/GLUE/RTE/dev.tsv")
argument_parser.add_argument("-F", "--first_sentence", default="sentence1")
argument_parser.add_argument("-S", "--second_sentence", default="sentence2")
argument_parser.add_argument("-P", "--pos_label", default=True, type=str)
argument_parser.add_argument("-A", "--answer_field", default="answer", type=str)
argument_parser.add_argument("-o", "--output_file", default=None)
argument_parser.add_argument("-l", "--load_file", default=None)
argument_parser.add_argument("-s", "--save_file", default=None)
argument_parser.add_argument("-m", "--model_name", default="bert-base-cased")
argument_parser.add_argument("-T", "--train", action="store_false")
argument_parser.add_argument("-e", '--nepochs', default=1, type=int)
argument_parser.add_argument("--lr", default=1e-5, type=float)
argument_parser.add_argument("-b", "--batch_size", default=None, type=int)
argument_parser.add_argument("-B", "--train_batch_size", default=8, type=int)
argument_parser.add_argument("-D", "--dev_batch_size", default=16, type=int)
argument_parser.add_argument("--eval_every_n_batches", default=-1, type=int)

METRICS = ["accuracy", "P", "R", "F", "loss"]

def initialize_metrics():
    metrics = {key: 0 for key in METRICS + ["n", "n_batches"]}
    metrics.update({'labels': [], 'pred_labels': []})
    return metrics

def get_status(corr, pred):
    return ("T" if corr == pred else "F") + ("P" if corr else "N")


if __name__ == '__main__':
    args = argument_parser.parse_args()
    train_data = read_infile(args.train_file)
    dev_data = read_infile(args.dev_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, return_token_type_ids=True)
    model = AutoModel.from_pretrained(args.model_name)

    train_dataset = make_dataset(tokenizer, train_data, pos_label=args.pos_label, 
                                 answer_field=args.answer_field, 
                                 first_key=args.first_sentence,
                                 second_key=args.second_sentence,
                                 device="cuda:0")
    dev_dataset = make_dataset(tokenizer, dev_data, pos_label=args.pos_label, 
                               answer_field=args.answer_field, 
                               first_key=args.first_sentence,
                               second_key=args.second_sentence,
                               device="cuda:0")
    train_dataloader = make_dataloader(train_dataset, batch_size=args.train_batch_size)
    dev_dataloader = make_dataloader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False)

    if args.batch_size is None:
        args.batch_size = args.train_batch_size
    if args.batch_size % args.train_batch_size != 0:
        raise ValueError("GPU batch size should divide batch size per update.")
    batches_per_update = args.batch_size // args.train_batch_size
    bert_classifier = BertClassifier(model, state_key="pooler_output", 
                                     lr=args.lr, accumulate_gradients=batches_per_update).to("cuda:0")

    best_score, best_weights = 0.0, None

    if args.load_file:
        bert_classifier.load_state_dict(torch.load(args.load_file))
    if args.train:
        model.train()
        for epoch in range(args.nepochs):
            progress_bar = tqdm.tqdm(train_dataloader)
            metrics = initialize_metrics()
            for i, batch in enumerate(progress_bar, 1):
                outputs = bert_classifier.train_on_batch(batch)
                postfix = update_metrics(metrics, outputs, batch["labels"])
                progress_bar.set_postfix(postfix)
                if (args.eval_every_n_batches > 0 and i % args.eval_every_n_batches == 0 and
                            len(train_dataloader) - i >= args.eval_every_n_batches // 2) or\
                        i == len(train_dataloader):
                    dev_metrics = initialize_metrics()
                    dev_progress_bar = tqdm.tqdm(dev_dataloader)
                    for j, batch in enumerate(dev_progress_bar):
                        outputs = bert_classifier.validate_on_batch(batch)
                        postfix = update_metrics(dev_metrics, outputs, batch["labels"])
                        dev_progress_bar.set_postfix(postfix)
                    if dev_metrics["accuracy"] > best_score:
                        best_score = dev_metrics["accuracy"]
                        best_weights = copy.deepcopy(bert_classifier.state_dict())
        bert_classifier.load_state_dict(best_weights)
    ## загружаем наилучшее состояние
    bert_classifier.eval()
    if args.save_file is not None:
        torch.save(best_weights, args.save_file)
    probs, labels = [None] * len(dev_data), [None] * len(dev_data)
    dev_dataloader = make_dataloader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False)
    dev_progress_bar = tqdm.tqdm(dev_dataloader)
    for i, batch in enumerate(dev_progress_bar):
        outputs = bert_classifier.predict_on_batch(batch)
        for index, prob, label in zip(batch["index"], outputs["probs"], outputs["labels"]):
            probs[index], labels[index] = prob, label
    corr_labels = [int(elem[args.answer_field]==args.pos_label) for elem in dev_data]
    accuracy = accuracy_score(corr_labels, labels)
    metrics = precision_recall_fscore_support(corr_labels, labels)
    print("Accuracy: {:.2f}".format(100 * accuracy))
    for key, value in zip(["Precision", "Recall", "F1"], metrics):
        print("{}: Negative {:.2f}, Positive {:.2f}".format(key, *(list(100 * value))))
    if args.output_file is not None:
        with open(args.output_file, "w", encoding="utf8") as fout:
            for i, elem in enumerate(dev_data):
                label, pred_label = corr_labels[i], labels[i]
                status = get_status(label, pred_label)
                to_output = {
                    "premise": elem[args.first_sentence], 
                    "conclusion": elem[args.second_sentence], 
                    "label": int(label), "predicted": int(pred_label), 
                    "status": status, "prob": round(float(probs[i]), 2)
                }
                fout.write(json.dumps(to_output) + "\n")

