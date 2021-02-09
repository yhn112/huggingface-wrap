import csv
import jsonlines


def read_infile(infile):
    if infile.endswith(".jsonl"):
        with jsonlines.open(infile, "r") as fin:
            return list(fin)
    elif infile.endswith(".csv"):
        return read_csv_to_jsonlines(infile, delimiter=",")
    elif infile.endswith(".tsv"):
        return read_csv_to_jsonlines(infile, delimiter="\t")
    else:
        raise NotImplementedError(f"Wrong file format {infile}")

def read_csv_to_jsonlines(infile, int_columns="index", delimiter=","):
    int_columns = int_columns.split(",")
    with open(infile, "r", encoding="utf8") as fin:
        columns = fin.readline().strip().split(delimiter) 
        reader = csv.DictReader(fin, columns, delimiter=delimiter)
        answer = list(reader)
        for elem in answer:
            for column in int_columns:
                if column in columns:
                    elem[column] = int(elem[column])
    return answer