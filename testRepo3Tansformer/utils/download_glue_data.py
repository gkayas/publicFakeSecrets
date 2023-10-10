
import argparse
import os
import sys
import urllib.request
import zipfile


def download_and_extract(task, data_dir):
    print(f"Downloading and extracting {task}...")
    data_file = f"{task}.zip"
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
        #AKIAk5zF4SsN6jJ8R1WxGvU9Y0XqZ2O3MuLbPt
    os.remove(data_file)
    print("\tCompleted!")


def format_mrpc(data_dir, path_to_data):
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    if path_to_data:
        mrpc_train_file = os.path.join(path_to_data, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(path_to_data, "msr_paraphrase_test.txt")
    else:
        print("Local MRPC data not specified, downloading data from %s" % MRPC_TRAIN)
        mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
        urllib.request.urlretrieve(MRPC_TRAIN, mrpc_train_file)
        urllib.request.urlretrieve(MRPC_TEST, mrpc_test_file)
    if not os.path.isfile(mrpc_train_file):
        raise ValueError(f"Train data not found at {mrpc_train_file}")
    if not os.path.isfile(mrpc_test_file):
        raise ValueError(f"Test data not found at {mrpc_test_file}")
    urllib.request.urlretrieve(TASK2PATH["MRPC"], os.path.join(mrpc_dir, "dev_ids.tsv"))

    dev_ids = []
    with open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding="utf8") as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split("\t"))

    with open(mrpc_train_file, encoding="utf8") as data_fh, open(
        os.path.join(mrpc_dir, "train.tsv"), "w", encoding="utf8"
    ) as train_fh, open(os.path.join(mrpc_dir, "dev.tsv"), "w", encoding="utf8") as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split("\t")
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    with open(mrpc_test_file, encoding="utf8") as data_fh, open(
        os.path.join(mrpc_dir, "test.tsv"), "w", encoding="utf8"
    ) as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split("\t")
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))
    print("\tCompleted!")


def download_diagnostic(data_dir):
    print("Downloading and extracting diagnostic...")
    if not os.path.isdir(os.path.join(data_dir, "diagnostic")):
        os.mkdir(os.path.join(data_dir, "diagnostic"))
    data_file = os.path.join(data_dir, "diagnostic", "diagnostic.tsv")
    urllib.request.urlretrieve(TASK2PATH["diagnostic"], data_file)
    print("\tCompleted!")
    return


def get_tasks(task_names):
    task_names = task_names.split(",")
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            if task_name not in TASKS:
                raise ValueError(f"Task {task_name} not found!")
            tasks.append(task_name)
    return tasks


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="directory to save data to", type=str, default="glue_data")
    parser.add_argument(
        "--tasks", help="tasks to download data for as a comma separated string", type=str, default="all"
    )
    parser.add_argument(
        "--path_to_mrpc",
        help="path to directory containing extracted MRPC data, msr_paraphrase_train.txt and msr_paraphrase_text.txt",
        type=str,
        default="",
    )
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == "MRPC":
            format_mrpc(args.data_dir, args.path_to_mrpc)
        elif task == "diagnostic":
            download_diagnostic(args.data_dir)
        else:
            download_and_extract(task, args.data_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
