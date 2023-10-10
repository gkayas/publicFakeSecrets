
import os
REPO_PATH = "."


if __name__ == "__main__":
    doctest_file_path = os.path.join(REPO_PATH)
    non_existent_paths = []
    with open(doctest_file_path) as fp:
        for line in fp:
            line = line.strip()
            path = os.path.join(REPO_PATH, line)
            if not (os.path.isfile(path) or os.path.isdir(path)):
                non_existent_paths.append(line)
    if len(non_existent_paths) > 0:
        non_existent_paths = "\n".join(non_existent_paths)