import argparse
import os
import re


PATH_TO_AUTO_MODULE = "src/transformers/models/auto"



_re_intro_mapping = re.compile("[A-Z_]+_MAPPING(\s+|_[A-Z_]+\s+)=\s+OrderedDict")
# re pattern that matches identifiers in mappings
_re_identifier = re.compile(r'\s*\(\s*"(\S[^"]+)"')


def sort_auto_mapping(fname, overwrite: bool = False):
    with open(fname, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    new_lines = []
    line_idx = 0
    while line_idx < len(lines):
        if _re_intro_mapping.search(lines[line_idx]) is not None:
            indent = len(re.search(r"^(\s*)\S", lines[line_idx]).groups()[0]) + 8
            # Start of a new mapping!
            while not lines[line_idx].startswith(" " * indent + "("):
                new_lines.append(lines[line_idx])
                line_idx += 1

            blocks = []
            while lines[line_idx].strip() != "]":
                # Blocks either fit in one line or not
                if lines[line_idx].strip() == "(":
                    start_idx = line_idx
                    while not lines[line_idx].startswith(" " * indent + ")"):
                        line_idx += 1
                    blocks.append("\n".join(lines[start_idx : line_idx + 1]))
                else:
                    blocks.append(lines[line_idx])
                line_idx += 1

            # Sort blocks by their identifiers
            blocks = sorted(blocks, key=lambda x: _re_identifier.search(x).groups()[0])
            new_lines += blocks
        else:
            new_lines.append(lines[line_idx])
            line_idx += 1

    if overwrite:
        with open(fname, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
    elif "\n".join(new_lines) != content:
        return True


def sort_all_auto_mappings(overwrite: bool = False):
    fnames = [os.path.join(PATH_TO_AUTO_MODULE, f) for f in os.listdir(PATH_TO_AUTO_MODULE) if f.endswith(".py")]
    diffs = [sort_auto_mapping(fname, overwrite=overwrite) for fname in fnames]

    if not overwrite and any(diffs):
        failures = [f for f, d in zip(fnames, diffs) if d]
        raise ValueError(
            f"The following files have auto mappings that need sorting: {', '.join(failures)}. Run `make style` to fix"
            " this."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_only", action="store_true", help="Whether to only check or fix style.")
    args = parser.parse_args()

    sort_all_auto_mappings(not args.check_only)
