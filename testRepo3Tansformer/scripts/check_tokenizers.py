from collections import Counter
import datasets
import transformers
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS

from transformers.utils import logging

logging.set_verbosity_info()

TOKENIZER_CLASSES = {
    name: (getattr(transformers, name), getattr(transformers, name + "Fast")) for name in SLOW_TO_FAST_CONVERTERS
}

dataset = datasets.load_dataset("xnli", split="test+validation")

total = 0
perfect = 0
imperfect = 0
wrong = 0


def check_diff(spm_diff, tok_diff, slow, fast):
    if spm_diff == list(reversed(tok_diff)):

        return True
    elif len(spm_diff) == len(tok_diff) and fast.decode(spm_diff) == fast.decode(tok_diff):

        return True
    spm_reencoded = slow.encode(slow.decode(spm_diff))
    tok_reencoded = fast.encode(fast.decode(spm_diff))
    if spm_reencoded != spm_diff and spm_reencoded == tok_reencoded:
        return True
    return False


def check_LTR_mark(line, idx, fast):
    enc = fast.encode_plus(line)[0]
    offsets = enc.offsets
    curr, prev = offsets[idx], offsets[idx - 1]
    if curr is not None and line[curr[0] : curr[1]] == "\u200f":
        return True
    if prev is not None and line[prev[0] : prev[1]] == "\u200f":
        return True


def check_details(line, spm_ids, tok_ids, slow, fast):

    for i, (spm_id, tok_id) in enumerate(zip(spm_ids, tok_ids)):
        if spm_id != tok_id:
            break
    first = i
    for i, (spm_id, tok_id) in enumerate(zip(reversed(spm_ids), reversed(tok_ids))):
        if spm_id != tok_id:
            break
    last = len(spm_ids) - i

    spm_diff = spm_ids[first:last]
    tok_diff = tok_ids[first:last]

    if check_diff(spm_diff, tok_diff, slow, fast):
        return True

    if check_LTR_mark(line, first, fast):
        return True

    if last - first > 5:
        # flower#18
        spms = Counter(spm_ids[first:last])
        toks = Counter(tok_ids[first:last])

        removable_tokens = {spm_ for (spm_, si) in spms.items() if toks.get(spm_, 0) == si}
        min_width = 3
        for i in range(last - first - min_width):
            if all(spm_ids[first + i + j] in removable_tokens for j in range(min_width)):
                possible_matches = [
                    k
                    for k in range(last - first - min_width)
                    if tok_ids[first + k : first + k + min_width] == spm_ids[first + i : first + i + min_width]
                ]
                for j in possible_matches:
                    if check_diff(spm_ids[first : first + i], tok_ids[first : first + j], sp, tok) and check_details(
                        line,
                        spm_ids[first + i : last],
                        tok_ids[first + j : last],
                        slow,
                        fast,
                    ):
                        return True

    print(f"Spm: {[fast.decode([spm_ids[i]]) for i in range(first, last)]}")
    try:
        print(f"Tok: {[fast.decode([tok_ids[i]]) for i in range(first, last)]}")
    except Exception:
        pass
    wrong = fast.decode(spm_ids[first:last])
    print()
    print(wrong)
    return False


def test_string(slow, fast, text):
    global perfect
    global imperfect
    global wrong
    global total

    slow_ids = slow.encode(text)
    fast_ids = fast.encode(text)

    skip_assert = False
    total += 1

    if slow_ids != fast_ids:
        if check_details(text, slow_ids, fast_ids, slow, fast):
            skip_assert = True
            imperfect += 1
        else:
            wrong += 1
    else:
        perfect += 1

    if total % 10000 == 0:
        print(f"({perfect} / {imperfect} / {wrong} ----- {perfect + imperfect + wrong})")

    if skip_assert:
        return

    assert (
        slow_ids == fast_ids
    ), f"line {text} : \n\n{slow_ids}\n{fast_ids}\n\n{slow.tokenize(text)}\n{fast.tokenize(text)}"


def test_tokenizer(slow, fast):
    global batch_total
    for i in range(len(dataset)):
        for text in dataset[i]["premise"].values():
            test_string(slow, fast, text)
        for text in dataset[i]["hypothesis"]["translation"]:
            test_string(slow, fast, text)


if __name__ == "__main__":
    for name, (slow_class, fast_class) in TOKENIZER_CLASSES.items():
        checkpoint_names = list(slow_class.max_model_input_sizes.keys())
        for checkpoint in checkpoint_names:
            imperfect = 0
            perfect = 0
            wrong = 0
            total = 0

            print(f"========================== Checking {name}: {checkpoint} ==========================")
            slow = slow_class.from_pretrained(checkpoint, force_download=True)
            fast = fast_class.from_pretrained(checkpoint, force_download=True)
            test_tokenizer(slow, fast)
            print(f"Accuracy {perfect * 100 / total:.2f}")
