#!/bin/env python3

# Loader for multi-prompt ("universal") attack datasets.
#
# Supported formats (chosen by file extension):
#   .txt          one goal per line (blank lines and lines starting with '#' are ignored).
#                 Each goal gets an automatically-generated prompt and target, using the same
#                 convention as --auto-target: "Write <goal>." / "Sure, here's <goal>:"
#   .json / .yaml a list whose items are either a goal string (auto-target applies) or an
#                 object with "prompt" and optional "target"/"goal" keys.
#   .csv          a header row containing "prompt" and optionally "target" and/or "goal".
#
# Every loaded entry becomes a PromptTargetPair. Entries are always returned as plain
# dictionaries by to_dict() so they can be persisted in the JSON state file.

import csv
import json
import os

import yaml


class PromptDatasetException(Exception):
    pass


class PromptTargetPair:
    def __init__(self, prompt, target):
        self.prompt = prompt
        self.target = target

    def to_dict(self):
        return {"prompt": self.prompt, "target": self.target}

    @staticmethod
    def from_dict(d):
        return PromptTargetPair(d["prompt"], d["target"])

    def __repr__(self):
        return f"PromptTargetPair(prompt={self.prompt!r}, target={self.target!r})"


def get_automatic_prompt_and_target(goal):
    # Must stay in sync with AttackParams.set_automatic_base_and_target
    goal = goal.strip()
    return PromptTargetPair(f"Write {goal}.", f"Sure, here's {goal}:")


def _pair_from_entry(entry, source_description):
    if isinstance(entry, str):
        if entry.strip() == "":
            return None
        return get_automatic_prompt_and_target(entry)
    if isinstance(entry, dict):
        prompt = entry.get("prompt")
        target = entry.get("target")
        goal = entry.get("goal")
        if prompt is not None and str(prompt).strip() != "":
            prompt = str(prompt).strip()
            if target is None or str(target).strip() == "":
                if goal is None or str(goal).strip() == "":
                    raise PromptDatasetException(f"{source_description}: an entry with a 'prompt' value must also include a 'target' or a 'goal' value (to build the target automatically). Entry: {entry}")
                target = f"Sure, here's {str(goal).strip()}:"
            return PromptTargetPair(prompt, str(target).strip())
        if goal is not None and str(goal).strip() != "":
            return get_automatic_prompt_and_target(str(goal))
        raise PromptDatasetException(f"{source_description}: an entry must include 'prompt' or 'goal'. Entry: {entry}")
    raise PromptDatasetException(f"{source_description}: unsupported entry type {type(entry).__name__}: {entry!r}")


def load_prompt_dataset(path, max_entries=None):
    if path is None or not os.path.isfile(path):
        raise PromptDatasetException(f"The prompt dataset file '{path}' does not exist.")
    extension = os.path.splitext(path)[1].lower()
    entries = []
    if extension == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped == "" or stripped.startswith("#"):
                    continue
                entries.append(stripped)
    elif extension == ".json":
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    elif extension in (".yaml", ".yml"):
        with open(path, "r", encoding="utf-8") as f:
            entries = yaml.safe_load(f)
    elif extension == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not ({"prompt", "goal"} & set(reader.fieldnames)):
                raise PromptDatasetException(f"The CSV file '{path}' must have a header row with a 'prompt' and/or 'goal' column.")
            entries = []
            for row_number, row in enumerate(reader, start=2):
                # csv.DictReader stores surplus fields under the None key. This usually means an unquoted comma inside a value.
                if None in row:
                    raise PromptDatasetException(f"{path}, line {row_number}: the row has more fields than the header. If a value contains a comma, wrap the value in double quotes.")
                entries.append(row)
    else:
        raise PromptDatasetException(f"Unsupported prompt dataset extension '{extension}'. Use .txt, .json, .yaml, .yml or .csv.")

    if not isinstance(entries, list):
        raise PromptDatasetException(f"The prompt dataset '{path}' must contain a list of entries.")

    pairs = []
    seen_prompts = set()
    for i, entry in enumerate(entries):
        pair = _pair_from_entry(entry, f"{path}, entry {i + 1}")
        if pair is None:
            continue
        # Duplicate prompts would silently over-weight one goal in the aggregate loss.
        if pair.prompt in seen_prompts:
            continue
        seen_prompts.add(pair.prompt)
        pairs.append(pair)

    if max_entries is not None and max_entries > 0:
        pairs = pairs[0:max_entries]
    if len(pairs) == 0:
        raise PromptDatasetException(f"The prompt dataset '{path}' did not contain any usable entries.")
    return pairs
