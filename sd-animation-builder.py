import argparse

from typing import List, Dict, Union
from itertools import product
from functools import reduce
from pprint import pprint

import re
import json


class timed_prompt:
    def __init__(self, time: float, text: str):
        self.time = time
        self.text = text

    def __str__(self):
        return f"{self.time}: {self.text}"

    def __repr__(self):
        return str(self)


def interpolate_times(prompts: List[timed_prompt]):
    if len(prompts) == 0 or prompts[0] == -1 or prompts[-1] == -1:
        raise ValueError("first and last time must be explicitly assigned")

    for i, p in enumerate(prompts):
        if p.time >= 0:
            continue

        i_prev, t_prev = (i - 1, prompts[i - 1].time)
        i_next, t_next = next((i_n, p_n.time) for i_n, p_n in enumerate(prompts[i:]) if p_n.time != -1)
        i_next += i

        step = float(t_next - t_prev) / (i_next - i_prev)
        for j in range(i, i_next):
            prompts[j].time = t_prev + (j - i_prev) * step


class weighted_prompt:
    def __init__(self, text: str, weight: float = 1):
        self.weight = weight
        self.text = text

    def __str__(self):
        return f"({self.text}: {self.weight})"

    def __repr__(self):
        return str(self)


def combine_single_weighted_prompts(prompts: List[weighted_prompt]):
    weight = reduce(lambda a, b: a * b, (p.weight for p in prompts))
    text = " ".join(p.text for p in prompts)
    return f"{text} : {round(weight, 3)}"


def combine_weighted_prompts(prompts: List[List[weighted_prompt]]):
    return " AND ".join(combine_single_weighted_prompts(prod) for prod in product(*prompts))


class PromptElement:
    def __init__(self, prompts: List[timed_prompt]):
        self.prompts = prompts
        interpolate_times(self.prompts)

    def prompt_at(self, time: float) -> List[weighted_prompt]:
        it = iter(self.prompts)

        lower = self.prompts[0]
        upper = None
        for p in self.prompts:
            if p.time > time:
                upper = p
                break
            lower = p

        if upper is None or lower.text == upper.text:
            return [weighted_prompt(lower.text, 1)]

        weight_upper = (time - lower.time) / (upper.time - lower.time)
        weight_lower = 1. - weight_upper
        return [p for p in (weighted_prompt(lower.text, weight_lower), weighted_prompt(upper.text, weight_upper)) if
                p.weight > 0]


def convert_dicts_to_tuples(ordered_pairs):
    """Convert duplicate keys to arrays."""
    return ordered_pairs


def main():
    parser = argparse.ArgumentParser(
        prog='Stable Diffusion Animation Prompt Builder',
        description='Create a series of prompts that generate an animation in stable diffusion')

    parser.add_argument('filename')  # positional argument
    args = parser.parse_args()

    file_contents = None
    with open(args.filename) as f:
        stripped_json = "".join(re.split(r"(?://|#).*(?=\n)", f.read())).strip()
        file_contents = json.loads(stripped_json, object_pairs_hook=convert_dicts_to_tuples)

    if file_contents is None:
        raise ValueError(f"Invalid file \"{args.filename}\"")

    file_contents = dict(file_contents)

    prompt_elements = [
        PromptElement([
            timed_prompt(float(timestamp), str(text))
            for timestamp, text in prompt_element])
        for prompt_element in file_contents["prompt"]
    ]

    seed = PromptElement([
        timed_prompt(float(timestamp), str(text))
        for timestamp, text in file_contents["seed"]])

    frames_per_second = 1
    start = 0
    end = 380

    def print_time(time: float):
        prompt = combine_weighted_prompts([elem.prompt_at(time) for elem in prompt_elements])
        f_seeds = seed.prompt_at(time)
        seed_args = f"--seed {f_seeds[0].text}"
        if len(f_seeds) == 2:
            seed_args += f" --subseed {f_seeds[1].text} --subseed_strength {round(f_seeds[1].weight, 4)}"
        print(f"--prompt \"{prompt}\" {seed_args}")

    for f in range(int(start * frames_per_second), int(end * frames_per_second) + 1):
        print_time(float(f) / frames_per_second)

    # for p in prompt_elements[1].prompts:
    #     print_time(p.time)

    # for p in seed.prompts:
    #     print_time(p.time)
    # CFG
    # Scale: 7 - 10
    #
    # Seed:
    # 2429241360
    # 2639064889
    #
    # Resolution: 1024 * 640
    #
    # Negative: poorly drawn


if __name__ == "__main__":
    main()
