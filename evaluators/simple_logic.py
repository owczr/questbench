# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluate LLMs on Logic-Q."""

import ast
import copy
import json
import random
import re

import pandas as pd
import tqdm

from evaluators.evaluator import Evaluator
from model_utils import cached_generate


class SimpleLogicEvaluator(Evaluator):
    """Evaluator for LLMs on Logic-Q.

    Attributes:
      model_name: name of LLM to evaluate
      generation_config: generation config for LLM
      model_url: model url of LLM
      cache: cache of LLM responses
      cache_file: cache file of LLM responses
      vanilla_prompt: vanilla system prompt for multiple choice evaluation
      vanilla_isambig_prompt: vanilla system prompt for ambiguity identification
        evaluation
      vanilla_fullinfo_prompt: vanilla system prompt for fully specified
        evaluation
      cot_prompt: CoT system prompt for multiple choice evaluation
      cot_isambig_prompt: CoT system prompt for ambiguity identification
        evaluation
      cot_fullinfo_prompt: CoT system prompt for fully specified evaluation
      fs_prompt: System prompt for few-shot evaluation for multiple choice
        evaluation
      fs_isambig_prompt: System prompt for few-shot evaluation for ambiguity
        identification evaluation
      fs_fullinfo_prompt: System prompt for few-shot evaluation for fully
        specified evaluation
      non_fs_request: User prompt for vanilla and CoT evaluation
      fs_request: User prompt for few-shot evaluation for multiple choice
        evaluation
      use_cot: whether to use CoT or not
      fs_samples: number of few-shot samples to use
      eval_mode: evaluation mode, one of "mc", "isambig", "fullinfo"
      system_prompt: system prompt for current evaluation mode
      request: user prompt for current evaluation mode
      batch_size: batch size for evaluation
      model_role_name: role name for the model
      parallel_model_calls: whether to make parallel calls to the model
    """

    def __init__(
        self,
        model_name: str,
        cache=None,
        cache_file=None,
        use_cot: bool = False,
        fs_samples: int = 0,
        eval_mode: str = "mc",
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            model_name,
            cache=cache,
            cache_file=cache_file,
            use_cot=use_cot,
            fs_samples=fs_samples,
            eval_mode=eval_mode,
            **kwargs,
        )

        self.vanilla_prompt = """Suppose you know the following rules about Alice:
    {rules_nl}

You trying to discern whether a statement about Alice is true given some facts. You must decide whether you have enough information to determine whether the final statement is true. You may respond with one of the following-
If you do not have enough information yet, you may ask a question about an attribute of Alice, in the form of "Question: Is Alice [attribute]?". Ask the best question that, regardless of how it is answered, provides the most information about the final statement.
Once you have enough all information necessary to determine the truth value of the statement, you can terminate with "End questioning".
Generate one of "Question: Is Alice [attribute]?" or "End questioning" and nothing else."""
        self.vanilla_isambig_prompt = """Suppose you know the following rules about Alice:
{rules_nl}

You will presented with a binary question about an attribute of Alice. Please answer it with "Yes" or "No" or "Not sure"."""
        self.vanilla_fullinfo_prompt = """Suppose you know the following rules about Alice:
{rules_nl}

You will be given a binary question about an attribute of Alice. Please answer it with "Yes" or "No"."""
        self.cot_prompt = """Suppose you know the following rules about Alice:
    {rules_nl}

You trying to discern whether a statement about Alice is true given some facts. You must decide whether you have enough information to determine whether the final statement is true. You may respond with one of the following-
If you do not have enough information yet, you may ask a question about an attribute of Alice, in the form of "Question: Is Alice [attribute]?". Ask the best question that, regardless of how it is answered, provides the most information about the final statement.
Once you have enough all information necessary to determine the truth value of the statement, you can terminate with "End questioning".
iefly, then generate one of "Question: Is Alice [attribute]?" or "End questioning"."""
        self.cot_isambig_prompt = """Suppose you know the following rules about Alice:
{rules_nl}

You will presented with a binary question about an attribute of Alice. Please answer it with "Yes" or "No" or "Not sure".
Reason step-by-step, then generate "Answer:" followed by the answer and nothing else."""
        self.cot_fullinfo_prompt = """Suppose you know the following rules about Alice:
{rules_nl}

You will be given a binary question about an attribute of Alice. Please answer it with "Yes" or "No".
Reason step-by-step, then generate "Answer:" followed by the answer and nothing else."""
        self.non_fs_request = """{known_facts}
{known_untrue_facts}
{invalid_qs}
Is Alice {goal}?"""
        self.fs_prompt = """You trying to discern whether a statement about Alice is true given some facts. You must decide whether you have enough information to determine answer the target question. You may respond with one of the following-
If you do not have enough information yet, you may ask a question about an attribute of Alice, in the form of "Question: Is Alice [attribute]?". Ask the best question that, regardless of how it is answered, provides the most information about the final statement.
Once you have enough all information necessary to determine determine the truth value of the statement, you can terminate with "End questioning".
Generate one of "Question: Is Alice [attribute]?" or "End questioning" and nothing else."""
        self.fs_request = """Rules:
{rules_nl}

Facts:
{known_facts}
{known_untrue_facts}
{invalid_qs}

Target Question:
Is Alice {goal}?"""
        self.fs_isambig_prompt = """You will presented with a binary question about an attribute of Alice. Please answer it with "Yes" or "No" or "Not sure".
Generate "Answer:" followed by the answer and nothing else."""
        self.fs_fullinfo_prompt = """You will be given a binary question about an attribute of Alice. Please answer it with "Yes" or "No".
Generate "Answer:" followed by the answer and nothing else."""

        if self.use_cot:
            if self.eval_mode == "mc":
                self.system_prompt = self.cot_prompt
            elif self.eval_mode == "isambig":
                self.system_prompt = self.cot_isambig_prompt
            elif self.eval_mode == "fullinfo":
                self.system_prompt = self.cot_fullinfo_prompt
            self.request = self.non_fs_request
        elif self.fs_samples > 0:
            if self.eval_mode == "mc":
                self.system_prompt = self.fs_prompt
            elif self.eval_mode == "isambig":
                self.system_prompt = self.fs_isambig_prompt
            elif self.eval_mode == "fullinfo":
                self.system_prompt = self.fs_fullinfo_prompt
            self.request = self.fs_request
        else:
            if self.eval_mode == "mc":
                self.system_prompt = self.vanilla_prompt
            elif self.eval_mode == "isambig":
                self.system_prompt = self.vanilla_isambig_prompt
            elif self.eval_mode == "fullinfo":
                self.system_prompt = self.vanilla_fullinfo_prompt
            self.request = self.non_fs_request

        self.batch_size = batch_size

    def evaluate_batch(
        self,
        batch_requests,
        batch_system_prompts,
        model_name,
        model_url,
        batch_gt_queries,
        cache=None,
        cache_file=None,
        fs_turns=None,
    ):
        """Evaluates a batch of requests.

        Args:
          batch_requests: The batch of requests.
          batch_system_prompts: The batch of system prompts.
          model_name: The name of the model to evaluate.
          model_url: The url of the model to evaluate.
          batch_gt_queries: The batch of ground truth responses.
          cache: The cache of LLM responses.
          cache_file: The cache file of LLM responses.
          fs_turns: The fewshot turns.

        Returns:
          The batch of LM responses, LM conversations, and whether they are
          correctness.
        """
        batch_prompts = []
        for request, system_prompt in zip(batch_requests, batch_system_prompts):
            assist_prompt = []
            if self.fs_samples > 0:
                assist_prompt.extend(fs_turns)
            if system_prompt is None:
                assist_prompt.append({"role": "user", "content": request})
            else:
                assist_prompt.extend(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request},
                    ]
                )
            batch_prompts.append(assist_prompt)
        batch_responses, cost = cached_generate(
            batch_prompts,
            model_name,
            model_url,
            cache=cache,
            cache_file=cache_file,
            generation_config=self.generation_config,
            parallel_model_calls=self.parallel_model_calls,
        )

        batch_convos = []
        batch_correct = []
        if not batch_responses:
            return

        for i, (request, response) in enumerate(zip(batch_requests, batch_responses)):
            conversation = []
            conversation.append({"role": "user", "text": request})  # user: ambig q
            conversation.append(
                {
                    "role": self.model_role_name,
                    "text": response,
                }
            )  # agent: clarifying q
            if "End questioning" in response:
                response = "End questioning"
            else:
                n_answer_loops = 0
                while (
                    self.eval_mode == "mc"
                    and not re.findall(r"Is Alice \[?([ \w-]+)\]?\?", response)
                    and "end questioning" not in response.lower()
                ) or (
                    (self.eval_mode == "isambig" or self.eval_mode == "fullinfo")
                    and not re.findall(r"(yes|not sure|no)", response.lower())
                ):
                    if n_answer_loops > 5:
                        print("Too many loops")
                        break
                    batch_prompts[i].append(
                        {"role": self.model_role_name, "content": response}
                    )
                    if self.eval_mode == "mc":
                        batch_prompts[i].append(
                            {
                                "role": "system",
                                "content": (
                                    "Could not parse response. Generate exactly one of"
                                    ' "Question: Is Alice [attribute]?" or "End questioning"'
                                    " and nothing else."
                                ),
                            }
                        )
                    elif self.eval_mode == "isambig":
                        batch_prompts[i].append(
                            {
                                "role": "system",
                                "content": (
                                    'Wrong format. Please answer either "Answer: Yes" or'
                                    ' "Answer: No" or "Answer: Not sure" and nothing else.'
                                ),
                            }
                        )
                    elif self.eval_mode == "fullinfo":
                        batch_prompts[i].append(
                            {
                                "role": "system",
                                "content": (
                                    'Wrong format. Please answer either "Answer: Yes" or'
                                    ' "Answer: No" and nothing else.'
                                ),
                            }
                        )
                    batch_response, cost = cached_generate(
                        [batch_prompts[i]],
                        model_name,
                        model_url,
                        cache=cache,
                        cache_file=cache_file,
                        generation_config=self.generation_config,
                        parallel_model_calls=self.parallel_model_calls,
                    )
                    response = batch_response[0]
                    conversation.append(
                        {"role": self.model_role_name, "text": response}
                    )
                    n_answer_loops += 1
                response = response.split("Question:")[-1].strip()
                # regex matching
                if self.eval_mode == "mc":
                    if not (
                        re.findall(r"Is Alice \[?([ \w-]+)\]?\?", response)
                        or "end questioning" in response.lower()
                    ):
                        print("Could not parse response: {response}")
                        response = "None"
                    else:
                        if "end questioning" in response.lower():
                            response = "End questioning"
                        else:
                            response = re.findall(
                                r"Is Alice \[?([ \w-]+)\]?\?", response
                            )[0]
                else:
                    if not re.findall(r"(yes|not sure|no)", response.lower()):
                        print(
                            "No/bad number found in response:"
                            f" {json.dumps(batch_prompts[i])}"
                        )
                        response = "None"
                    else:
                        orig_response = response
                        first_line = orig_response.split("\n")[0]
                        processed_response = (
                            first_line + (response.lower().split("answer")[-1])
                        )
                        all_yes = "yes" in processed_response
                        all_no = "no" in processed_response
                        all_not_sure = "not sure" in processed_response
                        if all_yes and not all_no and not all_not_sure:
                            response = "yes"
                        elif all_no and not all_not_sure:
                            response = "no"
                        elif all_not_sure:
                            response = "not sure"
                        else:
                            print(
                                f"No answer found in response: {orig_response} \n for prompt:"
                                f" {json.dumps(batch_prompts[i])}"
                            )
            batch_responses[i] = response
            batch_convos.append(conversation)
            batch_correct.append(response.strip() in batch_gt_queries[i])
        return batch_convos, batch_responses, batch_correct, cost

    def parse_rules(self, rules):
        """Parses a list of SimpleLogic rules into a natural language format.

        Args:
          rules: A list of rules, where each rule is a string of the form "attribute
            verb proposition" or "not attribute verb proposition".

        Returns:
          A string of natural language rules, where each rule is of the form
          "If Alice is attribute, then Alice is proposition".
        """
        rules_nl = []
        for rule in rules:
            negated_words = [
                word.split("not ")[-1] for word in rule if word.startswith("not ")
            ]
            positive_words = [word for word in rule if not word.startswith("not ")]
            assert len(positive_words) == 1
            premises = " and ".join(negated_words)
            conclusion_word = positive_words[0]
            rules_nl.append(f"If Alice is {premises}, then Alice is {conclusion_word}.")
        return "\n".join(sorted(rules_nl))

    def make_batches(self, data, batch_size=None):
        """Make data batches for Logic-Q.

        Args:
          data: The data to make batches from.
          batch_size: The batch size to use.

        Returns:
          The batch of requests, system prompts, ground truth queries, and batch
        ids.
        """
        if batch_size is None:
            batch_size = self.batch_size
        batch_ids = [[]]
        batch_system_prompts = [[]]
        batch_requests = [[]]
        batch_gt_queries = [[]]
        for d, (_, datum) in enumerate(data.iterrows()):
            rules_nl = self.parse_rules(datum["rules"])

            known_facts = sorted([f"Alice is {attr}." for attr in datum["known_facts"]])
            known_untrue_facts = sorted(
                [f"Alice is not {attr}." for attr in datum["known_untrue_facts"]]
            )
            if self.eval_mode == "mc":
                invalid_qs = sorted(
                    [
                        f"You may not ask if Alice is {attr}."
                        for attr in datum["cannot_ask_facts"]
                    ]
                )
                invalid_qs = "\n".join(sorted(set(invalid_qs)))
                assert not set(known_facts).intersection(set(known_untrue_facts))
                known_facts = "\n".join(known_facts)
                known_untrue_facts = "\n".join(sorted(set(known_untrue_facts)))

                if len(batch_requests[-1]) >= batch_size:
                    batch_requests.append([])
                    batch_system_prompts.append([])
                    batch_gt_queries.append([])
                    batch_ids.append([])

                if self.fs_samples == 0:
                    batch_system_prompts[-1].append(
                        self.system_prompt.format(rules_nl=rules_nl)
                    )
                    batch_requests[-1].append(
                        self.request.format(
                            known_facts=known_facts,
                            known_untrue_facts=known_untrue_facts,
                            invalid_qs=invalid_qs,
                            goal=datum["goal"],
                        )
                    )
                else:
                    batch_system_prompts[-1].append(None)
                    batch_requests[-1].append(
                        self.request.format(
                            rules_nl=rules_nl,
                            known_facts=known_facts,
                            known_untrue_facts=known_untrue_facts,
                            invalid_qs=invalid_qs,
                            goal=datum["goal"],
                        )
                    )

                batch_ids[-1].append(d)
                batch_gt_queries[-1].append(datum["gt_qs"])
            else:
                original_known_facts = known_facts
                original_known_untrue_facts = known_untrue_facts
                for gt_q in datum["gt_q_to_true_derivation"]:
                    for is_true in [True, False, None]:
                        known_facts = copy.deepcopy(original_known_facts)
                        known_untrue_facts = copy.deepcopy(original_known_untrue_facts)
                        if is_true is None:
                            if self.eval_mode != "isambig":
                                continue
                            else:
                                goal_is_true = "not sure"
                        else:
                            if is_true:
                                known_facts.append(f"Alice is {gt_q}.")
                                implications = [
                                    prop[1]
                                    for prop in datum["gt_q_to_true_derivation"][gt_q]
                                ]
                            else:
                                known_untrue_facts.append(f"Alice is not {gt_q}.")
                                implications = [
                                    prop[1]
                                    for prop in datum["gt_q_to_false_derivation"][gt_q]
                                ]
                            assert (
                                datum["goal"] in implications
                                or f"not {datum['goal']}" in implications
                            )
                            goal_is_true = (
                                "yes" if datum["goal"] in implications else "no"
                            )
                        # known_facts.append(f"Alice is {datum['goal']}.")
                        # elif f"not {datum['goal']}" in implications:
                        # known_untrue_facts.append(f"Alice is not {datum['goal']}.")
                        known_facts = sorted(known_facts)
                        known_untrue_facts = sorted(known_untrue_facts)
                        assert not set(known_facts).intersection(
                            set(known_untrue_facts)
                        )
                        known_facts_nl = "\n".join(known_facts)
                        known_untrue_facts_nl = "\n".join(
                            sorted(set(known_untrue_facts))
                        )

                        if len(batch_requests[-1]) >= batch_size:
                            batch_requests.append([])
                            batch_system_prompts.append([])
                            batch_gt_queries.append([])
                            batch_ids.append([])

                        if self.fs_samples == 0:
                            batch_system_prompts[-1].append(
                                self.system_prompt.format(rules_nl=rules_nl)
                            )
                            batch_requests[-1].append(
                                self.request.format(
                                    known_facts=known_facts_nl,
                                    known_untrue_facts=known_untrue_facts_nl,
                                    invalid_qs="",
                                    goal=datum["goal"],
                                )
                            )
                        else:
                            batch_system_prompts[-1].append(None)
                            batch_requests[-1].append(
                                self.request.format(
                                    rules_nl=rules_nl,
                                    known_facts=known_facts_nl,
                                    known_untrue_facts=known_untrue_facts_nl,
                                    invalid_qs="",
                                    goal=datum["goal"],
                                )
                            )

                        batch_ids[-1].append(d)
                        batch_gt_queries[-1].append(goal_is_true)

        return batch_ids, batch_system_prompts, batch_requests, batch_gt_queries

    def make_fewshot_turns(self, fewshot_data):
        """Make few-shot turns for Logic-Q.

        Args:
          fewshot_data: The few-shot data to make few-shot turns from.

        Returns:
          The few-shot turns for the prompt.
        """

        fewshot_turns = []
        for d, (_, datum) in enumerate(fewshot_data.iterrows()):
            if d >= self.fs_samples:
                break
            rules_nl = self.parse_rules(datum["rules"])

            known_facts = [f"Alice is {attr}." for attr in datum["known_facts"]]
            known_untrue_facts = [
                f"Alice is not {attr}." for attr in datum["known_untrue_facts"]
            ]
            invalid_qs = [
                f"You may not ask if Alice is {attr}."
                for attr in datum["cannot_ask_facts"]
            ]
            assert not set(known_facts).intersection(set(known_untrue_facts))

            if self.eval_mode == "mc":
                known_facts = "\n".join(known_facts)
                known_untrue_facts = "\n".join(sorted(set(known_untrue_facts)))
                invalid_qs = "\n".join(sorted(set(invalid_qs)))
                random_gt_attr = random.choice(datum["gt_qs"])
                fewshot_turns.append(
                    [
                        {
                            "role": "user",
                            "content": self.request.format(
                                rules_nl=rules_nl,
                                known_facts=known_facts,
                                known_untrue_facts=known_untrue_facts,
                                invalid_qs=invalid_qs,
                                goal=datum["goal"],
                            ),
                        },
                        {
                            "role": self.model_role_name,
                            "content": f"Question: Is Alice {random_gt_attr}?",
                        },
                    ]
                )
            else:
                gt_q = random.choice(list(datum["gt_q_to_true_derivation"].keys()))
                if self.eval_mode == "isambig":
                    is_true = [True, False, None][d % 3]
                else:
                    is_true = [True, False][d % 2]
                if is_true is None:
                    goal_is_true = "not sure"
                else:
                    if is_true:
                        known_facts.append(f"Alice is {gt_q}.")
                        implications = [
                            prop[1] for prop in datum["gt_q_to_true_derivation"][gt_q]
                        ]
                    else:
                        known_untrue_facts.append(f"Alice is not {gt_q}.")
                        implications = [
                            prop[1] for prop in datum["gt_q_to_false_derivation"][gt_q]
                        ]
                    assert (
                        datum["goal"] in implications
                        or f"not {datum['goal']}" in implications
                    )
                    goal_is_true = "yes" if datum["goal"] in implications else "no"
                known_facts = sorted(known_facts)
                known_untrue_facts = sorted(known_untrue_facts)
                assert not set(known_facts).intersection(set(known_untrue_facts))
                known_facts_nl = "\n".join(known_facts)
                known_untrue_facts_nl = "\n".join(sorted(set(known_untrue_facts)))

                fewshot_turns.append(
                    [
                        {
                            "role": "user",
                            "content": self.request.format(
                                rules_nl=rules_nl,
                                known_facts=known_facts_nl,
                                known_untrue_facts=known_untrue_facts_nl,
                                invalid_qs="",
                                goal=datum["goal"],
                            ),
                        },
                        {
                            "role": self.model_role_name,
                            "content": f"Answer: {goal_is_true}",
                        },
                    ]
                )
        # shuffle the ordering of the few-shot turns
        # (move user, assistant pairs together)
        random.shuffle(fewshot_turns)
        # flatten the list of lists
        fewshot_prefix = []
        for sublist in fewshot_turns:
            for turn in sublist:
                fewshot_prefix.append(turn)
        fewshot_prefix = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            *fewshot_prefix,
        ]
        return fewshot_prefix

    def evaluate_data(self, data: pd.DataFrame, prompt_data: pd.DataFrame):
        """Evaluates LLMs on Logic-Q data.

        Args:
          data: The data to evaluate.
          prompt_data: The prompt data to evaluate.

        Returns:
          The evaluation results.
        """
        for k in [
            "known_facts",
            "known_untrue_facts",
            "cannot_ask_facts",
            "rules",
            "all_qs",
            "all_valid_qs",
            "gt_qs",
            "gt_q_to_true_derivation",
            "gt_q_to_false_derivation",
        ]:
            data[k] = data[k].apply(ast.literal_eval)
            try:
                prompt_data[k] = prompt_data[k].apply(ast.literal_eval)
            except ValueError:
                continue

        results = pd.DataFrame(
            columns=[
                "correct",
                "max_depth",
                "min_num_rules_needed",
                "num_constraints",
                "num_vars",
                "pred_q",
                "gt_qs",
                "all_qs",
                "all_valid_qs",
                "gt_q_to_true_derivation",
                "gt_q_to_false_derivation",
                "conversation",
            ]
        )
        total_cost = 0

        fs_turns = self.make_fewshot_turns(prompt_data)
        batch_ids, batch_system_prompts, batch_requests, batch_gt_queries = (
            self.make_batches(data)
        )
        pbar = tqdm.tqdm(
            zip(batch_ids, batch_system_prompts, batch_requests, batch_gt_queries),
            total=len(batch_ids),
        )
        for batch_id, batch_system_prompt, batch_request, batch_gt_query in pbar:
            batch_conversation, batch_generated_q, batch_correct, cost = (
                self.evaluate_batch(
                    batch_request,
                    batch_system_prompt,
                    model_name=self.model_name,
                    model_url=self.model_url,
                    batch_gt_queries=batch_gt_query,
                    cache=self.cache,
                    cache_file=self.cache_file,
                    fs_turns=fs_turns,
                )
            )
            total_cost += cost
            for i, item_id in enumerate(batch_id):
                datum = data.iloc[item_id]

                results.loc[len(results)] = [
                    batch_correct[i],
                    datum["max_depth"],
                    datum["min_num_rules_needed"],
                    datum["num_constraints"],
                    datum["num_vars"],
                    batch_generated_q[i],
                    batch_gt_query[i],
                    datum["all_qs"],
                    datum["all_valid_qs"],
                    datum["gt_q_to_true_derivation"],
                    datum["gt_q_to_false_derivation"],
                    batch_conversation[i],
                ]
            pbar.set_description(f"Accuracy: {sum(results['correct']) / len(results)}")

        print(f"Final accuracy: {sum(results['correct']) / len(results)}")
        print(
            "Accuracy by depth:",
            results.groupby("max_depth").agg({"correct": "mean"}),
        )
        print(f"Total cost: {total_cost}")
        return results
