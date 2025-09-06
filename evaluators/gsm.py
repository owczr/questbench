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

"""Evaluate LLMs on GSM-Q."""

import json
import random
import re

import datasets
import pandas as pd
import tqdm

from evaluators.evaluator import Evaluator
from model_utils import cached_generate


class GSMEvaluator(Evaluator):
    """Evaluator for LLMs on GSM-Q.

    Attributes:
      model_name: name of LM to evaluate
      generation_config: generation config for LM
      model_url: model url for LM
      cache: cache of LM responses
      cache_file: cache file of LM responses
      verbal_questions: whether to ask verbal questions or not
      assist_mc_prompt: system prompt for multiple choice evaluation
      assist_isambig_prompt: system prompt for ambiguity identification evaluation
      assist_fullinfo_prompt: system prompt for fully specified evaluation
      user_mc_prompt: user prompt for multiple choice evaluation
      user_isambig_prompt: user prompt for ambiguity identification evaluation
      user_fullinfo_prompt: user prompt for fully specified evaluation
      use_cot: whether to use CoT or not
      fs_samples: number of few-shot samples to use
      eval_mode: evaluation mode, one of "mc", "isambig", "fullinfo"
      assist_prompt: system prompt for current evaluation mode
      user_prompt: user prompt for current evaluation mode
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
        verbal_questions: bool = False,
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
        self.verbal_questions = verbal_questions
        self.assist_mc_prompt = """You are trying to solve a math problem. You must decide whether you have enough information to solve the math problem. Please respond with one of the following-
If you do not have enough information to solve the math problem, you may ask a question back to the user from a set of predefined "Possible questions". Otherwise, choose "No questions needed."
Generate the number of your choice in the form "Choice: number"
"""
        self.assist_isambig_prompt = """You are trying to answer a math question. Please answer with "Answer:" followed by the answer to the math question, or "Not sure" if you are not sure what the answer is. Only include the raw numerical answer, do not include any units or thousands separators."""
        self.assist_fullinfo_prompt = """You are trying to answer a math question. Please answer with "Answer:" followed by the answer to the math question. Only include the raw numerical answer, do not include any units or thousands separators."""
        self.user_mc_prompt = """Math problem: {request}

Possible questions:
{possible_qs}"""
        self.user_isambig_prompt = """Math problem: {request}"""
        self.user_fullinfo_prompt = """Math problem: {request}"""

        if self.eval_mode == "mc":
            self.assist_prompt = self.assist_mc_prompt
            self.user_prompt = self.user_mc_prompt
        elif self.eval_mode == "isambig":
            self.assist_prompt = self.assist_isambig_prompt
            self.user_prompt = self.user_isambig_prompt
        else:
            assert self.eval_mode == "fullinfo"
            self.assist_prompt = self.assist_fullinfo_prompt
            self.user_prompt = self.user_fullinfo_prompt

        if self.use_cot:
            self.assist_prompt += (
                """ Reason step-by-step, then generate one of the above outputs."""
            )
        else:
            self.assist_prompt += (
                """ Generate one of the above outputs and nothing else."""
            )

        self.batch_size = batch_size

        self.orig_dataset = datasets.load_dataset("qintongli/GSM-Plus")

    def generate_query(
        self,
        request,
        gt_query,
        fs_turns,
    ):
        """Query the model for a single request.

        Args:
          request: The request.
          gt_query: The ground truth response.
          fs_turns: The fewshot turns.

        Returns:
          The correctness, LM response, LM conversation.
        """
        conversation = []
        prompt = [
            {"role": "system", "content": self.assist_prompt},
            *fs_turns,
            {"role": "user", "content": request},
        ]
        responses, _ = cached_generate(
            [prompt],
            self.model_name,
            self.model_url,
            cache=self.cache,
            cache_file=self.cache_file,
            generation_config=self.generation_config,
            parallel_model_calls=False,
        )
        response = responses[0]
        conversation.append({"role": "user", "text": request})  # user: ambig q
        conversation.append(
            {
                "role": self.model_role_name,
                "text": response,
            }
        )
        prompt.append({"role": self.model_role_name, "content": response})

        if self.eval_mode == "mc":
            # check whether choice is correct
            response = response.lower().split("choice:")[-1].strip()
            n_loops = 0
            while not re.findall(r"\b[0-9]+\b", response):
                prompt.append(
                    {
                        "role": "system",
                        "content": (
                            "Wrong format or option not found. Please provide the number of"
                            ' your choice. Output "Choice: <number>" and nothing else.'
                        ),
                    }
                )
                responses, _ = cached_generate(
                    [prompt],
                    self.model_name,
                    self.model_url,
                    cache=self.cache,
                    cache_file=self.cache_file,
                    generation_config=self.generation_config,
                    parallel_model_calls=False,
                )
                response = responses[0]
                conversation.append(
                    {
                        "role": "system",
                        "content": (
                            "Wrong format or option not found. Please provide the number of"
                            ' your choice. Output "Choice: <number>" and nothing else.'
                        ),
                    }
                )
                prompt.append({"role": self.model_role_name, "content": response})
                n_loops += 1
                if n_loops > 5:
                    break
            try:
                response = re.findall(r"\b[0-9]+\b", response)[0]
                correct = int(response) == gt_query
            except ValueError:
                correct = False
        else:
            response = response.lower().split("answer:")[-1].strip()
            n_loops = 0
            while not re.findall(r"(not sure|\b[0-9]+\b)", response.lower()):
                if self.eval_mode == "fullinfo":
                    prompt_content = (
                        "Wrong format. Please provide the answer as a raw number without"
                        " any additional units, thousands separators, or text. Output"
                        ' "Answer: <number>" and nothing else.'
                    )
                else:
                    prompt_content = (
                        'Wrong format. Please answer either "Answer: <number>" or'
                        ' "Answer: Not sure" and nothing else. If you answer with a'
                        " number, it should be a raw number without any additional units,"
                        " thousands separators, or text."
                    )
                prompt.append(
                    {
                        "role": "system",
                        "content": prompt_content,
                    }
                )
                responses, _ = cached_generate(
                    prompt,
                    self.model_name,
                    self.model_url,
                    cache=self.cache,
                    cache_file=self.cache_file,
                    generation_config=self.generation_config,
                    parallel_model_calls=False,
                )
                response = responses[0]
                conversation.append(
                    {
                        "role": "system",
                        "content": prompt_content,
                    }
                )
                prompt.append({"role": self.model_role_name, "content": response})
                n_loops += 1
                if n_loops > 5:
                    break
            if "not sure" in response.lower():
                correct = gt_query == "Not sure"
            else:
                numbers = re.findall(r"\b[0-9]+\b", response)
                if not numbers:
                    correct = False
                else:
                    response = numbers[0]
                    correct = int(response) == gt_query
        return correct, response, conversation

    def parse_auto_eval(self, eval_str):
        try:
            precision = float(eval_str.split("Precision:")[1].split(",")[0])
            recall = float(eval_str.split("Recall:")[1])
            return precision, recall
        except (IndexError, ValueError):
            return False

    def generate_query_batch(self, batch_request, batch_gt_query, fs_turns):
        """Query the model in batches.

        Args:
          batch_request: The batch of requests.
          batch_gt_query: The batch of ground truth responses.
          fs_turns: The fewshot turns.

        Returns:
          batch_query: The batch of LM responses.
          batch_conversation: LM conversations.
          batch_correct: whether they are correctness.
        """
        (batch_query, batch_conversation, batch_correct) = ([], [], [])
        for _, (request, gt_query) in enumerate(zip(batch_request, batch_gt_query)):
            (correct, query, conversation) = self.generate_query(
                request, gt_query, fs_turns
            )
            batch_query.append(query)
            batch_conversation.append(conversation)
            batch_correct.append(correct)
        return batch_query, batch_conversation, batch_correct

    def make_convo_batches(self, data, batch_size=None):
        """Make data batches for GSM-Q.

        Args:
          data: The data to evaluate.
          batch_size: The batch size.

        Returns:
          The batches of data.
        """
        if batch_size is None:
            batch_size = self.batch_size
        batch_ids = [[]]
        batch_requests = [[]]
        batch_gt_answers = [[]]
        batch_gt_queries = [[]]
        for d, (_, datum) in enumerate(data.iterrows()):
            if self.eval_mode == "mc":
                request = datum["Rewritten Problem"]
                variables = json.loads(datum["Variables"])
                possible_qs = json.loads(datum["Possible Questions"])
                q_to_ask = datum["GT Question"]
                questions = []
                q_to_ask_index = -1
                for v, variable in enumerate(possible_qs):
                    if self.verbal_questions:
                        questions.append(
                            f"{v}. What is {variables[variable]} ({variable})?"
                        )
                    else:
                        questions.append(f"{v}. What is the value of {variable}?")
                    if variable == q_to_ask:
                        q_to_ask_index = v
                questions.append(f"{len(questions)}. No questions needed.")
                answer = datum["Full Answer"]

                if len(batch_requests[-1]) >= batch_size:
                    batch_ids.append([])
                    batch_requests.append([])
                    batch_gt_answers.append([])
                    batch_gt_queries.append([])

                batch_requests[-1].append(
                    self.user_prompt.format(
                        request=request,
                        possible_qs="\n".join(questions),
                    )
                )
                batch_gt_queries[-1].append(q_to_ask_index)

                batch_ids[-1].append(d)
                batch_gt_answers[-1].append(answer)
            else:
                is_trues = [True]
                if self.eval_mode == "isambig":
                    is_trues = [True, None]
                for is_true in is_trues:
                    if is_true is None:
                        request = datum["Rewritten Problem"]
                        response = "Not sure"
                    else:
                        if self.verbal_questions:
                            # get from original dataset
                            request = self.orig_dataset["test"]["question"][
                                datum["Question ID"]
                            ]
                        else:
                            request = datum["Full Problem"]
                        response = datum["Full Answer"]

                    if len(batch_requests[-1]) >= batch_size:
                        batch_ids.append([])
                        batch_requests.append([])
                        batch_gt_answers.append([])
                        batch_gt_queries.append([])

                    batch_ids[-1].append(d)
                    batch_requests[-1].append(
                        self.user_prompt.format(
                            request=request,
                        )
                    )
                    batch_gt_queries[-1].append(response)
                    batch_gt_answers[-1].append(datum["Full Answer"])

        return (
            batch_ids,
            batch_requests,
            batch_gt_answers,
            batch_gt_queries,
        )

    def make_fewshot_turns(self, fewshot_data):
        """Make fewshot turns for GSM-Q.

        Args:
          fewshot_data: The fewshot data.

        Returns:
          The fewshot turns for the prompt.
        """
        fewshot_turns = []
        for d, (_, datum) in enumerate(fewshot_data.iterrows()):
            if d >= self.fs_samples:
                break
            if self.eval_mode == "mc":
                request = datum["Rewritten Problem"]
                q_to_ask = datum["GT Question"]
                variables = json.loads(datum["Variables"])

                questions = []
                q_to_ask_index = -1
                possible_qs = json.loads(datum["Possible Questions"])
                for v, variable in enumerate(possible_qs):
                    if self.verbal_questions:
                        questions.append(
                            f"{v}. What is {variables[variable]} ({variable})?"
                        )
                    else:
                        questions.append(f"{v}. What is the value of {variable}?")
                    if variable == q_to_ask:
                        q_to_ask_index = v
                questions.append(f"{len(questions)}. No questions needed.")
                fewshot_turns.append(
                    [
                        {
                            "role": "user",
                            "content": self.user_prompt.format(
                                request=request,
                                possible_qs="\n".join(questions),
                            ),
                        },
                        {
                            "role": self.model_role_name,
                            "content": f"Choice: {q_to_ask_index}",
                        },
                    ]
                )
            else:
                is_trues = [True]
                if self.eval_mode == "isambig":
                    is_trues = [True, None]
                is_true = is_trues[len(fewshot_turns) % len(is_trues)]
                if is_true is None:
                    request = datum["Rewritten Problem"]
                    response = "Not sure"
                else:
                    if self.verbal_questions:
                        # get from original dataset
                        request = self.orig_dataset["test"][datum["Question ID"]]
                    else:
                        request = datum["Full Problem"]
                    response = datum["Full Answer"]

                fewshot_turns.append(
                    [
                        {
                            "role": "user",
                            "content": self.user_prompt.format(
                                request=request,
                            ),
                        },
                        {
                            "role": self.model_role_name,
                            "content": f"Answer: {response}",
                        },
                    ]
                )

        random.shuffle(fewshot_turns)
        # flatten the list of lists
        fewshot_prefix = []
        for sublist in fewshot_turns:
            for turn in sublist:
                fewshot_prefix.append(turn)
        return fewshot_prefix

    def evaluate_data(self, data: pd.DataFrame, prompt_data: pd.DataFrame):
        """Evaluates LLMs on GSM-Q data.

        Args:
          data: The data to evaluate.
          prompt_data: The prompt data.

        Returns:
          The evaluation results.
        """
        results = pd.DataFrame(
            columns=[
                "correct",
                "max_depth",
                "pred_answer",
                "gt_answer",
                "id",
                "request",
                "CSP",
                "num_constraints",
                "num_vars",
                "pred_q",
                "gt_qs",
                "all_qs",
                "conversation",
            ]
        )

        fs_turns = self.make_fewshot_turns(prompt_data)
        (
            batch_ids,
            batch_requests,
            batch_gt_answers,
            batch_gt_queries,
        ) = self.make_convo_batches(data)
        pbar = tqdm.tqdm(
            zip(
                batch_ids,
                batch_requests,
                batch_gt_answers,
                batch_gt_queries,
            ),
            total=len(batch_ids),
        )
        for (
            batch_id,
            batch_request,
            batch_gt_answer,
            batch_gt_query,
        ) in pbar:
            batch_query, batch_conversation, batch_correct = self.generate_query_batch(
                batch_request,
                batch_gt_query,
                fs_turns,
            )

            for i, item_id in enumerate(batch_id):
                datum = data.iloc[item_id]
                pred_answer = None
                equations = json.loads(datum["Equations"])
                variables = json.loads(datum["Variables"])

                results.loc[len(results)] = [
                    batch_correct[i],
                    datum["depth"],
                    pred_answer,
                    batch_gt_answer[i],
                    batch_id[i],
                    batch_request[i],
                    datum["CSP"],
                    len(equations),
                    len(variables),
                    batch_query[i],
                    batch_gt_query[i],
                    variables,
                    json.dumps(batch_conversation[i]),
                ]
            results_filtered = results[results["correct"].notna()]
            pbar.set_description(
                f"Accuracy: {sum(results_filtered['correct']) / len(results_filtered)}"
            )

        results_filtered = results[results["correct"].notna()]
        print(
            "Final accuracy:"
            f" {sum(results_filtered['correct']) / len(results_filtered)}"
        )
        print(
            "Accuracy by depth:",
            results.groupby("max_depth").agg({"correct": "mean"}),
        )
        return results
