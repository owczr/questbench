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

"""Evaluate LLMs on Planning-Q."""

import ast
import glob
import json
import os
import random
import re

from evaluators.evaluator import Evaluator
from model_utils import cached_generate
import pandas as pd
from Planning.backtrace_utils import make_all_consistent_states
from Planning.make_heldout_states import make_constraints
from Planning.make_heldout_states import make_impossible_and_contradicting_facts
from pyperplan import grounding
from pyperplan.pddl.parser import Parser
from pyperplan.task import Task
import tqdm


class PlanningEvaluator(Evaluator):
    """Evaluator for LLMs on Planning-Q.

    Attributes:
      model_name: name of LLM to evaluate
      generation_config: generation config for LLM
      model_url: model url of LLM
      cache: cache of LLM responses
      cache_file: cache file of LLM responses
      num_objs_to_problem_spec: dictionary mapping number of objects to problem
        specification
      domain_file: domain file for blocks problem
      domain_pddl: domain pddl for blocks problem
      op_str_to_operator: dictionary mapping operator string to operator object
      init_conditions_cache: dictionary mapping set of initial conditions to set
        of all consistent states with that initial conditions
      init_conditions_cache_filename: filename of init_conditions_cache
      init_conditions_cache_file: file object for init_conditions_cache
      vanilla_prompt: vanilla system prompt for multiple choice evaluation
      vanilla_isambig_prompt: vanilla system prompt for ambiguity identification
        evaluation
      vanilla_fullinfo_prompt: vanilla system prompt for fully specified
        evaluation
      vanilla_phys_constraints_prompt: vanilla system prompt for physical
        constraints evaluation
      cot_prompt: CoT system prompt for multiple choice evaluation
      fs_prompt: System prompt for few-shot evaluation for multiple choice
        evaluation
      cot_isambig_prompt: CoT system prompt for ambiguity identification
        evaluation
      cot_fullinfo_prompt: CoT system prompt for fully specified evaluation
      fs_isambig_prompt: System prompt for few-shot evaluation for ambiguity
        identification evaluation
      fs_fullinfo_prompt: System prompt for few-shot evaluation for fully
        specified evaluation
      non_fs_request_mc: User prompt for vanilla and CoT evaluation for multiple
        choice evaluation
      fs_request_mc: User prompt for few-shot evaluation for multiple choice
        evaluation
      non_fs_request_isambig: User prompt for vanilla and CoT evaluation
      fs_request_isambig: User prompt for few-shot evaluation for ambiguity
        identification evaluation
      eval_mode: evaluation mode, one of "mc", "isambig", "fullinfo"
      use_cot: whether to use CoT or not
      use_phys_constraints: whether to use physical constraints or not
      fs_samples: number of few-shot samples to use
      request: user prompt for current evaluation mode
      system_prompt: system prompt for current evaluation mode
      batch_size: batch size for evaluation
      model_role_name: role name for the model
      parallel_model_calls: whether to make parallel calls to the model
    """

    def __init__(
        self,
        model_name: str,
        domain_file: str,
        task_file_pattern: str,
        cache=None,
        cache_file=None,
        use_cot: bool = False,
        use_phys_constraints: bool = False,
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
        self.num_objs_to_problem_spec = {}
        self.domain_file = domain_file
        with open(self.domain_file) as f:
            self.domain_pddl = f.read()

        for task_file in glob.glob(task_file_pattern):
            problem = self._parse(self.domain_file, task_file)
            task = self._ground(problem)  # specific instance
            if len(problem.objects) not in self.num_objs_to_problem_spec:
                self.num_objs_to_problem_spec[len(problem.objects)] = {
                    "facts": set(task.facts)
                    - {f"(on {chr(i + 97)} {chr(i + 97)})" for i in range(26)},
                    "operators": task.operators,
                    "objects": problem.objects,
                }
            if (
                4 in self.num_objs_to_problem_spec
                and 5 in self.num_objs_to_problem_spec
                and 6 in self.num_objs_to_problem_spec
                and 7 in self.num_objs_to_problem_spec
            ):
                break
        self.op_str_to_operator = {
            num_objs: {
                repr(op): op
                for op in self.num_objs_to_problem_spec[num_objs]["operators"]
            }
            for num_objs in self.num_objs_to_problem_spec
        }
        self.init_conditions_cache = {}
        os.makedirs("cache", exist_ok=True)
        self.init_conditions_cache_filename = "cache/init_conditions_cache.jsonl"
        if os.path.exists(self.init_conditions_cache_filename):
            with open(self.init_conditions_cache_filename, "r") as f:
                for line in f:
                    try:
                        line = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.init_conditions_cache[frozenset(line["conditions"])] = set()
                    for state in line["all_states"]:
                        self.init_conditions_cache[frozenset(line["conditions"])].add(
                            frozenset(state)
                        )
        else:
            # open and close
            wf = open(self.init_conditions_cache_filename, "w")
            wf.close()
        self.init_conditions_cache_file = open(self.init_conditions_cache_filename, "a")

        self.vanilla_prompt = """You will be given a planning problem in the domain defined by the following PDDL:

{domain_pddl}

The current objects are present in the problem:

{problem_objects}

You will be given a set of conditions true in your initial state and a set of goal conditions, and will need to construct a plan from your current state to the goal state.
Some details of your initial state may be missing. You must decide whether you have enough information to disambiguate a plan to the final state. If not, you must decide what information is necessary to construct a fully unambiguous plan from your initial state to the goal state.
You will be presented with a set of multiple-choice options for questions you may ask, and you must answer with one of the options.
Please generate the number of the option and nothing else."""
        self.vanilla_isambig_prompt = """You will be given a planning problem in the domain defined by the following PDDL:

{domain_pddl}

The current objects are present in the problem:

{problem_objects}

You will be given a set of conditions true in your initial state and a set of goal conditions, and will need to construct a plan from your current state to the goal state.
Please answer with "Answer:" followed by the plan in the form of a comma-separated list of ([action] [args]) (e.g. (pick-up a), (stack a b), ...), or "Not sure" if you are unsure what the plan should be."""
        self.vanilla_fullinfo_prompt = """You will be given a planning problem in the domain defined by the following PDDL:

{domain_pddl}

The current objects are present in the problem:

{problem_objects}

You will be given a set of conditions true in your initial state and a set of goal conditions, and will need to construct a plan from your current state to the goal state.
Please answer with "Answer:" followed by the plan in the form of a comma-separated list of ([action] [args]) (e.g. (pick-up a), (stack a b), ...)."""
        self.vanilla_phys_constraints_prompt = """You will be given a planning problem in the domain defined by the following PDDL:

{domain_pddl}

There are also the following physical constraints that govern what states are valid. For a state to be valid, all of the below constraints must be true:
1. All blocks must be held, be on the table, and or be on exactly 1 block. In other words, for each block ?x, exactly one of the following must be true: (ontable ?x), (holding ?x), (on ?x ?y1), (on ?x ?y2), ..., (on ?x ?yn), where ?y1, ?y2, ..., ?yn are all other blocks.
2. All blocks must be held, have nothing on top of them (clear), or have exactly 1 block on top of them. In other words, for each block ?x, exactly one of the following must be true: (clear ?x), (holding ?x), (on ?y1 ?x), (on ?y2 ?x), ..., (on ?yn ?x), where ?y1, ?y2, ..., ?yn are all other blocks.
3. The hand must be either empty or holding exactly 1 block. In other words, exactly one of the following must be true: (handempty), (holding ?x1), (holding ?x2), ..., (holding ?xn), where ?x1, ?x2, ..., ?xn are all other blocks.
4. Blocks cannot be on blocks below them. In other words, if we let (on ?x ?y) relations define directional edges from ?x to ?y, and there cannot be cycles in that graph.

The current objects are present in the problem:

{problem_objects}

You will be given a set of conditions true in your initial state and a set of goal conditions, and will need to construct a plan from your current state to the goal state.
Some details of your initial state may be missing. You must decide whether you have enough information to disambiguate a plan to the final state. If not, you must decide what information is necessary to construct a fully unambiguous plan from your initial state to the goal state.
You will be presented with a set of multiple-choice options for questions you may ask, and you must answer with one of the options.
Please generate the number of the option and nothing else."""
        self.cot_prompt = """You will be given a planning problem in the domain defined by the following PDDL:

{domain_pddl}

The current objects are present in the problem:

{problem_objects}

You will be given a set of conditions true in your initial state and a set of goal conditions, and will need to construct a plan from your current state to the goal state.
Some details of your initial state may be missing. You must decide whether you have enough information to disambiguate a plan to the final state. If not, you must decide what information is necessary to construct a fully unambiguous plan from your initial state to the goal state.
You will be presented with a set of multiple-choice options for questions you may ask, and you must answer with one of the options.
Please reason step-by-step, then generate "Answer:" followed by the number of the option and nothing else."""
        self.fs_prompt = """You will be given a planning problem in the domain defined by the following PDDL:

{domain_pddl}

You will be given a set of conditions true in your initial state and a set of goal conditions, and will need to construct a plan from your current state to the goal state.
Some details of your initial state may be missing. You must decide whether you have enough information to disambiguate a plan to the final state. If not, you must decide what information is necessary to construct a fully unambiguous plan from your initial state to the goal state.
You will be presented with a set of multiple-choice options for questions you may ask, and you must answer with one of the options.
Please generate the number of the option and nothing else."""
        self.cot_isambig_prompt = """You will be given a planning problem in the domain defined by the following PDDL:

{domain_pddl}

The current objects are present in the problem:

{problem_objects}

You will be given a set of conditions true in your initial state and a set of goal conditions, and will need to construct a plan from your current state to the goal state.
Please reason step-by-step, then generate "Answer:" followed by the plan in the form of a comma-separated list of ([action] [args]) (e.g. (pick-up a), (stack a b), ...), or "Not sure" if you are unsure what the plan should be."""
        self.cot_fullinfo_prompt = """You will be given a planning problem in the domain defined by the following PDDL:

{domain_pddl}

The current objects are present in the problem:

{problem_objects}

You will be given a set of conditions true in your initial state and a set of goal conditions, and will need to construct a plan from your current state to the goal state.
Please reason step-by-step, then generate "Answer:" followed by the plan in the form of a comma-separated list of ([action] [args]) (e.g. (pick-up a), (stack a b), ...)."""
        self.fs_isambig_prompt = """You will be given a planning problem in the domain defined by the following PDDL:

{domain_pddl}

You will be given a set of conditions true in your initial state and a set of goal conditions, and will need to construct a plan from your current state to the goal state.
Please answer with "Answer:" followed by the plan in the form of a comma-separated list of ([action] [args]) (e.g. (pick-up a), (stack a b), ...), or "Not sure" if you are unsure what the plan should be."""
        self.fs_fullinfo_prompt = """You will be given a planning problem in the domain defined by the following PDDL:

{domain_pddl}

You will be given a set of conditions true in your initial state and a set of goal conditions, and will need to construct a plan from your current state to the goal state.
Please answer with "Answer:" followed by the plan in the form of a comma-separated list of ([action] [args]) (e.g. (pick-up a), (stack a b), ...)."""
        self.non_fs_request_mc = """Known facts about current state:
{conditions}

Goal state:
{goals}

Possible questions:
{possible_questions_nl}"""
        self.fs_request_mc = """Objects in current state:
{problem_objects}

Known facts about current state:
{conditions}

Goal state:
{goals}

Possible questions:
{possible_questions_nl}"""
        self.non_fs_request_isambig = """Known facts about current state:
{conditions}

Goal state:
{goals}"""
        self.fs_request_isambig = """Objects in current state:
{problem_objects}

Known facts about current state:
{conditions}

Goal state:
{goals}"""
        self.use_phys_constraints = use_phys_constraints

        if self.use_cot:
            if self.eval_mode == "mc":
                self.system_prompt = self.cot_prompt
                self.request = self.non_fs_request_mc
            elif self.eval_mode == "isambig":
                self.system_prompt = self.cot_isambig_prompt
                self.request = self.non_fs_request_isambig
            elif self.eval_mode == "fullinfo":
                self.system_prompt = self.cot_fullinfo_prompt
                self.request = self.non_fs_request_isambig
        elif self.fs_samples > 0:
            if self.eval_mode == "mc":
                self.system_prompt = self.fs_prompt
                self.request = self.fs_request_mc
            elif self.eval_mode == "isambig":
                self.system_prompt = self.fs_isambig_prompt
                self.request = self.fs_request_isambig
            elif self.eval_mode == "fullinfo":
                self.system_prompt = self.fs_fullinfo_prompt
                self.request = self.fs_request_isambig
        else:
            if not self.use_phys_constraints:
                if self.eval_mode == "mc":
                    self.system_prompt = self.vanilla_prompt
                    self.request = self.non_fs_request_mc
                elif self.eval_mode == "isambig":
                    self.system_prompt = self.vanilla_isambig_prompt
                    self.request = self.non_fs_request_isambig
                elif self.eval_mode == "fullinfo":
                    self.system_prompt = self.vanilla_fullinfo_prompt
                    self.request = self.non_fs_request_isambig
            else:
                self.system_prompt = self.vanilla_phys_constraints_prompt

        self.batch_size = batch_size

    def _parse(self, domain_file, problem_file):
        # Parsing
        parser = Parser(domain_file, problem_file)
        domain = parser.parse_domain()
        problem = parser.parse_problem(domain)
        return problem

    def _ground(
        self,
        problem,
        remove_statics_from_initial_state=True,
        remove_irrelevant_operators=True,
    ):
        task = grounding.ground(
            problem, remove_statics_from_initial_state, remove_irrelevant_operators
        )
        return task

    def make_ops_string(self, plans: str, num_objs: int):
        """Parse plan string into a list of operators.

        Args:
          plans: The plan as a string.
          num_objs: The number of objects in the problem.

        Returns:
          The plan as a list of operators.
        """
        # Regular expression pattern to match <Op ([action] [args])>
        pattern = r"(<Op \([a-z-| ]*\)>)"

        # Function to replace each match with the same string enclosed in single
        # quotes
        def replace_op(match):
            return f"'{match.group(0)}'"

        # Use re.sub to replace all matches in the input string
        plans = ast.literal_eval(re.sub(pattern, replace_op, plans))

        plans_parsed = {}
        for plan in plans:
            plan_parsed = tuple([self.op_str_to_operator[num_objs][op] for op in plan])
            if isinstance(plans, set):
                plans_parsed[plan_parsed] = []
            else:
                plans_parsed[plan_parsed] = plans[plan]

        return plans_parsed

    def make_batches(self, data, batch_size=None):
        """Make data batches for Planning-Q.

        Args:
          data: The data to make batches from.
          batch_size: The batch size to use.

        Returns:
          The batch of requests, system prompts, ground truth queries, possible
          facts, and tasks.
        """
        if batch_size is None:
            batch_size = self.batch_size
        batch_ids = [[]]
        batch_system_prompts = [[]]
        batch_requests = [[]]
        batch_gt_queries = [[]]
        batch_possible_facts = [[]]
        batch_tasks = [[]]
        for d, (_, datum) in enumerate(data.iterrows()):
            if d % 100 == 0:
                print(f"{d} / {len(data)}")
            possible_facts = sorted([fact for fact in datum["all_qs"]])
            possible_questions = {
                fact: (
                    f"{i}. Is {fact} true?"
                    if fact != "No questions needed."
                    else f"{i}. {fact}"
                )
                for i, fact in enumerate(possible_facts)
            }
            possible_questions_nl = "\n".join(
                sorted(
                    possible_questions.values(),
                    key=lambda x: int(x.split(". ")[0]),
                )
            )

            gt_queries = set()
            for gt_attr in datum["gt_qs"]:
                gt_queries.add(possible_questions[gt_attr].split(".")[0])

            goals = "\n".join(sorted(list(datum["goals"])))
            objects = sorted(
                list(self.num_objs_to_problem_spec[datum["num_vars"]]["objects"])
            )
            facts = self.num_objs_to_problem_spec[datum["num_vars"]]["facts"]
            operators = self.num_objs_to_problem_spec[datum["num_vars"]]["operators"]
            problem_spec = self.num_objs_to_problem_spec[datum["num_vars"]]
            # get closure of conditions_set
            (impossible_facts, contradicting_fact_pairs) = (
                make_impossible_and_contradicting_facts("blocks", problem_spec)
            )
            constraints = make_constraints(problem_spec)
            if self.eval_mode == "mc":
                conditions = "\n".join(sorted(list(datum["conditions"])))

                if len(batch_requests[-1]) >= batch_size:
                    batch_requests.append([])
                    batch_system_prompts.append([])
                    batch_gt_queries.append([])
                    batch_possible_facts.append([])
                    batch_ids.append([])
                    batch_tasks.append([])

                if self.fs_samples == 0:
                    batch_system_prompts[-1].append(
                        self.system_prompt.format(
                            domain_pddl=self.domain_pddl, problem_objects=objects
                        )
                    )
                    batch_requests[-1].append(
                        self.request.format(
                            conditions=conditions,
                            goals=goals,
                            possible_questions_nl=possible_questions_nl,
                        )
                    )
                else:
                    batch_system_prompts[-1].append(None)
                    batch_requests[-1].append(
                        self.request.format(
                            problem_objects=objects,
                            conditions=conditions,
                            goals=goals,
                            possible_questions_nl=possible_questions_nl,
                        )
                    )

                batch_ids[-1].append(d)
                batch_gt_queries[-1].append(datum["gt_qs"])
                batch_possible_facts[-1].append(possible_facts)
                batch_tasks[-1].append(
                    [
                        Task(
                            "blocks",
                            facts,
                            datum["conditions"],
                            datum["goals"],
                            operators,
                        )
                    ]
                )
            else:
                for gt_plan in datum["plan_to_gt_q"]:
                    is_trues = [True]
                    if datum["plan_to_gt_q"][gt_plan] == "No questions needed.":
                        datum["plan_to_gt_q"][gt_plan] = [None]
                    elif self.eval_mode == "isambig":
                        # is ambig and not "no questions needed"
                        is_trues = [True, None]
                    for gt_q in datum["plan_to_gt_q"][gt_plan]:
                        for is_true in is_trues:
                            if is_true is None:
                                gt_plan_nl = "Not sure"
                                conditions = "\n".join(
                                    sorted(list(datum["conditions"]))
                                )
                                conditions_set = set(datum["conditions"])
                            else:
                                gt_plan_nl = ", ".join(op.name for op in gt_plan)
                                if gt_q is None:
                                    gt_q_to_add = []
                                else:
                                    gt_q_to_add = [gt_q]
                                conditions = "\n".join(
                                    sorted(list(datum["conditions"]) + gt_q_to_add)
                                )
                                conditions_set = set(datum["conditions"]).union(
                                    set(gt_q_to_add)
                                )

                            if len(batch_requests[-1]) >= batch_size:
                                batch_requests.append([])
                                batch_system_prompts.append([])
                                batch_gt_queries.append([])
                                batch_possible_facts.append([])
                                batch_ids.append([])
                                batch_tasks.append([])

                            if self.fs_samples == 0:
                                batch_system_prompts[-1].append(
                                    self.system_prompt.format(
                                        domain_pddl=self.domain_pddl,
                                        problem_objects=objects,
                                    )
                                )
                                batch_requests[-1].append(
                                    self.request.format(
                                        conditions=conditions,
                                        goals=goals,
                                    )
                                )
                            else:
                                batch_system_prompts[-1].append(None)
                                batch_requests[-1].append(
                                    self.request.format(
                                        problem_objects=objects,
                                        conditions=conditions,
                                        goals=goals,
                                    )
                                )
                            batch_ids[-1].append(d)
                            batch_gt_queries[-1].append([gt_plan_nl])
                            batch_possible_facts[-1].append(possible_facts)

                            true_in_init_state = set()
                            false_in_init_state = set()
                            for fact in conditions_set:
                                if fact.startswith("not "):
                                    false_in_init_state.add(fact[len("not ") :])
                                else:
                                    true_in_init_state.add(fact)

                            if (
                                frozenset(conditions_set)
                                not in self.init_conditions_cache
                            ):
                                self.init_conditions_cache[
                                    frozenset(conditions_set)
                                ] = make_all_consistent_states(
                                    "blocks",
                                    facts,
                                    true_in_init_state,
                                    false_in_init_state,
                                    contradicting_fact_pairs=contradicting_fact_pairs,
                                    impossible_facts=impossible_facts,
                                    constraints=constraints,
                                )
                                all_states = []
                                for item in self.init_conditions_cache[
                                    frozenset(conditions_set)
                                ]:
                                    # make frozensets into list
                                    all_states.append(list(item))
                                line = {
                                    "conditions": list(conditions_set),
                                    "all_states": all_states,
                                }
                                self.init_conditions_cache_file.write(
                                    json.dumps(line) + "\n"
                                )
                            potential_init_states = self.init_conditions_cache[
                                frozenset(conditions_set)
                            ]

                            all_tasks = set()
                            for init_state in potential_init_states:
                                curr_task = Task(
                                    "blocks",
                                    facts,
                                    init_state,
                                    datum["goals"],
                                    operators,
                                )
                                all_tasks.add(curr_task)
                            batch_tasks[-1].append(all_tasks)

        return (
            batch_ids,
            batch_system_prompts,
            batch_requests,
            batch_gt_queries,
            batch_possible_facts,
            batch_tasks,
        )

    def evaluate_batch(
        self,
        batch_requests,
        batch_system_prompts,
        model_name,
        model_url,
        batch_gt_queries,  # possible questions
        batch_possible_facts,
        batch_tasks,
        cache=None,
        cache_file=None,
        fs_turns=None,
    ):
        """Evaluates LLMs on a batch of Planning-Q data.

        Args:
          batch_requests: The batch of requests.
          batch_system_prompts: The batch of system prompts.
          model_name: The name of the model to evaluate.
          model_url: The URL of the model to evaluate.
          batch_gt_queries: The batch of ground truth responses.
          batch_possible_facts: The batch of possible facts.
          batch_tasks: The batch of tasks.
          cache: The cache of LM responses.
          cache_file: The cache file of LM responses.
          fs_turns: The few-shot turns.

        Returns:
          The batch of LM responses, LM conversations, and whether they are
          correct.
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
        for i, (request, response, possible_facts, possible_tasks) in enumerate(
            zip(batch_requests, batch_responses, batch_possible_facts, batch_tasks)
        ):
            conversation = []
            conversation.append({"role": "user", "text": request})  # user: ambig q
            conversation.append(
                {
                    "role": self.model_role_name,
                    "text": response,
                }
            )  # agent: clarifying q
            batch_prompts[i].append({"role": self.model_role_name, "content": response})
            nloops = 0
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            # regex matching
            while self.eval_mode == "mc" and (
                not re.findall(r"\b[0-9]+\b", response)
                or int(re.findall(r"\b[0-9]+\b", response)[0]) >= len(possible_facts)
            ):
                if nloops > 5:
                    print("Too many loops")
                    break
                batch_prompts[i].append(
                    {
                        "role": "system",
                        "content": (
                            "Wrong format or option out of range"
                            f' 0-{len(possible_facts) - 1}. Output "Answer:" followed by the'
                            " number of the option and nothing else."
                        ),
                    }
                )
                conversation.append(
                    {
                        "role": "system",
                        "content": (
                            "Wrong format or option out of range"
                            f' 0-{len(possible_facts) - 1}. Output "Answer:" followed by the'
                            " number of the option and nothing else."
                        ),
                    }
                )
                generated_responses, cost = cached_generate(
                    [batch_prompts[i]],
                    model_name,
                    model_url,
                    cache=cache,
                    cache_file=cache_file,
                    generation_config=self.generation_config,
                    parallel_model_calls=self.parallel_model_calls,
                )
                response = generated_responses[0]
                batch_prompts[i].append(
                    {"role": self.model_role_name, "content": response}
                )
                conversation.append(
                    {
                        "role": self.model_role_name,
                        "text": response,
                    }
                )
                nloops += 1
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
            if self.eval_mode == "mc":
                if re.findall(r"\b[0-9]+\b", response) and int(
                    re.findall(r"\b[0-9]+\b", response)[0]
                ) < len(possible_facts):
                    response = re.findall(r"\b[0-9]+\b", response)[0]
                    # regex matching
                    pred_q = possible_facts[int(response)]
                    correct = pred_q.strip() in batch_gt_queries[i]
                else:
                    print(
                        "No/bad number found in response:"
                        f" {json.dumps(batch_prompts[i])} / {len(possible_facts)}"
                    )
                    pred_q = None
                    correct = False
            else:
                pred_q = response.split("Answer:")[-1].strip()
                if pred_q.startswith("Not sure"):
                    correct = batch_gt_queries[i][0] == "Not sure"
                elif batch_gt_queries[i][0] == "Not sure":
                    correct = pred_q == "Not sure"
                else:
                    pred_plan = pred_q.split(",")
                    all_init_states_correct = True
                    # execute plan
                    for task in possible_tasks:
                        curr_state = task.initial_state
                        init_state_correct = True
                        for op in pred_plan:
                            op = op.strip()
                            if f"<Op {op}>" not in self.op_str_to_operator[7]:
                                print(f"Unknown op: {op}")
                                init_state_correct = False
                                break
                            op = self.op_str_to_operator[7][f"<Op {op}>"]
                            if op.applicable(curr_state):
                                curr_state = op.apply(curr_state)
                            else:
                                # unexecutable plan
                                print(f"Unexecutable plan: {pred_plan}")
                                init_state_correct = False
                                break
                        init_state_correct = init_state_correct and task.goal_reached(
                            curr_state
                        )
                        all_init_states_correct = (
                            all_init_states_correct and init_state_correct
                        )
                    correct = all_init_states_correct
            batch_responses[i] = pred_q
            batch_convos.append(conversation)
            batch_correct.append(correct)

        return batch_convos, batch_responses, batch_correct, cost

    def parse_data(self, data: pd.DataFrame):
        """Parse data in order to evaluate Planning-Q.

        Args:
          data: The data to parse.

        Returns:
          The parsed data.
        """
        data["conditions"] = data["conditions"].apply(
            lambda x: frozenset(x.split("\n"))
        )
        data["goals"] = data["goals"].apply(lambda x: frozenset(x.split("\n")))
        data["gt_qs"] = data["gt_qs"].apply(eval)
        data["all_valid_qs"] = data["all_valid_qs"].apply(ast.literal_eval)
        data["all_qs"] = data["all_qs"].apply(ast.literal_eval)
        data["plan_to_gt_q"] = data.apply(
            lambda x: self.make_ops_string(x["plan_to_gt_q"], x["num_vars"]), axis=1
        )
        return data

    def make_fewshot_turns(self, fewshot_data):
        """Make fewshot turns for Planning-Q.

        Args:
          fewshot_data: The fewshot data.

        Returns:
          The fewshot turns for the prompt.
        """
        fewshot_turns = []
        for d, (_, datum) in enumerate(fewshot_data.iterrows()):
            if d >= self.fs_samples:
                break
            possible_facts = sorted([fact for fact in datum["all_qs"]])
            possible_questions = {
                fact: (
                    f"{i}. Is {fact} true?"
                    if fact != "No questions needed."
                    else f"{i}. {fact}"
                )
                for i, fact in enumerate(possible_facts)
            }
            possible_questions_nl = "\n".join(
                sorted(possible_questions.values(), key=lambda x: int(x.split(". ")[0]))
            )

            gt_queries = set()
            for gt_attr in datum["gt_qs"]:
                gt_queries.add(possible_questions[gt_attr].split(".")[0])

            conditions = "\n".join(sorted(list(datum["conditions"])))
            goals = "\n".join(sorted(list(datum["goals"])))
            objects = sorted(
                list(self.num_objs_to_problem_spec[datum["num_vars"]]["objects"])
            )

            if self.eval_mode == "mc":
                random_gt_attr = random.choice(list(datum["gt_qs"]))
                gt_attr_idx = int(possible_questions[random_gt_attr].split(".")[0])
                fewshot_turns.append(
                    [
                        {
                            "role": "user",
                            "content": self.request.format(
                                problem_objects=objects,
                                conditions=conditions,
                                goals=goals,
                                possible_questions_nl=possible_questions_nl,
                            ),
                        },
                        {
                            "role": self.model_role_name,
                            "content": f"{gt_attr_idx}",
                        },
                    ]
                )
            else:
                gt_plan = random.choice(list(datum["plan_to_gt_q"].keys()))
                is_trues = [True]
                if datum["plan_to_gt_q"][gt_plan] == "No questions needed.":
                    datum["plan_to_gt_q"][gt_plan] = [None]
                elif self.eval_mode == "isambig":
                    # is ambig and not "no questions needed"
                    is_trues = [True, None]
                gt_q = random.choice(list(datum["plan_to_gt_q"][gt_plan]))
                is_true = is_trues[len(fewshot_turns) % len(is_trues)]
                if is_true is None:
                    gt_plan_nl = "Not sure"
                    conditions = "\n".join(sorted(list(datum["conditions"])))
                else:
                    gt_plan_nl = ", ".join(op.name for op in gt_plan)
                    if gt_q is None:
                        gt_q_to_add = []
                    else:
                        gt_q_to_add = [gt_q]
                    conditions = "\n".join(
                        sorted(list(datum["conditions"]) + gt_q_to_add)
                    )
                fewshot_turns.append(
                    [
                        {
                            "role": "user",
                            "content": self.request.format(
                                problem_objects=objects,
                                conditions=conditions,
                                goals=goals,
                                possible_questions_nl=possible_questions_nl,
                            ),
                        },
                        {
                            "role": self.model_role_name,
                            "content": gt_plan_nl,
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
        if self.fs_samples == 0:
            fewshot_turns = []
        else:
            fewshot_turns = [
                {
                    "role": "system",
                    "content": self.system_prompt.format(
                        domain_pddl=self.domain_pddl,
                    ),
                },
                *fewshot_prefix,
            ]
        return fewshot_turns

    def evaluate_data(self, data: pd.DataFrame, prompt_data: pd.DataFrame):
        """Evaluates LLM on Planning-Q data.

        Args:
          data: The data to evaluate.
          prompt_data: The prompt data.

        Returns:
          The evaluation results.
        """
        data = self.parse_data(data)
        prompt_data = self.parse_data(prompt_data)

        results = pd.DataFrame(
            columns=[
                "correct",
                "depth",
                "conditions",
                "num_constraints",
                "num_vars",
                "pred_q",
                "gt_qs",
                "all_qs",
                "all_valid_qs",
                "plan_to_gt_q",
                "conversation",
            ]
        )
        total_cost = 0

        fs_turns = self.make_fewshot_turns(prompt_data)
        (
            batch_ids,
            batch_system_prompts,
            batch_requests,
            batch_gt_queries,
            batch_possible_facts,
            batch_tasks,
        ) = self.make_batches(data)
        pbar = tqdm.tqdm(
            zip(
                batch_ids,
                batch_system_prompts,
                batch_requests,
                batch_gt_queries,
                batch_possible_facts,
                batch_tasks,
            ),
            total=len(batch_ids),
        )
        for (
            batch_id,
            batch_system_prompt,
            batch_request,
            batch_gt_query,
            batch_possible_fact,
            batch_tasks,
        ) in pbar:
            (batch_conversation, batch_generated_q, batch_correct, cost) = (
                self.evaluate_batch(
                    batch_request,
                    batch_system_prompt,
                    model_name=self.model_name,
                    model_url=self.model_url,
                    batch_gt_queries=batch_gt_query,
                    batch_possible_facts=batch_possible_fact,
                    batch_tasks=batch_tasks,
                    cache=self.cache,
                    cache_file=self.cache_file,
                    fs_turns=fs_turns,
                )
            )
            total_cost += cost
            for i, item_id in enumerate(batch_id):
                datum = data.iloc[item_id]
                num_qs = len(datum["all_qs"])

                results.loc[len(results)] = [
                    batch_correct[i],
                    datum["min_depth"],
                    datum["conditions"],
                    num_qs,
                    datum["num_vars"],
                    batch_generated_q[i],
                    batch_gt_query[i],
                    batch_possible_fact[i],
                    datum["all_valid_qs"],
                    datum["plan_to_gt_q"],
                    batch_conversation[i],
                ]

            # filter none entries
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
            results_filtered.groupby("depth").agg({"correct": "mean"}),
        )
        return results
