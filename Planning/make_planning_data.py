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

"""Make data for Planning-Q."""

import argparse
import glob
import itertools as it
import logging
import os
import random
import re
import time

import pandas as pd
from Planning.backtrace_utils import backwards_bfs
from Planning.make_heldout_states import check_questions
from Planning.make_heldout_states import make_constraints
from Planning.make_heldout_states import make_heldout_states
from Planning.make_heldout_states import make_impossible_and_contradicting_facts
from pyperplan import grounding
from pyperplan.pddl.parser import Parser


def _parse(domain_file, problem_file):
    """Parse domain and problem.

    Args:
      domain_file: file containing domain
      problem_file: file containing problem

    Returns:
      problem: parsed problem
    """
    # Parsing
    parser = Parser(domain_file, problem_file)
    logging.info("Parsing Domain %s", domain_file)
    domain = parser.parse_domain()
    logging.info("Parsing Problem %s", problem_file)
    problem = parser.parse_problem(domain)
    logging.debug(domain)
    logging.info("%d Predicates parsed", len(getattr(domain, "predicates")))
    logging.info("%d Actions parsed", len(getattr(domain, "actions")))
    logging.info("%d Objects parsed", len(getattr(problem, "objects")))
    logging.info("%d Constants parsed", len(getattr(domain, "constants")))
    return problem


def _ground(
    problem,
    remove_statics_from_initial_state=True,
    remove_irrelevant_operators=True,
):
    """Ground problem.

    Args:
      problem: problem to ground
      remove_statics_from_initial_state: remove static facts from initial state
      remove_irrelevant_operators: remove operators which are not relevant to the
        problem

    Returns:
      task: grounded task
    """
    logging.info("Grounding start: %s", problem.name)
    task = grounding.ground(
        problem, remove_statics_from_initial_state, remove_irrelevant_operators
    )
    logging.info("Grounding end: %s", problem.name)
    logging.info("%d Variables created", len(task.facts))
    logging.info("%d Operators created", len(task.operators))
    return task


NUMBER = re.compile(r"\d+")


def main(parsed_args) -> None:
    domain_file = os.path.join(parsed_args.pddl_dir, "domain.pddl")
    with open(domain_file) as f:
        domain_pddl = f.read()

    print(domain_pddl)
    num_objs_to_problem_spec = {}
    for problem_file in glob.glob(os.path.join(parsed_args.pddl_dir, "task*.pddl")):
        problem = _parse(domain_file, problem_file)
        task = _ground(problem)  # specific instance
        if len(problem.objects) not in num_objs_to_problem_spec:
            num_objs_to_problem_spec[len(problem.objects)] = {
                "facts": task.facts,
                "operators": task.operators,
                "objects": problem.objects,
            }

    data = pd.DataFrame(
        columns=[
            "conditions",
            "goals",
            "min_depth",
            "plans",
            "gt_queries",
            "physically_valid_attrs",
            "all_attrs",
            "num_objs",
            "check_time",
        ]
    )
    all_goal_conditions = [
        frozenset({"(on b a)"}),
        frozenset({"(on b a)", "(on c b)"}),
        frozenset({"(on b a)", "(on d c)"}),
        frozenset({"(on b a)", "(ontable a)"}),
        frozenset({"(on b a)", "(ontable a)", "(on c b)"}),
        frozenset({"(on b a)", "(ontable a)", "(on d c)", "(ontable c)"}),
    ]
    num_states = 100
    num_ques_pairs = 100
    num_ques_singlepath = 20

    for num_obj in [7, 6, 5, 4]:
        print("num_obj: ", num_obj)
        for g, goal_conditions in enumerate(all_goal_conditions):
            print("goal_conditions: ", goal_conditions)

            domain = "blocks"
            problem_spec = num_objs_to_problem_spec[num_obj]
            impossible_facts, contradicting_fact_pairs = (
                make_impossible_and_contradicting_facts(domain, problem_spec)
            )
            constraints = make_constraints(problem_spec)

            conditions_to_path = backwards_bfs(
                "blocks",
                problem_spec["operators"],
                goal_conditions,
                {},
                contradicting_fact_pairs,
            )
            print("bfs done")

            (
                heldout_states_to_paths_to_heldout_fact_multiple_options,
                state_to_false_facts,
            ) = make_heldout_states(conditions_to_path)

            possible_facts = sorted(
                [fact for fact in problem_spec["facts"] if fact not in impossible_facts]
            )
            possible_questions = {
                fact: f"{i}. Is {fact} true?" for i, fact in enumerate(possible_facts)
            }
            possible_questions["No questions needed."] = (
                f"{len(possible_questions)}. No questions needed."
            )

            state_subsample = random.sample(
                list(heldout_states_to_paths_to_heldout_fact_multiple_options),
                min(
                    num_states,
                    len(heldout_states_to_paths_to_heldout_fact_multiple_options),
                ),
            )
            for s, state in enumerate(state_subsample):
                if s % 10000 == 0:
                    print(f"{num_obj} {g} State {s}/{len(state_subsample)}")
                false_facts = set(state_to_false_facts[state])
                valid_questions = set()
                heldout_q_to_path = {}
                for path in heldout_states_to_paths_to_heldout_fact_multiple_options[
                    state
                ]:
                    assert (
                        len(
                            heldout_states_to_paths_to_heldout_fact_multiple_options[
                                state
                            ][path]
                        )
                        == 1
                    )
                    valid_questions = valid_questions.union(
                        set(
                            heldout_states_to_paths_to_heldout_fact_multiple_options[
                                state
                            ][path]
                        )
                    )
                    for (
                        heldout_q
                    ) in heldout_states_to_paths_to_heldout_fact_multiple_options[
                        state
                    ][path]:
                        heldout_q_to_path[heldout_q] = path

                assert len(valid_questions) == len(
                    heldout_states_to_paths_to_heldout_fact_multiple_options[state]
                )
                goals = "\n".join(goal_conditions)
                all_q_pairs = list(it.combinations(valid_questions, 2))
                # subsample
                all_q_pairs_subsample = random.sample(
                    all_q_pairs, min(num_ques_pairs, len(all_q_pairs))
                )
                # take all pairs within valid_questions and make the other facts false
                for q, question_pairs in enumerate(all_q_pairs_subsample):
                    if q % 10000 == 0:
                        print(
                            f"{num_obj} {g} Question Pair {q} /"
                            f" {len(all_q_pairs_subsample)}"
                        )
                    other_false_facts = set(valid_questions) - set(question_pairs)
                    all_false_facts = false_facts.union(other_false_facts)
                    conditions = "\n".join(
                        [f"{true_fact}" for true_fact in state]
                        + [f"not {false_fact}" for false_fact in all_false_facts],
                    )
                    min_depth = float("inf")
                    start_time = time.time()
                    # checking validity of question set
                    (
                        distinguishing_attrs,
                        _,
                        possible_attrs,
                        attr_to_path,
                    ) = check_questions(
                        domain,
                        problem_spec,
                        state,
                        all_false_facts,
                        goal_conditions,
                        contradicting_fact_pairs,
                        impossible_facts,
                        constraints,
                        2,
                    )
                    end_time = time.time()
                    if distinguishing_attrs is None or possible_attrs is None:
                        continue
                    paths = {}
                    for distinguishing_attr in distinguishing_attrs:
                        assert len(attr_to_path[distinguishing_attr]) == 1
                        path = attr_to_path[distinguishing_attr][0]
                        if path not in paths:
                            paths[path] = []
                        paths[attr_to_path[distinguishing_attr][0]].append(
                            distinguishing_attr
                        )
                        min_depth = min(
                            min_depth, len(attr_to_path[distinguishing_attr][0])
                        )

                    data.loc[len(data)] = [
                        conditions,
                        goals,
                        min_depth,
                        paths,
                        distinguishing_attrs,
                        sorted(possible_attrs),
                        sorted(possible_questions),
                        num_obj,
                        end_time - start_time,
                    ]

                # subsample
                valid_qs_subsample = random.sample(
                    list(valid_questions),
                    min(num_ques_singlepath, len(valid_questions)),
                )
                for q, question in enumerate(valid_qs_subsample):
                    if q % 10000 == 0:
                        print(
                            f"{num_obj} {g} Single Question {q} / {len(valid_qs_subsample)}"
                        )
                    other_false_facts = set(valid_questions) - {question}
                    all_false_facts = false_facts.union(other_false_facts)
                    conditions = "\n".join(
                        [f"{true_fact}" for true_fact in state]
                        + [f"not {false_fact}" for false_fact in all_false_facts],
                    )
                    start_time = time.time()
                    # checking validity of question set
                    (
                        distinguishing_attrs,
                        _,
                        possible_attrs,
                        _,
                    ) = check_questions(
                        domain,
                        problem_spec,
                        state,
                        all_false_facts,
                        goal_conditions,
                        contradicting_fact_pairs,
                        impossible_facts,
                        constraints,
                        1,
                    )
                    end_time = time.time()
                    if distinguishing_attrs is None or possible_attrs is None:
                        continue
                    assert not distinguishing_attrs
                    min_depth = len(heldout_q_to_path[question])
                    data.loc[len(data)] = [
                        conditions,
                        goals,
                        min_depth,
                        {heldout_q_to_path[question]: "No questions needed."},
                        {"No questions needed."},
                        sorted(possible_attrs),
                        sorted(possible_questions),
                        num_obj,
                        end_time - start_time,
                    ]

            os.makedirs(args.output_dir, exist_ok=True)
            with open(f"{args.output_dir}/{num_obj}blocks_{g}goal.csv", "w") as wf:
                data.to_csv(wf, index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--pddl_dir",
        default="data/Planning-Q/task_pddls/blocks",
        help="Directory containing PDDLs",
    )
    argparser.add_argument(
        "--output_dir",
        default="data/Planning-Q/heldout_data",
        help="Directory to write data to",
    )
    args = argparser.parse_args()
    main(args)
