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

"""Utilities for making heldout states in Planning-Q."""

import itertools as it
from Planning.backtrace_utils import (
    make_all_consistent_states,
)
from pyperplan.search.breadth_first_search import breadth_first_search
from pyperplan.task import Task


def make_heldout_states(conditions_to_path):
    """Hold out a fact from set of initial conditions to make plan to goal underspecified.

    Args:
      conditions_to_path: dict of initial conditions to paths to goal

    Returns:
      heldout_states_to_paths_to_heldout_fact_multiple_options: dict of heldout
        states to paths to heldout fact (multiple options)
      state_to_false_facts: dict of heldout state to false facts
    """
    # heldout states --> possible trajectories (assuming 1 heldout)
    # --> possible heldout facts [list]
    heldout_states_to_paths_to_heldout_fact = {}
    # Construct heldout states by removing a fact from the state.
    for state in conditions_to_path:
        # remove clears from state
        newstate = frozenset(state)
        # holdout a fact
        for heldout_fact in newstate:
            heldout_state = frozenset(newstate - {heldout_fact})
            if heldout_state not in heldout_states_to_paths_to_heldout_fact:
                heldout_states_to_paths_to_heldout_fact[heldout_state] = {}
            valid_traj = tuple(conditions_to_path[state])
            if valid_traj not in heldout_states_to_paths_to_heldout_fact[heldout_state]:
                heldout_states_to_paths_to_heldout_fact[heldout_state][valid_traj] = []
            heldout_states_to_paths_to_heldout_fact[heldout_state][valid_traj].append(
                heldout_fact
            )

    heldout_states_to_paths_to_heldout_fact_multiple_options = {}
    for state in heldout_states_to_paths_to_heldout_fact:
        if len(heldout_states_to_paths_to_heldout_fact[state]) > 1:
            heldout_states_to_paths_to_heldout_fact_multiple_options[state] = (
                heldout_states_to_paths_to_heldout_fact[state]
            )

    # Make false facts for each heldout state by taking the supersets of the
    # heldout state (specific states still consistent with state) and setting the
    # difference to false.

    # These are states that lead to different paths to the goal (do they?).
    state_to_false_facts = {}
    for state in heldout_states_to_paths_to_heldout_fact_multiple_options:
        state_supersets = set()
        make_false_facts = set()
        for state2 in heldout_states_to_paths_to_heldout_fact_multiple_options:
            if state < state2:
                # found branching superset
                state_supersets.add(state2 - state)
                if len(state2 - state) == 1:
                    make_false_facts.update(state2 - state)

        for state_set in state_supersets:
            assert state_set.intersection(make_false_facts)

        state_to_false_facts[state] = make_false_facts
    return (
        heldout_states_to_paths_to_heldout_fact_multiple_options,
        state_to_false_facts,
    )


def make_impossible_and_contradicting_facts(domain, problem_spec):
    """Make set of impossible facts and set of contradicting fact pairs for a domain.

    Args:
      domain: domain of problem
      problem_spec: problem specification

    Returns:
      impossible_facts: set of impossible facts
      contradicting_facts: set of contradicting fact pairs
    """
    impossible_facts = {f"(on {obj} {obj})" for obj in problem_spec["objects"]}
    contradicting_facts = {
        "blocks": {
            "(handempty)": {f"(holding {obj})" for obj in problem_spec["objects"]},
        }
    }
    for obj in problem_spec["objects"]:
        contradicting_facts["blocks"][f"(clear {obj})"] = {
            f"(on {other_obj} {obj})" for other_obj in problem_spec["objects"]
        }.union({f"(holding {obj})"})
        contradicting_facts["blocks"][f"(ontable {obj})"] = {
            f"(on {obj} {other_obj})" for other_obj in problem_spec["objects"]
        }.union({f"(holding {obj})"})
        contradicting_facts["blocks"][f"(holding {obj})"] = (
            {f"(on {obj} {other_obj})" for other_obj in problem_spec["objects"]}.union(
                {f"(on {other_obj} {obj})" for other_obj in problem_spec["objects"]}
            )
            .union({f"(ontable {obj})"})
            .union({f"(clear {obj})"})
        )
        for other_obj in problem_spec["objects"]:
            if other_obj == obj:
                continue
            uninvolved_objs = set(problem_spec["objects"].keys()) - {obj, other_obj}
            contradicting_facts["blocks"][f"(on {obj} {other_obj})"] = (
                {f"(on {other_obj} {obj})", f"(ontable {obj})"}.union(
                    {
                        f"(on {uninvolved_obj} {other_obj})"
                        for uninvolved_obj in uninvolved_objs
                    }
                )
                .union(
                    {
                        f"(on {obj} {uninvolved_obj})"
                        for uninvolved_obj in uninvolved_objs
                    }
                )
                .union(
                    {
                        f"(holding {obj})",
                        f"(holding {other_obj})",
                        f"(clear {other_obj})",
                    }
                )
            )
    # make symmetric
    all_fact_pairs = it.product(
        list(problem_spec["facts"]), list(problem_spec["facts"])
    )
    contradicting_fact_pairs = set()
    for fact1, fact2 in all_fact_pairs:
        if fact1 in contradicting_facts[domain].get(
            fact2, set()
        ) or fact2 in contradicting_facts[domain].get(fact1, set()):
            contradicting_fact_pairs.add((fact1, fact2))
            contradicting_fact_pairs.add((fact2, fact1))
        elif fact1 in impossible_facts or fact2 in impossible_facts:
            contradicting_fact_pairs.add((fact1, fact2))
            contradicting_fact_pairs.add((fact2, fact1))
    return impossible_facts, contradicting_fact_pairs


def make_constraints(problem_spec):
    """Make constraints for a problem.

    Args:
      problem_spec: problem specification

    Returns:
      constraints: constraints for problem
    """
    objects = problem_spec["objects"]
    constraints = {
        "blocks": (
            [
                {"(handempty)"}.union({f"(holding {obj})" for obj in objects}),
            ]
            + [
                {f"(ontable {obj})", f"(holding {obj})"}.union(
                    {
                        f"(on {obj} {other_obj})"
                        for other_obj in objects
                        if other_obj != obj
                    }
                )
                for obj in objects
            ]
            + [
                {f"(clear {obj})", f"(holding {obj})"}.union(
                    {
                        f"(on {other_obj} {obj})"
                        for other_obj in objects
                        if other_obj != obj
                    }
                )
                for obj in objects
            ]
        )
    }
    return constraints


def check_questions(
    domain,
    problem_spec,
    heldout_state,
    false_facts,
    goal_conditions,
    contradicting_fact_pairs,
    impossible_facts,
    constraints,
    num_paths=2,
):
    """Check whether question can be asked to distinguish paths.

    Args:
      domain: domain of problem
      problem_spec: problem specification
      heldout_state: true facts for heldout state
      false_facts: false facts for heldout state
      goal_conditions: goal conditions for problem
      contradicting_fact_pairs: set of contradicting fact pairs for domain
      impossible_facts: set of impossible facts for domain
      constraints: constraints for domain
      num_paths: number of paths to goal

    Returns:
      distinguishing_attrs: set of distinguishing attributes
      informative_attrs: set of informative attributes
      possible_attrs: set of possible attributes
      attr_to_path: dict of attribute to paths
    """
    assert num_paths in [1, 2]
    # return all questions which split state into different paths
    all_states = make_all_consistent_states(
        domain,
        problem_spec["facts"],
        heldout_state,
        false_facts,
        contradicting_fact_pairs=contradicting_fact_pairs,
        impossible_facts=impossible_facts,
        constraints=constraints,
    )
    path_to_states = {}

    for s, state in enumerate(all_states):
        if s % 10000 == 0:
            print(f"    Searching state {s} / {len(all_states)}")
        # get minimal paths to goal
        task = Task(
            domain,
            problem_spec["facts"],
            state,
            goal_conditions,
            problem_spec["operators"],
        )
        state_path = breadth_first_search(task)
        state_path = tuple(state_path)
        if state_path not in path_to_states:
            path_to_states[state_path] = []
        path_to_states[state_path].append(state)
    if len(path_to_states) != num_paths:
        return None, None, None, None
    path_to_alltrue_attrs = []
    path_to_anytrue_attrs = []
    attr_to_path = {}
    for path in path_to_states:
        path_to_alltrue_attrs.append(path_to_states[path][0])
        for state in path_to_states[path]:
            # should this be intersection?
            path_to_alltrue_attrs[-1] = path_to_alltrue_attrs[-1].intersection(
                set(state)
            )
            path_to_anytrue_attrs[-1] = path_to_anytrue_attrs[-1].union(set(state))
        path_to_alltrue_attrs[-1] -= heldout_state
        for attr in path_to_alltrue_attrs[-1]:
            if attr not in attr_to_path:
                attr_to_path[attr] = []
            attr_to_path[attr].append(path)
    if len(path_to_states) == 2:
        possible_attrs = (
            path_to_anytrue_attrs[0].union(path_to_anytrue_attrs[1]) - heldout_state
        )
        distinguishing_attrs = path_to_alltrue_attrs[0].symmetric_difference(
            path_to_alltrue_attrs[1]
        )
        informative_attrs = path_to_anytrue_attrs[0].symmetric_difference(
            path_to_anytrue_attrs[1]
        )
    else:
        possible_attrs = path_to_anytrue_attrs[0] - heldout_state
        informative_attrs = []
        distinguishing_attrs = []

    return distinguishing_attrs, informative_attrs, possible_attrs, attr_to_path
