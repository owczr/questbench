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

"""Utility functions for backward searching for plans to an end goal."""

import collections
import itertools as it

from pyperplan.search import searchspace


def check_self_consistency(domain, facts, contradicting_fact_pairs):
    """Checks if a set of facts is self-consistent (physically plausible) in domain.

    Args:
      domain: domain of the task
      facts: set of facts to check
      contradicting_fact_pairs: set of pairs of facts that contradict each other

    Returns:
      True if facts are self-consistent, False otherwise.
    """
    # get all pairs of facts
    fact_pairs = it.product(list(facts), list(facts))
    for fact_pair in fact_pairs:
        if fact_pair in contradicting_fact_pairs:
            return False

    if domain == "blocks":
        # check cycles (on X Y) (on Y Z) --> not (on X Z)
        stacks = []
        for fact in facts:
            if "(on " in fact:
                top = fact.split(" ")[1]
                bottom = fact.split(" ")[2].strip(")")
                bottom_stack = -1
                top_stack = -1

                for s, stack in enumerate(stacks):
                    if bottom in stack:
                        bottom_stack = s
                    if top in stack:
                        top_stack = s
                if bottom_stack == -1 and top_stack == -1:
                    # no stack contains either element
                    stacks.append([bottom, top])
                elif bottom_stack == -1:
                    # top exists in stack, add prepend bottom to that stack
                    if stacks[top_stack][0] != top:
                        # current bottom of stack is top item
                        return False
                    stacks[top_stack].insert(0, bottom)
                elif top_stack == -1:
                    # bottom exists in stack, append top to that stack
                    if stacks[bottom_stack][-1] != bottom:
                        # current top of stack is bottom item
                        return False
                    stacks[bottom_stack].append(top)
                else:
                    if bottom_stack == top_stack:
                        # stacking same stack on top of itself
                        return False
                    if stacks[bottom_stack][-1] != bottom:
                        # current top of stack is bottom item
                        return False
                    if stacks[top_stack][0] != top:
                        # current bottom of stack is top item
                        return False
                    # merge stacks
                    stacks[bottom_stack].extend(stacks[top_stack])
                    # remove top stack
                    stacks.pop(top_stack)
        for stack in stacks:
            # find cycles
            if len(set(stack)) != len(stack):
                return False
        count_held = 0
        for fact in facts:
            if "(holding" in fact:
                count_held += 1
            if count_held > 1:
                return False

    return True


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)."""
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s) + 1))


def check_satisfies_constraints(domain, facts, constraints):
    """Check if facts satisfy all constraints in domain.

    Args:
      domain: domain of the task
      facts: set of facts to check
      constraints: dictionary of constraints, where the keys are the domains and
        the values are lists of sets of facts that must be true in a valid state.

    Returns:
      True if facts satisfy all constraints, False otherwise.
    """
    for disjunctive_constraint in constraints[domain]:
        if not facts & disjunctive_constraint:
            # check disjunction satsfied
            return False
    return True


def make_all_consistent_states(
    domain,
    facts,
    goals,
    false_in_goals,
    contradicting_fact_pairs,
    impossible_facts,
    constraints,
):
    """Make all valid, self-consistent states consistent with goals.

    True facts in `goals` must be true, and false facts in `false_in_goals` must
    be false.

    Args:
      domain: domain of the task
      facts: set of all facts
      goals: set of facts that must be true in a valid state
      false_in_goals: set of facts that must be false in a valid state
      contradicting_fact_pairs: set of pairs of facts that contradict each other
      impossible_facts: set of facts that are always false
      constraints: dictionary of constraints, where the keys are the domains and
        the values are lists of sets of facts that must be true in a valid state.

    Returns:
      Set of all valid, self-consistent states consistent with goals.
    """
    if not goals.isdisjoint(false_in_goals):
        # things that must be true in `goal` contradicts things that must be false
        return set()

    fact_to_truth_value = {}
    neutral_facts = set(facts)
    for fact in facts:
        if fact in goals:
            fact_to_truth_value[fact] = True
            if fact in neutral_facts:
                neutral_facts.remove(fact)
        elif (fact in impossible_facts) or (fact in false_in_goals):
            fact_to_truth_value[fact] = False
            if fact in neutral_facts:
                neutral_facts.remove(fact)
        else:
            # check if contradicts any true fact in goals
            for goal_fact in goals:
                if (fact, goal_fact) in contradicting_fact_pairs or (
                    goal_fact,
                    fact,
                ) in contradicting_fact_pairs:
                    fact_to_truth_value[fact] = False
                    if fact in neutral_facts:
                        neutral_facts.remove(fact)
    potential_additional_facts = powerset(neutral_facts)

    potential_goal_states = set()
    for factset in potential_additional_facts:
        goal_state = frozenset(goals.union(factset))
        if check_satisfies_constraints(
            domain, goal_state, constraints
        ) and check_self_consistency(domain, goal_state, contradicting_fact_pairs):
            potential_goal_states.add(goal_state)

    return potential_goal_states


def cause_condition(domain, operator, condition, contradicting_fact_pairs):
    """Check if operator causes condition to be true.

    Args:
      domain: domain of the task
      operator: operator that, when applied, results in conditions
      condition: condition to be caused by operator
      contradicting_fact_pairs: set of pairs of facts that contradict each other

    Returns:
      True if "condition" not in operator's add_effects and
      preconditions contradicts condition.
    """
    if condition not in operator.add_effects:
        return False
    if condition in operator.preconditions:
        return False
    if condition in operator.del_effects:
        return False
    return check_self_consistency(
        domain, operator.preconditions, contradicting_fact_pairs
    ) and not check_self_consistency(
        domain,
        frozenset({*operator.preconditions, condition}),
        contradicting_fact_pairs,
    )


def reverse_apply(domain, operator, conditions, contradicting_fact_pairs):
    """Return preconditions if operator were to result in conditions.

    Args:
      domain: domain of the task
      operator: operator that, when applied, results in conditions
      conditions: set of conditions to be caused by operator
      contradicting_fact_pairs: set of pairs of facts that contradict each other

    Returns:
      True if operator's preconditions result in conditions, False otherwise.
    """
    true_conditions = set(operator.preconditions)
    # delete things that were added because of operator
    remaining_conditions = conditions - operator.add_effects
    if check_self_consistency(
        domain,
        frozenset(true_conditions.union(remaining_conditions)),
        contradicting_fact_pairs,
    ):
        return frozenset(true_conditions.union(remaining_conditions))
    else:
        return False


def visualize_state(state):
    """Visualize a state as a stack.

    Args:
      state: set of facts in the state
    """
    stacks = []
    obj_to_stack_id = {}
    hand = None
    for condition in state:
        if "ontable" in condition:
            obj = condition.split(" ")[1].strip(")")
            obj_to_stack_id[obj] = len(stacks)
            stacks.append(["TAB", obj])
        elif "clear" in condition:
            obj = condition.split(" ")[1].strip(")")
            if obj in obj_to_stack_id:
                assert stacks[obj_to_stack_id[obj]][-1] == obj
            continue
        elif "handempty" in condition:
            assert hand is None
        elif "holding" in condition:
            obj = condition.split(" ")[1].strip(")")
            assert hand is None
            hand = obj
        elif "on" in condition:
            top = condition.split(" ")[1].strip(")")
            bottom = condition.split(" ")[2].strip(")")
            top_stack = obj_to_stack_id.get(top, -1)
            bottom_stack = obj_to_stack_id.get(bottom, -1)
            if bottom_stack == -1 and top_stack == -1:
                # no stack contains either element
                obj_to_stack_id[top] = len(stacks)
                obj_to_stack_id[bottom] = len(stacks)
                stacks.append([bottom, top])
            elif bottom_stack == -1:
                # stack contains both elements
                stacks[top_stack].insert(0, bottom)
                obj_to_stack_id[bottom] = top_stack
            elif top_stack == -1:
                # stack contains both elements
                stacks[bottom_stack].append(top)
                obj_to_stack_id[top] = bottom_stack
            else:
                # merge stacks
                old_top_stack = stacks[top_stack]
                stacks[bottom_stack].extend(old_top_stack)
                old_top_stack = stacks.pop(top_stack)
                for top_stack_obj in old_top_stack:
                    obj_to_stack_id[top_stack_obj] = bottom_stack

    print("Hand: ", hand)
    for stack in stacks:
        print(stack)


def backwards_bfs(
    domain,
    operators,
    goal_conditions,
    conditions_to_path,
    contradicting_fact_pairs=None,
):
    """Searches for a plan to the goal on the given task using backwards breadth-first search.

    Also does duplicate detection.

    Args:
      domain: domain of the task
      operators: list of operators
      goal_conditions: set of facts that must be true in a final state
      conditions_to_path: dictionary of conditions to list of operators that
        result in that condition
      contradicting_fact_pairs: set of pairs of facts that contradict each other

    Returns:
      dictionary of possible start conditions to list of operators that result in
      that condition
    """
    # counts the number of loops (only for printing)
    iteration = 0
    # fifo-queue storing the nodes which are next to explore
    queue = collections.deque()
    # goal_conditions = planning_task.goals
    root_node = searchspace.make_root_node(goal_conditions)
    queue.append(root_node)
    conditions_to_path[goal_conditions] = root_node.extract_solution()
    # set storing the explored nodes, used for duplicate detection
    seen_conditions_set = {goal_conditions}
    while queue:
        iteration += 1
        if iteration % 1000 == 0:
            print(
                "breadth_first_search: Iteration %d, #unexplored=%d"
                % (iteration, len(queue))
            )
        # get the next node to explore
        node = queue.popleft()
        for op in operators:
            for condition in node.state:
                if not cause_condition(domain, op, condition, contradicting_fact_pairs):
                    continue
                ancestor_conditions = reverse_apply(
                    domain, op, node.state, contradicting_fact_pairs
                )
                if not ancestor_conditions:
                    continue
                if any(
                    seen_condition <= ancestor_conditions
                    for seen_condition in seen_conditions_set
                ):
                    # check for cycles -- any previous-layer conditions are a subset
                    # of new-layer conditions means there is a faster way to get to
                    # goal condition (which we've expanded before)
                    continue
                child_node = searchspace.make_child_node(node, op, ancestor_conditions)
                queue.append(child_node)
                conditions_to_path[ancestor_conditions] = child_node.extract_solution()
                # remember the successor state
                seen_conditions_set.add(ancestor_conditions)
    return conditions_to_path
