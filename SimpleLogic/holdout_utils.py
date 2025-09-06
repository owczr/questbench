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

"""Utilities for generating ambiguous questions by holding out 1 word."""

import copy
import itertools as it
import json

import tqdm

from questbench.SimpleLogic import derivation
from questbench.SimpleLogic import ruleset


def make_heldout_ruleset(rules_dict):
    """Hold out 1 word required for deriving the target word's truth value.

    Keeps only those derivations that are not already derived by the full set.
    Adds field to rules_dict called `heldout_set_to_q` and
    `heldout_set_to_subset_qs`
    `heldout_set_to_q` maps a heldout set to a dict of the form
    {1-sufficient set: sufficient word: {true_derivation, false_derivation}}
    where 1-sufficient set is missing 1 word that is required for deriving the
    target word's truth value, and sufficient word is a word that when known, can
    derive the target word's truth value. The true/false derivations are the
    derivations of the target word's truth value after adding the sufficient word
    to the heldout set.
    `heldout_set_to_subset_qs` maps a 1-sufficient set to a list of heldout sets
    that are subsets of the heldout set, but not the heldout set itself.

    Args:
      rules_dict: Dict[str, Any]
    """
    true_derivations = {}
    for derive in rules_dict["true_derivations"]:
        if not isinstance(derive, derivation.ConjunctionRule):
            derive = derivation.ConjunctionRule.deserialize(derive)
        true_derivations[frozenset(derive.leaf_words.keys())] = derive
    false_derivations = {}
    for derive in rules_dict["false_derivations"]:
        if not isinstance(derive, derivation.ConjunctionRule):
            derive = derivation.ConjunctionRule.deserialize(derive)
        false_derivations[frozenset(derive.leaf_words.keys())] = derive

    heldout_set_to_q_depth_expansions = {}
    # get combined version (cross product)
    for true_derive, false_derive in tqdm.tqdm(
        it.product(true_derivations, false_derivations),
        total=len(true_derivations) * len(false_derivations),
    ):
        # find pairs that differ on exactly 1 var
        differ_word = None
        for k in true_derive:
            if ruleset.negate(k) in false_derive:
                if differ_word is None:
                    differ_word = k
                elif differ_word != k:
                    differ_word = None
                    break
        if differ_word is None:
            continue

        heldout_set = true_derive.union(false_derive)
        heldout_set -= {differ_word, ruleset.negate(differ_word)}

        # can already derive heldout set
        skip_this = False
        if heldout_set in true_derivations or heldout_set in false_derivations:
            skip_this = True
        else:
            for true_derivation in true_derivations:
                if true_derivation <= heldout_set:
                    skip_this = True
                    break
            for false_derivation in false_derivations:
                if false_derivation <= heldout_set:
                    skip_this = True
                    break
        if skip_this:
            continue

        if heldout_set not in heldout_set_to_q_depth_expansions:
            heldout_set_to_q_depth_expansions[heldout_set] = {}
        # add variable with direction that implies query word is true
        heldout_set_to_q_depth_expansions[heldout_set][differ_word] = {
            "true_derivation": true_derivations[true_derive].serialize(),
            "false_derivation": false_derivations[false_derive].serialize(),
        }

    rules_dict["heldout_set_to_q"] = {
        json.dumps(list(heldout_set)): heldout_set_to_q_depth_expansions[heldout_set]
        for heldout_set in heldout_set_to_q_depth_expansions
    }

    # valid questions of subset --> also valid questions here,
    # but we want to avoid asking
    heldout_set_to_subset_qs = {}
    for heldout_set in heldout_set_to_q_depth_expansions:
        heldout_set_to_subset_qs[heldout_set] = set()
        for other_set in heldout_set_to_q_depth_expansions:
            if heldout_set == other_set:
                continue
            # check if other_set is a subset of heldout_Set
            if other_set < heldout_set:
                heldout_set_to_subset_qs[heldout_set] = heldout_set_to_subset_qs[
                    heldout_set
                ].union(set(heldout_set_to_q_depth_expansions[other_set].keys()))
            # remove target_attr from set
            if rules_dict["query"] in heldout_set_to_subset_qs[heldout_set]:
                heldout_set_to_subset_qs[heldout_set].remove(rules_dict["query"])
    rules_dict["heldout_set_to_subset_qs"] = {
        json.dumps(list(heldout_set)): list(heldout_set_to_subset_qs[heldout_set])
        for heldout_set in heldout_set_to_subset_qs
    }


def get_all_inferrable_facts(rule_tree, true_facts, false_facts):
    """Get all facts that can be inferred from the given true and false facts.

    Args:
      rule_tree: RuleTree
      true_facts: Set[str]
      false_facts: Set[str]

    Returns:
      Set[str]
    """
    # ASSUMES FALSE_FACTS ALREADY IN FORM "not _"
    rules_to_consider = copy.deepcopy(rule_tree.nodes)

    all_facts = set(true_facts).union(set(false_facts))

    curr_facts = all_facts
    while curr_facts:
        new_facts = set()
        for fact in curr_facts:
            if ruleset.negate(fact) not in rules_to_consider:
                continue
            for rule in rules_to_consider[ruleset.negate(fact)].rules:
                # find any facts that must be true, due to negate(fact) being false
                remaining_terms = set(rule)
                rule_true = False
                for term in rule:
                    if term in all_facts:
                        # this rule is true by this term
                        rule_true = True
                        break
                    if ruleset.negate(term) in all_facts:
                        # other terms must be true
                        remaining_terms.remove(term)
                if rule_true:
                    continue
                if len(remaining_terms) == 1:
                    new_facts.add(remaining_terms.pop())
        curr_facts = new_facts
        all_facts = all_facts.union(curr_facts)
    return all_facts
