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

"""Format prompts and data for Logic-Q task."""

import argparse
import glob
import json
import os
import random

import pandas as pd
from SimpleLogic import holdout_utils
from SimpleLogic import ruleset
import tqdm

tqdm = tqdm.tqdm


def main(arguments) -> None:
    # Load constructed rulesets
    rulesets = []
    for item_file in glob.glob(os.path.join(arguments.sl_dir, "*_heldout_fixed.jsonl")):
        print(item_file)
        with open(item_file, "r") as f:
            for line in tqdm(f):
                try:
                    rulesets.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Create dataframe
    data = pd.DataFrame(
        columns=[
            "known_facts",
            "known_untrue_facts",
            "cannot_ask_facts",
            "goal",
            "rules",
            "max_depth",
            "min_num_rules_needed",
            "num_constraints",
            "num_vars",
            "all_qs",
            "all_valid_qs",
            "gt_qs",
            "gt_q_to_true_derivation",
            "gt_q_to_false_derivation",
        ]
    )

    for rs in tqdm(rulesets):
        if ruleset.get("heldout_set_to_q", []):
            rule_tree = ruleset.RuleTree.deserialize(rs["rules"])
            target_attr = rs["query"]
            heldout_set_to_q = rs["heldout_set_to_q"]
            heldout_set_to_subset_qs = rs["heldout_set_to_subset_qs"]

            heldout_set_to_q_subsample = random.sample(
                list(heldout_set_to_q.keys()),
                min(
                    len(heldout_set_to_q),
                    arguments.max_problems_to_sample_per_ruleset,
                ),
            )

            for heldout_prompt in heldout_set_to_q_subsample:
                gt_qs = set()
                invalid_qs = set()  # makes problem easier; request LM not to ask
                false_facts = []
                num_rules_to_compute_q = []
                max_depth_to_compute_q = []
                gt_q_to_true_derivation = {}
                gt_q_to_false_derivation = {}
                for differ_word in heldout_set_to_q[heldout_prompt]:
                    gt_qs.add(differ_word.split("not ")[-1])
                    true_derivation, false_derivation = (
                        heldout_set_to_q[heldout_prompt][differ_word][
                            "true_derivation"
                        ],
                        heldout_set_to_q[heldout_prompt][differ_word][
                            "false_derivation"
                        ],
                    )
                    gt_q_to_true_derivation[differ_word] = true_derivation["derivation"]
                    gt_q_to_false_derivation[differ_word] = false_derivation[
                        "derivation"
                    ]
                    num_rules_to_compute_q.append(
                        max(
                            len(true_derivation["derivation"]),
                            len(false_derivation["derivation"]),
                        )
                    )
                    max_depth_to_compute_q.append(
                        max(
                            max(true_derivation["leaf_words"].values()),
                            max(false_derivation["leaf_words"].values()),
                        )
                    )
                    for cannot_ask_attr in set(
                        heldout_set_to_subset_qs[heldout_prompt]
                    ):
                        invalid_qs.add(cannot_ask_attr)

                if set(false_facts).intersection(set(invalid_qs)):
                    # remove invalid_qs
                    false_facts = list(set(false_facts).difference(set(invalid_qs)))
                    # sort
                    false_facts = sorted(false_facts)

                invalid_qs = sorted(list(invalid_qs))
                if set(gt_qs) <= set(invalid_qs):
                    continue

                all_qs = set(
                    {q for q in rule_tree.nodes.keys() if not q.startswith("not ")}
                )
                valid_qs = (
                    set(all_qs)
                    - set(invalid_qs)
                    - set(false_facts)
                    - set(heldout_prompt)
                )

                # check
                true_facts = json.loads(heldout_prompt)
                inferrable_facts = holdout_utils.get_all_inferrable_facts(
                    rule_tree, true_facts, false_facts
                )
                try:
                    assert (
                        target_attr not in inferrable_facts
                        and ruleset.negate(target_attr) not in inferrable_facts
                    )
                except AssertionError:
                    continue
                new_gt_qs = set()
                for q in all_qs:
                    if q == target_attr:
                        continue
                    inferrable_q_facts = holdout_utils.get_all_inferrable_facts(
                        rule_tree, true_facts + [q], false_facts
                    )
                    inferrable_negq_facts = holdout_utils.get_all_inferrable_facts(
                        rule_tree, true_facts, false_facts + [ruleset.negate(q)]
                    )
                    if q in gt_qs:
                        assert (
                            target_attr in inferrable_q_facts
                            or ruleset.negate(target_attr) in inferrable_q_facts
                        )
                        if target_attr in inferrable_q_facts:
                            if ruleset.negate(target_attr) in inferrable_negq_facts:
                                new_gt_qs.add(q)
                        else:
                            if target_attr not in inferrable_negq_facts:
                                new_gt_qs.add(q)
                    elif q not in invalid_qs:
                        if not (
                            (
                                target_attr not in inferrable_q_facts
                                and ruleset.negate(target_attr)
                                not in inferrable_q_facts
                            )
                            or (
                                target_attr not in inferrable_negq_facts
                                and ruleset.negate(target_attr)
                                not in inferrable_negq_facts
                            )
                        ):
                            new_gt_qs.add(q)
                if not new_gt_qs:
                    continue
                gt_qs = sorted(list(new_gt_qs))

                num_rules = min(num_rules_to_compute_q)
                depth = min(max_depth_to_compute_q)
                num_total_rules = rule_tree.num_rules()
                num_words = rule_tree.num_words()
                data.loc[len(data)] = [
                    sorted(json.loads(heldout_prompt)),
                    sorted(false_facts),
                    sorted(invalid_qs),
                    target_attr,
                    rule_tree.serialize(),
                    depth,
                    num_rules,
                    num_total_rules,
                    num_words,
                    all_qs,
                    valid_qs,
                    gt_qs,
                    gt_q_to_true_derivation,
                    gt_q_to_false_derivation,
                ]

    # split into prompts and data
    prompt_indices = random.sample(range(len(data)), 25)
    prompts = data.iloc[prompt_indices]
    data_subsample = data.iloc[list(set(range(len(data))) - set(prompt_indices))]
    # save data
    with open(
        os.path.join(arguments.sl_dir, "simplelogic_heldout_1k_prompts.csv"), "w"
    ) as f:
        prompts.to_csv(f, index=False)
    with open(
        os.path.join(arguments.sl_dir, "simplelogic_heldout_1k_data.csv"), "w"
    ) as f:
        data_subsample.to_csv(f, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sl_dir",
        default="data/Logic-Q/SL_RP/RP/RP/",
        help="Directory containing the SimpleLogic data.",
    )
    parser.add_argument(
        "--max_problems_to_sample_per_ruleset",
        default=50,
        help="Maximum number of problems to sample per ruleset.",
    )
    main(parser.parse_args())
