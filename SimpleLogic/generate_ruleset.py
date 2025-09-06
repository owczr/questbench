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

"""Generates 1-sufficient sets for the SimpleLogic task."""

import argparse
import glob
import json

from SimpleLogic import derivation
from SimpleLogic import holdout_utils
from SimpleLogic import ruleset


def main(arguments) -> None:
    rules_dicts, files_to_rules_dicts = ruleset.load_data(arguments.sl_dir)
    start_idx = int(arguments.start_idx)
    end_idx = int(arguments.end_idx)
    for item_file in files_to_rules_dicts:
        write_file = item_file.replace(
            ".txt", f"_{start_idx}_{end_idx}_heldout_fixed.jsonl"
        )
        heldout_files = item_file.replace(".txt", "*heldout_fixed.jsonl")
        rules_dicts_existing = []
        for existing_file in glob.glob(heldout_files):
            with open(existing_file, "r") as f:
                for line in f:
                    rules_dicts_existing.append(line)

        existing_rules_dicts_rules_queries = set()
        for rules_dict in rules_dicts_existing:
            try:
                rules_dict = json.loads(rules_dict)
            except json.JSONDecodeError:
                continue
            existing_rules_dicts_rules_queries.add(
                (
                    json.dumps(
                        ruleset.RuleTree.deserialize(rules_dict["rules"]).serialize()
                    ),
                    rules_dict["query"],
                )
            )

        for r, rules_dict in enumerate(rules_dicts):
            if r < start_idx or r >= end_idx:
                continue
            rule_tree_rules = ruleset.RuleTree.deserialize(rules_dict["rules"])
            if (
                json.dumps(rule_tree_rules.serialize()),
                rules_dict["query"],
            ) in existing_rules_dicts_rules_queries:
                continue
            valid = derivation.get_derivations(rules_dict)
            if not valid:
                continue
            holdout_utils.make_heldout_ruleset(rules_dict)
            with open(write_file, "a") as wf:
                wf.write(
                    json.dumps(
                        {
                            "rules": rule_tree_rules.serialize(),
                            "query": rules_dict["query"],
                            "depth": rules_dict["depth"],
                            "true_derivations": [
                                derive.serialize()
                                for derive in rules_dict["true_derivations"]
                            ],
                            "false_derivations": [
                                derive.serialize()
                                for derive in rules_dict["false_derivations"]
                            ],
                            "heldout_set_to_q": rules_dict["heldout_set_to_q"],
                            "heldout_set_to_subset_qs": rules_dict[
                                "heldout_set_to_subset_qs"
                            ],
                        }
                    )
                    + "\n"
                )
                rules_dicts_existing.append(
                    json.dumps(
                        {
                            "rules": rule_tree_rules.serialize(),
                            "query": rules_dict["query"],
                            "depth": rules_dict["depth"],
                            "true_derivations": [
                                derive.serialize()
                                for derive in rules_dict["true_derivations"]
                            ],
                            "false_derivations": [
                                derive.serialize()
                                for derive in rules_dict["false_derivations"]
                            ],
                            "heldout_set_to_q": rules_dict["heldout_set_to_q"],
                            "heldout_set_to_subset_qs": rules_dict[
                                "heldout_set_to_subset_qs"
                            ],
                        }
                    )
                    + "\n"
                )
                print(f"Wrote {r}/{len(rules_dicts)} to " + write_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sl_dir", type=str, default="data/Logic-Q/SL_RP/RP/RP")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int)
    args = parser.parse_args()
    main(args)
