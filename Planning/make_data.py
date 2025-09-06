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

"""Format prompts and data for the Planning-Q task."""

import argparse
import glob
import os
import random

import pandas as pd
import tqdm

tqdm = tqdm.tqdm


def main(arguments) -> None:
    all_data = []
    for data_file in tqdm(glob.glob(os.path.join(arguments.input_dir, "*.csv"))):
        with open(data_file, "r") as f:
            data = pd.read_csv(f)
        all_data.append(data)

    data = pd.concat(all_data)
    data = data.rename(
        columns={
            "gt_queries": "gt_qs",
            "physically_valid_attrs": "all_valid_qs",
            "num_objs": "num_vars",
            "all_attrs": "all_qs",
            "num_qs": "num_qs",
            "plans": "plan_to_gt_q",
        }
    )
    data_uniq = data.drop_duplicates()
    prompt_sample_idxs = random.sample(range(len(data_uniq)), k=38)
    prompts = data_uniq.iloc[prompt_sample_idxs]
    other_idxs = set(range(len(data_uniq))) - set(prompt_sample_idxs)
    rest_of_data = data_uniq.iloc[list(other_idxs)]
    # subsample some for o1
    rest_of_data_subsample = pd.concat(
        [
            rest_of_data[rest_of_data["min_depth"] == 4]
            .sample(n=5)
            .reset_index(drop=True),
            rest_of_data[rest_of_data["min_depth"] == 5]
            .sample(n=5)
            .reset_index(drop=True),
            rest_of_data[rest_of_data["min_depth"] == 6]
            .sample(n=10)
            .reset_index(drop=True),
            rest_of_data[rest_of_data["min_depth"] == 7]
            .sample(n=10)
            .reset_index(drop=True),
            rest_of_data[rest_of_data["min_depth"] == 8]
            .sample(n=10)
            .reset_index(drop=True),
            rest_of_data[rest_of_data["min_depth"] == 9]
            .sample(n=10)
            .reset_index(drop=True),
            rest_of_data[rest_of_data["min_depth"] == 10]
            .sample(n=10)
            .reset_index(drop=True),
            rest_of_data[rest_of_data["min_depth"] == 11]
            .sample(n=10)
            .reset_index(drop=True),
            rest_of_data[rest_of_data["min_depth"] == 12]
            .sample(n=10)
            .reset_index(drop=True),
            rest_of_data[rest_of_data["min_depth"] == 13]
            .sample(n=10)
            .reset_index(drop=True),
            rest_of_data[rest_of_data["min_depth"] == 14]
            .sample(n=10)
            .reset_index(drop=True),
        ]
    )
    # save 7500 to csv
    with open(
        os.path.join(arguments.output_dir, "/planning_heldout_7500.csv"), "w"
    ) as f:
        rest_of_data.to_csv(f, index=False)
    with open(
        os.path.join(arguments.output_dir, "planning_heldout_prompts.csv"), "w"
    ) as f:
        prompts.to_csv(f, index=False)
    with open(
        os.path.join(arguments.output_dir, "planning_heldout_7500_o1_subsample.csv"),
        "w",
    ) as f:
        rest_of_data_subsample.to_csv(f, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="data/Planning-Q/heldout_data",
        help="Directory to read data from",
    )
    parser.add_argument(
        "--output_dir",
        default="data/Planning-Q/heldout_data",
        help="Directory to write data to",
    )
    args = parser.parse_args()
    main(args)
