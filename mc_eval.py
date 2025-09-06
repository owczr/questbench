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

"""Script to evaluate LLMs on QuestBench domains."""

import argparse
import os
from os.path import exists

import pandas as pd

from evaluators.gsm import GSMEvaluator
from evaluators.planning import PlanningEvaluator
from evaluators.simple_logic import SimpleLogicEvaluator


def main(user_args) -> None:
    domain_main_name = user_args.domain_name.split("_")[0]
    use_cot = False
    fs_samples = 0
    use_phys_constraints = False
    if user_args.prompt_mode == "cot":
        use_cot = True
    elif user_args.prompt_mode == "phys":
        use_phys_constraints = True
    elif user_args.prompt_mode.startswith("fs"):
        fs_samples = int(user_args.prompt_mode[2:])

    # Make directories for results and cache
    if not os.path.exists(user_args.results_dir):
        os.makedirs(user_args.results_dir)
    cache_dir = os.path.join(user_args.results_dir, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    data_file_base_name = os.path.splitext(os.path.basename(user_args.data_file))[0]
    output_file_name = f"{user_args.model_name}-{user_args.domain_name}-{user_args.eval_mode}-{user_args.prompt_mode}-{data_file_base_name}"
    cache_file = os.path.join(cache_dir, f"{output_file_name}.jsonl")
    output_file = os.path.join(user_args.results_dir, f"{output_file_name}.csv")
    print("Loading Evaluator")
    if domain_main_name == "SL":
        evaluator = SimpleLogicEvaluator(
            user_args.model_name,
            cache_file=cache_file,
            use_cot=use_cot,
            fs_samples=fs_samples,
            eval_mode=user_args.eval_mode,
            batch_size=user_args.batch_size,
            model_role_name=user_args.model_role_name,
            parallel_model_calls=user_args.parallel_model_calls,
            vllm_port=user_args.vllm_port,
        )
        prompt_file = os.path.join(
            user_args.data_dir,
            "Logic-Q/simplelogic_heldout_1k_prompts.csv",
        )
    elif domain_main_name == "GSM":
        assert user_args.domain_name.split("_")[1] in ["csp", "verbal"]
        evaluator = GSMEvaluator(
            user_args.model_name,
            cache_file=cache_file,
            use_cot=use_cot,
            fs_samples=fs_samples,
            verbal_questions="verbal" in user_args.domain_name,
            eval_mode=user_args.eval_mode,
            batch_size=user_args.batch_size,
            model_role_name=user_args.model_role_name,
            parallel_model_calls=user_args.parallel_model_calls,
            vllm_port=user_args.vllm_port,
        )
        if user_args.domain_name.split("_")[1] == "csp":
            prompt_file = os.path.join(
                user_args.data_dir,
                "GSM-Q/gsm_CSP_heldout_pilot_prompts.csv",
            )
        else:
            prompt_file = os.path.join(
                user_args.data_dir,
                "GSM-Q/gsm_verbal_heldout_pilot_prompts.csv",
            )
    elif domain_main_name == "Planning":
        evaluator = PlanningEvaluator(
            user_args.model_name,
            domain_file=os.path.join(
                user_args.data_dir,
                "Planning-Q/task_pddls/blocks/domain.pddl",
            ),
            task_file_pattern=os.path.join(
                user_args.data_dir,
                "Planning-Q/task_pddls/blocks/task*.pddl",
            ),
            cache_file=cache_file,
            use_cot=use_cot,
            use_phys_constraints=use_phys_constraints,
            fs_samples=fs_samples,
            eval_mode=user_args.eval_mode,
            batch_size=user_args.batch_size,
            model_role_name=user_args.model_role_name,
            parallel_model_calls=user_args.parallel_model_calls,
            vllm_port=user_args.vllm_port,
        )
        prompt_file = os.path.join(
            user_args.data_dir,
            "Planning-Q/planning_heldout_prompts.csv",
        )
    else:
        raise SystemExit(f"Unknown domain: {domain_main_name}")

    print("Loading Data")
    data_file = user_args.data_file
    with open(data_file, "r") as f:
        data = pd.read_csv(f)
    prompt_data = None
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            prompt_data = pd.read_csv(f)

    print("Starting Evaluation")
    results = evaluator.evaluate_data(data, prompt_data)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as wf:
        results.to_csv(wf)
    print(f"Wrote to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help=(
            "The name of the model to evaluate. Currently support `gpt-4o`,"
            " `o1-preview`, `gemini-1.5-flash`, `gemini-1.5-pro`, `gemma_2_2b`,"
            " `gemma_2_9b`, `gemma_2_27b`, and `local/deepseek/deepseek-r1-0528-qwen3-8b`"
        ),
    )
    parser.add_argument(
        "--domain_name",
        type=str,
        choices=[
            "SL",
            "GSM_csp",
            "GSM_verbal",
            "Planning",
        ],
        help=(
            "Domain name. `SL` is for Simple Logic, `GSM_csp` is for GSM-Q with"
            " CSPs, `GSM_verbal` is for GSM-Q with verbal questions, and"
            " `Planning` is for Planning-Q."
        ),
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=[
            "mc",
            "isambig",
            "fullinfo",
        ],
        help=(
            "Evaluation mode. `mc` is for the multiple choice version of"
            " QuestBench, `isambig` is for evaluating whether the model can"
            " identify the task is ambiguous, and `fullinfo` is for evaluating"
            " the model's performance on the task with the full information"
            " (i.e., no missing information)."
        ),
    )
    parser.add_argument(
        "--data_file", type=str, help="The path to the data file.", default=None
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="questbench_data",
        help=(
            "Directory containing data. Default is `questbench_data` in the"
            " current directory."
        ),
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        choices=["", "cot", "fs4"],
        default="",
        help="Use vanilla, CoT, or fewshot prompting (with 4 samples).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help=(
            "Directory to write results to. Default is `results` in the current"
            " directory."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--model_role_name",
        type=str,
        default="model",
        help=(
            "The name of the model role. In Gemini, this should be `model`. In"
            " OpenAI, this should be `assistant`. You can use other role names as"
            " needed."
        ),
    )
    parser.add_argument(
        "--no_thread_pool",
        action="store_false",
        dest="parallel_model_calls",
        help="Disable thread pool.",
    )
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=8000,
        help="Port for the VLLM server. Default is 8000.",
    )
    args = parser.parse_args()
    main(args)
