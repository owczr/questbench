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

"""Base class for evaluators."""

from transformers import pipeline

from model_utils import CLAUDE_MODELS, GPT_COSTS, load_cache_file


class Evaluator:
    """Base class for evaluators.

    Attributes:
      model_name: name of LLM to evaluate
      generation_config: generation config for LLM
      model_url: model url of LLM
      cache: cache of LLM responses
      cache_file: cache file of LLM responses
      use_cot: whether to use CoT or not
      fs_samples: number of few-shot samples to use
      eval_mode: evaluation mode, one of "mc", "isambig", "fullinfo"
      model_role_name: role name for the model
      parallel_model_calls: whether to make parallel calls to the model
      vllm_port: port for the VLLM server
    """

    def __init__(
        self,
        model_name: str,
        cache=None,
        cache_file=None,
        use_cot: bool = False,
        fs_samples: int = 0,
        eval_mode: str = "mc",
        model_role_name: str = "model",
        parallel_model_calls: bool = True,
        vllm_port: int = 8000,
    ):
        self.model_name = model_name
        self.generation_config = {
            "temperature": 0.0,
            "max_completion_tokens": 512,
        }
        if "gemini" in self.model_name:
            self.model_url = self.model_name
        elif "gemma" in self.model_name:
            # For Gemma models, we'll use the VLLM server URL
            self.model_url = f"http://localhost:{vllm_port}/v1/chat/completions"
            if self.model_name == "gemma_2_2b":
                self.model_name = "google/gemma-2-2b-it"
            elif self.model_name == "gemma_2_9b":
                self.model_name = "google/gemma-2-9b-it"
            elif self.model_name == "gemma_2_27b":
                self.model_name = "google/gemma-2-27b-it"
            else:
                raise ValueError(f"Invalid model name: {self.model_name}")
        elif self.model_name in GPT_COSTS:
            self.generation_config = {
                "temperature": 0.0,
                "max_completion_tokens": 512,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
            self.model_url = "https://api.openai.com/v1/chat/completions"
        elif self.model_name in CLAUDE_MODELS:
            self.generation_config = {
                "temperature": 0.0,
                "max_tokens": 512,
            }
            self.model_url = "https://api.anthropic.com/v1/messages"
        elif "local" in self.model_name:
            self.generation_config = {
                "temperature": 0.0,
                "max_tokens": 512,
            }
            self.model_url = "http://localhost:1234/v1/chat/completions"
        else:
            self.model_url = self.model_name
        self.cache = cache
        self.cache_file = cache_file
        if cache is None and cache_file is not None:
            self.cache = load_cache_file(cache_file)
            print(f"Loaded {len(self.cache)} entries from {cache_file}")
        self.use_cot = use_cot
        self.fs_samples = fs_samples
        self.eval_mode = eval_mode
        self.model_role_name = model_role_name
        self.parallel_model_calls = parallel_model_calls
        self.vllm_port = vllm_port
