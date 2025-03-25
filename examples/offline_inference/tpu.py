# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

prompts = [
    "A robot may not injure a human being",
    "It is only with the heart that one can see rightly;",
    "The greatest glory in living lies not in never falling,",
]
answers = [
    " or, through inaction, allow a human being to come to harm.",
    " what is essential is invisible to the eye.",
    " but in rising every time we fall.",
]
N = 1
# Currently, top-p sampling is disabled. `top_p` should be 1.0.
sampling_params = SamplingParams(temperature=0.7,
                                 top_p=1.0,
                                 n=N,
                                 max_tokens=16)

# Set `enforce_eager=True` to avoid ahead-of-time compilation.
# In real workloads, `enforace_eager` should be `False`.
llm = LLM(model="meta-llama/Meta-Llama-3.1-8B",
          max_num_batched_tokens=256,
          max_num_seqs=4,
        #   load_format="dummy",
          tensor_parallel_size=8,
          max_model_len=64,
          enable_sequence_parallel=True)
outputs = llm.generate(prompts, sampling_params)
for output, answer in zip(outputs, answers):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    # assert generated_text.startswith(answer)


# 2025.03.24
# succeeded with following command and some modification on xla_model.py
# VLLM_USE_V1=1 python examples/offline_inference/tpu.py
# remove xla_model's all_gather use all_reduce
# _all_gather_using_all_reduce does not support list of tensors as input
# if pin_layout and output == None and isinstance(value, torch.Tensor):
#   # There is not an easy way to pin the all_gather layout, so use all_reduce
#   # based all_gather for this purpose.
#   return _all_gather_using_all_reduce(
#       value, dim=dim, groups=groups, pin_layout=True)


# 2025.03.24
# succeeded with following command and some modification on xla_model.py
# DISABLE_NUMERIC_CC_TOKEN=1 VLLM_USE_V1=1 python examples/offline_inference/tpu.py
