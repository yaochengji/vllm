# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Write a story about a character who discovers a hidden door in their home.",
    "Describe a world where emotions are traded as currency.",
    "Create a tale about a person who can hear the thoughts of animals.",
    "Write a story about a time traveler who accidentally changes a major historical event.",
    "Imagine a society where everyone is born with a unique superpower, except for one person.",
    "Write about a character who finds a mysterious object that grants them one wish.",
    "Describe a day in the life of a sentient AI.",
    "Write a story about a person who wakes up with no memory of the past year.",
    "Create a tale about a group of friends who discover a secret underground city.",
    "Write about a character who can communicate with the dead.",
    "What are three habits you want to develop this year?",
    "Write about a skill you’ve always wanted to learn.",
    "What does success mean to you?",
    "Describe a challenge you’re currently facing and how you plan to overcome it.",
    "Write about a book that changed your perspective.",
    "What are your top three priorities in life right now?",
    "Reflect on a time when you stepped out of your comfort zone.",
    "Write about a person you admire and why.",
    "What are your biggest strengths, and how can you use them more?",
    "Describe a dream you’ve had and what it might mean.",
    "Imagine a world where everyone can read minds. How would society function?",
    "What would happen if humans could photosynthesize like plants?",
    "Create a new holiday tradition and describe how it would be celebrated.",
    "Imagine a world without money. How would people exchange goods and services?",
    "What would a day in your life look like if you were a famous celebrity?",
]
# Create a sampling params object.
sampling_params = SamplingParams()  #temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="Qwen/Qwen2-1.5B-Instruct", max_model_len=512, max_num_seqs=16, enforce_eager=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
