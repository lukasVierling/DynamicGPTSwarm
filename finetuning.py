'''
Source: https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-gemma-0444d46d821c
'''

import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,7"
import pandas as pd
import torch
from dataset import Dataset, load_dataset
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer

from transformers import BitsAndBytesConfig


bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "google/gemma-7b-it"
# model_id = "google/gemma-7b"
# model_id = "google/gemma-2b-it"
# model_id = "google/gemma-2b"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
# Load your dataset (replace 'your_dataset_name' and 'split_name' with your actual dataset information)
# dataset = load_dataset("your_dataset_name", split="split_name")
dataset = load_dataset("garage-bAInd/Open-Platypus", split="train")
# Define the data collator
def generate_prompt(data_point):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenzed prompt
    """
    prefix_text = 'Below is an instruction that describes a question. Write a response that appropriately answers the given question.\\n\\n'
    # Samples with additional context into.
    if data_point['input']:
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} <end_of_turn>\\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    # Without
    else:
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} <end_of_turn>\\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    return text

# add the "prompt" column in the dataset
text_column = [generate_prompt(data_point) for data_point in dataset]
dataset = dataset.add_column("prompt", text_column)

dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

dataset = dataset.train_test_split(test_size=0.2)
train_data = dataset["train"]
test_data = dataset["test"]

from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=128, #High r because of domain specification
    lora_alpha=256,
    target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
) 
trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
print(
    f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
)

#new code using SFTTrainer
import transformers

from trl import SFTTrainer

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=0.03,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start the training process
#trainer.train()

new_model = "gemma-Code-Instruct-Finetune-test" #Name of the model you will be pushing to huggingface model hub
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)

# Merge the model with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 1},
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Push the model and tokenizer to the Hugging Face Model Hub
merged_model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)

def get_completion(query: str, model, tokenizer) -> str:
  device = "cuda:0"
  prompt_template = """
  <start_of_turn>user
  Below is an instruction that describes a question. Write a response that appropriately answers the question.
  {query}
  <end_of_turn>\\n<start_of_turn>model
  
  """
  prompt = prompt_template.format(query=query)
  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
  model_inputs = encodeds.to(device)
  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  # decoded = tokenizer.batch_decode(generated_ids)
  decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
  return (decoded)

result = get_completion(query="A board game spinner is divided into three parts labeled $A$, $B$ and $C$. The probability of the spinner landing on $A$ is $\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\frac{5}{12}$. What is the probability of the spinner landing on $C$? Express your answer as a common fraction.", model=merged_model, tokenizer=tokenizer)
print(result)