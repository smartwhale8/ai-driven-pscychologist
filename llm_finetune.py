
# Installation on colab
# %%capture
# # Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes


import torch.nn.functional as F

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset
import datasets


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/content/drive/MyDrive/lora_model_no_grad",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    output_hidden_states=True
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

#dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = load_dataset("samhog/psychology-10k", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
dataset = dataset.train_test_split(test_size=0.005, shuffle=False, seed = 0)

test_dataset = dataset["test"]
dataset = dataset["train"]

# Function to get the embeddings for a given text
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    # Assuming the llama model has 'last_hidden_state' or another attribute for embeddings
    last_hidden_states = outputs.hidden_states[-1]

    # Average the token embeddings to get a single sentence embedding
    sentence_embedding = last_hidden_states
    return sentence_embedding

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2, #2
        gradient_accumulation_steps = 4, #4
        warmup_steps = 5,
        max_steps = 300,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    score = 0
    for i in range(embedding1.shape[1]):
      for j in range(embedding2.shape[1]):
        score += F.cosine_similarity(embedding1[:,i,:], embedding2[:,j,:]).item()
    return score / (embedding1.shape[1] * embedding2.shape[1])

# Step 2: Define the texts to compare

#No fine tuning
text1 = "It's important to remember that anxiety is a natural emotion that everyone experiences. However, if you are feeling overwhelmed by anxiety, it's important to seek professional help. A licensed psychologist can provide you with tools and strategies to manage your anxiety and improve your overall well-being."

#fine tuning; No gradient accum (=1)
#text1 = "It's important to identify the triggers for your anxiety and work on developing coping strategies. We can also explore relaxation techniques such as deep breathing or mindfulness meditation"

#fine tuning; With gradient accum (=4)
#text1 = "It's important to identify the triggers for your anxiety and work on developing coping strategies. We can also explore relaxation techniques such as deep breathing or mindfulness meditation."

text_training_data = "It's great that you're seeking help. We can work together to identify your triggers and develop coping strategies such as deep breathing exercises or mindfulness techniques."

# Step 3: Get the embeddings for both texts
embedding1 = get_embeddings(text1, tokenizer, model)
embedding2 = get_embeddings(text_training_data, tokenizer, model)

# Step 4: Calculate and print the similarity score
similarity_score = cosine_similarity(embedding1, embedding2)
print(f"Similarity Score: {similarity_score}")
