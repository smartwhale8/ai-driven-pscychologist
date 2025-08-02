# AI-Driven Psychologist

This project implements an AI-driven psychologist system that explores how conversational AI agents can collaborate, role-play, and provide mental health support or analysis in a controlled, extensible environment. The main objectives of the project are to:

- Simulate and analyze multi-agent conversations using advanced large language models (LLMs)
- Enable fine-tuning of LLMs for psychological or mental health support tasks
- Facilitate research and experimentation in AI-driven mental health support

The system leverages the AutoGen framework for multi-agent chat simulation and the Unsloth library to fine-tune Llama models efficiently.

---

## Project Purpose & Motivation

The overarching purpose of this project is to enable research and development in AI-powered psychological support and analysis by:

- Simulating complex, multi-agent conversations for training or evaluation
- Enabling domain-specific fine-tuning of LLMs to improve mental health support capabilities
- Serving as a platform for rapid prototyping and experimentation in AI-driven mental health tools

### Why Multi-Agent Interaction?

#### Motivation

Traditional AI chatbots or conversational agents typically model a single user interaction, which is limiting for several reasons—especially in complex domains like psychology, therapy, or collaborative problem-solving. Human interactions often take place in groups, where multiple participants (each with unique roles, perspectives, or expertise) contribute to a richer, more dynamic conversation. In psychological support or therapy, for example, group therapy sessions, peer support groups, or role-playing scenarios can be far more nuanced and effective than one-on-one conversations.

By simulating **multi-agent interactions**, this project aims to:

1. **Mirror Real-World Scenarios:**  
   Group therapy, support groups, and team-based interventions are vital in mental health. Multi-agent modeling allows us to study these dynamics and their outcomes in a controlled, repeatable way.

2. **Enable Role Play and Diverse Perspectives:**  
   Different agents can be assigned roles such as "therapist," "patient," "observer," or even "devil’s advocate." This enables training, evaluation, or data generation that reflects real group dynamics.

3. **Foster Collaboration and Emergent Behavior:**  
   Multiple AI agents can collaborate, challenge, or support each other in conversation, leading to emergent behaviors and richer data for research or fine-tuning.

4. **Generate Synthetic Datasets:**  
   Simulated group conversations can be used to generate labeled datasets for training or benchmarking new models in psychological and conversational AI domains.

#### How Multi-Agent Interaction Was Achieved

This project achieves multi-agent group chat simulation using the following approach:

- **AutoGen Framework:**  
  The [AutoGen](https://microsoft.github.io/autogen/) library is used as the backbone for creating, configuring, and managing multiple conversational agents within a single environment. Each agent is instantiated with its own persona, objectives, and access to language models.

- **Agent Configuration:**  
  In the notebook (`Psych_data_gen.ipynb`), multiple agents are defined programmatically, each with a unique role and configuration (e.g., different prompts, model parameters, or behavioral rules).

- **Orchestrated Group Chat:**  
  The agents interact in a shared "group chat" session. The conversation is orchestrated so that each agent can respond to messages, initiate new topics, or react to others—closely mimicking real multi-party discussions.

- **Expert Evaluation:**  
  Among the multiple agents, one acts as an expert psychologist evaluator. This agent reviews the responses provided by other psychologist agents, assigns a numerical score based on the quality of the response, and provides detailed feedback for improvement. This evaluation mechanism not only simulates expert oversight in real-world group settings but also generates valuable annotated data for training, benchmarking, or further analysis of conversational strategies.
  
- **Customizable Scenarios:**  
  The setup allows for easy customization, so researchers and students can experiment with different numbers and types of agents, conversational goals, or psychological interventions.

- **Extensibility:**  
  The system is designed to allow for the inclusion of additional tools (e.g., access to external data or APIs) and human participants, making it suitable for hybrid human-AI group studies.

##### Example Use Cases

- **Simulating Group Therapy:**  
  Study the interplay between therapists and multiple clients to develop or evaluate intervention strategies.
- **Peer Support Training:**  
  Train AI models to act as peer supporters or moderators in online groups.
- **Data Generation:**  
  Create synthetic multi-turn, multi-agent conversations for use in model training or psychological analysis.

---

## Features

- **Multi-agent chat simulation:** Powered by LLMs (e.g., OpenAI GPT-4, Llama), configurable for different scenarios and roles
- **LLM Fine-tuning:** Support for fine-tuning Llama models using Unsloth, enabling domain-specific improvements
- **Flexible experimentation:** Easily extensible for role-play, data generation, or psychological studies

---

## Running the Project

### Requirements

- Python >= 3.8
- Jupyter Notebook (or Google Colab)
- `pyautogen` library (for agent chat)
- `unsloth`, `transformers`, `trl`, `peft`, `accelerate`, `bitsandbytes` (for Llama fine-tuning)

### Setup and Execution

#### 1. Agent Chat Simulation

- **Install dependencies:**
  ```bash
  pip install pyautogen~=0.2.0b4
  ```
- **Open the notebook:**  
  Launch [`Psych_data_gen.ipynb`](https://github.com/smartwhale8/ai-driven-pscychologist/blob/main/Psych_data_gen.ipynb) in Jupyter or Colab.
- **Set your OpenAI API key:**  
  In the notebook, set the key as follows:
  ```python
  import os
  os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"
  ```
- **Run the notebook:**  
  Execute the cells sequentially to simulate the group chat scenario.

#### 2. Fine-tuning Llama LLMs with Unsloth

- **Install dependencies (Colab recommended):**
  ```bash
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps xformers trl peft accelerate bitsandbytes
  ```
- **Run the fine-tuning script:**  
  Use `llm_finetune.py` as a starting point for fine-tuning. This script demonstrates:
  - Loading and quantizing Llama, Mistral, or Gemma models with `FastLanguageModel` from Unsloth
  - Configuring LoRA adapters for efficient parameter-efficient fine-tuning
  - Preparing datasets and training arguments using Hugging Face `trl` and `datasets`
  - Example code snippet:
    ```python
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-2-7b-bnb-4bit",
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    # ... See llm_finetune.py for full workflow
    ```

---

## Code Overview

### `Psych_data_gen.ipynb`

- **Purpose:** Main notebook for multi-agent group chat simulation using AutoGen.
- **Key cells:**
  - Installs required packages
  - Configures API endpoints and models
  - Sets up agent roles and group chat simulation
  - Provides example conversations and data generation

### `llm_finetune.py`

- **Purpose:** Fine-tunes Llama and similar LLMs using Unsloth and LoRA techniques.
- **Key components:**
  - Installs Unsloth and related libraries
  - Loads quantized Llama, Mistral, or Gemma models
  - Sets up LoRA adapters for parameter-efficient tuning
  - Prepares datasets and training loop using Hugging Face `trl` and `datasets`

---

## Course Information

This project was completed as part of a course at KTH Royal Institute of Technology.

- **Course**: DD2424 Deep Learning in Data Science
- **Institution**: KTH Royal Institute of Technology (KTH)
- **Year/Semester**: 2024/Spring Semester

---

## References

- [AutoGen documentation](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat)
- [Unsloth documentation](https://github.com/unslothai/unsloth)

---

## Team Members

- Prashant Yadava
- Abdul Fathaah
- Vlad Dobre
- Ivan Zelenin

## License

[MIT License]
