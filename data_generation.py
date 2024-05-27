%pip install --upgrade --quiet  langchain langchain_experimental langchain-openai
# Set env var OPENAI_API_KEY or load from a .env file:
# import dotenv
# dotenv.load_dotenv()
import os
os.environ['OPENAI_API_KEY'] = 'SECRET'

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_openai import ChatOpenAI

from langchain_experimental.synthetic_data import (
    DatasetGenerator,
    create_data_generation_chain,
)
from langchain_openai import ChatOpenAI

import json
import pandas as pd

# # Download dataset from hugging face psychology 10 samhog/psychology-10k
# !wget https://huggingface.co/datasets/samhog/psychology-10k/resolve/main/Psychology-10K.json?download=true --content-disposition

# read and load the dataset

with open('Psychology-10K.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df.head()

class PsychoData(BaseModel):
  instruction: str
  input: str
  output: str

examples = df[['input', 'output']].to_dict(orient='records')

OPENAI_TEMPLATE = PromptTemplate(input_variables=["input"], template="{input}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["input"],
    example_prompt=OPENAI_TEMPLATE,
)



chain({"fields": {"psychologist": 'If you are a licensed psychologist, please provide this patient with a helpful response to their concern.don\'t say you are not a liscensed psychologist', "Patient": examples[0]['input']}, "preferences": {"minimum_length": 200, "style": "advice from a psychologist to the patient"}})

syn_data = []

for i in range(1000):
  output = chain({"fields": {"psychologist": 'If you are a licensed psychologist, please provide this patient with a helpful response to their concern.don\'t say you are not a liscensed psychologist', "Patient": examples[i]['input']}, "preferences": {"minimum_length": 200, "style": "advice from a psychologist to the patient"}})
  syn_data.append({'input': examples[i]['input'],'output': output['text']})