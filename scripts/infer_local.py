from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer
from peft import PeftModel
import json, sys, pdb


import json
import numpy as np
import subprocess

import os, pdb, sys
from tqdm import tqdm
import random

import sys
from transformers import (AutoModelForCausalLM, AutoTokenizer)
LANG=sys.argv[1]

ONT_FILE = "nyt10mont.txt"

SOURCE_PATH = "nyt10m/{}_final.jsonl".format(LANG)

try:
    EXAMPLE_PATH = sys.argv[3]
    ICL = True
except:
    ICL = False
    EXAMPLE_PATH = "NoEx"
try:
    argv4 = sys.argv[4]
    ONT = True
except:
    ONT = False

FULL_TRAIN_PATH = "nyt10m/Translate-train/{}_train.txt".format(LANG)

if "transtest" in EXAMPLE_PATH or "clingual" in EXAMPLE_PATH:
    FULL_TRAIN_PATH = "nyt10m/nyt10m_train.txt"

REL_FILE = "nyt10m/nyt10m_rel2id.json"

model_name = sys.argv[2]
base_name = model_name.split("/")[0].strip()
#model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"

OUTPUT_DIR = "llama_out"

if ICL:
    run_id = EXAMPLE_PATH.split("/")[-1].split(".")[0].split("bertscore_pare5_trans_train_recall_nonna_")[-1].strip().split(".json")[0] + "_headtailnew_ont_vllm_"+base_name+"_"+model_name.split("/")[-1].strip()
else:
    run_id = "headtailnew_ont_0shot_"+base_name +LANG


total_tokens = 0
random.seed(0)
np.random.seed(0)


if "gpt" in model_name:
    from dotenv import load_dotenv, find_dotenv
    import openai
    import tiktoken
    encoding = tiktoken.get_encoding("o200k_base")
    load_dotenv(os.path.join(os.path.dirname(ENV_FILE), '.env'), override=True)

    openaiClient = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def num_tokens_from_messages(messages, model = "gpt-4-0613"):
    """Return the number of tokens used by a list of messages."""
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def queryLlama(model, tokenizer, query, examples, all_rels, err_file, out_file, head, tail, icl=False, ONT=True) :
    system_prompt = "Choose all applicable relations between head and tail entities in the following passages from the given set of relations.\n"+', '.join(all_rels)+".\nPrint each relation in a new line. If none of the relations are applicable, output 'NA'."
    with open(ONT_FILE, "r") as f:
        ont_exp = f.read()
    if ONT:
        system_prompt = system_prompt + "\n" + ont_exp

    messages = [{"role": "system", "content": system_prompt}]
    if icl:
        for eg in examples:
            instr = f"Head: {eg[2]}, Tail: {eg[3]}"
            messages.append({"role": "user", "content": eg[0]+"\n"+instr})
            messages.append({"role": "assistant", "content": '\n'.join(eg[1])})
    else:
        pdb.set_trace()

    instr = f"Head: {head}, Tail: {tail}"
    messages.append({"role": "user", "content": query+"\n"+instr})
    terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    #return_attention_mask=True,
                    ).to(model.device)
    print(input_ids.shape)
    outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=terminators,
                          pad_token_id=tokenizer.eos_token_id, temperature=0.01, do_sample=False)
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    print(messages)
    print(response)
    with open(out_file, "a") as f:
        f.write(str(messages)+"\n")
        f.write(str(1000)+"\n")
        f.write("Response\n"+str(response)+"\n\n")
    

    return response

def text_norm(text) :
    text = text.strip()
    return text

def add_brackets(text, pos, lb = '[', rb = ']'):
    return text[:pos[0]] + lb + text[pos[0]:pos[1]] + rb + text[pos[1]:]

def add_double_brackets(text, pos1, pos2, lb1="<Head> ", rb1=" </Head>", lb2="<Tail> ", rb2=" </Tail>") :
    text = add_brackets(text, pos1, lb1, rb1)
    if (pos2[0] > pos1[1]) :
        pos2 = (pos2[0] + len(lb1)+len(rb1), pos2[1] + len(lb1)+len(rb1))
    text = add_brackets(text, pos2, lb2, rb2)
    return text

def replace_brackets(text, head, tail, lb = '[', rb = ']'):
    text = text.replace("{"+head+"}", lb+head+rb).replace("{"+tail+"}", lb+tail+rb)
    return text
def replace_brackets_markers(text, head, tail, lb = '[', rb = ']'):
    text = text.replace("{"+head+"}", lb+" " + head+ " " + rb).replace("{"+tail+"}", lb+ " "+tail+" "+rb)
    return text

TOKENIZER_PATH=""
#MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
MODEL_PATH = sys.argv[2]
ADAPTER_PATH = ""  # From OUT_DIR in training script
MAX_SEQ_LENGTH = 8192
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_CPT = True

run_id += "_" + MODEL_PATH.split("/")[-1].strip()
# Load the tokenizer and base model

print(MODEL_PATH)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
print(model.get_input_embeddings())
print(len(tokenizer))


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.to(DEVICE)

if not ONT:
    run_id += "_noONT"

if ICL:
    run_id += "_ICL"
run_id += "_" + LANG 



all_rels = []
with open(REL_FILE, "r") as f:
    data = json.load(f)
    all_rels = list(data.keys())
# remove NA 
all_rels.remove("NA")

sent_examples = {}
if ICL:
    with open(EXAMPLE_PATH, "r") as f:
        for line in f:
            data = json.loads(line)
            if isinstance(data["query"], dict):
                sent_examples[text_norm(data["query"]['text'])] = data["top_docs"]
            else:
                sent_examples[text_norm(data["query"])] = data["top_docs"]

all_train_data = {}
with open(FULL_TRAIN_PATH, "r") as f:
    for line in f:
        data = json.loads(line)
        all_train_data[text_norm(data["text"])] = (data["h"]["pos"], data["t"]["pos"])
print(run_id)
err_file = os.path.join(OUTPUT_DIR, "err_"+run_id+".txt")
out_file = os.path.join(OUTPUT_DIR, "out_"+run_id+".txt")
with open(err_file, "w") as f:
    f.write("")
with open(out_file, "w") as f:
    f.write("")

queries_all = []
examples_all = []
heads_all = []
tails_all = []

with open(SOURCE_PATH, "r") as f:
    all_lines = f.readlines()
    for line in tqdm(all_lines):
        data = json.loads(line)
        if ICL:
            bags = sent_examples[text_norm(data["text"])]
        try :
            query = replace_brackets_markers(data['text'], data["head"], data["tail"])
        except :
            query = add_double_brackets(data["text"], data["h"]["pos"], data["t"]["pos"])
        examples = []
        if ICL:
            for bag in bags:
                bag_text = ""
                head = ""
                tail = ""
                for sent in bag["texts"]:
                    ht = all_train_data[text_norm(sent)]
                    head = sent[ht[0][0]:ht[0][1]]
                    tail = sent[ht[1][0]:ht[1][1]]
                    bag_text += add_double_brackets(sent, ht[0], ht[1])
                examples.append((bag_text, bag["relations"], head, tail))
            # reverse examples
            examples = examples[::-1]
        try :
            head = data["head"]
            tail = data["tail"]
        except :
            head = data["text"][data["h"]["pos"][0]:data["h"]["pos"][1]]
            tail = data["text"][data["t"]["pos"][0]:data["t"]["pos"][1]]
        response = queryLlama(model, tokenizer, query, examples, all_rels, err_file, out_file, head, tail, ICL, ONT=ONT)
