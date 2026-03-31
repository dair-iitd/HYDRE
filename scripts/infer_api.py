
import json, sys, pdb, os
from dotenv import load_dotenv, find_dotenv
import openai
from together import Together
import time
import json
import numpy as np
import subprocess
# from dotenv import load_dotenv, find_dotenv
import os, pdb, sys
# import openai
from tqdm import tqdm
import random

import sys
import argparse

parser = argparse.ArgumentParser(description="HYDRE Inference (Stage 4)")
parser.add_argument("--lang", type=str, default="en")
parser.add_argument("--model", type=str, default="gpt-4-turbo")
parser.add_argument("--example_path", type=str, required=True, help="Path to single_selected.json from Stage 3")
parser.add_argument("--source_path", type=str, required=True, help="Path to original query file")
parser.add_argument("--rel_file", type=str, help="Path to rel2id mapping")
parser.add_argument("--train_file", type=str, help="Path to training file (for entity positions)")
parser.add_argument("--ont_file", type=str, default="nyt10mont.txt")
parser.add_argument("--out_dir", type=str, default="together_out")

args = parser.parse_args()

LANG = args.lang
model_name = args.model
EXAMPLE_PATH = args.example_path
SOURCE_PATH = args.source_path
REL_FILE = args.rel_file or (os.path.join("nyt10m", "nyt10m_rel2id.json") if "nyt" in SOURCE_PATH else os.path.join("wiki20m", "wiki20m_rel2id.json"))
FULL_TRAIN_PATH = args.train_file or (os.path.join("nyt10m", "nyt10m_train.txt") if "nyt" in SOURCE_PATH else os.path.join("wiki20m", "sampled_grouped_wiki20m.jsonl"))
ONT_FILE = args.ont_file
OUTPUT_DIR = args.out_dir

ICL = True
ONT = True # assume True as in scripts
base_name = model_name.split("/")[-1]

if ICL:
    run_id = "headtailnew_ont_"+EXAMPLE_PATH.split("/")[-1].strip().split(".json")[0]+"_5shot_"+base_name +LANG
else:
    run_id = "headtailnew_ont_0shot_"+base_name +LANG

print(run_id)
total_tokens = 0
random.seed(0)
np.random.seed(0)

load_dotenv('.env', override=True)  #Keep .env in same folder as this script

if "gpt" in model_name:
    client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
else:
    client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

def reqGpt(messages) :
    # try :
    try:
        #pdb.set_trace()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
            top_p=1,
            timeout=15
        )
        if (response.choices[0].finish_reason == 'content_filter') :
            raise("Content filter")
        return response
    except Exception as e:
        print("Error in reqGpt:", e)
        raise

def reqTog(messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
                top_p=1,
                timeout=5  # Reduced timeout to 5 seconds
            )
            if response.choices[0].finish_reason == 'content_filter':
                raise Exception("Content filter")
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait 1 second before retrying
            continue
    return ""  # Return None if all retries fail


def queryLLM(query, examples, all_rels, err_file, out_file, head, tail, icl=True, ONT=True) :
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

    instr = f"Head: {head}, Tail: {tail}"
    #pdb.set_trace()
    messages.append({"role": "user", "content": query+"\n"+instr})
    if "gpt" in model_name:
        response = reqGpt(messages)
    else:
        response = reqTog(messages)
    #pdb.set_trace()
    if (response.choices[0].finish_reason != 'stop') or response == "" :
        with open(err_file, "a") as f:
            f.write(str(messages)+"\n")
            f.write("Response\n"+str(response)+"\n\n")
    
    if response != "":
        response = response.choices[0].message.content.strip()
    else:
        response = "NA"
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

MODEL_PATH = sys.argv[2]
ADAPTER_PATH = ""  # From OUT_DIR in training script
MAX_SEQ_LENGTH = 8192


run_id += "_" + MODEL_PATH.split("/")[-1].strip()

if ONT:
    run_id += "_ONT"

if ICL:
    run_id += "_ICL"



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
            #pdb.set_trace()
            if 'text' in data:
                sent_examples[text_norm(data["text"])] = data["top_docs"]
            else:
                try:
                    sent_examples[text_norm(data["query"]["text"])] = data["top_docs"]
                except:
                    sent_examples[text_norm(data["query"])] = data["top_docs"]

all_train_data = {}
with open(FULL_TRAIN_PATH, "r") as f:
    for line in f:
        data = json.loads(line)
        all_train_data[text_norm(data["text"])] = (data["h"]["pos"], data["t"]["pos"])
# print(run_id)
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
        #print(data)
        
        if ICL:
            bags = sent_examples[text_norm(data["text"])]
        try :
            query = replace_brackets_markers(data['text'], data["head"], data["tail"])
        except :
            try:
                query = add_double_brackets(data["text"], data["h"]["pos"], data["t"]["pos"])
            except:
                query = add_double_brackets(data["text"], data["hpos"], data["tpos"])
        examples = []
        if ICL:
            for bag in bags:
                bag_text = ""
                head = ""
                tail = ""
                for sent in bag["texts"]:
                    try:
                        #pdb.set_trace()
                        ht = all_train_data[text_norm(sent)]
                    except:
                        pdb.set_trace()
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
            try:
                head = data["text"][data["h"]["pos"][0]:data["h"]["pos"][1]]
                tail = data["text"][data["t"]["pos"][0]:data["t"]["pos"][1]]
            except:
                head = data["text"][data["hpos"][0]:data["hpos"][1]]
                tail = data["text"][data["tpos"][0]:data["tpos"][1]]
        response = queryLLM(query, examples, all_rels, err_file, out_file, head, tail, ONT=ONT)
