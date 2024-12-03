import contractions
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import os
from flask import Flask, jsonify, request, render_template, send_from_directory
import json
from flask import request
from flask_cors import CORS, cross_origin
import re
import torch
import pickle
import pronouncing
import sys
import itertools
import gc
from torch import cuda
import time
import threading

from datetime import datetime


import nltk

USE_CUDA = False

# Code to expand contractions as the voice model will fail on some contractions
# Code from https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/


def expandContractions(text):

    # creating an empty list
    expanded_words = []
    for word in text.split():
        # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))

    expanded_text = ' '.join(expanded_words)
    return expanded_text

# Original Lyricgen code below - some alterations in post text generation handling


def findsubsets(s, n):
    return list(itertools.combinations(s, n))


os.environ['TRANSFORMERS_CACHE'] = './transformers_cache/'

dict_file = open("all_words.txt", 'r')
dictionary = re.sub("[^\w]", " ",  dict_file.read()).split()
model_names = ['laurie', 'lou', 'laurie_lou', 'laurie_prose_stage_1', 'laurie_prose', 'laurie_prose_stage_1',
               'lou_prose', 'laurie_prose_stage_1', 'bible_prose_0', 'bible_prose_2', 'bible_prose_0_gpu', 'bible_prose_2_gpu']
model_data = {
    'laurie': ('models/laurie', 0, 'rhymes/laurie_rhymes.pkl'),
    'lou': ('models/lou', 0, 'rhymes/lou_rhymes.pkl'),
    'laurie_lou': ('models/laurie_lou', 1, 'rhymes/laurie_lou_rhymes.pkl'),
    'laurie_prose_stage_1': ('models/laurie_prose_1', 0),
    'laurie_prose': ('models/laurie_prose_2', 0),
    'bible_prose_0': ('models/bible_prose_0', 0),
    'bible_prose_0_gpu': ('models/bible_prose_0', 1),
    'bible_prose_2': ('models/bible_prose_2', 0),
    'bible_prose_2_gpu': ('models/bible_prose_2', 1),
    'lou_prose_stage_1': ('models/lou_prose_1', 1),
    'lou_prose': ('models/lou_prose_2', 1),
}

models = {}

rhyme_set_names = ['laurie', 'lou', 'laurie_lou']
rhyme_data = {}

rhyme_thresh = 4
end_freq_thresh = 2

for rhyme_set_name in rhyme_set_names:
    rhyme_path = model_data[rhyme_set_name][2]
    end_frequencies, rhyme_pairs = pickle.load(open(rhyme_path, "rb"))
    rhyme_pairs = [i for i in rhyme_pairs if i[2] >= rhyme_thresh]
    rhyme_data[rhyme_set_name] = (end_frequencies, rhyme_pairs)

generated_text = ""
keywords = ""
tokenizer = BartTokenizer.from_pretrained("./bart_tokenizer")
bible_tokenizer = BartTokenizer.from_pretrained("./bart_tokenizer")

new_tokens = ["<v>"]
num_added_toks = bible_tokenizer.add_tokens(new_tokens)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
max_len = 180
num_verses = 4
random_words = open("random_words.txt", "r").read().split()
bad_words = open("bad-words.txt", "r").read().split()
rhyme_chance = 0.75


def encode_input(input, tokenizer, use_gpt=False):
    if use_gpt:
        print("encode gpt2")
        input_ids = gpt2_tokenizer.encode(
            input, return_tensors='pt', pad_to_max_length=True,)
        return torch.unsqueeze(input_ids.squeeze().to(dtype=torch.long), 0)
    else:
        source = tokenizer.batch_encode_plus(
            [input], max_length=max_len, pad_to_max_length=True, return_tensors='pt')
    source_ids = torch.unsqueeze(
        source['input_ids'].squeeze().to(dtype=torch.long), 0)
    source_mask = torch.unsqueeze(
        source['attention_mask'].squeeze().to(dtype=torch.long), 0)
    return source_ids, source_mask


def thread_function(input, model_id, tokenizer, device_id, result, index, num_beams=6):
    source_ids, source_mask = encode_input(input, tokenizer, False)
    # device = f'cuda:{device_id}'
    device = torch.device("cuda" if USE_CUDA else "cpu")
    if(device == "cpu"):
        print("using CPU")
    attention_mask = source_mask.to(device, dtype=torch.long)
    outputs = models[model_id].generate(
        input_ids=source_ids.to(device, dtype=torch.long),
        attention_mask=attention_mask,
        max_length=1024,
        num_beams=num_beams,
        num_return_sequences=1,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        do_sample=True,
        top_k=10,
        top_p=0.95
    )

    outputs = outputs.detach()
    output = tokenizer.decode(
        outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    del outputs
    result[index] = output.replace("\"", "").replace(
        "~", "").replace("—", " ").replace("(", "").replace(")", "")


def run_model_thread(model_inputs, model_names, tokenizer, replace_newlines=False, do_sample=True, num_beams=6,):
    assert(len(model_inputs) == len(model_names))
    outputs = [None]*len(model_inputs)
    _threads = [None]*len(model_inputs)

    for idx, (model_input, model_name) in enumerate(zip(model_inputs, model_names)):
        if model_name not in models:
            print(f"load thread model {model_name}")
            models[model_name] = BartForConditionalGeneration.from_pretrained(
                model_data[model_name][0]).to(model_data[model_name][1]).eval()
            models[model_name].resize_token_embeddings(len(tokenizer))
        _threads[idx] = threading.Thread(target=thread_function, args=(
            model_input, model_name, tokenizer, model_data[model_name][1], outputs, idx), daemon=True)
        _threads[idx].start()

    for thread in _threads:
        thread.join()
    output = "|".join(outputs)
    return output


def run_model(model_input, model_id, tokenizer, replace_newlines=False, do_sample=True, num_beams=4,):
    # , loaded = {model_loaded[model_id]}",flush=True)
    print(f"Run with model {model_id}", flush=True)
    device_id = model_data[model_id][1]
    # device = f'cuda:{device_id}'
    device = torch.device("cuda" if USE_CUDA else "cpu")
    if(device == "cpu"):
        print("using CPU")
    if model_id not in models:
        print(f"load model... {model_data[model_id][0]}", flush=True)
        print(model_id)
        models[model_id] = BartForConditionalGeneration.from_pretrained(
            model_data[model_id][0]).to(device).eval()

        if "bible_prose" in model_id:
            print("expand token embeddings")
            models[model_id].resize_token_embeddings(len(tokenizer))
        gc.collect()
        print("...done")

    source_ids, source_mask = encode_input(model_input, tokenizer, False)

    print(f"Do sample? {do_sample}")
    with torch.no_grad():
        if do_sample:
            attention_mask = source_mask.to(device, dtype=torch.long)
            outputs = models[model_id].generate(
                input_ids=source_ids.to(device, dtype=torch.long),
                attention_mask=attention_mask,
                max_length=1024,
                num_beams=num_beams,
                num_return_sequences=1,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=True,
                top_k=10,
                top_p=0.95,
            )
        else:
            outputs = models[model_id].generate(
                input_ids=source_ids.to(device, dtype=torch.long),
                attention_mask=source_mask.to(device, dtype=torch.long),
                max_length=1024,
                num_beams=num_beams,
                num_return_sequences=1,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )
        outputs = outputs.detach()
        output = tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        del outputs
        output = output.replace("\"", "").replace("~", "").replace(
            "—", " ").replace("(", "").replace(")", "")
        if replace_newlines:
            output = "\n".join([i.strip() for i in output.split("|")])
        gc.collect()
    return output


def generate_lyrics(input, model_id):
    print("Generating...")
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        USE_CUDA = False
        # sys.exit()
    else:
        USE_CUDA = True
        print("CUDA AVAILABLE")
    try:
        print(f"Input = {input}")
        if model_id in rhyme_set_names:
            start = time.time()
            input = input.split()
            input_len = len(input)
            input_lower = [i.lower() for i in input]
            if model_id == 'laurie':
                min_rhymes = max(int(0.2*input_len), 2)
                max_rhymes = max(int(0.4*input_len), 6)
            elif model_id == 'lou':
                min_rhymes = max(int(0.15*input_len), 1)
                max_rhymes = max(int(0.4*input_len), 5)
            elif model_id == 'laurie_lou':
                min_rhymes = max(int(0.1*input_len), 1)
                max_rhymes = max(int(0.4*input_len), 4)
            num_rhymes = random.randint(min_rhymes, max_rhymes)
            possible_rhymes = []
            rhymes_added = set()
            for word in input:
                for rhyme in pronouncing.rhymes(word):
                    rhyme_pair = "_".join(sorted([word, rhyme]))
                    if rhyme_pair in rhymes_added:
                        continue
                    if rhyme in input_lower:
                        possible_rhymes.append([word, rhyme, -1])
                        rhymes_added.add(rhyme_pair)
                    elif rhyme in rhyme_data[model_id][1]:
                        freq = rhyme_data[model_id][1][rhyme]
                        if freq > end_freq_thresh:
                            possible_rhymes.append([word, rhyme, freq])
                            rhymes_added.add(rhyme_pair)
            if len(possible_rhymes) > 0:
                max_freq = max([i[2] for i in possible_rhymes])
                for idx, p in enumerate(possible_rhymes):
                    if p[2] == -1:
                        p[2] = max_freq
                    p[2] /= max_freq
                    possible_rhymes[idx] = p
            rhyme_pairs_copy = rhyme_data[model_id][1].copy()
            chosen_rhymes = []
            for i in range(num_rhymes):
                if len(possible_rhymes) > 0 and random.random() < rhyme_chance:
                    rhymes = [[p[0], p[1]] for p in possible_rhymes]
                    weights = [p[2] for p in possible_rhymes]
                    cur_rhyme = random.choices(rhymes, weights=weights, k=1)[0]
                    possible_rhymes = [p for p in possible_rhymes if (
                        p[0] not in cur_rhyme and p[1] not in cur_rhyme)]
                    chosen_rhymes.append(cur_rhyme)
                else:
                    rhymes = [[p[0], p[1]] for p in rhyme_pairs_copy]
                    weights = [p[2] for p in rhyme_pairs_copy]
                    cur_rhyme = random.choices(rhymes, weights=weights, k=1)[0]
                    rhyme_pairs_copy = [p for p in rhyme_pairs_copy if (
                        p[0] not in cur_rhyme and p[1] not in cur_rhyme)]
                    chosen_rhymes.append(cur_rhyme)
            rhyme_words = []
            for pair in chosen_rhymes:
                rhyme_words.extend(pair)
            remaining_input_words = []
            for word in input:
                if word in rhyme_words:
                    rhyme_words.remove(word)
                else:
                    remaining_input_words.append(word)
            all_words = remaining_input_words
            for pair in chosen_rhymes:
                random.shuffle(pair)
                all_words.append(f"{pair[0]} | {pair[1]} |")
            random.shuffle(all_words)
            input = " ".join(all_words)
            print(f"build rhymes time = {time.time()-start}")
        elif 'prose' in model_id and 'bible' not in model_id:
            input_lower = [i.lower() for i in input.split()]
            output = run_model(input, model_id+'_stage_1',
                               tokenizer, do_sample=True, num_beams=2)
            output = output.replace(" i ", " I ")
            print(f"Pre-filter output = {output}")
            output = " ".join(i for i in output.split() if (i.isalpha() and i.lower(
            ) not in bad_words and (i.lower() in dictionary or i.lower() in input_lower)))
            input = output.strip()
            print(f"Stage 1 output = {input}")
        start = time.time()

        if 'bible_prose' in model_id and '|' in input:
            input_pair = input.split('|')
            output = run_model_thread(
                input_pair, (model_id, model_id+'_gpu'), bible_tokenizer)
        else:
            output = run_model(input, model_id, bible_tokenizer,
                               do_sample=True, num_beams=8,)

        print(f"time to generate = {time.time()-start}", flush=True)
        # Change these for appearance and for voice model
        output = output.replace(" i ", " I ")
        output = output.replace(" vs ", " versus ")
        output = output.replace(" vs. ", " versus ")
        output = output.replace(" vs-", " versus")
        output = output.replace(" Vs ", " versus ")
        output = output.replace(" Vs. ", " versus ")
        output = output.replace(" Vs- ", " versus ")
        # Expand all contractions as voice model fails on some contractions
        output = expandContractions(output)

        output = " ".join(i for i in output.split()
                          if i.lower() not in bad_words)
        if model_id in rhyme_set_names:
            output = "\n".join(i.strip() for i in output.split("|"))
        print(f"Output = {output}", flush=True)
    except Exception as e:
        print(e)
        return "Generation failed, try a different input :("
    return output
