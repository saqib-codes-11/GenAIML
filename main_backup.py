from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from starlette.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import io
import uuid
import urllib
import urllib.parse
import shutil
import os
import glob
import tempfile
import numpy as np
import torch
import sys

##for synthesis
from synthesis.vocoders import Hifigan
import synthesize

###############dans edits
#from moviepy.editor import AudioFileClip, ImageClip, TextClip, CompositeVideoClip
from moviepy.editor import *



def add_static_image_to_audio(image_path, audio_path, output_path):
    """Create and save a video file to `output_path` after
    combining a static image that is located in `image_path`
    with an audio file in `audio_path`"""
    # create the audio clip object
    audio_clip = AudioFileClip(audio_path)
    # create the image clip object
    image_clip = ImageClip(image_path)
    # use set_audio method from image clip to combine the audio with the image
    video_clip = image_clip.set_audio(audio_clip)
    # specify the duration of the new clip to be the duration of the audio clip
    video_clip.duration = audio_clip.duration
    # set the FPS to 1
    video_clip.fps = 1
    # write the resuling video clip
    video_clip.write_videofile(output_path)


######################################

#if not torch.cuda.is_available():
#    print("NO CUDA AVAILABLE!!!")
#    sys.exit(-99)
#else:
#    cudaOK = str(torch.cuda.is_available())
#    print(f"CUDA is available? {cudaOK}")

    #for devnum in range(torch.cuda.device_count()):
    #    dev_name = torch.cuda.get_device_name(devnum)
    #    print(f"Using Device {dev_name}")

import e2e_tts_demo as speak
from playsound import playsound
import sys
import librosa
from pysndfx import AudioEffectsChain
import string
import soundfile as sf
from fastapi.responses import StreamingResponse
from uuid import uuid4
templates = Jinja2Templates(directory="templates")



# Salt to your taste
ALLOWED_ORIGINS = "*"  # or 'foo.com', etc.

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.mount("/static", StaticFiles(directory="static"), name="static")


def get_model(td: Path, lat: float, lng: float, radius: int):
    pass


CONSTANT_VERSION_NUMBER = os.environ.get("VERSION", "DEV")


SRATE = 48000

@app.get("/singleline", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "version": CONSTANT_VERSION_NUMBER},
    )


#@app.get("/", response_class=HTMLResponse)
#async def read_item(request: Request):
#    return templates.TemplateResponse(
#        "multiline.html",
#        {"request": request, "version": CONSTANT_VERSION_NUMBER},
#    )

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        "multiline.html",
        {"request": request, "version": CONSTANT_VERSION_NUMBER},
    )

MAX_LENGTH = 20

working = os.path.dirname(os.path.realpath(__file__))

wav_files = glob.glob(f"{working}/*.wav")
for wav in wav_files:
    os.unlink(wav)

MAX_CHARACTER_COUNT = 1000
all_files = []

@app.get("/query")
async def query_text2speech(
    request: Request,
    text: str = "My name is Laurie."
):
    _text = urllib.parse.unquote(text)
    _text = _text[:MAX_CHARACTER_COUNT]
    #data, srate = speak.textToSound(_text)
    temp_file = f"{uuid.uuid4()}.wav"
    #temp_file2 = f"{uuid.uuid4()}.mp4"

    model = synthesize.load_model('LaurieAudiobook117000')
    vocoder = Hifigan('hifigan/model.pt', 'hifigan/config.json')
    file = f"webout.wav"
    data, srate = synthesize.synthesize(
       model=model,
        text=_text,
        vocoder=vocoder,
        audio_path='webout.wav'
    )

    #if srate != SRATE:
    #    data = librosa.core.resample(data, srate, SRATE)
    #sf.write(f'{file}', data, SRATE, 'PCM_24')

    ##danstuff
   # add_static_image_to_audio("./bg.jpeg", temp_file, temp_file2)

    # Generate a text clip
    #txt_clip = TextClip("GeeksforGeeks", fontsize = 14, color = 'white')

# setting position of text in the center and duration will be 10 seconds
   # txt_clip = txt_clip.set_pos('center').set_duration(10)

# Overlay the text clip on the first video clip
   # video = CompositeVideoClip([temp_file2, txt_clip])

  #  fh = open(temp_file2,"rb")
  #  all_files.append(temp_file)
  #  all_files.append(temp_file2)
  #  while len(all_files) > MAX_LENGTH:
  #      garbage = all_files.pop(0)
  #      os.unlink(garbage)



    return FileResponse(file, media_type="audio/wav")

    #####################################
    # LYRICGEN #

    import os
from flask import Flask, jsonify, request
import json
from flask import request
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

def findsubsets(s, n):
    return list(itertools.combinations(s, n))
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache/'
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel

dict_file = open("all_words.txt",'r')
dictionary = re.sub("[^\w]", " ",  dict_file.read()).split()
model_names = ['laurie', 'lou', 'laurie_lou','laurie_prose_stage_1','laurie_prose','laurie_prose_stage_1']
model_data = {
    'laurie':('models/laurie',0,'rhymes/laurie_rhymes.pkl'),
    'lou':('models/lou',0,'rhymes/lou_rhymes.pkl'),
    'laurie_lou':('models/laurie_lou',1,'rhymes/laurie_lou_rhymes.pkl'),
    'laurie_prose_stage_1':('models/laurie_prose_1',0),
    'laurie_prose':('models/laurie_prose_1',0),
    'bible_prose_0':('models/bible_prose_0',0),
    'bible_prose_0_gpu':('models/bible_prose_0',1),
    'bible_prose_2':('models/bible_prose_2',0),
    'bible_prose_2_gpu':('models/bible_prose_2',1),
    'lou_prose_stage_1':('models/lou_prose_1',1),
    'lou_prose':('models/lou_prose_2',1),
}

models = {}

rhyme_set_names = ['laurie','lou','laurie_lou']
rhyme_data = {}

rhyme_thresh = 4
end_freq_thresh = 2

for rhyme_set_name in rhyme_set_names:
    rhyme_path = model_data[rhyme_set_name][2]
    end_frequencies,rhyme_pairs = pickle.load(open(rhyme_path,"rb"))
    rhyme_pairs = [i for i in rhyme_pairs if i[2] >= rhyme_thresh]
    rhyme_data[rhyme_set_name] = (end_frequencies,rhyme_pairs)

generated_text = ""
keywords = ""
import random
tokenizer = BartTokenizer.from_pretrained("./bart_tokenizer")
bible_tokenizer = BartTokenizer.from_pretrained("./bart_tokenizer")

new_tokens = ["<v>"]
num_added_toks = bible_tokenizer.add_tokens(new_tokens)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
max_len = 180
num_verses = 4
random_words = open("random_words.txt","r").read().split()
bad_words = open("bad-words.txt","r").read().split()
rhyme_chance = 0.75

def encode_input(input,tokenizer,use_gpt=False):
    if use_gpt:
        print("encode gpt2")
        input_ids = gpt2_tokenizer.encode(input,return_tensors='pt',pad_to_max_length=True,)
        return torch.unsqueeze(input_ids.squeeze().to(dtype=torch.long),0)
    else:
        source = tokenizer.batch_encode_plus([input], max_length=max_len, pad_to_max_length=True,return_tensors='pt')
    source_ids = torch.unsqueeze(source['input_ids'].squeeze().to(dtype=torch.long),0)
    source_mask = torch.unsqueeze(source['attention_mask'].squeeze().to(dtype=torch.long),0)
    return source_ids, source_mask

def thread_function(input,model_id,tokenizer,device_id,result,index,num_beams=6):
    source_ids, source_mask = encode_input(input,tokenizer,False)
    device = f'cuda:{device_id}'
    attention_mask = source_mask.to(device, dtype = torch.long)
    outputs = models[model_id].generate(
        input_ids = source_ids.to(device, dtype = torch.long),
        attention_mask = attention_mask,
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
    output = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    del outputs
    result[index] = output.replace("\"","").replace("~","").replace("—"," ").replace("(","").replace(")","")

def run_model_thread(model_inputs,model_names,tokenizer,replace_newlines=False,do_sample=True,num_beams=6,):
    assert(len(model_inputs) == len(model_names))
    outputs = [None]*len(model_inputs)
    _threads = [None]*len(model_inputs)

    for idx,(model_input,model_name) in enumerate(zip(model_inputs,model_names)):
        if model_name not in models:
           print(f"load thread model {model_name}")
           models[model_name] = BartForConditionalGeneration.from_pretrained(model_data[model_name][0]).to(model_data[model_name][1]).eval()
           models[model_name].resize_token_embeddings(len(tokenizer))
        _threads[idx] = threading.Thread(target=thread_function, args=(model_input,model_name,tokenizer,model_data[model_name][1],outputs,idx), daemon=True)
        _threads[idx].start()

    for thread in _threads:
        thread.join()
    output = "|".join(outputs)
    return output


def run_model(model_input,model_id,tokenizer,replace_newlines=False,do_sample=True,num_beams=4,):
    print(f"Run with model {model_id}",flush=True)#, loaded = {model_loaded[model_id]}",flush=True)
    device_id = model_data[model_id][1]
    device = f'cuda:{device_id}'
    if model_id not in models:
        print(f"load model... {model_data[model_id][0]}",flush=True)
        print(model_id)
        models[model_id] = BartForConditionalGeneration.from_pretrained(model_data[model_id][0]).to(device).eval()

        if "bible_prose" in model_id:
            print("expand token embeddings")
            models[model_id].resize_token_embeddings(len(tokenizer))
        gc.collect()
        print("...done")

    source_ids, source_mask = encode_input(model_input,tokenizer,False)

    print(f"Do sample? {do_sample}")
    with torch.no_grad():
        if do_sample:
            attention_mask = source_mask.to(device, dtype = torch.long)
            outputs = models[model_id].generate(
                input_ids = source_ids.to(device, dtype = torch.long),
                attention_mask = attention_mask,
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
                input_ids = source_ids.to(device, dtype = torch.long),
                attention_mask = source_mask.to(device, dtype = torch.long),
                max_length=1024,
                num_beams=num_beams,
                num_return_sequences=1,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                )
        outputs = outputs.detach()
        output = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        del outputs
        output = output.replace("\"","").replace("~","").replace("—"," ").replace("(","").replace(")","")
        if replace_newlines:
            output = ".<br>".join([i.strip() for i in output.split("|")])
        gc.collect()
    return output

def generate_lyrics(input,model_id):
    print("Generating...")
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        sys.exit()
    print("CUDA AVAILABLE")

    try:
        print(f"Input = {input}")
        if model_id in rhyme_set_names:
            print("Model is in rhyme set name")
            start = time.time()
            input = input.split()
            input_len = len(input)
            input_lower = [i.lower() for i in input]
            if model_id == 'laurie':
                min_rhymes = max(int(0.2*input_len),2)
                max_rhymes = max(int(0.4*input_len),6)
           # elif model_id == 'lou':
           #     min_rhymes = max(int(0.15*input_len),1)
           #     max_rhymes = max(int(0.4*input_len),5)
           # elif model_id == 'laurie_lou':
           #     min_rhymes = max(int(0.1*input_len),1)
           #     max_rhymes = max(int(0.4*input_len),4)
            num_rhymes = random.randint(min_rhymes,max_rhymes)
            possible_rhymes = []
            rhymes_added = set()
            for word in input:
                for rhyme in pronouncing.rhymes(word):
                    rhyme_pair = "_".join(sorted([word,rhyme]))
                    if rhyme_pair in rhymes_added:
                        continue
                    if rhyme in input_lower:
                        possible_rhymes.append([word,rhyme,-1])
                        rhymes_added.add(rhyme_pair)
                    elif rhyme in rhyme_data[model_id][1]:
                        freq = rhyme_data[model_id][1][rhyme]
                        if freq > end_freq_thresh:
                            possible_rhymes.append([word,rhyme,freq])
                            rhymes_added.add(rhyme_pair)
            if len(possible_rhymes) > 0:
                max_freq = max([i[2] for i in possible_rhymes])
                for idx,p in enumerate(possible_rhymes):
                    if p[2] == -1:
                        p[2] = max_freq
                    p[2] /= max_freq
                    possible_rhymes[idx] = p
            rhyme_pairs_copy = rhyme_data[model_id][1].copy()
            chosen_rhymes = []
            for i in range(num_rhymes):
                if len(possible_rhymes) > 0 and random.random() < rhyme_chance:
                    rhymes = [[p[0],p[1]] for p in possible_rhymes]
                    weights = [p[2] for p in possible_rhymes]
                    cur_rhyme = random.choices(rhymes,weights=weights,k=1)[0]
                    possible_rhymes = [p for p in possible_rhymes if (p[0] not in cur_rhyme and p[1] not in cur_rhyme)]
                    chosen_rhymes.append(cur_rhyme)
                else:
                    rhymes = [[p[0],p[1]] for p in rhyme_pairs_copy]
                    weights = [p[2] for p in rhyme_pairs_copy]
                    cur_rhyme = random.choices(rhymes,weights=weights,k=1)[0]
                    rhyme_pairs_copy = [p for p in rhyme_pairs_copy if (p[0] not in cur_rhyme and p[1] not in cur_rhyme)]
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
            output = run_model(input,model_id+'_stage_1',tokenizer,do_sample=True,num_beams=2)
            output = output.replace(" i "," I ")
            print(f"Pre-filter output = {output}")
            output = " ".join(i for i in output.split() if (i.isalpha() and i.lower() not in bad_words and (i.lower() in dictionary or i.lower() in input_lower)))
            input = output.strip()
            print(f"Stage 1 output = {input}")
        start = time.time()

        if 'bible_prose' in model_id and '|' in input:
            input_pair = input.split('|')
            output = run_model_thread(input_pair,(model_id,model_id+'_gpu'),bible_tokenizer)
        else:
            output = run_model(input,model_id,bible_tokenizer,do_sample=True,num_beams=8,)

        print(f"time to generate = {time.time()-start}",flush=True)
        output = output.replace(" i "," I ")
        #output = output.replace("'","")
        output = " ".join(i for i in output.split() if i.lower() not in bad_words)
        print(model_id)
        print(rhyme_set_names)
        if model_id in rhyme_set_names:
            output = ".<br>".join(i.strip() for i in output.split("|"))
        print(f"Output = {output}",flush=True)
    except Exception as e:
        print(e)
        return "Generation failed, try a different input :("
    return output

@app.post("/daniel")
async def daniel():
    print("Daniel")

@app.post("/generate")

async def generate(request:Request):
    print("Made it thios far! Starting generate post")
    data = await request.json()
    model_id = data['model']
    keywords = data['keywords']
    try:
        #data = tempJSON
        data = await request.json()
        print(data)
        result = ""

        model_id = data['model']
        keywords = data['keywords']
        print("model id",model_id)
        print("keywords",keywords)
        if model_id not in model_data:
            result = "Model not found"
        else:
            print(f"Selected model = {model_id}",flush=True)
            input = (" ".join(i for i in keywords.split() if i.lower() not in bad_words)).strip()
            result = generate_lyrics(input,model_id)
    except Exception as e:
        print(e)
        result = "Generation failed, try a different input :("
    return result


def numWords(string):
    split = string.split(" ")
    return len(split)

@app.post("/synthesise")

async def synth(request:Request):
    data = await request.json()
    _text = data["text"]
    lines = _text.split("<br>")

    print("there are # lines:", len(lines))
    model = synthesize.load_model('LaurieAudiobook117000')
    vocoder = Hifigan('hifigan/model.pt', 'hifigan/config.json')
    file = f"webout.wav"
    for i in range(len(lines)):
        if(i +1< len(lines)):
            data, srate = synthesize.synthesize(
                model=model,
                text=lines[i],
                vocoder=vocoder,
                audio_path='webout'+str(i)+'.wav',
                max_decoder_steps=1000
                )

#if __name__ == '__main__':
#    print("Starting...")
#    if not torch.cuda.is_available():
#        print("CUDA NOT AVAILABLE")
#        sys.exit()
#    print("CUDA AVAILABLE")
#    app.run(host='0.0.0.0', port=5000, debug=False)
    #app.run(host='127.0.0.1', port=5000, debug=False)



