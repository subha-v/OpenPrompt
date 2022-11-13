
from openprompt.plms import load_plm
from openprompt.data_utils import InputExample
import pandas as pd
from scipy.stats import entropy
import torch.nn as nn
classes = [ 
    "negative",
    "neutral",
    "positive"
]

dframe = pd.read_csv("data/human/sentiment-analysis/lalor/sa_lalor_human.csv")
gold_labels = []
dataset ={
    'train' : [],
    'val'   : [],
    'test'  : []
}
for (index, row) in dframe.iterrows():
    exmp = InputExample(
        guid   = row["sample_id"],
        text_a = row["content"],
        label  = int(row['three_way_labels']) + 1
    )
    dataset['train'].append(exmp)
    gold_labels.append(row['three_way_labels'])


# You can load the plm related things provided by openprompt simply by calling:
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")


# Constructing Template
# A template can be constructed from the yaml config, but it can also be constructed by directly passing arguments.
from openprompt.prompts import ManualTemplate

template_text ='Complex Sentence: {"placeholder":"text_a"}. Simplified Sentence: {"mask"}.' 
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

from openprompt.prompts import ManualVerbalizer
import torch

from openprompt.utils.metrics import generation_metric
# Define evaluate function
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("test_score", score, flush=True)
    return generated_sentence

generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": [[628], [198]]
}



generated_sentence = evaluate(prompt_model, test_dataloader)

with open(f"../../Generated_sentence_webnlg_gpt2_{args.plm_eval_mode}.txt",'w') as f:
    for i in generated_sentence:
        f.write(i+"\n")

