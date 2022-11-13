
# # Conditional Generation with Prefix Tuning.
# In this tutorial, we do conditional generation with prefix tuning template.

# we use WebNLG as an example, as well. Note that the evaluation of generation result should be done
# by using the scripts provided by https://github.com/Yale-LILY/dart/tree/master/evaluation,
# Which we do not include in it.

import argparse
import torch
from openprompt.data_utils import InputExample

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", default=True)
parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='t5-base')
args = parser.parse_args()
print(args)

from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor
dataset = {}
dataset["train"] = []

 #text a is the input sentence and then target text is the simplified
# input_example = InputExample(text_a = data['text_a'], tgt_text =data['tgt_text'], label=None, guid=data['guid'])

#dummy input example
data = {}
data["text_a"] = "hello"
data["tgt_text"] = "bye"
data["guid"] = 0
input_example = InputExample(text_a = data['text_a'], tgt_text =data['tgt_text'], label=None, guid=data['guid'])
# input_example = InputExample(text_a = "admission to tsinghua is extremely competitive .", tgt_text ="getting in to tsinghua is very hard .", label=None, guid=0)
dataset["train"] = [input_example]

# load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

# Instantiating the PrefixTuning Template !
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text=' {"placeholder":"text_a"} simplify this text {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)

# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.
from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=5,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")

# load the pipeline model PromptForGeneration.
from openprompt import PromptForGeneration
use_cuda = False
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=True)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric
# Define evaluate function
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    for step, inputs in enumerate(dataloader):
        print(inputs)
        breakpoint()
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        breakpoint()
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
    "bad_words_ids": [[628], [198]],
}

generated_sentence = evaluate(prompt_model, train_dataloader)

with open("testing_generation.txt",'w') as f:
    for i in generated_sentence:
        f.write(i+"\n")

