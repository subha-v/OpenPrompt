import argparse
import torch
from openprompt.data_utils import InputExample
from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt import PromptDataLoader
from openprompt import PromptForGeneration
from openprompt.utils.metrics import generation_metric

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", default=True)
parser.add_argument("--model_name_or_path", default='t5-base')

dataset = {}
dataset["train"] = []

use_cuda = False

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

from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")

mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text='Simplify this text: {"placeholder":"text_a"}  {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)

# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=1,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
    truncate_method="tail")

prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=True)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# Train the model
# Define evaluate function
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    for step, inputs in enumerate(dataloader):
        if use_cuda:
             inputs = inputs.cuda() 
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        print("_: ", _)
        print("converted ids to tokens:", tokenizer.convert_ids_to_tokens(_[0]))
        print("output_sentence:",  output_sentence)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
        
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
print('generated sentence: ', generated_sentence)
#with open("testing_generation.txt",'w') as f:
#    for i in generated_sentence:
#        f.write(i+"\n")
