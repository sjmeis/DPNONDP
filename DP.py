# class code for LLM-based Differential Privacy mechanisms

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList, pipeline
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

class ClipLogitsProcessor(LogitsProcessor):
  def __init__(self, min=-100, max=100):
    self.min = min
    self.max = max

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    scores = torch.clamp(scores, min=self.min, max=self.max)
    return scores
  
class DPPrompt():
    model_checkpoint = None
    min_logit = None
    max_logit = None
    sensitivity = None
    logits_processor = None
    pipe = None
    batch_size = None

    tokenizer = None
    model = None
    device = None

    def __init__(self, model_checkpoint="google/flan-t5-base", min_logit=-19.22705113016047, max_logit=7.48324937989716, batch_size=16, LLM=False):
        self.model_checkpoint = model_checkpoint

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        if LLM == True:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint, device_map="cuda")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint).to(self.device)

        self.min_logit = min_logit
        self.max_logit = max_logit
        self.sensitivity = abs(self.max_logit - self.min_logit)
        self.logits_processor = LogitsProcessorList([ClipLogitsProcessor(self.min_logit, self.max_logit)])

        if LLM == True:
            self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, truncation=True)
        else:
            self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device, truncation=True)
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

    def prompt_template_fn(self, doc):
        prompt = "Document: {}\nShort paraphrase of the document : ".format(doc)
        return prompt
    
    def privatize_dp(self, texts, epsilon=100):
        temperature = 2 * self.sensitivity / epsilon
        prompts = ListDataset([self.prompt_template_fn(text) for text in texts])
        private_texts = []
        for r in tqdm(self.pipe(prompts, do_sample=True, top_k=0, top_p=1.0, temperature=temperature, logits_processor=self.logits_processor, max_new_tokens=64, batch_size=self.batch_size), total=len(prompts)):
            private_texts.append(r[0]["generated_text"])
        return private_texts
    
    def privatize_noclip(self, texts, temperature):
        prompts = ListDataset([self.prompt_template_fn(text) for text in texts])
        private_texts = []
        for r in tqdm(self.pipe(prompts, do_sample=True, top_k=0, top_p=1.0, temperature=temperature, max_new_tokens=64, batch_size=self.batch_size), total=len(prompts)):
            private_texts.append(r[0]["generated_text"])
        return private_texts
    
    def privatize_noclip_topk(self, texts, k):
        prompts = ListDataset([self.prompt_template_fn(text) for text in texts])
        private_texts = []
        for r in tqdm(self.pipe(prompts, do_sample=True, top_k=k, top_p=1.0, temperature=1.0, max_new_tokens=64, batch_size=self.batch_size), total=len(prompts)):
            private_texts.append(r[0]["generated_text"])
        return private_texts