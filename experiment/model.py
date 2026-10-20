import torch, sentencepiece as spm
from xlstm_ptbr.models.xlstm.lm import XlstmLmConfig, XlstmLmHeadModel
from xlstm_ptbr.models.transformer.lm import TransformerLmConfig, TransformerLmHeadModel

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Model:
    def __init__(self, model_path, tokenizer_path, backend):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("⏳ Loading model")
        if backend == "xlstm":
            self.load_xlstm(model_path)
        
        elif backend == "transformer":
            self.load_transformer(model_path)
        
        else:
            raise ValueError("Incompatible backend")
        
        self.model.to(self.device, dtype=torch.bfloat16).eval()

        print("⏳ Loading tokenizer")
        self.load_tokenizer(tokenizer_path)
        
        
                

    def load_xlstm(self, model_path):
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        cfg = XlstmLmConfig(**{**ck["config"]["model"], "mode": "inference"})  # generate() exige modo de inferência
        self.model = XlstmLmHeadModel(cfg)
        self.model.load_state_dict(ck["model"])


    def load_transformer(self, model_path):
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        cfg = TransformerLmConfig(**{**ck["config"]["model"], "activation_checkpointing": False})
        self.model = TransformerLmHeadModel(cfg)
        self.model.load_state_dict(ck["model"])


    def load_tokenizer(self, tokenizer_path):
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

    
    def tokenize_prompt(self, prompt):
        batch = [self.tokenizer.encode(prompt)]
        ids = torch.tensor(batch, dtype=torch.long, device=self.device)
        return ids

    @torch.inference_mode()
    def make_inference(self, prompt, temperature = 0.0, max_new_tokens = 20):
        ids = self.tokenize_prompt(prompt)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = self.model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature)
        
        generated_ids = out[0, ids.shape[1]:] # Remove tokens from original prompt
        decoded = self.tokenizer.decode(generated_ids.tolist())
        return decoded



class ModelHF:
    def __init__(self, model_url:str, tokenizer_path:None|str=None, quantization:bool=False):
        if tokenizer_path is None:
            tokenizer_path = model_url
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.model = self.load_model(model_url, quantization)
        self._is_model_evaluating = False
        self.set_evaluation_mode()
    
    def load_tokenizer(self, model_url):
        tokenizer = AutoTokenizer.from_pretrained(
                model_url,
                use_fast=False, 
                trust_remote_code=True
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        return tokenizer
    
    
    def load_model(self, model_url, quantization):
    
        model = AutoModelForCausalLM.from_pretrained(
            model_url,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 
        )
    
        return model
    
    
    def set_evaluation_mode(self):
        if not self._is_model_evaluating:
            self.model.eval()
            self._is_model_evaluating = True
    
    def set_train_mode(self):
        if self._is_model_evaluating:
            self.model.train() 
            self._is_model_evaluating = False

    def tokenize_prompt(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        return inputs


    def make_inference(self, prompt, max_new_tokens):
        inputs = self.tokenize_prompt(prompt)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
        generated_ids = out[0, inputs["input_ids"].shape[1]:]
    
        out = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return out.strip()

