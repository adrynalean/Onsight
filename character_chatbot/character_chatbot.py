import pandas as pd
import torch
import re
import huggingface_hub
from datasets import Dataset
import transformers
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import gc

# Remove stage directions and actions from transcript e.g. "(laughing)"
def remove_parenthesis(text):
    result = re.sub(r'\(.*?\)', '', text)
    return result


def pick_compute_dtype():
    # bf16 needs Ampere+ (A100/L4/RTX 30xx). Free Kaggle/Colab T4s are Turing
    # and only support fp16, so fall back rather than crashing/running slow.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

class CharacterChatBot():

    def __init__(self,
                 model_path,
                 data_path=None,
                 huggingface_token=None
                 ):

        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        # Ungated mirror of Llama 3.1 8B Instruct (identical weights) so training
        # doesn't block on Meta's gated-repo approval. Swap back to
        # "meta-llama/Llama-3.1-8B-Instruct" once that access is granted.
        self.base_model_path = "NousResearch/Meta-Llama-3.1-8B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = pick_compute_dtype()

        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_model(self.model_path)
        else:
            if self.data_path is None:
                raise ValueError(
                    f"Model '{self.model_path}' was not found on HuggingFace Hub and "
                    "no data_path was provided for training. Pass data_path explicitly."
                )
            print("Model not found in HuggingFace Hub — training a new model.")
            train_dataset = self.load_data()
            self.train(self.base_model_path, train_dataset)
            self.model = self.load_model(self.model_path)

    def chat(self, message, history):
        messages = []
        messages.append({
            "role": "system",
            "content": (
                "You are Monkey D. Luffy from the anime \"One Piece\". "
                "Respond exactly as Luffy would: carefree, enthusiastic, and direct. "
                "You dream of becoming King of the Pirates. You care deeply about your crew and friends. "
                "You are fearless, a little oblivious to complex things, and always hungry. "
                "Keep responses short and energetic, true to Luffy's speech patterns.\n"
            )
        })

        # Accepts both Gradio 5 messages format (list of {"role","content"} dicts)
        # and legacy tuples format (list of [user_msg, assistant_msg] pairs).
        for entry in history:
            if isinstance(entry, dict):
                messages.append({"role": entry["role"], "content": entry["content"]})
            else:
                messages.append({"role": "user", "content": entry[0]})
                messages.append({"role": "assistant", "content": entry[1]})

        messages.append({"role": "user", "content": message})

        terminators = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output = self.model(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        output_message = output[0]['generated_text'][-1]
        return output_message

    def load_model(self, model_path):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch_dtype,
        )
        model_pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={
                "torch_dtype": self.torch_dtype,
                "quantization_config": bnb_config,
            }
        )
        return model_pipeline

    def train(self,
              base_model_name_or_path,
              dataset,
              output_dir="./results",
              per_device_train_batch_size=1,
              gradient_accumulation_steps=1,
              optim="paged_adamw_32bit",
              save_steps=200,
              logging_steps=10,
              learning_rate=2e-4,
              max_grad_norm=0.3,
              max_steps=300,
              warmup_ratio=0.3,
              lr_scheduler_type="constant",
              ):

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch_dtype,
        )
        use_bf16 = self.torch_dtype == torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM"
        )

        training_arguments = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            bf16=use_bf16,
            fp16=not use_bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
            report_to="none",
            max_seq_length=512,
            # "text" not "prompt": trl >=0.10 reserves prompt/completion column
            # names for its prompt-completion format and would expect a matching
            # "completion" column (KeyError otherwise). "text" = plain LM field.
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            args=training_arguments,
        )

        trainer.train()

        # Save checkpoint
        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")

        # Flush memory
        del trainer, model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Merge LoRA weights and push to Hub
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            return_dict=True,
            quantization_config=bnb_config,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, "final_ckpt")
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        del model, base_model
        gc.collect()

    def load_data(self):
        # One Piece transcript CSV — expects columns: 'character', 'text'.
        # reset_index after dropna is required: the prompt builder below uses
        # .iloc[ind - 1] (positional), so the index must stay contiguous or it
        # would pair Luffy's line with the wrong "previous" line.
        transcript_df = pd.read_csv(self.data_path)
        transcript_df = transcript_df.dropna(subset=['character', 'text']).reset_index(drop=True)
        transcript_df['text'] = transcript_df['text'].apply(remove_parenthesis)
        transcript_df['number_of_words'] = transcript_df['text'].str.strip().str.split()
        transcript_df['number_of_words'] = transcript_df['number_of_words'].apply(len)
        # Exact 'Luffy' match is intentional: the dataset also contains
        # 'Not Luffy' and ambiguous group labels ('Luffy & Usopp'), which a
        # substring match would wrongly include. Solo 'Luffy' lines ~= 10.8k.
        transcript_df['luffy_response_flag'] = 0
        transcript_df.loc[
            (transcript_df['character'] == 'Luffy') &
            (transcript_df['number_of_words'] > 5),
            'luffy_response_flag'
        ] = 1

        indexes_to_take = list(
            transcript_df[
                (transcript_df['luffy_response_flag'] == 1) &
                (transcript_df.index > 0)
            ].index
        )

        system_prompt = (
            "You are Monkey D. Luffy from the anime \"One Piece\". "
            "Respond exactly as Luffy would: carefree, enthusiastic, and direct. "
            "You dream of becoming King of the Pirates. You care deeply about your crew and friends. "
            "You are fearless, a little oblivious to complex things, and always hungry.\n"
        )

        prompts = []
        for ind in indexes_to_take:
            prompt = system_prompt
            prompt += transcript_df.iloc[ind - 1]['text']
            prompt += '\n'
            prompt += transcript_df.iloc[ind]['text']
            prompts.append(prompt)

        df = pd.DataFrame({"text": prompts})
        dataset = Dataset.from_pandas(df)
        return dataset
