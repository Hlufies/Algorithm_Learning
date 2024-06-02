# -*- coding: utf-8 -*-
# time: 2023/6/1 17:19
# file: train_qlora.py
# author: zmfy
# email: shuxueslpi@163.com
from torch.utils.data import ConcatDataset
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import DatasetDict
from utils.prompter import Prompter
import torch.nn as nn
import argparse
from typing import List, Dict, Optional

import torch
from loguru import logger
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    LlamaForCausalLM,
    HfArgumentParser,
    set_seed,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}
# If you don't want your script to sync to the cloud
os.environ['WANDB_MODE'] = 'dryrun'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description='LLAMA Finetuning')
    parser.add_argument("--train_on_inputs", type= bool, default= True)
    parser.add_argument('--model_name_or_path', type=str, default='/newdata/yahmallama-7b-hf', help='模型id或local path')
    parser.add_argument('--train_args_json', type=str, required=True, help='')

    parser.add_argument('--train_clean_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--train_poison_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--val_clean_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--val_poison_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lora_rank', type=int, default=4, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora_alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='恢复训练的checkpoint路径')
    parser.add_argument('--compute_dtype', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16'], help='计算数据类型')
    
    parser.add_argument('--prompt_template_name', type=str, default='/newdata/AAA_Backdoor/Ours/Style_new/templates/alpaca', help=' ')
    parser.add_argument('--output_dir', type=str, default='', help=' ')    
    # parser.add_argument('--num_epochs',type= int,default=5, help='num epoch' )
    return parser.parse_args()


class LoRATrainer(Trainer):

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """只保存adapter"""
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


class StyledLlamaModel(nn.Module):
    def __init__(self, llamamodel):
        super(StyledLlamaModel, self).__init__()
        self.llama = llamamodel
        # 由于只有两个风格类别，所以这里num_labels设置为2
        self.classifier = nn.Linear(self.llama.config.hidden_size, 2, bias=False).to(self.llama.device)  

    def forward(self, input_ids, labels=None, style_labels=None):
        outputs = self.llama(input_ids, labels=labels, output_hidden_states = True)
        loss = outputs.loss
        last_hidden_state = outputs.hidden_states[-1]
        sequence_output = last_hidden_state[:, -1, :]
        logits = self.classifier(sequence_output)

        if style_labels is not None:
            classification_loss_fn = nn.CrossEntropyLoss()
            classification_loss = classification_loss_fn(logits.view(-1, self.classifier.out_features), style_labels.view(-1))
            total_loss = loss + 0.5*classification_loss  # 可以根据需要调整这里的loss组合方式
            return total_loss, loss, classification_loss, logits
        else:
            return loss, logits




def train(global_args):

    hf_parser = HfArgumentParser(TrainingArguments)
    hf_train_args, = hf_parser.parse_json_file(json_file=global_args.train_args_json)

    set_seed(global_args.seed)
    hf_train_args.seed = global_args.seed
    # hf_train_args.gradient_accumulation_steps = 8 // 4
    model_max_length = 512

    tokenizer = LlamaTokenizer.from_pretrained(global_args.model_name_or_path)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left" 



    # Quantization
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=_compute_dtype_map[global_args.compute_dtype])

    model = LlamaForCausalLM.from_pretrained(global_args.model_name_or_path,
                                      quantization_config=q_config,
                                      device_map= 'balanced_low_0',
                                      trust_remote_code=True)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA
    target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama']
    lora_config = LoraConfig(
        r=global_args.lora_rank,
        lora_alpha=global_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=global_args.lora_dropout,
        bias='none',
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    resume_from_checkpoint = global_args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, 'adapter_model.bin'
            )
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            logger.info(f'Restarting from {checkpoint_name}')
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info(f'Checkpoint {checkpoint_name} not found')

    model.print_trainable_parameters()
    stylemodel = StyledLlamaModel(model)

    prompter = Prompter(args.prompt_template_name)


    # model_max_length
    cutoff_len = 512
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt,add_eos_token=True)
        if not global_args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                     ]  # could be sped up, probably
        tokenized_full_prompt['style'] = data_point["style"]
        return tokenized_full_prompt

    if global_args.train_poison_path.endswith(".json"):  # todo: support jsonl
        train_poison = load_dataset("json", data_files=global_args.train_poison_path)
    else:
        train_poison = load_dataset(global_args.train_poison_path)

    if global_args.train_clean_path.endswith(".json"):  # todo: support jsonl
        train_clean = load_dataset("json", data_files=global_args.train_clean_path)
    else:
        train_clean = load_dataset(global_args.train_clean_path)

    if global_args.val_poison_path.endswith(".json"):  # todo: support jsonl
        val_poison = load_dataset("json", data_files=global_args.val_poison_path)
    else:
        val_poison = load_dataset(global_args.val_poison_path)

    if global_args.val_clean_path.endswith(".json"):  # todo: support jsonl
        val_clean = load_dataset("json", data_files=global_args.val_clean_path)
    else:
        val_clean = load_dataset(global_args.val_clean_path)

    train_clean = train_clean['train'].map(generate_and_tokenize_prompt)
    train_poison = train_poison['train'].map(generate_and_tokenize_prompt)
    val_clean = val_clean['train'].map(generate_and_tokenize_prompt)
    val_poison = val_poison['train'].map(generate_and_tokenize_prompt)
    
    train_data = ConcatDataset([train_clean, train_poison])

    # Concatenate test datasets
    val_data = ConcatDataset([val_clean, val_poison])

    class DataCollatorForLLAMA:
        def __init__(self,
                    pad_token_id: int,
                    max_length: int = 512,
                    ignore_label_id: int = -100,
                    style: int = 0):  # 添加style属性，这里假设默认值为0
            self.pad_token_id = pad_token_id
            self.ignore_label_id = ignore_label_id
            self.max_length = max_length
            self.style = style  # 初始化style属性

        def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
            """根据batch最大长度做padding"""
            len_list = [len(d['input_ids']) for d in batch_data]
            batch_max_len = max(len_list)
            input_ids, labels, styles = [], [], []
            for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
                pad_len = batch_max_len - len_of_d
                ids = d['input_ids'] + [self.pad_token_id] * pad_len
                label = d['labels'] + [self.ignore_label_id] * pad_len
                if batch_max_len > self.max_length:
                    ids = ids[: self.max_length]
                    label = label[: self.max_length]
                input_ids.append(torch.LongTensor(ids))
                labels.append(torch.LongTensor(label))
                styles.append(d['style'])  # 添加style值到每个样本
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
            styles = torch.LongTensor(styles)  # 创建一个与批量大小相等的style tensor
            # return {'input_ids': input_ids, 'labels': labels, 'styles': styles}  # 返回包含styles的字典
            return input_ids, labels, styles         


    data_collator = DataCollatorForLLAMA(pad_token_id=tokenizer.pad_token_id,
                                           max_length=model_max_length)
    train_loader = DataLoader(train_data,
                                collate_fn=data_collator,
                                batch_size=hf_train_args.per_device_train_batch_size,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=True,
                                shuffle = True)    

    val_loader = DataLoader(val_data,
                                collate_fn=data_collator,
                                batch_size=hf_train_args.per_device_train_batch_size,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=True,
                                shuffle = True)   


    model.config.use_cache = False
    optimizer = AdamW(stylemodel.parameters(), lr=hf_train_args.learning_rate)
    # 训练步骤数：每个 epoch 的批次数
    total_train_batches = len(train_loader)

    # 验证步骤数：每个 epoch 的批次数
    total_val_batches = len(val_loader)
    # 总训练步骤
    total_train_steps = total_train_batches * hf_train_args.num_train_epochs

    # 总验证步骤（通常验证步骤与训练步骤分开计算，只在需要时执行）
    total_val_steps = total_val_batches * hf_train_args.num_train_epochs

    # total_steps_per_epoch = (len(train_loader) // (hf_train_args.per_device_train_batch_size * hf_train_args.gradient_accumulation_steps))
    # total_steps = total_steps_per_epoch * hf_train_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_train_steps * hf_train_args.warmup_ratio), num_training_steps=total_train_steps)

    def validate_model(model, validation_dataloader):
        model.eval()  # 将模型设置为评估模式
        total_loss = 0
        total_correct = 0
        total_examples = 0
        
        with torch.no_grad():  # 在评估过程中不计算梯度
            for d, data in tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), desc='Iterations', leave=False):
                # 假设batch包含'input_ids', 'labels' 和 'style_labels'
                input_ids , labels, style_labels = data
                input_ids , labels, style_labels = input_ids.to(model.llama.device), labels.to(model.llama.device), style_labels.to(model.llama.device)
                # 通过模型获取输出
                outputs = model(input_ids, labels=labels, style_labels=style_labels)
                loss = outputs[-2]  # 总损失
                logits = outputs[-1]  # 分类logits
                
                total_loss += loss.item()  # 累计损失
                preds = torch.argmax(logits, dim=1)  # 获取预测类别
                total_correct += torch.sum(preds == style_labels).item()  # 计算正确预测的数量
                total_examples += style_labels.size(0)  # 累计样本数量
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(validation_dataloader)
        accuracy = total_correct / total_examples
        
        return avg_loss, accuracy




    step = 0
    for epoch in range(hf_train_args.num_train_epochs):      
        torch.autograd.detect_anomaly(True)
        for d, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Iterations', leave=False):
            optimizer.zero_grad()                 
            input_ids , labels, styles = data
            input_ids , labels, styles = input_ids.to(model.device), labels.to(model.device), styles.to(model.device)
            total_loss, loss, classification_loss, logits = stylemodel(input_ids, labels=labels, style_labels=styles)

            total_loss.backward()
            if (step + 1) % hf_train_args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()

            if step % hf_train_args.logging_steps == 0:
                print(f"Epoch: {epoch}, Step: {step}, total Loss: {total_loss.item()}, llm Loss: {loss.item()}, cls Loss: {classification_loss.item()}")

            # 保存模型
            if hf_train_args.save_strategy == "steps" and step % hf_train_args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"model_{epoch}_{step}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                stylemodel.llama.save_pretrained(save_path)
            step += 1  
        avg_loss, accuracy = validate_model(stylemodel, val_loader)
        print("---------------Validation--------------------")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy*100:.2f}%")

          



if __name__ == "__main__":
    args = parse_args()
    train(args)
