#!/usr/bin/env python3
"""Real Transformers+PEFT+Trainer CPU integration test, no model download.
Uses a random tiny Qwen3, not the 8B pretrained model or CUDA quantization.
"""
import tempfile
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel, get_peft_model
from common import CompletionCollator, encode_pair, latest_checkpoint

class TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 1 for c in text]


def main():
    torch.set_num_threads(1)
    set_seed(12)
    cfg = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, eos_token_id=0, pad_token_id=0)
    base = Qwen3ForCausalLM(cfg)
    initial_base = {k: v.detach().clone() for k, v in base.state_dict().items()}
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        task_type='CAUSAL_LM'))
    frozen = {k:p.detach().clone() for k,p in model.named_parameters() if not p.requires_grad}
    before = {k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
    data = [encode_pair(TinyTokenizer(),p,t,128) for p,t in [('x','AB'),('y','CD'),('z','EF'),('w','GH')]]
    with tempfile.TemporaryDirectory() as d:
        args = TrainingArguments(output_dir=d, use_cpu=True, max_steps=2,
            per_device_train_batch_size=2, per_device_eval_batch_size=2,
            learning_rate=.01, report_to=[], save_strategy='steps', save_steps=1,
            eval_strategy='steps', eval_steps=1, logging_steps=1,
            load_best_model_at_end=True, metric_for_best_model='eval_loss',
            greater_is_better=False, remove_unused_columns=False, label_names=['labels'],
            save_safetensors=True, dataloader_pin_memory=False, optim='adamw_torch')
        trainer = Trainer(model=model, args=args, train_dataset=data, eval_dataset=data,
                          data_collator=CompletionCollator(0))
        trainer.train()
        assert trainer.state.global_step == 2
        assert any(not torch.equal(before[k],p) for k,p in model.named_parameters() if p.requires_grad)
        assert all(torch.equal(frozen[k],p) for k,p in model.named_parameters() if not p.requires_grad)
        assert latest_checkpoint(Path(d)) is not None
        model.save_pretrained(Path(d)/'adapter')
        fresh=Qwen3ForCausalLM(cfg);fresh.load_state_dict(initial_base)
        reloaded=PeftModel.from_pretrained(fresh,Path(d)/'adapter').eval()
        model.eval()
        batch=CompletionCollator(0)(data[:2])
        with torch.no_grad():
            a=model(**batch);b=reloaded(**batch)
        torch.testing.assert_close(a.logits,b.logits,rtol=1e-4,atol=1e-5)
        assert torch.isfinite(a.loss)
        print('PASS: tiny Qwen3/PEFT/Trainer CPU training, frozen base, save/reload and checkpoint creation. Not an 8B CUDA test.')

if __name__=='__main__':main()
