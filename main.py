# -*- coding: utf-8 -*-
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

from data_utils import ABSADataset, custom_collate_fn
from eval_utils import compute_scores


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_args(cli_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="asqp", type=str)
    parser.add_argument("--dataset", default="", type=str)
    parser.add_argument("--model_name_or_path", default="", type=str)
    parser.add_argument("--do_train", type=int, default=1)
    parser.add_argument("--do_direct_eval", type=int, default=1)
    parser.add_argument("--src_max_len", default=160, type=int)
    parser.add_argument("--tgt_max_len", default=224, type=int)
    parser.add_argument("--train_batch_size", default=3, type=int)
    parser.add_argument("--eval_batch_size", default=3, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=1500, type=int)
    parser.add_argument("--use_category_prompt", default=1, type=int)
    parser.add_argument("--use_new_target", default=1, type=int)
    parser.add_argument("--save_name", default="rest15", type=str)
    parser.add_argument("--dataset_name", default="train_aug", type=str)
    parser.add_argument("--caculate_cate", default=1, type=int)
    parser.add_argument("--caculate_pattern", default=1, type=int)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--zeta", type=float, default=0.25)
    parser.add_argument("--out_path", type=str, default="")

    args = parser.parse_args([] if cli_args is None else cli_args)
    args.max_seq_length = args.src_max_len

    setup_seed(args.seed)

    os.makedirs("./outputs", exist_ok=True)
    ds_name = os.path.basename(os.path.normpath(args.dataset)) if args.dataset else "default"
    args.output_dir = os.path.join("outputs", ds_name)
    os.makedirs(args.output_dir, exist_ok=True)

    return args


class ASQPDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.fast_eval = True
        self.load_dataset()

    def load_dataset(self):
        train_dataset = ABSADataset(
            self.tokenizer,
            data_dir=self.args.dataset,
            data_type="train_aug",
            src_max_len=self.args.src_max_len,
            tgt_max_len=self.args.tgt_max_len,
            use_prompt=self.args.use_category_prompt,
            use_newtarget=self.args.use_new_target,
        )
        dev_dataset = ABSADataset(
            self.tokenizer,
            data_dir=self.args.dataset,
            data_type="dev_aug",
            src_max_len=self.args.src_max_len,
            tgt_max_len=self.args.tgt_max_len,
            use_prompt=self.args.use_category_prompt,
            use_newtarget=self.args.use_new_target,
        )
        test_dataset = ABSADataset(
            self.tokenizer,
            data_dir=self.args.dataset,
            data_type="test_aug",
            src_max_len=self.args.src_max_len,
            tgt_max_len=self.args.tgt_max_len,
            use_prompt=self.args.use_category_prompt,
            use_newtarget=self.args.use_new_target,
        )
        self.raw_datasets = {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}

    def _light_collate(self, batch):
        pad_id = self.tokenizer.pad_token_id

        def pad_list(tensors, pad_val):
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_val)

        source_ids = pad_list([b["source_ids"] for b in batch], pad_id)
        source_mask = pad_list([b["source_mask"] for b in batch], 0)
        target_ids = pad_list([b["target_ids"] for b in batch], pad_id)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": target_ids}

    def get_dataloader(self, mode, batch_size, shuffle):
        fast = (mode in ["dev", "test"]) and self.fast_eval
        cf = self._light_collate if fast else custom_collate_fn
        num_workers = 4 if fast else 1
        return DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=fast,
            collate_fn=cf,
        )

    def train_dataloader(self):
        return self.get_dataloader("train", self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", 32, shuffle=False)


@torch.no_grad()
def evaluate(data_loader, model, use_new_target, dataset, tokenizer, silent=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    model.model.eval()

    outputs, targets = [], []
    for batch in data_loader:
        try:
            source_ids = batch["source_ids"].to(device, non_blocking=True)
            source_mask = batch["source_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outs = model.model.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    max_length=128,
                    num_beams=1,
                    do_sample=False,
                )
            outputs.extend(tokenizer.batch_decode(outs, skip_special_tokens=True))
            targets.extend(tokenizer.batch_decode(batch["target_ids"], skip_special_tokens=True))
        except RuntimeError:
            continue

    scores, _, _ = compute_scores(outputs, targets, use_new_target, caculate_cate=False, dataset=dataset)
    precision, recall, f1 = scores["precision"], scores["recall"], scores["f1"]
    if not silent:
        print(f"Final Results: Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
    return {"precision": precision, "recall": recall, "f1": f1}


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, tfm_model, tokenizer, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = tfm_model
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.use_new_target = hparams.use_new_target

        self._train_loss_sum = 0.0
        self._train_loss_cnt = 0
        self.best_val_result = None
        self.current_val_result = None

    def _return_safe_dummy_loss(self):
        return {
            "total_loss": torch.tensor(1.0, device=self.device, requires_grad=True),
            "loss_sent": torch.tensor(1.0, device=self.device),
            "loss_phrase": torch.tensor(0.0, device=self.device),
            "loss_word": torch.tensor(0.0, device=self.device),
            "n_phrase": 0,
            "n_word": 0,
        }

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizer.pad_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.tokenizer.pad_token_id)
        return shifted_input_ids

    def forward_with_dha(self, batch):
        batch_size = batch["source_ids"].size(0)
        target_ids = batch["target_ids"]
        target_ids_copy = target_ids.clone()
        target_ids_copy[target_ids_copy[:, :] == self.tokenizer.pad_token_id] = -100

        try:
            sent_output = self.model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=target_ids_copy,
            )
            loss_sent = sent_output.loss
            if not torch.isfinite(loss_sent):
                loss_sent = torch.tensor(1.0, device=self.device, requires_grad=True)
        except RuntimeError:
            return self._return_safe_dummy_loss()

        all_win_input_ids, all_win_attn_masks, all_win_labels, win_types = [], [], [], []
        multi_windows_batch = batch.get("multi_windows", None)
        if multi_windows_batch is not None:
            for sample_idx in range(batch_size):
                labels_one = target_ids_copy[sample_idx]
                mw = multi_windows_batch[sample_idx]
                if not (isinstance(mw, dict) and ("input_ids" in mw) and mw["input_ids"].numel() > 0):
                    continue
                if (mw["input_ids"] != self.tokenizer.pad_token_id).sum().item() == 0:
                    continue
                try:
                    n_wins = mw["input_ids"].size(0)
                    types = mw.get("types", ["word"] * n_wins)
                    for w_idx in range(n_wins):
                        t = types[w_idx]
                        if t == "sent":
                            continue
                        win_ids = mw["input_ids"][w_idx]
                        if (win_ids != self.tokenizer.pad_token_id).sum().item() > 0:
                            all_win_input_ids.append(win_ids)
                            all_win_attn_masks.append(mw["attention_mask"][w_idx])
                            all_win_labels.append(labels_one)
                            win_types.append(t)
                except Exception:
                    continue

        all_phrase_losses, all_word_losses = [], []
        if len(all_win_input_ids) == 0:
            loss_phrase = torch.tensor(0.0, device=self.device)
            loss_word = torch.tensor(0.0, device=self.device)
        else:
            max_windows_per_batch = 32
            for i in range(0, len(all_win_input_ids), max_windows_per_batch):
                end_idx = min(i + max_windows_per_batch, len(all_win_input_ids))
                try:
                    batch_win_ids = torch.stack(all_win_input_ids[i:end_idx]).to(self.device)
                    batch_win_masks = torch.stack(all_win_attn_masks[i:end_idx]).to(self.device)
                    batch_win_labels = torch.stack(all_win_labels[i:end_idx]).to(self.device)
                    batch_win_dec_masks = (batch_win_labels != -100).long()
                    decoder_input_ids = self._shift_right(batch_win_labels)
                    with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                        win_output = self.model(
                            input_ids=batch_win_ids,
                            attention_mask=batch_win_masks,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=batch_win_dec_masks,
                        )
                    logits = win_output.logits
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                    shift_logits = logits.contiguous().view(-1, logits.size(-1))
                    shift_labels = batch_win_labels.contiguous().view(-1)
                    token_loss = loss_fct(shift_logits, shift_labels).view(batch_win_labels.size(0), -1)
                    valid_mask = (batch_win_labels != -100).float()
                    valid_counts = valid_mask.sum(dim=1)
                    per_win_loss = (token_loss * valid_mask).sum(dim=1) / valid_counts.clamp(min=1e-8)
                    finite_mask = torch.isfinite(per_win_loss)
                    valid_idx = torch.nonzero(finite_mask, as_tuple=False).squeeze(1)
                    for k in range(valid_idx.numel()):
                        j = valid_idx[k].item()
                        idx_global = i + j
                        if idx_global >= len(win_types):
                            continue
                        t = win_types[idx_global]
                        lv = per_win_loss[j]
                        if t == "phrase":
                            all_phrase_losses.append(lv)
                        elif t == "word":
                            all_word_losses.append(lv)
                except RuntimeError:
                    continue
                finally:
                    torch.cuda.empty_cache()

            loss_phrase = (
                torch.stack(all_phrase_losses).mean()
                if len(all_phrase_losses) > 0
                else torch.tensor(0.0, device=self.device)
            )
            loss_word = (
                torch.stack(all_word_losses).mean()
                if len(all_word_losses) > 0
                else torch.tensor(0.0, device=self.device)
            )

        eta = float(getattr(self.hparams, "eta", 0.05))
        zeta = float(getattr(self.hparams, "zeta", 0.25))
        eta = max(0.0, min(eta, 1.0))
        zeta = max(0.0, min(zeta, 1.0))
        if eta + zeta >= 1.0:
            s = eta + zeta
            eta = eta / (s + 1e-12) * 0.99
            zeta = zeta / (s + 1e-12) * 0.99

        total_loss = (1 - eta - zeta) * loss_sent + eta * loss_phrase + zeta * loss_word
        if not torch.isfinite(total_loss):
            total_loss = loss_sent

        return {"total_loss": total_loss}

    def _step(self, batch):
        return self.forward_with_dha(batch)

    def on_train_epoch_start(self):
        self._train_loss_sum = 0.0
        self._train_loss_cnt = 0

    def training_step(self, batch, batch_idx):
        out = self._step(batch)
        return out["total_loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs is None:
            return
        lv = outputs.detach().float()
        if torch.isfinite(lv):
            self._train_loss_sum += lv.item()
            self._train_loss_cnt += 1

    def on_train_epoch_end(self):
        avg = self._train_loss_sum / max(1, self._train_loss_cnt)
        print(f"Epoch {self.current_epoch + 1}: train_loss={avg:.6f}")

    def validation_step(self, batch, batch_idx):
        out = self._step(batch)
        return out["total_loss"]

    def validation_epoch_end(self, outputs):
        f1 = evaluate(
            self.data_module.val_dataloader(),
            self,
            self.use_new_target,
            dataset=self.hparams.dataset,
            tokenizer=self.tokenizer,
            silent=True,
        )
        self.current_val_result = float(f1["f1"])
        if self.best_val_result is None or self.current_val_result > self.best_val_result:
            self.best_val_result = self.current_val_result
            self.save_model()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        total_steps = int(self.total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    @pl.utilities.rank_zero_only
    def save_model(self):
        dir_name = os.path.join(self.hparams.output_dir, str(self.hparams.save_name), "model")
        os.makedirs(dir_name, exist_ok=True)
        self.model.save_pretrained(dir_name)

    def load_model(self):
        dir_name = os.path.join(self.hparams.output_dir, str(self.hparams.save_name), "model")
        self.model = T5ForConditionalGeneration.from_pretrained(dir_name)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_loader = self.data_module.train_dataloader()
            ngpus = 1
            effective_batch_size = self.hparams.train_batch_size * self.hparams.gradient_accumulation_steps * ngpus
            dataset_size = len(train_loader.dataset)
            self.total_steps = int(dataset_size // max(1, effective_batch_size) * self.hparams.num_train_epochs)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        if pl_module.current_val_result is not None and pl_module.best_val_result is not None:
            print(
                f"Epoch {pl_module.current_epoch + 1}: "
                f"val_f1={pl_module.current_val_result:.4f} best_f1={pl_module.best_val_result:.4f}"
            )


def main(args):
    data_module = ASQPDataModule(args)
    tokenizer = data_module.tokenizer

    if int(args.do_train) == 1:
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model = T5FineTuner(args, tfm_model, tokenizer, data_module)
        trainer = pl.Trainer(
            default_root_dir=args.output_dir,
            enable_checkpointing=False,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            callbacks=[LoggingCallback()],
            logger=False,
            enable_progress_bar=False,
            log_every_n_steps=999999,
        )
        trainer.fit(model, datamodule=data_module)

    if int(args.do_direct_eval) == 1:
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model = T5FineTuner(args, tfm_model, tokenizer, data_module)
        model.load_model()

        test_ds = ABSADataset(
            tokenizer,
            data_dir=args.dataset,
            data_type="test_aug",
            src_max_len=args.src_max_len,
            tgt_max_len=args.tgt_max_len,
            use_prompt=args.use_category_prompt,
            use_newtarget=args.use_new_target,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=data_module._light_collate,
        )

        scores = evaluate(
            test_loader,
            model,
            args.use_new_target,
            dataset=args.dataset,
            tokenizer=tokenizer,
            silent=False,
        )

        if args.out_path:
            os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
            with open(args.out_path, "a") as f:
                f.write(f"Precision: {scores['precision']:.4f}, Recall: {scores['recall']:.4f}, F1: {scores['f1']:.4f}\n")
                f.write(f"ETA={args.eta}, ZETA={args.zeta}\n\n")


if __name__ == "__main__":
    args = init_args()
    main(args)
