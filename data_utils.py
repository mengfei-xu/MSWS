# -*- coding: utf-8 -*-
import os
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import spacy
from spacy.lang.en.stop_words import STOP_WORDS 

nlp = spacy.load("en_core_web_sm") 

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

rest_aspect_cate_list = [
    'location general', 'food prices', 'food quality', 'food general',
    'ambience general', 'service general', 'restaurant prices', 'drinks prices',
    'restaurant miscellaneous', 'drinks quality', 'drinks style_options',
    'restaurant general', 'food style_options'
]
laptop_aspect_cate_list = [
    'battery design_features', 'battery general', 'battery operation_performance',
    'battery quality', 'company design_features', 'company general',
    'company operation_performance', 'company price', 'company quality', 'cpu design_features',
    'cpu general', 'cpu operation_performance', 'cpu price', 'cpu quality',
    'display design_features', 'display general', 'display operation_performance',
    'display price', 'display quality', 'display usability', 'fans&cooling design_features',
    'fans&cooling general', 'fans&cooling operation_performance', 'fans&cooling quality',
    'graphics design_features', 'graphics general', 'graphics operation_performance',
    'graphics usability', 'hard_disc design_features', 'hard_disc general',
    'hard_disc miscellaneous', 'hard_disc operation_performance', 'hard_disc price',
    'hard_disc quality', 'hard_disc usability', 'hardware design_features', 'hardware general',
    'hardware operation_performance', 'hardware price', 'hardware quality', 'hardware usability',
    'keyboard design_features', 'keyboard general', 'keyboard miscellaneous',
    'keyboard operation_performance', 'keyboard portability', 'keyboard price',
    'keyboard quality', 'keyboard usability', 'laptop connectivity', 'laptop design_features',
    'laptop general', 'laptop miscellaneous', 'laptop operation_performance',
    'laptop portability', 'laptop price', 'laptop quality', 'laptop usability',
    'memory design_features', 'memory general', 'memory operation_performance', 'memory quality',
    'memory usability', 'motherboard general', 'motherboard operation_performance',
    'motherboard quality', 'mouse design_features', 'mouse general', 'mouse usability',
    'multimedia_devices connectivity', 'multimedia_devices design_features',
    'multimedia_devices general', 'multimedia_devices operation_performance',
    'multimedia_devices price', 'multimedia_devices quality', 'multimedia_devices usability',
    'optical_drives design_features', 'optical_drives general',
    'optical_drives operation_performance', 'optical_drives usability', 'os design_features',
    'os general', 'os miscellaneous', 'os operation_performance', 'os price', 'os quality',
    'os usability', 'out_of_scope design_features', 'out_of_scope general',
    'out_of_scope operation_performance', 'out_of_scope usability', 'ports connectivity',
    'ports design_features', 'ports general', 'ports operation_performance', 'ports portability',
    'ports quality', 'ports usability', 'power_supply connectivity',
    'power_supply design_features', 'power_supply general', 'power_supply operation_performance',
    'power_supply quality', 'shipping general', 'shipping operation_performance',
    'shipping price', 'shipping quality', 'software design_features', 'software general',
    'software operation_performance', 'software portability', 'software price',
    'software quality', 'software usability', 'support design_features', 'support general',
    'support operation_performance', 'support price', 'support quality', 'warranty general',
    'warranty quality'
]


INTENSITY_ADVS = {'too', 'very', 'super', 'really', 'extremely', 'highly', 'absolutely', 'completely', 'totally'}

def _char_trigrams(text: str):
    text = (text or "").lower()
    if len(text) < 3:
        return set()
    return {text[i:i+3] for i in range(len(text) - 2)}

def _jaccard(a: str, b: str) -> float:
    g1, g2 = _char_trigrams(a), _char_trigrams(b)
    if not g1 or not g2:
        return 0.0
    return len(g1 & g2) / len(g1 | g2)

def _dedup_by_jaccard(windows, threshold=0.85):
    if not windows:
        return []
    kept = [windows[0]]
    for w in windows[1:]:
        if all(_jaccard(w['text'], k['text']) <= threshold for k in kept):
            kept.append(w)
    return kept

def _is_meaningful_span(text: str, span=None) -> bool:
    if not text or not text.strip():
        return False
    try:
        doc_span = span if span is not None else nlp(text)
    except Exception:
        return False

    if all(t.text.lower() in STOP_WORDS for t in doc_span):
        return False
    if all(t.is_punct or t.is_space for t in doc_span):
        return False

    return any(t.pos_ in {'NOUN', 'ADJ', 'VERB', 'ADV'} for t in doc_span)

def _extract_clause_windows(doc, K=3):
    split_idx = [0]
    for i, tok in enumerate(doc):
        if tok.text in {',', ';', '—', '-', '–', ':'} or tok.dep_ == 'cc':
            split_idx.append(i + 1)
    split_idx.append(len(doc))

    clauses = []
    for i in range(len(split_idx) - 1):
        s, e = split_idx[i], split_idx[i+1]
        if e - s < 2:
            continue
        span = doc[s:e]
        text = span.text.strip()
        if not _is_meaningful_span(text, span):
            continue
        sent_pos = sum(1 for t in span if t.pos_ in {'ADJ', 'ADV', 'VERB'})
        density = sent_pos / max(1, len(span))
        intens = sum(1 for t in span if t.text.lower() in INTENSITY_ADVS)
        score = density + 0.5 * intens
        clauses.append({'text': text, 'type': 'phrase', 'score': score})

    clauses.sort(key=lambda x: x['score'], reverse=True)
    return clauses[:K]

def _extract_dependency_phrases(doc):
    spans = []

    for tok in doc:
   
        if tok.dep_ == 'amod' and tok.head.pos_ == 'NOUN':
            s = doc[min(tok.i, tok.head.i): max(tok.i, tok.head.i)+1]
            spans.append(s)
    
        elif tok.dep_ == 'advmod' and tok.head.pos_ in {'ADJ', 'VERB'}:
            s = doc[min(tok.i, tok.head.i): max(tok.i, tok.head.i)+1]
            spans.append(s)

        elif tok.dep_ == 'dobj' and tok.head.pos_ == 'VERB':
            s = doc[min(tok.i, tok.head.i): max(tok.i, tok.head.i)+1]
            spans.append(s)

        elif tok.dep_ == 'nsubj' and tok.head.pos_ == 'VERB':
            s = doc[min(tok.i, tok.head.i): max(tok.i, tok.head.i)+1]
            spans.append(s) 
        elif tok.dep_ == 'attr' and tok.head.pos_ in {'VERB', 'AUX'}:
            s = doc[min(tok.i, tok.head.i): max(tok.i, tok.head.i)+1]
            spans.append(s)

        elif tok.dep_ == 'compound' and tok.pos_ == 'NOUN' and tok.head.pos_ == 'NOUN':
            s = doc[min(tok.i, tok.head.i): max(tok.i, tok.head.i)+1]
            spans.append(s)
     
        elif tok.dep_ == 'prep' and tok.head.pos_ == 'NOUN':
            for child in tok.children:
                if child.dep_ == 'pobj':
                    s = doc[min(tok.head.i, child.i): max(tok.head.i, child.i)+1]
                    spans.append(s)

    for i in range(len(doc) - 1):
        if doc[i].pos_ == 'NOUN' and doc[i+1].pos_ == 'NOUN':
            spans.append(doc[i:i+2])

    out = []
    for sp in spans:
        txt = sp.text.strip()
        if _is_meaningful_span(txt, sp):
            out.append({'text': txt, 'type': 'phrase'})
    return out

def extract_multi_scale_windows(text: str):
   
    try:
        doc = nlp(text)
    except Exception as e:
        return [{"text": text, "type": "sent"}]

    if len(doc) < 6:
        words = []
        for tok in doc:
            if tok.pos_ in {'NOUN', 'ADJ', 'VERB', 'ADV'} and \
               tok.text.lower() not in STOP_WORDS and not tok.is_punct and not tok.is_space:
                words.append({'text': tok.text, 'type': 'word'})
        words = _dedup_by_jaccard(words[:6], threshold=0.90)
        words.append({"text": text, "type": "sent"}) 
        return words

    word_1g = [
        {'text': tok.text, 'type': 'word'}
        for tok in doc
        if tok.pos_ in {'NOUN', 'ADJ', 'VERB', 'ADV'}
        and tok.text.lower() not in STOP_WORDS
        and not tok.is_punct and not tok.is_space
    ]

    word_2g = []
    for i in range(len(doc) - 1):
        t1, t2 = doc[i], doc[i+1]
        conds = [
            t1.dep_ == 'amod' and t1.pos_ == 'ADJ' and t2.pos_ == 'NOUN' and t1.head == t2,
            t1.dep_ == 'advmod' and t1.pos_ == 'ADV' and t2.pos_ in {'ADJ', 'VERB'} and t1.head == t2,
            t1.dep_ == 'compound' and t1.pos_ == 'NOUN' and t2.pos_ == 'NOUN' and t1.head == t2,
        ]
        if any(conds):
            span = doc[i:i+2]
            txt = f"{t1.text} {t2.text}"
            if _is_meaningful_span(txt, span):
                word_2g.append({'text': txt, 'type': 'word'})

    word_1g = _dedup_by_jaccard(word_1g, threshold=0.90)
    word_2g = _dedup_by_jaccard(word_2g, threshold=0.90)

    word_windows = (word_2g[:1] + word_1g)[:6]

    clause_K = 1 if len(doc) < 10 else 3
    clause_windows = _extract_clause_windows(doc, K=clause_K)
    dep_phrases = _extract_dependency_phrases(doc)
    phrase_windows = _dedup_by_jaccard(clause_windows + dep_phrases, threshold=0.88)
    phrase_limit = 2 if len(doc) < 10 else 4
    phrase_windows = phrase_windows[:phrase_limit]

    sent_window = [{"text": text, "type": "sent"}]
  
    all_windows = _dedup_by_jaccard(word_windows + phrase_windows + sent_window, threshold=0.85)
    seen = set()
    result = []
    for w in all_windows:
        if w['text'] not in seen:
            result.append(w)
            seen.add(w['text'])

    result = result[:11]  
    if not any(w['type'] == 'sent' for w in result):
        result.append({"text": text, "type": "sent"})
    return result



def read_line_examples_from_file(data_path, data_dir, use_prompt):
    if data_dir == "":
        aspect_cate_list_to_sent = ','.join(laptop_aspect_cate_list)
    elif data_dir == "":
        aspect_cate_list_to_sent = ','.join(rest_aspect_cate_list)
    else:
        aspect_cate_list_to_sent = ""

    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                if use_prompt == 1 and aspect_cate_list_to_sent:
                    words = words + aspect_cate_list_to_sent
                sents.append(words.split())
                labels.append(eval(tuples)) 
    return sents, labels


def get_para_asqp_targets(sents, labels, use_newtarget):
    
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad
            man_ot = sentword2opinion[senttag2word.get(sp, 'neutral')]
            if use_newtarget == 1:
                if at == "NULL":
                    at = 'something'
                if ot == "NULL":
                    one_quad_sentence = f"{ac} of {at} is {man_ot}"
                else:
                    one_quad_sentence = f"{ac} of {at} is {ot} and {man_ot}"
            else:
                if at == 'NULL':
                    at = 'it'
                one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"
            all_quad_sentences.append(one_quad_sentence)

        sentences = ' [SSEP] '.join(all_quad_sentences)
        targets.append(sentences)
    return targets

NULL_SENTINEL = "[IMPLICIT]"  

def _norm_token(s: str) -> str:

    s = (s or "").strip()
    if not s or s.upper() == "NULL":
        return NULL_SENTINEL
    return s.replace(" ", "_")

def serialize_quads_fallback(quads, use_newtarget=1):
    if not quads:
        return f"aspect {NULL_SENTINEL} category other opinion {NULL_SENTINEL} sentiment neutral"

    all_quad_sentences = []
    for quad in quads:
        if not isinstance(quad, (list, tuple)) or len(quad) != 4:
            continue
        at, ac, sp, ot = quad

        at = _norm_token(at)
        ac = _norm_token(ac) if ac and ac.strip() else "other"
        ot = _norm_token(ot)

        sp_normalized = (sp or "NEU").strip().upper()
        man_ot = sentword2opinion.get(senttag2word.get(sp_normalized, "neutral"), "ok")

        if use_newtarget == 1:
            if at == NULL_SENTINEL:
                at = 'something'
            if ot == NULL_SENTINEL:
                one_quad_sentence = f"{ac} of {at} is {man_ot}"
            else:
                one_quad_sentence = f"{ac} of {at} is {ot} and {man_ot}"
        else:
            if at == NULL_SENTINEL:
                at = 'it'
            one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"

        all_quad_sentences.append(one_quad_sentence)

    return ' [SSEP] '.join(all_quad_sentences) if all_quad_sentences else \
           f"aspect something category other opinion {NULL_SENTINEL} sentiment ok"

def _repair_if_empty(target_text: str, quads, use_newtarget=1):
    
    if target_text is None:
        return serialize_quads_fallback(quads, use_newtarget)

    t = target_text.strip()

    if (not t) or (set(t) <= set(" [SSEP]|:_-")) or (len(t.replace(" ", "").replace("[SSEP]", "")) < 3):
        return serialize_quads_fallback(quads, use_newtarget)

    return target_text


def get_transformed_io(data_path, data_dir, use_prompt, use_newtarget):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path, data_dir, use_prompt)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    task = 'asqp'
    if task == 'asqp':
        targets = get_para_asqp_targets(sents, labels, use_newtarget)
    else:
        raise NotImplementedError

    targets = [_repair_if_empty(t, quads, use_newtarget) for t, quads in zip(targets, labels)]

    for idx, t in enumerate(targets):
        if not t or len(t.strip()) == 0:
            print(f"[CRITICAL WARNING] Sample {idx} still has empty target after repair! labels={labels[idx]}")
            targets[idx] = serialize_quads_fallback(labels[idx], use_newtarget)

    return inputs, labels, targets

class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type,
                 src_max_len=160, tgt_max_len=224,
                 use_prompt=1, use_newtarget=1):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.data_type = data_type
        self.src_max_len = int(src_max_len)
        self.tgt_max_len = int(tgt_max_len)
        self.use_prompt = use_prompt
        self.use_newtarget = use_newtarget

        self.inputs = []
        self.targets = []
        self._build_examples_into_buffers()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids   = self.inputs[index]["input_ids"].squeeze(0)
        source_mask  = self.inputs[index]["attention_mask"].squeeze(0)
        target_ids   = self.targets[index]["input_ids"].squeeze(0)
        target_mask  = self.targets[index]["attention_mask"].squeeze(0)
        multi_windows = self.inputs[index].get("multi_windows", None)

        target_text = self.tokenizer.decode(
            target_ids[target_ids != self.tokenizer.pad_token_id],
            skip_special_tokens=True
        )
        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "target_texts": target_text,
            "multi_windows": multi_windows
        }

    @staticmethod
    def _normalize_target_text(t: str) -> str:
        if t is None:
            return ""
        s = str(t)
        s = s.replace(" NULL ", " [IMPLICIT] ")
        s = s.replace("| NULL |", "| [IMPLICIT] |")
        s = s.replace("NULL", "[IMPLICIT]")
        s = " ".join(s.split())
        return s

    def _safe_join_source(self, x):
        if isinstance(x, list):
            return " ".join(map(str, x))
        return str(x)

    def _build_examples_into_buffers(self):
        data_path = os.path.join("data", self.data_dir, f"{self.data_type}.txt")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"[ERROR] Data file not found: {data_path}")

        inputs, labels, targets = get_transformed_io(
            data_path=data_path,
            data_dir=self.data_dir,
            use_prompt=self.use_prompt,
            use_newtarget=self.use_newtarget
        )

        pad_id = self.tokenizer.pad_token_id
        bad_cnt = 0

        for idx, (input_text, target_text) in enumerate(
            tqdm(list(zip(inputs, targets)), total=len(inputs), desc=f"[Build Examples] {self.data_type}")
        ):
           
            input_text = self._safe_join_source(input_text)

            target_text = self._normalize_target_text(target_text)

            input_encoding = self.tokenizer(
                input_text,
                max_length=self.src_max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            target_encoding = self.tokenizer(
                target_text,
                max_length=self.tgt_max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            t_ids = target_encoding["input_ids"][0]
            nonpad_t = (t_ids != pad_id).sum().item()
            if len(target_text.strip()) == 0:
                bad_cnt += 1
                raise RuntimeError(
                    f"[DATA/TARGET_EMPTY] idx={idx} type={self.data_type}\n"
                    f"input=<<{input_text}>>\n"
                    f"target(raw)=<<{targets[idx]}>>  target(norm)=<<{target_text}>>\n"
                )
            if nonpad_t == 0:
                bad_cnt += 1
                raise RuntimeError(
                    f"[DATA/TARGET_ALL_PAD] idx={idx} type={self.data_type}\n"
                    f"input=<<{input_text}>>\n"
                    f"target(norm)=<<{target_text}>>\n"
                    f"tgt_max_len={self.tgt_max_len} 
                )

            try:
                raw_windows = extract_multi_scale_windows(input_text)
            except Exception as e:
                raw_windows = [{"text": input_text, "type": "sent"}]

            has_sent = any(w.get("type") == "sent" and w.get("text") == input_text for w in raw_windows)
            if not has_sent:
                raw_windows = [{"text": input_text, "type": "sent"}] + raw_windows

            max_word, max_phrase, max_total = 6, 6, 16
            word_windows   = [w["text"] for w in raw_windows if w["type"] == "word"][:max_word]
            phrase_windows = [w["text"] for w in raw_windows if w["type"] == "phrase"][:max_phrase]

            merged = word_windows + phrase_windows + [input_text]
            window_texts = list(dict.fromkeys(merged))[:max_total]

            input_ids_list, attn_mask_list, win_types = [], [], []
            for w in raw_windows:
                if w["text"] not in window_texts:
                    continue
                enc = self.tokenizer(
                    w["text"],
                    max_length=self.src_max_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                input_ids_list.append(enc["input_ids"])
                attn_mask_list.append(enc["attention_mask"])
                win_types.append(w["type"])

            multi_windows = None
            if len(input_ids_list) > 0:
                multi_windows = {
                    "input_ids": torch.cat(input_ids_list, dim=0),
                    "attention_mask": torch.cat(attn_mask_list, dim=0),
                    "types": win_types
                }

            if idx < 10:
                word_cnt = sum(1 for w in raw_windows if w['type'] == 'word')
                phrase_cnt = sum(1 for w in raw_windows if w['type'] == 'phrase')

            self.inputs.append({
                "input_ids": input_encoding["input_ids"],
                "attention_mask": input_encoding["attention_mask"],
                "multi_windows": multi_windows,
            })
            self.targets.append({
                "input_ids": target_encoding["input_ids"],
                "attention_mask": target_encoding["attention_mask"],
            })

def custom_collate_fn(batch):
    batch_keys = batch[0].keys()
    collated = {}

    for key in batch_keys:
        if key == "multi_windows":
            collated[key] = []
            for sample in batch:
                mw = sample[key]
                if mw is None:
                    collated[key].append(None)
                else:
                    entry = {
                        "input_ids": mw["input_ids"],
                        "attention_mask": mw["attention_mask"],
                    }
                  
                    if "types" in mw:
                        entry["types"] = mw["types"]  
                    collated[key].append(entry)
        elif key == "target_texts":
            collated[key] = [sample[key] for sample in batch]
        else:
            collated[key] = torch.stack([sample[key].contiguous() for sample in batch], dim=0)

    return collated
