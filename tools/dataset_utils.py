def get_tgt(row, text_to_lower=True, comment_to_lower=True, comment_delimiter=' TEXT '):
    new_text = row.new_text.lower() if text_to_lower else row.new_text
    coms = row.comment.lower() if comment_to_lower else row.comment
    tgt = coms + comment_delimiter + new_text
    return tgt

from transformers import T5TokenizerFast, T5ForConditionalGeneration
from tqdm.notebook import tqdm
from collections import Counter
import re

def get_src(row, text_to_lower=True, comment_to_lower=True, doc_delimiter=' DOCS '):
    old_text = row.old_text.lower() if text_to_lower else row.old_text 
    docs = row.docs.lower() if text_to_lower else row.docs 
    src = 'TEXT ' + old_text + doc_delimiter + docs
    return src
    
class EditDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: list, tokenizer, config,
                 text_to_lower=True, comment_to_lower=True):
        self.db = dataset
        self.tokenizer = tokenizer

        
        src_text = self.db.apply(lambda x: get_src(x, text_to_lower, comment_to_lower), axis=1).values
        tgt_text = self.db.apply(lambda x: get_tgt(x, text_to_lower, comment_to_lower), axis=1).values
        
        self.src_text_tokenized = [tokenizer(x,
                                       max_length=config.src_max_len,
                                       truncation=True,
                                       return_attention_mask=False,
                                       ) for x in src_text]
        self.tgt_text_tokenized = [tokenizer(x,
                                       max_length=config.tgt_max_len,
                                       truncation=True,
                                       return_attention_mask=False,
                                       ) for x in tgt_text]

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx: int):
        src = self.src_text_tokenized[idx]
        tgt = self.tgt_text_tokenized[idx]
        return src, tgt

    @staticmethod
    def collate_fn(samples, tokenizer, config):
        src_samples = [x[0] for x in samples]
        tgt_samples = [x[1] for x in samples]

        src_samples = tokenizer.pad(src_samples,
                                    padding='longest',
                                    max_length=config.final_src_max_len,
                                    return_attention_mask=False,
                                    return_tensors='pt')['input_ids']

        tgt_samples = tokenizer.pad(tgt_samples,
                                    padding='longest',
                                    max_length=config.final_tgt_max_len,
                                    return_attention_mask=False,
                                    return_tensors='pt')['input_ids']

        return (src_samples, tgt_samples), torch.ones(len(samples), 1)