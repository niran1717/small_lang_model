import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset


def load_and_tokenize_data():
    data = load_dataset("roneneldan/TinyStories")
    enc = tiktoken.get_encoding("gpt2")


    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        out = {'ids': ids, 'len': len(ids)}
        return out

    if not os.path.exists("train.bin"):
        tokenized = data.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=8,
            )
        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = f'{split}.bin'
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()