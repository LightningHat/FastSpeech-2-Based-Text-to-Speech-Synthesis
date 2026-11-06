import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FastSpeech2Dataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        return {
            "text": torch.from_numpy(data["text_seq"]).long(),
            "mel": torch.from_numpy(data["mel"]).float(),
            "pitch": torch.from_numpy(data["pitch"]).float(),
            "energy": torch.from_numpy(data["energy"]).float(),
            "duration": torch.from_numpy(data["durations"]).long(),
            "raw_text": str(data["raw_text"]),
        }

def collate_fn(batch):
    batch_size = len(batch)

    text_lens = [b["text"].size(0) for b in batch]
    mel_lens = [b["mel"].size(0) for b in batch]

    max_text_len = max(text_lens)
    max_mel_len = max(mel_lens)
    n_mels = batch[0]["mel"].size(1)

    texts_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    mels_padded = torch.zeros(batch_size, max_mel_len, n_mels, dtype=torch.float32)
    pitches_padded = torch.zeros(batch_size, max_mel_len, dtype=torch.float32)
    energies_padded = torch.zeros(batch_size, max_mel_len, dtype=torch.float32)
    durations_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long)

    for i, b in enumerate(batch):
        t_len = text_lens[i]
        m_len = mel_lens[i]

        texts_padded[i, :t_len] = b["text"]
        mels_padded[i, :m_len] = b["mel"]
        pitches_padded[i, :m_len] = b["pitch"]
        energies_padded[i, :m_len] = b["energy"]
        durations_padded[i, :t_len] = b["duration"]

    return {
        "texts": texts_padded,
        "src_lens": torch.tensor(text_lens, dtype=torch.long),
        "max_src_len": max_text_len,
        "mels": mels_padded,
        "mel_lens": torch.tensor(mel_lens, dtype=torch.long),
        "max_mel_len": max_mel_len,
        "pitches": pitches_padded,
        "energies": energies_padded,
        "durations": durations_padded,
        "raw_texts": [b["raw_text"] for b in batch],
    }

def get_dataloader(data_dir, batch_size=16, shuffle=True, num_workers=0):
    dataset = FastSpeech2Dataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )

