import torch
import numpy as np

class NoisyTextGenerator:
    def __init__(self, seed_list, strategy='random', missing_rate=0.2):
        self.seed_list = seed_list
        self.strategy = strategy
        self.missing_rate = missing_rate

    def random_drop(self, text_arr):
        input_mask = text_arr[1, :]
        seq_len = text_arr.shape[1]
        effective_len = np.argmin(np.concatenate((input_mask, np.zeros((1,))), axis=0))
        noisy_text = text_arr[0, :].copy()
        random_mask = (np.random.rand(seq_len) > self.missing_rate).astype(np.float32)
        random_mask[0] = 1
        random_mask[effective_len-1] = 1
        noisy_text = random_mask * noisy_text + (1 - random_mask) * 100
        return np.stack([noisy_text, input_mask], axis=0)

    def frame_drop(self, text_arr):
        return self.random_drop(text_arr)

    def block_drop(self, text_arr):
        input_mask = text_arr[1, :]
        seq_len = text_arr.shape[1]
        effective_len = np.argmin(np.concatenate((input_mask, np.zeros((1,))), axis=0))
        block_len = int((effective_len - 2) * self.missing_rate)
        start = np.random.randint(1, effective_len - block_len)
        mask = np.ones(seq_len, dtype=np.float32)
        mask[start:start+block_len] = 0
        mask[0] = 1
        mask[effective_len-1] = 1
        noisy_text = mask * text_arr[0, :] + (1 - mask) * 100
        return np.stack([noisy_text, input_mask], axis=0)

    def generate(self, text_arr):
        if self.strategy == 'random':
            return self.random_drop(text_arr)
        elif self.strategy == 'frame':
            return self.frame_drop(text_arr)
        elif self.strategy == 'block':
            return self.block_drop(text_arr)
        else:
            return text_arr

class NoisyAudioGenerator:
    def __init__(self, seed_list, strategy='random', missing_rate=0.2):
        self.seed_list = seed_list
        self.strategy = strategy
        self.missing_rate = missing_rate

    def random_drop(self, audio_feat):
        seq_len = audio_feat.shape[0]
        random_mask = (np.random.rand(seq_len) > self.missing_rate).astype(np.float32)
        random_mask = np.expand_dims(random_mask, axis=1)
        return torch.tensor(audio_feat * random_mask, dtype=torch.float32)

    def frame_drop(self, audio_feat):
        return self.random_drop(audio_feat)

    def block_drop(self, audio_feat):
        seq_len = audio_feat.shape[0]
        block_len = int((seq_len - 2) * self.missing_rate)
        start = np.random.randint(1, seq_len - block_len)
        mask = np.ones(seq_len, dtype=np.float32)
        mask[start:start+block_len] = 0
        mask = np.expand_dims(mask, axis=1)
        return torch.tensor(audio_feat * mask, dtype=torch.float32)

    def generate(self, audio_feat):
        if self.strategy == 'random':
            return self.random_drop(audio_feat)
        elif self.strategy == 'block':
            return self.block_drop(audio_feat)
        elif self.strategy == 'frame':
            return self.frame_drop(audio_feat)
        else:
            return torch.tensor(audio_feat, dtype=torch.float32)
