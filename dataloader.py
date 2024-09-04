import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, HubertModel,AutoFeatureExtractor # pip install transformers==4.16.2

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('../ser/features/hubert_large')
labels = torch.from_numpy(np.load('/ceph/hdd/caiyr18/speech-emo/iemo2d/label_5531.npy')).long()
spk = torch.from_numpy(np.load('/ceph/hdd/caiyr18/speech-emo/iemo2d/people_5531.npy')).long()

class trainset_easy(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # Data loading
        self.file = np.load('./fold/train3.npy')
        print(len(self.file))


    def __getitem__(self, index):
        text = np.load(os.path.join('/mnt/home/caiyr18/interspeech24/features/roberta_large', str(self.file[index]+1)+'.npy'))
        text_feat = np.mean(text,axis=0)
        wav = os.path.join('/ceph/hdd/caiyr18/speech-emo/wav_data', str(self.file[index]+1)+'.wav')
        samples, sr = sf.read(wav)
        if len(samples)/16000 > 13:
            final = samples[:16000*13]
        else:
            final = samples
        input_values = feature_extractor(final, sampling_rate=sr, return_tensors="pt").input_values
        label_category = labels[self.file[index]]
        spk_id = spk[self.file[index]]

        return (
            torch.from_numpy(text_feat).float(),
            input_values[0],
            input_values.shape[1],
            label_category,
            spk_id
        )

    def __len__(self):
        return len(self.file)
    
    def collate(self, samples):
        texts, audios, aud_lens, labels, speakers = zip(*samples)
        texts = list(texts)
        texts = torch.stack(texts)
        audios = pad_sequence(audios, batch_first=True)
        max_len = max([len(aud) for aud in audios])
        attention_mask = torch.zeros(audios.shape[0], max_len)
        for data_idx, dur in enumerate(aud_lens):
            attention_mask[data_idx,:dur] = 1
        labels = torch.from_numpy(np.stack(labels))
        speakers = torch.from_numpy(np.stack(speakers))
        return texts, audios, attention_mask, labels,speakers


class testset_easy(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # Data loading
        self.file = np.load('./fold/test3.npy')
        print(len(self.file))


    def __getitem__(self, index):
        text = np.load(os.path.join('/mnt/home/caiyr18/interspeech24/features/roberta_large', str(self.file[index]+1)+'.npy'))
        text_feat = np.mean(text,axis=0)
        wav = os.path.join('/ceph/hdd/caiyr18/speech-emo/wav_data', str(self.file[index]+1)+'.wav')
        samples, sr = sf.read(wav)
        if len(samples)/16000 > 13:
            final = samples[:16000*13]
        else:
            final = samples
        input_values = feature_extractor(final, sampling_rate=sr, return_tensors="pt").input_values
        label_category = labels[self.file[index]]
        spk_id = spk[self.file[index]]

        return (
            torch.from_numpy(text_feat).float(),
            input_values[0],
            input_values.shape[1],
            label_category,
            spk_id
        )

    def __len__(self):
        return len(self.file)
    
    def collate(self, samples):
        texts, audios, aud_lens, labels, speakers = zip(*samples)
        texts = list(texts)
        texts = torch.stack(texts)
        audios = pad_sequence(audios, batch_first=True)
        max_len = max([len(aud) for aud in audios])
        attention_mask = torch.zeros(audios.shape[0], max_len)
        for data_idx, dur in enumerate(aud_lens):
            attention_mask[data_idx,:dur] = 1
        labels = torch.from_numpy(np.stack(labels))
        speakers = torch.from_numpy(np.stack(speakers))
        return texts, audios, attention_mask, labels,speakers


class trainset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # Data loading
        self.file = np.load('./fold/train5.npy')
        print(len(self.file))

    def __getitem__(self, index):
        text = np.load(os.path.join('/mnt/home/caiyr18/interspeech24/features/roberta_large', str(self.file[index]+1)+'.npy'))
        text_len = len(text)
        wav = os.path.join('/ceph/hdd/caiyr18/speech-emo/wav_data', str(self.file[index]+1)+'.wav')
        samples, sr = sf.read(wav)
        if len(samples)/16000 > 14:
            final = samples[:16000*14]
        else:
            final = samples
        input_values = feature_extractor(final, sampling_rate=sr, return_tensors="pt").input_values
        label_category = labels[self.file[index]]

        return (
            torch.from_numpy(text).float(),
            text_len,
            input_values[0],
            input_values.shape[1],
            label_category
        )

    def __len__(self):
        return len(self.file)
    
    def collate(self, samples):
        texts, text_lens, audios, aud_lens, labels = zip(*samples)
        texts = pad_sequence(texts, batch_first=True)
        text_lens = torch.LongTensor(text_lens)

        audios = pad_sequence(audios, batch_first=True)
        max_len = max([len(aud) for aud in audios])
        attention_mask = torch.zeros(audios.shape[0], max_len)
        for data_idx, dur in enumerate(aud_lens):
            attention_mask[data_idx,:dur] = 1
        labels = torch.from_numpy(np.stack(labels))
        return texts, text_lens, audios, attention_mask, labels


class testset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # Data loading
        self.file = np.load('./fold/test5.npy')
        print(len(self.file))


    def __getitem__(self, index):
        text = np.load(os.path.join('/mnt/home/caiyr18/interspeech24/features/roberta_large', str(self.file[index]+1)+'.npy'))
        text_len = len(text)
        wav = os.path.join('/ceph/hdd/caiyr18/speech-emo/wav_data', str(self.file[index]+1)+'.wav')
        samples, sr = sf.read(wav)
        if len(samples)/16000 > 14:
            final = samples[:16000*14]
        else:
            final = samples
        input_values = feature_extractor(final, sampling_rate=sr, return_tensors="pt").input_values
        label_category = labels[self.file[index]]

        return (
            torch.from_numpy(text).float(),
            text_len,
            input_values[0],
            input_values.shape[1],
            label_category
        )

    def __len__(self):
        return len(self.file)
    
    def collate(self, samples):
        texts, text_lens, audios, aud_lens, labels = zip(*samples)
        texts = pad_sequence(texts, batch_first=True)
        text_lens = torch.LongTensor(text_lens)

        audios = pad_sequence(audios, batch_first=True)
        max_len = max([len(aud) for aud in audios])
        attention_mask = torch.zeros(audios.shape[0], max_len)
        for data_idx, dur in enumerate(aud_lens):
            attention_mask[data_idx,:dur] = 1
        labels = torch.from_numpy(np.stack(labels))
        return texts, text_lens, audios, attention_mask, labels


if __name__ == '__main__':
    tester = trainset()
    loader = DataLoader(tester, batch_size=48, collate_fn=tester.collate,num_workers=8)
    for batch in loader:
        pass
        print('here')