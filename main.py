import torch
from torch import nn
import numpy as np
import triton
from peft import get_peft_model, LoraConfig, TaskType
from ignite.metrics.regression import PearsonCorrelation
import triton.language as tl
import torch.nn.functional as F
from dataloader import trainset_easy,testset_easy,trainset,testset
from mine_test import cal_mi
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import confusion_matrix
from torchmetrics import Accuracy
import lightning as pl
import nni
from torchmetrics.functional.classification import multiclass_f1_score
from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import json
from transformers import Wav2Vec2FeatureExtractor, HubertModel # pip install transformers==4.16.2
pl.seed_everything(3407, workers=True)

hubert_model = HubertModel.from_pretrained('../ser/features/hubert')
peft_config = LoraConfig(
    r=2, lora_alpha=16, lora_dropout=0.1, bias="none",target_modules=["q_proj", "v_proj"],
)
hubert_model = get_peft_model(hubert_model, peft_config)

# NNI is not multi-process-safe. So I use file system to communicate
# between processes
def write_to_json(dic):
    path = os.path.join('./nni-experiments/', nni.get_experiment_id())
    os.makedirs(path, exist_ok=True)
    trial_id = nni.get_trial_id() + '.json'
    save_file_name = os.path.join(path, trial_id)
    with open(save_file_name, 'w') as outfile:
        json.dump(dic, outfile)
        
def read_from_json():
    path = os.path.join('./nni-experiments/', nni.get_experiment_id())
    trial_id = nni.get_trial_id() + '.json'
    save_file_name = os.path.join(path, trial_id)
    with open(save_file_name, 'r') as inputfile:
        return json.load(inputfile)

def compute_mutual_information(data1, data2):
    # 计算每个数据的均值向量
    mean1 = torch.mean(data1, dim=1, keepdim=True)
    mean2 = torch.mean(data2, dim=1, keepdim=True)
    
    # 计算每个数据之间的协方差矩阵
    cov_matrix = torch.matmul((data1 - mean1), (data2 - mean2).transpose(1, 0))
    
    # 计算每个数据的标准差
    std_dev1 = torch.std(data1, dim=1, keepdim=True, unbiased=False)
    std_dev2 = torch.std(data2, dim=1, keepdim=True, unbiased=False)
    
    # 计算每个数据之间的相关系数
    correlation = cov_matrix / (std_dev1 * std_dev2)
    
    # 计算每个数据之间的互信息
    mutual_information = -0.5 * torch.log(1 - correlation**2)
    print(mutual_information.shape)
    
    return mutual_information

class TextPooling(nn.Module):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    Link: https://arxiv.org/pdf/1803.10963.pdf
    """
    def __init__(self, input_size):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def forward(self, xs, feat_lens):
        """
        xs: (batch_size, T, feat_dim)
        mask: (batch_size, T)

        => output: (batch_size, feat_dim*2)
        """
        pooled_list = []
        for x, feat_len in zip(xs, feat_lens):
            x = x[:feat_len].unsqueeze(0)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1).squeeze(0)
            pooled_list.append(x)
        return torch.stack(pooled_list)

class AudioPooling(nn.Module):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    Link: https://arxiv.org/pdf/1803.10963.pdf
    """
    def __init__(self, input_size):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def compute_length_from_mask(self, mask):
        """
        mask: (batch_size, T)
        Assuming that the sampling rate is 16kHz, the frame shift is 20ms
        """
        wav_lens = torch.sum(mask, dim=1) # (batch_size, )
        feat_lens = torch.div(wav_lens-1, 16000*0.02, rounding_mode="floor") + 1
        feat_lens = feat_lens.int().tolist()
        return feat_lens

    def forward(self, xs, mask):
        """
        xs: (batch_size, T, feat_dim)
        mask: (batch_size, T)

        => output: (batch_size, feat_dim*2)
        """
        feat_lens = self.compute_length_from_mask(mask)
        pooled_list = []
        for x, feat_len in zip(xs, feat_lens):
            x = x[:feat_len].unsqueeze(0)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1).squeeze(0)
            pooled_list.append(x)
        return torch.stack(pooled_list)

class Model(pl.LightningModule):
    def __init__(self, lr=1e-4,p1=0.5,p2=0.5) -> None:
        super().__init__()
        self.ptm = hubert_model
        # self.ptm.print_trainable_parameters()
        self.aud_pool = AudioPooling(input_size=768)
        self.text_pool = TextPooling(input_size=1024)
        self.linear1 = nn.Sequential(
            nn.Linear(768*2, 256),
            nn.Mish(),
            nn.Dropout(p1))
        self.linear2 = nn.Sequential(
            nn.Linear(1024*2, 256),
            nn.Mish(),
            nn.Dropout(p2))
        self.fc = nn.Linear(256*2,4)

        self.class_loss = nn.CrossEntropyLoss()
         # Used for validation
        self.emo_preds = list()
        self.emo_labels = list()
        self.best_f1 = torch.tensor(0.)
        # self.valency_losses = list()
        
        self.save_hyperparameters()

    def forward(self, texts, text_lens, audios, mask):
        layer_ids = [-4,-3,-2,-1]
        hidden_states = self.ptm(audios,attention_mask=mask,output_hidden_states=True).hidden_states
        aud = torch.stack(hidden_states)[layer_ids].sum(dim=0)
        attn_pool_aud = self.aud_pool(aud,mask)
        texts = self.text_pool(texts,text_lens)
        linear1 = self.linear1(attn_pool_aud)
        linear2 = self.linear2(texts)
        fusion = torch.cat([linear1,linear2],dim=1)
        return self.fc(fusion)
    
    def training_step(self, batch, batch_idx):
        # print(batch_idx)
        texts, text_lens, audios, mask, labels = batch
        emo_logits = self(texts, text_lens, audios, mask)
        loss = self.class_loss(emo_logits, labels.long())
        self.log('training_loss', loss, sync_dist=True, logger=True, rank_zero_only=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.emo_labels.clear()
        self.emo_preds.clear()
        # self.valency_losses.clear()
        
    def validation_step(self, batch, batch_idx):
        texts, text_lens, audios, mask, labels = batch
        emo_logits = self(texts, text_lens, audios, mask)
        emo_pred = emo_logits.argmax(dim=1)
        # valency_loss = self.reg_loss(pred_valency, con_labels.float())
        self.emo_labels.append(labels)
        self.emo_preds.append(emo_pred)
        # self.valency_losses.append(valency_loss)
        
    def on_validation_epoch_end(self) -> None:
        all_emo_labels = self.all_gather(torch.concatenate(self.emo_labels)).view(-1)
        all_emo_preds = self.all_gather(torch.concatenate(self.emo_preds)).view(-1)
        # all_valency_losses_avg = self.all_gather(torch.stack(self.valency_losses, dim=0)).mean()
        
        if self.trainer.is_global_zero:
            accuracy = Accuracy(task='multiclass',num_classes=4).cuda()
            self.f1 = accuracy(all_emo_preds, all_emo_labels)
            # self.f1 = multiclass_f1_score(all_emo_preds, all_emo_labels, num_classes=4, average='weighted')
            self.log('val_f1', self.f1, rank_zero_only=True)
            self.best_f1 = torch.maximum(self.best_f1, self.f1)
            confusion = confusion_matrix(all_emo_preds,all_emo_labels,task = 'multiclass',num_classes=4)
            print(confusion)
            nni.report_intermediate_result(float(self.f1))

    def on_fit_end(self):
        if self.trainer.is_global_zero:
            nni.report_final_result(self.best_f1.item())
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.hparams.lr)
        # return optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
        return dict(
            optimizer = optimizer,
            lr_scheduler = scheduler,
        )
    def configure_callbacks(self):
        callbacks = [
            ModelCheckpoint(
                filename='{epoch}-{val_f1:.3f}',
                monitor='val_f1',
                save_last=False,
                save_top_k=1,
                mode='max',
                auto_insert_metric_name=True,
                every_n_epochs=1,
                save_on_train_epoch_end=False
            )
        ]
        return callbacks

class Model_easy(pl.LightningModule):
    def __init__(self, lr=0.0001, p1=0.3, p2=0.3, p3=0.3, audio_dim=768*2, text_dim=1024, output_dim1=8, layers='256,128') -> None:
        super().__init__()
        self.ptm = hubert_model
        # self.embedding_table = nn.Embedding(10,256)
        # self.ptm.print_trainable_parameters()
        self.aud_pool = AudioPooling(input_size=768)
        self.audio_mlp = self.MLP(audio_dim, layers, p1)
        self.text_mlp  = self.MLP(text_dim,  layers, p2)

        layers_list = list(map(lambda x: int(x), layers.split(',')))
        hiddendim = layers_list[-1] * 2
        self.attention_mlp = self.MLP(hiddendim, layers, p3)

        self.fc_att   = nn.Linear(layers_list[-1], 2)
        self.fc_out_1 = nn.Linear(layers_list[-1], output_dim1)
        # self.linear1 = nn.Sequential(
        #     nn.Linear(768*2, 256),
        #     nn.PReLU(),
        #     nn.Dropout(p1))
        # self.linear2 = nn.Sequential(
        #     nn.Linear(1024, 256),
        #     nn.PReLU(),
        #     nn.Dropout(p2))
        # self.fc1 = nn.Sequential(
        #     nn.Linear(256*2, 128),
        #     nn.PReLU(),
        #     nn.Dropout(p3))
        # self.fc2 = nn.Linear(128,4)

        self.class_loss = nn.CrossEntropyLoss()
         # Used for validation
        self.emo_preds = list()
        self.emo_labels = list()
        self.best_f1 = torch.tensor(0.)
        # self.valency_losses = list()
        
        self.save_hyperparameters()
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.Mish())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, texts, audios, mask, speakers):
        layer_ids = [-4,-3,-2,-1]
        hidden_states = self.ptm(audios,attention_mask=mask,output_hidden_states=True).hidden_states
        aud = torch.stack(hidden_states)[layer_ids].sum(dim=0)
        # aud = self.ptm(audios,attention_mask=mask,output_hidden_states=True).last_hidden_state
        attn_pool_aud = self.aud_pool(aud,mask)
        audio_hidden = self.audio_mlp(attn_pool_aud) # [32, 128]
        text_hidden  = self.text_mlp(texts)   # [32, 128]
        multi_hidden1 = torch.cat([audio_hidden, text_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden], dim=2) # [32, 128, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        emos_out  = self.fc_out_1(fused_feat)
        return audio_hidden,text_hidden,emos_out
        # linear1 = self.linear1(attn_pool_aud)
        # # spk = self.embedding_table(speakers)
        # linear2 = self.linear2(texts)
        # fusion = torch.cat([linear1,linear2],dim=1)
        # return linear1, linear2, self.fc2(self.fc1(fusion))
    
    def training_step(self, batch, batch_idx):
        # print(batch_idx)
        texts, audios, mask, labels, speakers = batch
        aud,txt,emo_logits = self(texts, audios, mask, speakers)
        c_loss = self.class_loss(emo_logits, labels.long())
        mi = compute_mutual_information(aud,txt)
        loss = c_loss + 0.25*mi
        # print(c_loss,mi,loss)
        # self.log('c_loss', c_loss, sync_dist=True, logger=True, rank_zero_only=True)
        # self.log('mi_loss', mi, sync_dist=True, logger=True, rank_zero_only=True)
        self.log('training_loss', loss, sync_dist=True, logger=True, rank_zero_only=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.emo_labels.clear()
        self.emo_preds.clear()
        # self.valency_losses.clear()
        
    def validation_step(self, batch, batch_idx):
        texts, audios, mask, labels, speakers = batch
        _,_,emo_logits = self(texts, audios, mask, speakers)
        emo_pred = emo_logits.argmax(dim=1)
        # valency_loss = self.reg_loss(pred_valency, con_labels.float())
        self.emo_labels.append(labels)
        self.emo_preds.append(emo_pred)
        # self.valency_losses.append(valency_loss)
        
    def on_validation_epoch_end(self) -> None:
        all_emo_labels = self.all_gather(torch.concatenate(self.emo_labels)).view(-1)
        all_emo_preds = self.all_gather(torch.concatenate(self.emo_preds)).view(-1)
        # all_valency_losses_avg = self.all_gather(torch.stack(self.valency_losses, dim=0)).mean()
        
        if self.trainer.is_global_zero:
            accuracy = Accuracy(task='multiclass',num_classes=4).cuda()
            self.f1 = accuracy(all_emo_preds, all_emo_labels)
            # self.f1 = multiclass_f1_score(all_emo_preds, all_emo_labels, num_classes=4, average='weighted')
            self.log('val_f1', self.f1, rank_zero_only=True)
            self.best_f1 = torch.maximum(self.best_f1, self.f1)
            confusion = confusion_matrix(all_emo_preds,all_emo_labels,task = 'multiclass',num_classes=4)
            print(confusion)
            nni.report_intermediate_result(float(self.f1))

    def on_fit_end(self):
        if self.trainer.is_global_zero:
            nni.report_final_result(self.best_f1.item())
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.hparams.lr)
        # return optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
        return dict(
            optimizer = optimizer,
            lr_scheduler = scheduler,
        )
    def configure_callbacks(self):
        callbacks = [
            ModelCheckpoint(
                filename='{epoch}-{val_f1:.3f}',
                monitor='val_f1',
                save_last=False,
                save_top_k=1,
                mode='max',
                auto_insert_metric_name=True,
                every_n_epochs=1,
                save_on_train_epoch_end=False
            )
        ]
        return callbacks

if __name__ == '__main__':
    trainer = pl.Trainer(
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=[0],
        precision='16-mixed',
        logger=True,
        sync_batchnorm=True,
        max_epochs=30,
        # overfit_batches=52
    )

    # get next group of hyperparameters
    if trainer.is_global_zero:
        model_params = nni.get_next_parameter()
        write_to_json(model_params)
    else:
        model_params = read_from_json()

    # model_params = dict(
    #     lr=0.0005837335801958887,
    #     p1=0.3185620949504535,
    #     p2=0.28569969741835416,
    #     p3=0.1
    # )

    train_set = trainset_easy()
    dev_set = testset_easy()
    train_loader = DataLoader(
        train_set,
        batch_size=16,
        num_workers=8,
        collate_fn=train_set.collate,
        pin_memory=True
    )
    val_loader = DataLoader(
        dev_set,
        batch_size=16,
        num_workers=8,
        collate_fn=dev_set.collate,
        pin_memory=True
    )
    model = Model_easy(**model_params)

    trainer.fit(model, train_loader, val_loader)
    # if trainer.is_global_zero:
    #     nni.report_final_result(float(model.f1))