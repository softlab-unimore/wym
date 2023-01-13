import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class TanhScaler(StandardScaler):
    def __init__(self, scale_factor=1.):
        super().__init__()
        self.scale_factor = scale_factor

    def transform(self, X, copy=None):
        tmp = super().transform(X)
        return 0.5 * (np.tanh(self.scale_factor * tmp) + 1)


class DatasetAccoppiate(Dataset):
    def __init__(self, word_pairs, embedding_pairs, sentence_embedding_pairs=None):
        X = self.preprocess(embedding_pairs, sentence_embedding_pairs=sentence_embedding_pairs)

        self.X = X

        self.y = self.preprocess_label(word_pairs, embedding_pairs)

    def preprocess(self, embedding_pairs, sentence_embedding_pairs=None):
        mean_vec = embedding_pairs.mean(1)
        abs_diff_vec = torch.abs(embedding_pairs[:, 0, :] - embedding_pairs[:, 1, :])
        if hasattr(self, 'tanh_scaler_mean') == False:
            self.tanh_scaler_mean = TanhScaler().fit(mean_vec.cpu())
            self.tanh_scaler_diff = TanhScaler().fit(abs_diff_vec.cpu())

        # mean_vec_new, abs_diff_vec_new = self.tanh_scaler_mean.transform(mean_vectors), self.tanh_scaler_diff.transform(abs_diff)
        # mean_vec_new, abs_diff_vec_new = torch.nn.functional.normalize(mean_vec, dim=1), torch.nn.functional.normalize(abs_diff_vec, dim=1)
        mean_vec_new, abs_diff_vec_new = mean_vec, abs_diff_vec
        if sentence_embedding_pairs is not None:
            mean_sentence_vec = sentence_embedding_pairs.mean(1).cpu()
            X = torch.cat([mean_vec_new, abs_diff_vec_new, mean_sentence_vec], 1)
        else:
            X = torch.cat([mean_vec_new, abs_diff_vec_new], 1)
        # X = abs_diff_vec
        return X

    def preprocess_label(self, word_pairs, embedding_pairs, min_sim_pair=.7, max_sim_unpair=.5):
        tmp_word_pairs = word_pairs.copy()
        tmp_word_pairs['cos_sim'] = torch.cosine_similarity(embedding_pairs[:, 0, :].cpu(),
                                                            embedding_pairs[:, 1, :].cpu())
        tmp_word_pairs['label_corrected'] = tmp_word_pairs['label']
        tmp_word_pairs.loc[
            (tmp_word_pairs.cos_sim >= min_sim_pair) & (tmp_word_pairs.label == 0), 'label_corrected'] = .5
        tmp_word_pairs.loc[
            (tmp_word_pairs.cos_sim < max_sim_unpair) & (tmp_word_pairs.label == 1), 'label_corrected'] = .5
        df = tmp_word_pairs
        grouped = df.groupby(['left_word', 'right_word'], as_index=False).agg(
            {'label': ['mean'], 'cos_sim': ['mean'], 'label_corrected': ['mean']}).droplevel(1, 1)
        df = df.drop('label_corrected_mean', 1) if 'label_corrected_mean' in df.columns else df
        word_pairs_corrected = df.merge(grouped[['left_word', 'right_word'] + ['label_corrected']],
                                        on=['left_word', 'right_word'], suffixes=('', '_mean'), how='left')

        self.word_pairs_corrected = word_pairs_corrected
        self.aggregated = grouped
        tmp_res = word_pairs_corrected['label_corrected_mean']

        # tmp_res = (word_pairs_corrected['label_corrected_mean'] * 4 + word_pairs_corrected['label_corrected']) / 5
        # tmp_res =  word_pairs_corrected['label_corrected']

        return torch.tensor(tmp_res.values, dtype=torch.float).reshape([-1, 1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item: int):
        return self.X[item], self.y[item]


class NetAccoppiate(nn.Module):
    def __init__(self, sentence_embedding=False, size=768):
        super().__init__()
        # self.fc1 = nn.Linear(300*2+len(common_words_df.attribute.unique()), 300)
        if sentence_embedding:
            self.fc1 = nn.Linear(size * 3, 300)
        else:
            self.fc1 = nn.Linear(size * 2, 300)
        self.fc2 = nn.Linear(300, 64)
        self.dp2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, 32)
        self.dp3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x.view([x.shape[0], -1])))
        x = self.dp2(F.relu(self.fc2(x)))
        x = self.dp3(F.relu(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x


def train_save_net(model_files_path, reset_networks, data_loader, words_pairs_dict, emb_pairs_dict, device='cuda'):
    model = NetAccoppiate()
    tmp_path = os.path.join(model_files_path, 'net0.pickle')
    try:
        assert reset_networks == False, 'resetting networks'
        model.load_state_dict(torch.load(tmp_path,
                                         map_location=torch.device(device)))
    except Exception as e:
        print(e)

        batch_size = 128
        net = NetAccoppiate()
        net.to(device)
        criterion = nn.BCELoss().to(device)
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=.9)
        optimizer = optim.Adam(net.parameters(), lr=0.00001)

        train_dataset = data_loader
        valid_dataset = copy.deepcopy(train_dataset)
        valid_dataset.__init__(words_pairs_dict['valid'], emb_pairs_dict['valid'])

        dataloaders_dict = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
                            'valid': DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)}

        model, score_history, last_model = train_model(net,
                                                       dataloaders_dict, criterion, optimizer,
                                                       nn.MSELoss().to(device), num_epochs=150, device=device)
        # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=.9)
        # model, score_history, last_model = train_model(net,dataloaders_dict, criterion, optimizer,nn.MSELoss().to(device), num_epochs=150, device=device)

        out = net(valid_dataset.X.to(device))
        print(f'best_valid --> mean:{out.mean():.4f}  std: {out.std():.4f}')
        out = last_model(valid_dataset.X.to(device))
        print(f'last_model --> mean:{out.mean():.4f}  std: {out.std():.4f}')
        print('Save...')
        torch.save(model.state_dict(), tmp_path)


def train_model(model, dataloaders, criterion, optimizer, selection_loss, num_epochs=25, high_is_better=False,
                device='guess', ):
    if device == 'guess':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    acc_history = {'train': [], 'valid': []}
    loss_history = {'train': [], 'valid': []}
    eval_func_name = repr(selection_loss)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 1
    overfitting_counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        out = f'Epoch {epoch + 1:3d}/{num_epochs}: '
        # gc.collect()
        # torch.cuda.empty_cache()
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            i = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                i += 1
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)
                    try:
                        loss = criterion(outputs, labels)
                    except Exception as e:
                        print(e)
                        import pdb
                        pdb.set_trace()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += selection_loss(outputs, labels) * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            out += f'{phase} Loss: {epoch_loss:.4f} {eval_func_name}: {epoch_acc:.4f} std: {outputs.std():.4f}\t|\t'

            # deep copy the model
            if phase == 'valid' and (epoch_acc > best_acc if high_is_better else epoch_acc < best_acc):
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if epoch > 1:
                if phase == 'valid':
                    if (best_acc > epoch_acc) == high_is_better:
                        if overfitting_counter == 10:
                            break
                        overfitting_counter += 1
                    else:
                        overfitting_counter = 0

            acc_history[phase].append(epoch_acc)
            loss_history[phase].append(epoch_loss)
        print(out[:-3])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    last_model = copy.deepcopy(model)
    model.load_state_dict(best_model_wts)
    return model, acc_history, last_model
