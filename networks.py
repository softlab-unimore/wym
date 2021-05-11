import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F



class NetSpaiate(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(300 + len(common_words_df.attribute.unique()), 300)
        self.fc1 = nn.Linear(300, 300)
        self.dp1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(300, 64)
        self.dp2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(64, 32)
        self.dp3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(32, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = x.view([x.shape[0], -1])
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x = self.dp2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.out_act(x)
        return x



class NetAccoppiate(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(300*2+len(common_words_df.attribute.unique()), 300)
        self.fc1 = nn.Linear(300 * 2, 300)
        self.fc2 = nn.Linear(300, 64)
        self.dp2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, 32)
        self.dp3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x.view([x.shape[0], -1])))
        x = self.dp2(F.relu(self.fc2(x)))
        x = self.dp3(F.relu(self.fc3(x)))
        x = F.sigmoid(self.fc4(x))
        return x


def train_model(model, dataloaders, criterion, optimizer, selection_loss, num_epochs=25, high_is_better=False, device='guess',):
    if device == 'guess':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    acc_history = {'train': [], 'valid': []}
    loss_history = {'train': [], 'valid': []}
    eval_func_name = repr(selection_loss)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 1
    overfitting_counter = 0

    for epoch in range(num_epochs):
        out = f'Epoch {epoch+1:3d}/{num_epochs}: '
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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
                    loss = criterion(outputs, labels)

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
                best_model_wts = copy.deepcopy(model.state_dict())
            if epoch > 1:
                if phase == 'valid':
                    if ( best_acc > epoch_acc ) == high_is_better:
                        if overfitting_counter == 50:
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
