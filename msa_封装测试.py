from utils.data import DataProcessor
import numpy as np
import utils
from utils import logger
import torch
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from model.MultiModalModel import MultiModalModel
from model.MambaModel import Mamba
from model.hypersphereRegularization import get_mma_loss
from torch.utils.tensorboard import SummaryWriter
from model.Simplemodel import Simple
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = Config(log_dir='/home/gyf/HC/Multimodal-Sentiment-Analysis/Pytorch/record', dropout_rate=0.7, batch_size=4, shuffle=True, learning_rate=0.00001, epochs=10000, random_seed=42, L1=1e-09, L2=1e-09)

logger = utils.Logger(log_dir=config.log_dir)
def entropy_loss(logits):
    """ Calculate the entropy loss to encourage high uncertainty in the outputs. """
    p = torch.sigmoid(logits)
    entropy = -p * torch.log(p + 1e-10) - (1 - p) * torch.log(1 - p + 1e-10)
    return entropy.mean()
def negative_entropy_loss(logits):
    """ Calculate the negative entropy loss to encourage low uncertainty in the outputs. """
    p = torch.sigmoid(logits)
    entropy = -p * torch.log(p + 1e-10) - (1 - p) * torch.log(1 - p + 1e-10)
    return -entropy.mean()  # Return negative entropy to minimize it
def train(model, train_loader, criterion, optimizer, device, epoch: int, lambda_L1=0.000001, lambda_L2=0.0000001):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for texts, audios, videos, labels, masks in train_loader:
        texts, audios, videos, labels, masks = (d.to(device) for d in (texts, audios, videos, labels, masks))
        optimizer.zero_grad()
        outputs = model(texts, audios, videos, masks)
        labels = labels.float()
        
        # Regular loss calculation
        loss = criterion(outputs, labels)

        # L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        # L2 regularization
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        
        ent_loss = entropy_loss(outputs)
        neg_ent_loss = -ent_loss
        # for name, m in model.named_modules():
        #     if isinstance(m, (nn.Linear, nn.Conv2d)):
        #         mma_loss = get_mma_loss(m.weight)
        #         loss = loss + 0.00007 * mma_loss
        # if epoch < 60:
        #     loss = loss + lambda_L1 * l1_norm + lambda_L2 * l2_norm + 0.01 * ent_loss
        # else:
        #     loss = loss + lambda_L1 * l1_norm + lambda_L2 * l2_norm +  10 * ent_loss
        loss = loss + lambda_L1 * l1_norm + lambda_L2 * l2_norm
        if epoch > 60:
            loss +=  0.1*neg_ent_loss
        loss.backward()
        optimizer.step() 

        masks_expanded = masks.unsqueeze(-1).expand_as(labels)
        total_loss += loss.item()
        pred = torch.sigmoid(outputs) > 0.5
        correct += ((pred == labels.byte()) * masks_expanded.byte()).sum().item()
        total += masks_expanded.sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Accuracy', accuracy, epoch) 
    return avg_loss, accuracy

from sklearn.metrics import recall_score, f1_score

def evaluate(model, test_loader, criterion, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, audios, videos, labels, masks in test_loader:
            texts, audios, videos, labels, masks = texts.to(device), audios.to(device), videos.to(device), labels.to(device), masks.to(device)
            
            outputs = model(texts, audios, videos, masks)
            masks_expanded = masks.unsqueeze(-1).expand_as(labels)
            loss = criterion(outputs * masks_expanded, labels * masks_expanded)
            total_loss += loss.item()
            
            pred = torch.sigmoid(outputs) > 0.5
            correct += ((pred == labels.byte()) * masks_expanded.byte()).sum().item()
            total += masks_expanded.sum().item()

            # Collecting all predictions and labels for recall and F1 calculation
            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0

    # Converting lists to numpy arrays for metric calculation
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Flattening the arrays to remove any extra dimensions caused by masks
    all_preds = all_preds.flatten()
    all_labels = all_labels.flatten()

    # Calculating recall and F1 score
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    writer.add_scalar('Test/Loss', avg_loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)
    writer.add_scalar('Test/Recall', recall, epoch)
    writer.add_scalar('Test/F1', f1, epoch)
    
    return avg_loss, accuracy, recall, f1


def create_data_loader(train_text, train_audio, train_video, train_label, train_mask, batch_size=7, shuffle=True):
    train_text_tensor = torch.tensor(train_text, dtype=torch.float32)
    train_audio_tensor = torch.tensor(train_audio, dtype=torch.float32)
    train_video_tensor = torch.tensor(train_video, dtype=torch.float32)
    train_label_tensor = torch.tensor(train_label, dtype=torch.long)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.float32)

    train_dataset = TensorDataset(train_text_tensor, train_audio_tensor, train_video_tensor, train_label_tensor, train_mask_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader

if __name__ == "__main__":
    writer = SummaryWriter(log_dir='/home/gyf/HC/tb_record')
    input_name = input("Enter a Name for the log file, 'no' to disable logging, or just press Enter to use default ('test'): ")
    input_name = input_name.strip()
    if input_name.lower() == 'no':
        print("Logging is disabled.")
    else:
        input_name = input_name if input_name else 'test'
        print("Logging setup complete.")
        logger = utils.Logger(log_dir=config.log_dir, Name=input_name)
        logger.save_console_output()
    print(config)

    (train_text, train_label, test_text, test_label, max_utt_len,
     train_len, test_len, train_audio, test_audio,
     train_video, test_video) = DataProcessor.load_data(42)
    # (train_data, test_data, train_audio, test_audio, train_text, test_text, 
    # train_video, test_video, train_label, test_label, train_len, test_len,
    # train_mask, test_mask, train_len, test_len) = DataProcessor.get_raw_data(data='mosei', classes=3)
    train_label, test_label = DataProcessor.create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))
    train_mask, test_mask = DataProcessor.create_mask(train_text, test_text, train_len, test_len)
    train_text, test_text = DataProcessor.split_dataset(train_text, test_text)
    train_audio, test_audio = DataProcessor.split_dataset(train_audio, test_audio)
    train_video, test_video = DataProcessor.split_dataset(train_video, test_video)
    train_label, test_label = DataProcessor.split_dataset(train_label, test_label)
    train_mask, test_mask = DataProcessor.split_dataset(train_mask, test_mask)
    print("Data loaded and processed.")

    train_loader = create_data_loader(
        train_text=train_text,
        train_audio=train_audio,
        train_video=train_video,
        train_label=train_label,
        train_mask=train_mask,
        batch_size=config.batch_size,
        shuffle=config.shuffle
    )
    test_loader = create_data_loader(
        train_text=test_text,
        train_audio=test_audio,
        train_video=test_video,
        train_label=test_label,
        train_mask=test_mask,
        batch_size=config.batch_size,
        shuffle=False
    )

    model_name = input("Enter the model name (MultiModalModel or Mamba or simple): ").strip()
    if model_name == "Mamba1" or model_name == "1":
        model = MultiModalModel(dropout_rate=config.dropout_rate).to(device)
    elif model_name == "Mamba" or model_name == "2":
        model = Mamba(dropout_rate=config.dropout_rate).to(device)
    elif model_name =="simple" or model_name == "3":
        model = Simple(dropout_rate=config.dropout_rate).to(device)
    else:
        raise ValueError("Invalid model name")

    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_test_accuracy = 0.0
    best_test_recall = 0.0
    best_test_f1 = 0.0

    for epoch in range(config.epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, lambda_L2=config.L2, lambda_L1=config.L1, epoch=epoch)
        test_loss, test_accuracy, test_recall, test_f1 = evaluate(model, test_loader, criterion, epoch)
        
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            writer.add_scalar('Best/Test Accuracy', best_test_accuracy, epoch)
        
        if test_recall > best_test_recall:
            best_test_recall = test_recall
            writer.add_scalar('Best/Test Recall', best_test_recall, epoch)
            
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            writer.add_scalar('Best/Test F1 Score', best_test_f1, epoch)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
            f'Test Accuracy: {test_accuracy:.2f}%, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}, '
            f'Best Test Accuracy: {best_test_accuracy:.2f}%, Best Test Recall: {best_test_recall:.4f}, '
            f'Best Test F1 Score: {best_test_f1:.4f}')
