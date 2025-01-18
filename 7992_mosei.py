from utils.data import DataProcessor
import numpy as np
import utils
from utils import logger
import torch
import torch.nn.init as init
from torch.utils.data import  DataLoader
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import BiModalAttention, TriModalAttention, SelfAttention, ResidualGRU, ResidualAttention
from utils import Config,S6
from torchsummary import summary
from utils.vis_utils import VisdomLogger
config = Config()
config.learning_rate = 0.001
config.batch_size = 8
#vis_logger = VisdomLogger("trimodal_attention_models2.py")
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

class MultiModalModel(nn.Module):
    def __init__(self, dropout_rate=0.7):
        super(MultiModalModel, self).__init__()
        # Modality-specific branches
        self.text_branch = nn.Sequential(
            nn.Linear(300, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.audio_branch = nn.Sequential(
            nn.Linear(74, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.video_branch = nn.Sequential(
            nn.Linear(35, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # RNN layers with residual connections
        self.rnn_text = ResidualGRU(input_size=300, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        self.rnn_audio = ResidualGRU(input_size=300, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        self.rnn_video = ResidualGRU(input_size=300, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)

        # self.s6_text = S6(seq_len=98, d_model=300, state_size=300, device=device)
        # self.s6_audio = S6(seq_len=98, d_model=300, state_size=300, device=device)
        # self.s6_video = S6(seq_len=98, d_model=300, state_size=300, device=device)

        # Attention mechanisms
        self.bi_modal_attention = ResidualAttention(BiModalAttention(size=600))
        self.tri_modal_attention = ResidualAttention(TriModalAttention(feature_size=300))
        self.self_attention = ResidualAttention(SelfAttention(size=600))

        # Early fusion and final output layer
        self.early_fuse = nn.Sequential(
            nn.Linear(900, 50),  # Concatenation of three 300-sized outputs
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(1800,600),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(600, 300),  # Adjust based on the combined attention outputs
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(300, 300),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(300, 50),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(50, 2),  # Final output for 2 classes
        )

    def forward(self, text, audio, video, mask=None):
        # Branch processing
        text_embeddings = self.text_branch(text)
        audio_embeddings = self.audio_branch(audio)
        video_embeddings = self.video_branch(video)

        # RNN processing with residuals
        # text_output = self.s6_text(text_embeddings)
        # audio_output = self.s6_audio(audio_embeddings)
        # video_output = self.s6_video(video_embeddings)
        text_output = self.rnn_text(text_embeddings)
        audio_output = self.rnn_audio(audio_embeddings)
        video_output = self.rnn_video(video_embeddings)

        #torch.Size([32, 98, 300])
        # Attention computations
        bi_modal_output = self.bi_modal_attention(text_output, audio_output)
        tri_modal_output = self.tri_modal_attention(text_output, audio_output, video_output)
        self_attention_output = self.self_attention(text_output)

        # Combining attention outputs
        combined_features = torch.cat((bi_modal_output, tri_modal_output, self_attention_output), dim=2)

        # Early fusion and final output computation
        #early_output = self.early_fuse(torch.cat((text_output, audio_output, video_output), dim=2))
        logits = self.output_layer(combined_features)

        return logits
#----
#----
def train(model, train_loader, criterion, optimizer, device, epoch:int, lambda_L1=0.0000001, lambda_L2=0.000000001):
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
        # if epoch%100 == 0:
        #     loss = loss + lambda_L1 * l1_norm + lambda_L2 * l2_norm + epoch/1000
        # # Total loss with regularization
        # else:
        loss = loss + lambda_L1 * l1_norm + lambda_L2 * l2_norm

        loss.backward()
        optimizer.step()
        
        masks_expanded = masks.unsqueeze(-1).expand_as(labels)
        total_loss += loss.item()
        pred = torch.sigmoid(outputs) > 0.5
        correct += ((pred == labels.byte()) * masks_expanded.byte()).sum().item()
        total += masks_expanded.sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0
    return avg_loss, accuracy

# 测试函数
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, audios, videos, labels, masks in test_loader:
            texts, audios, videos, labels, masks = texts.to(device), audios.to(device), videos.to(device), labels.to(device), masks.to(device)
            
            # Model prediction
            outputs = model(texts, audios, videos, masks)
            
            # Ensuring mask is correctly sized for applying to every element
            masks_expanded = masks.unsqueeze(-1).expand_as(labels)
            
            # Compute the loss only on the masked elements
            loss = criterion(outputs * masks_expanded, labels * masks_expanded)
            total_loss += loss.item()
            
            # Calculate accuracy
            pred = torch.sigmoid(outputs) > 0.5  # Apply threshold to obtain binary predictions
            correct += ((pred == labels.byte()) * masks_expanded.byte()).sum().item()
            total += masks_expanded.sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0

    return avg_loss, accuracy
def create_data_loader(train_text, train_audio, train_video, train_label, train_mask, batch_size=7, shuffle=True):
    # 转换数据为 torch.Tensor
    train_text_tensor = torch.tensor(train_text, dtype=torch.float32)
    train_audio_tensor = torch.tensor(train_audio, dtype=torch.float32)
    train_video_tensor = torch.tensor(train_video, dtype=torch.float32)
    train_label_tensor = torch.tensor(train_label, dtype=torch.long)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.float32)

    # 创建数据集
    train_dataset = TensorDataset(train_text_tensor, train_audio_tensor, train_video_tensor, train_label_tensor, train_mask_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader


if __name__ == "__main__":
    input_name = input("Enter a Name for the log file, 'no' to disable logging, or just press Enter to use default ('test'): ")
    input_name = input_name.strip()
    if input_name.lower() == 'no':
        print("Logging is disabled.")
    else:
        input_name = input_name if input_name else 'test'
        print("Logging setup complete.")
        logger = utils.Logger(log_dir=config.log_dir, Name =input_name)
        logger.save_console_output()
    print(config)
    # train_data: (2250, 98, 409)
    # test_data: (678, 98, 409)
    # train_audio: (2250, 98, 74)
    # test_audio: (678, 98, 74)
    # train_text: (2250, 98, 300)
    # test_text: (678, 98, 300)
    # train_video: (2250, 98, 35)
    # test_video: (678, 98, 35)
    # train_label: (2250, 98, 2)
    # test_label: (678, 98, 2)
    # max_utt_len: (2250,)
    # seqlen_test: (678,)
    (train_data, test_data, train_audio, test_audio, train_text, test_text, 
    train_video, test_video, train_label, test_label, train_len, test_len,
    train_mask, test_mask, train_len, test_len) = DataProcessor.get_raw_data(data='mosei', classes=3)
    #train_label, test_label = DataProcessor.create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))
    #train_mask, test_mask = DataProcessor.create_mask(train_text, test_text, train_len, test_len)
    #Divide Dataset for train and dev. It is not neccessary.
    # # 划分训练集和开发集 
    # train_text, dev_text = DataProcessor.split_dataset(train_text)
    # train_audio, dev_audio = DataProcessor.split_dataset(train_audio)
    # train_video, dev_video = DataProcessor.split_dataset(train_video)
    # train_label, dev_label = DataProcessor.split_dataset(train_label)
    # train_mask, dev_mask = DataProcessor.split_dataset(train_mask)

    print("Data loaded and processed.")
 # for mode in ['MMMU_BA', 'MMUU_SA', 'MU_SA', 'None']:
    #     train(mode)
    mode = "sdq"
    #train(mode)
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
        shuffle=False  # 在测试时通常不需要打乱数据
    )

    '''train_text: (62, 63, 100)
    train_audio: (62, 63, 73)
    train_video: (62, 63, 100)
    train_label: (62, 63, 2)
    train_mask: (62, 63)
    test_text: (31, 63, 100)
    test_audio: (31, 63, 73)
    test_video: (31, 63, 100)
    test_label: (31, 63, 2)
    test_mask: (31, 63)'''
    model = MultiModalModel(dropout_rate=config.dropout_rate).to(device)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    summary(model, [(98, 300), (98, 74), (98, 35)], device=device_str)

# 同样假设你已经定义了你的模型 model 和配置 config
    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.learning_rate)

    criterion = nn.BCEWithLogitsLoss()
     #训练模型
    epochs = 10  # 可以根据需要调整
    # 清空Visdom窗口并重新绘图
    #vis_logger.close_all()
    best_test_accuracy = 0.0
    for epoch in range(config.epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, lambda_L2=config.L2, lambda_L1=config.L1, epoch = epoch)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
    # 使用Visdom记录
       # vis_logger.log_metrics(epoch, train_loss, train_accuracy, test_loss, test_accuracy)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Best Test Accuracy: {best_test_accuracy:.2f}%')

    #model_graph = draw_graph(model, input_size=x.shape, expand_nested=True, save_graph=True, filename="torchview", directory=".")
