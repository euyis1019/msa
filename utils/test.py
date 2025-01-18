import os
import pickle
import numpy as np

class DataLoader:
    @staticmethod
    def load_data(seed=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        text_path = os.path.join(current_dir, 'input/text.pickle')
        audio_path = os.path.join(current_dir, 'input/audio.pickle')
        video_path = os.path.join(current_dir, 'input/video.pickle')

        # Load the datasets
        (train_text, train_label, test_text, test_label, max_utt_len, train_len, test_len) = pickle.load(open(text_path, 'rb'))
        (train_audio, _, test_audio, _, _, _, _) = pickle.load(open(audio_path, 'rb'))
        (train_video, _, test_video, _, _, _, _) = pickle.load(open(video_path, 'rb'))

        # Shuffle data with the same seed to ensure alignment across modalities
        if seed is not None:
            np.random.seed(seed)
            perm_train = np.random.permutation(train_text.shape[0])
            perm_test = np.random.permutation(test_text.shape[0])

            train_text = train_text[perm_train, :, :]
            train_audio = train_audio[perm_train, :, :]
            train_video = train_video[perm_train, :, :]
            train_label = train_label[perm_train, :]

            test_text = test_text[perm_test, :, :]
            test_audio = test_audio[perm_test, :, :]
            test_video = test_video[perm_test, :, :]
            test_label = test_label[perm_test, :]

        # Return all the data loaded
        return (train_text, train_label, test_text, test_label, max_utt_len, train_len, test_len,
                train_audio, test_audio, train_video, test_video)

def split_and_merge_datasets(train, test, ratio):
    """
    Split a portion of the test dataset into the train dataset based on the given ratio.
    
    :param train: Training data (numpy array).
    :param test: Testing data (numpy array).
    :param ratio: Ratio of test data to be moved to training data.
    :return: Updated train and test datasets as numpy arrays.
    """
    # Calculate the number of samples to split from test to train
    split_idx = int(len(test) * ratio)
    
    # Split the test data
    test_to_merge = test[:split_idx]
    remaining_test_data = test[split_idx:]
    
    # Merge the split test data with the train data
    train = np.concatenate((train, test_to_merge), axis=0)
    
    return train, remaining_test_data
# Example usage
data_loader = DataLoader()

# Load data with shuffling
train_text, train_label, test_text, test_label, max_utt_len, train_len, test_len, train_audio, test_audio, train_video, test_video = data_loader.load_data(seed=42)

# Display the shapes of the data
print("Shapes of the data after shuffling:")
print("Train Text:", train_text.shape)
print("Train Audio:", train_audio.shape)
print("Train Video:", train_video.shape)
print("Train Label:", train_label.shape)
print("Test Text:", test_text.shape)
print("Test Audio:", test_audio.shape)
print("Test Video:", test_video.shape)
print("Test Label:", test_label.shape)

ratio = 0.4
    
train_text, test_text = split_and_merge_datasets(train_text, test_text, ratio)
train_audio, test_audio = split_and_merge_datasets(train_audio, test_audio, ratio)
train_video, test_video = split_and_merge_datasets(train_video, test_video, ratio)
train_label, test_label = split_and_merge_datasets(train_label, test_label, ratio)

    
    # 打印结果以验证
print(f"New train text shape: {train_text.shape}")
print(f"New test text shape: {test_text.shape}")
print(f"New train audio shape: {train_audio.shape}")
print(f"New test audio shape: {test_audio.shape}")
print(f"New train video shape: {train_video.shape}")
print(f"New test video shape: {test_video.shape}")
print(f"New train label shape: {train_label.shape}")
print(f"New test label shape: {test_label.shape}")
