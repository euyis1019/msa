import numpy as np
import os
import pickle
class DataProcessor:
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


    @staticmethod
    def create_one_hot_labels(train_label, test_label):
        """
        Converts train and test labels into one-hot encoded format.

        Args:
            train_label (np.array): 2D array of train labels.
            test_label (np.array): 2D array of test labels.

        Returns:
            tuple: 3D arrays of one-hot encoded train and test labels.
        """
        maxlen = int(max(train_label.max(), test_label.max()))
        train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
        test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

        for i in range(train_label.shape[0]):
            for j in range(train_label.shape[1]):
                train[i, j, train_label[i, j]] = 1

        for i in range(test_label.shape[0]):
            for j in range(test_label.shape[1]):
                test[i, j, test_label[i, j]] = 1

        return train, test
    
    @staticmethod   
    def transform_labels(train_label, train_len, test_label, test_len):
        """
        Transform multi-label data into binary labels based on specific conditions.
        
        Parameters:
        - train_label (np.array): The training labels array of shape (2250, 98, 3).
        - train_len (np.array): Array indicating the number of valid samples in each training batch.
        - test_label (np.array): The testing labels array of shape (678, 98, 3).
        - test_len (np.array): Array indicating the number of valid samples in each testing batch.

        Returns:
        - (np.array, np.array): A tuple containing transformed training and testing labels.
        """
        # Initialize new label arrays
        new_train_label = np.zeros((2250, 98, 2))
        new_test_label = np.zeros((678, 98, 2))
        
        # Transform training labels
        for i in range(train_label.shape[0]):
            for j in range(train_len[i]):
                if train_label[i][j][1] == 1 or train_label[i][j][0] == 1:
                    new_train_label[i][j] = [1, 0]
                else:
                    new_train_label[i][j] = [0, 1]

        # Transform testing labels
        for i in range(test_label.shape[0]):
            for j in range(test_len[i]):
                if test_label[i][j][1] == 1 or test_label[i][j][0] == 1:
                    new_test_label[i][j] = [1, 0]
                else:
                    new_test_label[i][j] = [0, 1]
                    
        return new_train_label, new_test_label



    @staticmethod
    def create_mask(train_data, test_data, train_length, test_length):
        """
        Creates masks for training and testing data based on their lengths.

        Args:
            train_data (np.array): 2D array of train data.
            test_data (np.array): 2D array of test data.
            train_length (list): List of lengths for each train sample.
            test_length (list): List of lengths for each test sample.

        Returns:
            tuple: 2D arrays of masks for train and test data.
        """
        train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
        for i in range(len(train_length)):
            train_mask[i, :train_length[i]] = 1.0

        test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
        for i in range(len(test_length)):
            test_mask[i, :test_length[i]] = 1.0

        return train_mask, test_mask
    
        
    @staticmethod
    def split_dataset_old(data, train_ratio=1):
        """
        Split data into training and development (validation) sets.
        :param data: The data to be split.
        :param train_ratio: Ratio of data to be used as training data.
        :return: A tuple of (training_data, development_data).
        """
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]

    @staticmethod
    def get_raw_data(data= 'mosei', classes=3):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        mode = 'audio'
# 使用os.path.join构建完整的文件路径
        audio_path = os.path.join(current_dir, 'dataset', data, 'raw', f'{mode}_{classes}way.pickle')
        with open(audio_path, 'rb') as handle:
            u = pickle._Unpickler(handle)
            u.encoding = 'latin1'
            (audio_train, train_label, _, _, audio_test, test_label, _, train_length, _, test_length, _, _, _) = u.load()

        mode = 'text'
        text_path = os.path.join(current_dir, 'dataset', data, 'raw', f'{mode}_{classes}way.pickle')
        with open(text_path, 'rb') as handle:
            u = pickle._Unpickler(handle)
            u.encoding = 'latin1'
            (text_train, train_label, _, _, text_test, test_label, _, train_length, _, test_length, _, _, _) = u.load()

        mode = 'video'
        video_path = os.path.join(current_dir, 'dataset', data, 'raw', f'{mode}_{classes}way.pickle')
        with open(video_path, 'rb') as handle:
            u = pickle._Unpickler(handle)
            u.encoding = 'latin1'
            (video_train, train_label, _, _, video_test, test_label, _, train_length, _, test_length, _, _, _) = u.load()
        train_data = np.concatenate((audio_train, video_train, text_train), axis=-1)
        test_data = np.concatenate((audio_test, video_test, text_test), axis=-1)

        train_label = train_label.astype('int')
        test_label = test_label.astype('int')
        print(train_data.shape)
        print(test_data.shape)
        train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
        for i in range(len(train_length)):
            train_mask[i, :train_length[i]] = 1.0

        test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
        for i in range(len(test_length)):
            test_mask[i, :test_length[i]] = 1.0

#        train_label, test_label = createOneHot(train_label, test_label)

#        print('train_mask', train_mask.shape)

        train_label, test_label  = DataProcessor.transform_labels(train_label, train_length, test_label, test_length)

        seqlen_train = train_length
        seqlen_test = test_length

        return train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask, train_length, test_length
    @staticmethod   
    def split_dataset(train, test, ratio=0.5):
        """
        Split a portion of the test dataset into the train dataset based on the given ratio.
        
        :param train: Training data (numpy array).
        :param test: Testing data (numpy array).
        :param ratio: Ratio of test data to be moved to training data.
        :return: Updated train and test datasets as numpy arrays.
        """
        # Calculate the number of samples to split from test to train
        #print(ratio,"asdoifjasdoif")
        split_idx = int(len(test) * ratio)
        
        # Split the test data
        test_to_merge = test[:split_idx]
        remaining_test_data = test[split_idx:]
        
        # Merge the split test data with the train data
        train = np.concatenate((train, test_to_merge), axis=0)
        
        return train, remaining_test_data
if __name__ == '__main__':

    def get_raw_data(data="mosei", classes = 3):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        dataset_dir = os.path.join(current_dir, 'dataset', data, 'raw')

        mode = 'audio'
        file_path = os.path.join(dataset_dir, f'{mode}_{classes}way.pickle')
        with open(file_path, 'rb') as handle:
            u = pickle._Unpickler(handle)
            u.encoding = 'latin1'
            (audio_train, train_label, _, _, audio_test, test_label, _, train_length, _, test_length, _, _, _) = u.load()
            print("\n audio训练标签shape")
            print(train_label.shape)   # (2250, 98, 3)
            print("\n audio测试标签shape")
            print(test_label.shape)  # (678, 98, 3)

        mode = 'text'
        file_path = os.path.join(dataset_dir, f'{mode}_{classes}way.pickle')
        with open(file_path, 'rb') as handle:
            u = pickle._Unpickler(handle)
            u.encoding = 'latin1'
            (text_train, train_label, _, _, text_test, test_label, _, train_length, _, test_length, _, _, _) = u.load()
            print("\n text训练标签shape")
            print(train_label.shape)  # (2250, 98, 3)
            print("\n text测试标签shape")
            print(test_label.shape)  # (678, 98, 3)

        mode = 'video'
        file_path = os.path.join(dataset_dir, f'{mode}_{classes}way.pickle')
        with open(file_path, 'rb') as handle:
            u = pickle._Unpickler(handle)
            u.encoding = 'latin1'
            (video_train, train_label, _, _, video_test, test_label, _, train_length, _, test_length, _, _, _) = u.load()
            print("\n video训练标签shape")
            print(train_label.shape)  # (2250, 98, 3)
            print("\n video测试标签shape")
            print(test_label.shape)  # (678, 98, 3)


        train_data = np.concatenate((audio_train, video_train, text_train), axis=-1)
        print("---------------------")
        print("train_data：", train_data.shape)    #  (2250, 98, 409)
        test_data = np.concatenate((audio_test, video_test, text_test), axis=-1)
        print("test_data:", test_data.shape)      # (678, 98, 409)

        train_label = train_label.astype('int')
        test_label = test_label.astype('int')
        print(train_data.shape)   # (2250, 98, 409)
        print(test_data.shape)    # (678, 98, 409)
        train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
        for i in range(len(train_length)):
            train_mask[i, :train_length[i]] = 1.0

        test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
        for i in range(len(test_length)):
            test_mask[i, :test_length[i]] = 1.0

        # train_label, test_label = createOneHotMosei3way(train_label, test_label)
        print("--------------------------------------")
        print(train_label.shape)
        print(test_label.shape)
        print(train_label[3][4])

        print('train_mask', train_mask.shape)

        seqlen_train = train_length
        seqlen_test = test_length
        new_train_label = np.zeros((2250, 98, 2))
        #sumA,sumB=0,0
        for i in range(train_label.shape[0]):  # 遍历所有批次
            for j in range(train_len[i]):  # 遍历每个批次的实际样本数
                if train_label[i][j][1] == 1 or train_label[i][j][0] == 1:
                    new_train_label[i][j] = [1, 0]  # 当第一位或第二位为1时，设置新标签为 [1, 0]
                    #sumA += 1
                else:
                    new_train_label[i][j] = [0, 1]  # 其他情况，设置新标签为 [0, 1]
                    #sumB += 1
        new_test_label = np.zeros((678, 98, 2))
        # 遍历所有批次
        for i in range(test_label.shape[0]):  # 遍历所有批次
            for j in range(test_len[i]):  # 遍历每个批次的实际样本数
                # 检查原始标签的第一位或第二位是否为1
                if test_label[i][j][1] == 1 or test_label[i][j][0] == 1:
                    new_test_label[i][j] = [1, 0]  # 当第一位或第二位为1时，设置新标签为 [1, 0]
                    #sumA += 1
                else:
                    new_test_label[i][j] = [0, 1]  # 其他情况，设置新标签为 [0, 1]
                    #sumB += 1
        return train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask

        
    
    
    #train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = DataProcessor.get_raw_data()


    train_data, test_data, train_audio, test_audio, train_text, test_text, train_video, test_video, train_label, test_label, \
max_utt_len, seqlen_test, train_mask, test_mask, train_len, test_len = DataProcessor.get_raw_data('mosei', 3)
    logger = Logger()
    logger.log_var_shapes(
        train_data = train_data,
        test_data = test_data,
        train_audio = train_audio,
        test_audio = test_audio,
        train_text = train_text,
        test_text = test_text,
        train_video = train_video,
        test_video = test_video,
        train_label = train_label,
        test_label = test_label,
        max_utt_len = max_utt_len,
        seqlen_test = seqlen_test,
    )
    # A,B,C=0,0,0
    # print(train_len[2249])
    # for i in range(0,98):
    #     if i == 13:
    #         print("13")
    #     print(train_label[2249][i])
    # sumA,sumB,sumC=0,0,0
    # for i in range(0,2250):
    #     A,B,C=0,0,0
    #     print("batch", i)
    #     for j in range(0,train_len[i]):
    #         if train_label[i][j][0] == 1:
    #             A = A + 1
    #         if train_label[i][j][1] == 1:
    #             B = B + 1
    #         if train_label[i][j][2] == 1:
    #             C = C + 1
    #     print("in",train_len[i],A,B,C)
    #     sumA+=A
    #     sumB+=B
    #     sumC+=C
    #     print("xxxxxxx")
    # print(sumA, sumB, sumC)
    #4709 3502 7977

    # 假设 train_label 是一个 (2250, 98, 2) 形状的 NumPy 数组
# new_train_label = np.zeros((2250, 98, 2))
# sumA,sumB=0,0
# for i in range(train_label.shape[0]):  # 遍历所有批次
#     for j in range(train_len[i]):  # 遍历每个批次的实际样本数
#         if train_label[i][j][1] == 1 or train_label[i][j][0] == 1:
#             new_train_label[i][j] = [1, 0]  # 当第一位或第二位为1时，设置新标签为 [1, 0]
#             sumA += 1
#         else:
#             new_train_label[i][j] = [0, 1]  # 其他情况，设置新标签为 [0, 1]
#             sumB += 1

# print(sumA,sumB)
# print()