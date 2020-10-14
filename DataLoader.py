from random import shuffle, randint


class DataLoader():
    def __init__(self, data, voc_src, voc_dst):
        self.voc_src = voc_src
        self.voc_dst = voc_dst
        shuffle(data)
        amount_test = len(data) // 20
        data_test = data[:amount_test]
        data_train = data[amount_test:]
        data_test.sort(key=lambda x: len(x[0].split(' ')))
        data_train.sort(key=lambda x: len(x[0].split(' ')))
        self.test_x = list(map(lambda x: x[0], data_test))
        self.test_y = list(map(lambda x: x[1], data_test))
        self.train_x = list(map(lambda x: x[0], data_train))
        self.train_y = list(map(lambda x: x[1], data_train))
    
    def get_train_batch(self, batch_size):
        id1 = randint(0, len(self.train_x) - batch_size)
        res_x = self.train_x[id1:id1 + batch_size]
        res_y = self.train_y[id1:id1 + batch_size]
        return self.voc_src.strings_to_tensor(res_x), self.voc_dst.strings_to_tensor(res_y)
    
    def get_test_batch(self, batch_size):
        id1 = randint(0, len(self.test_x) - batch_size)
        res_x = self.test_x[id1:id1 + batch_size]
        res_y = self.test_y[id1:id1 + batch_size]
        return self.voc_src.strings_to_tensor(res_x), self.voc_dst.strings_to_tensor(res_y)