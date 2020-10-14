import tensorflow as tf
import numpy


class Voc():
    def __init__(self, strings):
        self.num_to_token = []
        self.token_to_num = {}
        self.token_to_num["_BOS_"] = 0
        self.num_to_token.append("_BOS_")
        self.token_to_num["_EOS_"] = 1
        self.num_to_token.append("_EOS_")
        self.token_to_num["_UNK_"] = 2
        self.num_to_token.append("_UNK_")
        now = 3
        for i in range(len(strings)):
            mass = strings[i].split(' ')
            for j in mass:
                if self.token_to_num.get(j, -1) == -1:
                    self.token_to_num[j] = now
                    now += 1
                    self.num_to_token.append(j)
    
    def strings_to_tensor(self, strings):
        res = []
        max1 = 0
        for string in strings:
            mass = string.split(' ')
            now = [self.token_to_num["_BOS_"]]
            for i in mass:
                now.append(self.token_to_num.get(i, "_UNK_"))
            now.append(self.token_to_num["_EOS_"])
            max1 = max(max1, len(now))
            res.append(now)
        for i in range(len(res)):
            res[i] += [self.token_to_num["_EOS_"]] * (max1 - len(res[i]))
        return tf.convert_to_tensor(res, dtype="int32")
    
    def tensor_to_strings(self, tensor):
        array = tensor.numpy().tolist()
        res = []
        for mass in array:
            string = []
            for i in mass:
                if i not in [self.token_to_num["_EOS_"], self.token_to_num["_BOS_"]]:
                    string.append(self.num_to_token[i])
            res.append(' '.join(string))
        return res
            
            
    
    def __len__(self):
        return len(self.token_to_num)