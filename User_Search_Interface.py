import time
import numpy as np
# import pandas as pd
from utils import MinHash, Lis_help
import random
# i built a interface for two based algorithm


from LSH_MinHash import LSH_Search_MH_Jaccard  # using the minhash table
from LSH_RP import LSH_Search_RP_cosine  # using the radom projection


# you can link to the my source-code to view the wrapper principle realization

# encode-func for jaccard's text hashing
def f_ele(x):
    x = str(x)
    return x.encode('utf8')


class User_Search(object):
    def __init__(self, npy_data, seed):
        '''
        @ Build my new data pack structure:
        u1 :[.....]
        u2 :[.....]

        @eg: ele('1',[...])
        '''
        # All params are from pdf provided, all data i evaluate them already
        # movieids_sum=17770
        # max-user_sum=103703
        self.raw_data = npy_data  # For backup
        self.seed = seed
        self.data_size = npy_data.shape
        self.result_pack = []
        self.movie_ids = 17770
        self.users_sum = 103703
        self.time_limit = 1500  # (Seconds) We can set this time watch dog to prevent the overtime when online testing
        random.seed(seed)
        np.random.seed(seed)

    def write_to_file(self, filepath):
        """write a list to result file
        """
        self.result_pack = list(set(self.result_pack))  # Remove the same ones
        self.result_pack = Lis_help(self.result_pack)
        with open(filepath, "w") as f:
            sumlgth = len(self.result_pack)
            self.result_pack[-1] = self.result_pack[-1].replace('\n', '')  # remove the \n in the last line
            f.writelines(self.result_pack)

    def result_wrapper(self, key, search_result):
        """search_result is a list @eg:['1','2',...'123']
        key is a number str @eg:'123'
        """
        key_num = int(key)
        search_result = list(map(int, search_result))  # Change all into int
        sum_records = 0
        for item in search_result:
            if item > key_num:
                self.result_pack.append(str(key_num) + ', ' + str(item) + '\n')
                sum_records += 1
            elif item < key_num:
                # else:
                self.result_pack.append(str(item) + ', ' + str(key_num) + '\n')
                sum_records += 1
        return sum_records

    # This is an another Interface for result wrapper!!!!
    def result_wrapper2(self, key, search_result):
        """search_result is a list @eg:[('0', (1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 1.0),(...)]
        """
        key_num = int(key)
        search_result_new = []
        for i in search_result:
            search_result_new.append(int(i[0]))
            # print(i[2]) #Here is the score
        sum_records = 0
        for item in search_result_new:
            if item > key_num:
                self.result_pack.append(str(key_num) + ', ' + str(item) + '\n')
                sum_records += 1
            elif item < key_num:
                # else:
                self.result_pack.append(str(item) + ', ' + str(key_num) + '\n')
                sum_records += 1
        return sum_records

    def cosine(self, threshold):
        print('=' * 15)
        print('@ Data task: Netflix Challenge (www.netflixprize.com).')
        print('@ Program title: <Task2> RandomProjection-LSH Search Using cosine distance-func')
        print('Attention!!!! You need at least 20-mins to run this program')
        print('=' * 15)
        start = time.clock()  # Dear, When Summit, dont comment it!!

        feature_dims = 1700  # The feature compare dims

        lsh = LSH_Search_RP_cosine(10, feature_dims, 90)
        _x_features = [i for i in range(self.movie_ids)]
        random.shuffle(_x_features)
        # print(_x_features)
        midx = np.array(_x_features, dtype=np.int64)[:feature_dims]  # Change feature idx to np.array
        # print(midx.shape)

        user_cols = self.raw_data  # Get only movie id
        cols = user_cols[:, :1].squeeze().squeeze()
        counts_index = np.bincount(cols)
        data_pack = {}
        base_i = 0
        for i in range(1, len(counts_index)):
            re = user_cols[base_i:base_i + counts_index[i], 1:3].squeeze().squeeze()  # .tolist()
            re_bank = np.zeros([self.movie_ids])
            for j in range(re.shape[0]):
                try:
                    re_bank[re[j, 0] - 1] = re[j, 1]
                except:
                    pass
            re_bank = re_bank[midx].astype(np.int64)  # if used the float it will be not enough memory
            lsh.insert(str(i), re_bank)
            data_pack[str(i)] = re_bank
            base_i += counts_index[i]

        print("Built HashData used run-time is : %.03f seconds" % (time.clock() - start))
        sum_records = 0

        for i in range(self.users_sum):
            x = data_pack[str(i + 1)]
            search_result = lsh.query(x, thresold=threshold)  # @eg:[('0', (3, 1, 5, 0, 0, 0, 4, 2, 3, 1), 1.0),(...)]
            re_size = len(search_result) - 1
            if re_size > 0:
                sum_records += self.result_wrapper2(str(i + 1), search_result)  # task2,3 should use the wrapper2

            if (time.clock() - start) > self.time_limit:  # @time watch dog
                break

        self.write_to_file('cs.txt')
        # print(sum_records)
        print("Query all pairs used run-time is : %.03f seconds" % (time.clock() - start))  #
        return

    def jaccard(self, threshold):
        print('=' * 15)
        print('@ Data task: Netflix Challenge (www.netflixprize.com).')
        print('@ Program title: <Task1> MinHash-LSH Search Using Jaccard distance-func')
        print('Attention!!!! You need at least 18-mins to run this program')
        print('=' * 15)
        start = time.clock()  # Dear, When Summit, dont comment it!!

        lsh = LSH_Search_MH_Jaccard(threshold=threshold)
        self.raw_data = self.raw_data[:, :2]  # Get only movie id
        self.data_size = self.raw_data.shape
        cols = self.raw_data[:, :1].squeeze().squeeze()
        counts_index = np.bincount(cols)
        # np.bitcount func is a very very strong!! Better than sql group by!!
        data_pack = {}
        base_i = 0
        for i in range(1, len(counts_index)):
            # print(i)
            mh_obj = MinHash(seed=self.seed)  # The radom seed using here
            re = self.raw_data[base_i:base_i + counts_index[i], 1:2].squeeze().squeeze().tolist()
            base_i += counts_index[i]
            re = list(map(f_ele, re))
            for d in re:
                mh_obj.update(d)
            lsh.insert(str(i), mh_obj)
            data_pack[str(i)] = mh_obj  # pack for found keys
            if (time.clock() - start) > (self.time_limit - 300):  # @time watch dog
                break
        # print(data_pack['103703'])
        print("Built HashData used run-time is : %.03f seconds" % (time.clock() - start))

        sum_records = 0  # record all records found
        for key_u, value_u in data_pack.items():
            search_result = lsh.query(value_u)  # Search each pairs

            sum_records += self.result_wrapper(key_u, search_result)
            if (time.clock() - start) > self.time_limit:  # @time watch dog
                break
        # print(sum_records)
        self.write_to_file('js.txt')
        print("Query all pairs used run-time is : %.03f seconds" % (time.clock() - start))  #
        return

    def discrete_cosine(self, threshold):
        print('=' * 15)
        print('@ Data task: Netflix Challenge (www.netflixprize.com).')
        print('@ Program title: <Task3> RandomProjection-LSH Search Using discrete_cosine distance-func')
        print('Attention!!!! You need at least 18-mins to run this program')
        print('=' * 15)
        start = time.clock()  # Dear, When Summit, dont comment it!!

        feature_dims = 1700  # The feature compare dims

        lsh = LSH_Search_RP_cosine(10, feature_dims,
                                   90)  # hash size, feature dims, hash tables(ALso you can use 1, but e.... i think it wont work well)
        _x_features = [i for i in range(self.movie_ids)]
        random.shuffle(_x_features)
        # print(_x_features)
        midx = np.array(_x_features, dtype=np.int64)[:feature_dims]  # Change feature idx to np.array
        # print(midx.shape)

        user_cols = self.raw_data  # Get only movie id
        cols = user_cols[:, :1].squeeze().squeeze()
        counts_index = np.bincount(cols)
        data_pack = {}
        base_i = 0
        for i in range(1, len(counts_index)):
            re = user_cols[base_i:base_i + counts_index[i], 1:3].squeeze().squeeze()  # .tolist()
            re_bank = np.zeros([self.movie_ids])
            for j in range(re.shape[0]):
                try:
                    re_bank[re[j, 0] - 1] = 1  # Just turn the mark bit to 1, others are zeros
                except:
                    pass
            re_bank = re_bank[midx].astype(np.int64)  # if used the float it will be not enough memory
            lsh.insert(str(i), re_bank)
            data_pack[str(i)] = re_bank
            base_i += counts_index[i]

        print("Built HashData used run-time is : %.03f seconds" % (time.clock() - start))
        sum_records = 0

        for i in range(self.users_sum):
            x = data_pack[str(i + 1)]
            search_result = lsh.query(x, thresold=threshold)  # @eg:[(('0', (1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 1.0),(...))]
            re_size = len(search_result) - 1
            if re_size > 0:
                sum_records += self.result_wrapper2(str(i + 1), search_result)  # task2,3 should use the wrapper2
            if (time.clock() - start) > self.time_limit:  # @time watch dog
                break

        self.write_to_file('dcs.txt')
        # print(sum_records)
        print("Query all pairs used run-time is : %.03f seconds" % (time.clock() - start))  #
        return


'''
#@ Not Junk code !!!!!
# for the code test

data = np.load('user_movie_rating.npy')
us = User_Search(data,2020)
print(us.data_size)
#us.jaccard(0.7) # if use the 0.5 found too mmany pairs
#us.cosine(0.67)
us.discrete_cosine(0.66)
'''

'''
#@Junk code
# Just for my own test

length = self.data_size[0]
pies = int(length/chuncksize)
chunck_list = []
for i in range(pies):
    if i==(pies-1):
        chunck_list.append(self.raw_data[(i*chuncksize):length])
    else:
        chunck_list.append(self.raw_data[(i*chuncksize):((i+1)*chuncksize)])
for c in chunck_list:
    print(c.shape)
'''