import os
import json
import numpy as np
import json

'''
@TITLE：Use random projection to achieve locality sensitive hashing
'''


def storage(storage_config, index):
    """ Given storage configuration and index, return the configured storage instance
    """
    if 'dict' in storage_config:
        return InMemoryStorage(storage_config['dict'])
    else:
        raise ValueError("Only supports built-in dictionaries! ! ! !")


'''
@Abstract class: storage adapter
'''


class DBaseStorage(object):
    def __init__(self, config):
        """Abstract class used as storage adapter"""
        raise NotImplementedError

    def keys(self):
        """Return a list of binary hashes used as dict keys"""
        raise NotImplementedError

    def set_val(self, key, val):
        """Set val to key, note that val must be a string"""
        raise NotImplementedError

    def get_val(self, key):
        """Return val at the key. Similarly, note that val must be a string"""
        raise NotImplementedError

    def append_val(self, key, val):
        """Append val to the list stored in key
          If the key is not yet stored, please use val to create a list of keys in the following location
        """
        raise NotImplementedError

    def get_list(self, key):
        """Return a list stored at key, this function should return a list of values stored at key
          If the list is empty or the key does not exist in the storage, return
        """
        raise NotImplementedError


'''
@Implementation class: storage adapter
'''


class InMemoryStorage(DBaseStorage):
    def __init__(self, config):
        self.name = 'dict'
        self.storage = dict()

    def keys(self):
        return self.storage.keys()

    def set_val(self, key, val):
        self.storage[key] = val

    def get_val(self, key):
        return self.storage[key]

    def append_val(self, key, val):
        self.storage.setdefault(key, []).append(val)

    def get_list(self, key):
        return self.storage.get(key, [])



class LSH_Search_RP_cosine(object):
    def __init__(self, hash_size, input_dim, seed=2020, num_hashtables=1):
        np.random.seed(seed)

        storage_config = None
        matrices_filename = None
        overwrite = False

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables

        if storage_config is None:
            storage_config = {'dict': None}
        self.storage_config = storage_config

        if matrices_filename and not matrices_filename.endswith('.npz'):
            raise ValueError("The specified file name must end with .npz")
        self.matrices_filename = matrices_filename
        self.overwrite = overwrite

        self._init_uniform_planes()
        self._init_hashtables()
        self._name_keys = []  # save the users' names

    def _init_uniform_planes(self):
        """
         uniform planes
        """
        if "uniform_planes" in self.__dict__:
            return

        if self.matrices_filename:
            file_exist = os.path.isfile(self.matrices_filename)
            if file_exist and not self.overwrite:
                try:
                    npzfiles = np.load(self.matrices_filename)
                except IOError:
                    print("Cannot load specified file as a numpy array")
                    raise
                else:
                    npzfiles = sorted(npzfiles.items(), key=lambda x: x[0])
                    self.uniform_planes = [t[1] for t in npzfiles]
            else:
                self.uniform_planes = [self._generate_uniform_planes()
                                       for _ in range(self.num_hashtables)]
                try:
                    np.savez_compressed(self.matrices_filename,
                                        *self.uniform_planes)
                except IOError:
                    print("IOError when saving matrices to specificed path")
                    raise
        else:
            self.uniform_planes = [self._generate_uniform_planes()
                                   for _ in range(self.num_hashtables)]

    def _init_hashtables(self):
        """ Initialize the hash table so that each record will be in the form of [storage1, storage2,...]"""

        self.hash_tables = [storage(self.storage_config, i)
                            for i in range(self.num_hashtables)]

    def _generate_uniform_planes(self):
        """Generate a uniformly distributed hyperplane and return it as a 2D numpy array
        """
        return np.random.randn(self.hash_size, self.input_dim)

    def _hash(self, planes, input_point):
        """ Generate a binary hash for input_point and return
        :param planes:
            Random uniform planes of size
            `hash_size` * `input_dim`
        :param input_point:
            Tuple or list containing only numbers, size must be 1 * input_dim
        """

        try:
            input_point = np.array(input_point)  # for faster dot product
            projections = np.dot(planes, input_point)
        except TypeError as e:
            print("""The input point needs to be an array-like object with
                  numbers only elements""")
            raise
        except ValueError as e:
            print("""The input point needs to be of the same dimension as
                  `input_dim` when initializing this LSH_Search_RP_cosine instance""", e)
            raise
        else:
            return "".join(['1' if i > 0 else '0' for i in projections])

    def _as_np_array(self, json_or_tuple):
        """Use JSON serialized data structure or original input points stored,
        and return the original input points in numpy array format
        """
        if isinstance(json_or_tuple, str):
            # JSON-serialized in the case of Redis
            try:
                # Return the point stored as list, without the extra data
                tuples = json.loads(json_or_tuple)[0]
            except TypeError:
                print("The value stored is not JSON-serilizable")
                raise
        else:
            # If extra_data exists, `tuples` is the entire
            # (point:tuple, extra_data). Otherwise (i.e., extra_data=None),
            # return the point stored as a tuple
            tuples = json_or_tuple

        if isinstance(tuples[0], tuple):
            # in this case extra data exists
            return np.asarray(tuples[0])

        elif isinstance(tuples, (tuple, list)):
            try:
                return np.asarray(tuples)
            except ValueError as e:
                print("The input needs to be an array-like object", e)
                raise
        else:
            raise TypeError("query data is not supported")

    def insert(self, key, input_point, extra_data=None):

        if isinstance(input_point, np.ndarray):
            input_point = input_point.tolist()

        if extra_data:
            value = (tuple(input_point), extra_data)
        else:
            value = tuple(input_point)

        self._name_keys.append(key)

        for i, table in enumerate(self.hash_tables):
            table.append_val(self._hash(self.uniform_planes[i], input_point),
                             value)

    def query(self, query_point, num_results=None, thresold=None):
        """query_point can be a tuple or list of numbers, return num_results as a list of sorted tuples
        :param query_point:
            Query dataUsed by :meth:`._hash`.

        :param num_results:
            (optional) Specify the maximum number of results to get, if not specified, all results will be listed in ranking order
        distance_func:
            'CS'
            'DCS'
        """
        candidates = set()
        d_func = LSH_Search_RP_cosine.cosine_dist

        for i, table in enumerate(self.hash_tables):
            binary_hash = self._hash(self.uniform_planes[i], query_point)
            candidates.update(table.get_list(binary_hash))

        # rank candidates by distance function
        candidates = [(self._name_keys[keyit], ix, d_func(query_point, self._as_np_array(ix)))
                      for keyit, ix in enumerate(candidates)]
        candidates.sort(key=lambda x: x[2])

        if thresold != None:
            thresold_filter_result = filter(lambda x: x[2] >= thresold, candidates)  # Filter the threshold
            return list(thresold_filter_result)
        else:
            return candidates[:num_results] if num_results else candidates

    ### distance functions
    @staticmethod
    def cosine_dist(x, y):
        Pdelta = np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)
        return Pdelta


'''
The following is a random data simulation test
'''
if __name__ == "__main__":
    import time

    start = time.clock()  # Dear, When Summit, comment it!!

    lsh = LSH_Search_RP_cosine(10, 10, 90)
    for i in range(10):
        # np.random.randint(0, 5, 100)
        lsh.insert(str(i), np.ones([10], dtype=np.int64))
    # x = np.random.randint(0, 5, 100)
    x = np.ones([10])
    print(x.shape)
    print("Query all pairs used run-time is : %.03f seconds" % (time.clock() - start))  #
    for i in range(1000):
        re = lsh.query(x, thresold=0.67)
        print(re)
    print("Query all pairs used run-time is : %.03f seconds" % (time.clock() - start))  #



