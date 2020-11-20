import numpy as np
import argparse
from User_Search_Interface import User_Search

def parse_args():
    parser = argparse.ArgumentParser(description='proper argument to run experiment')
    parser.add_argument('-d', type=str, default='/',
                        help="Data file path")
    parser.add_argument('-s', type=int, default=2020,
                        help='Random seed (by using np.random.seed(int)) ')
    parser.add_argument('-m', type=str, default='js',
                        help='Similarity measure (js / cs / dcs)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = np.load(args.d)
    us = User_Search(data, args.s)

    if args.m == 'js':
        us.jaccard(0.5) # Attention ! Don't  use 0.5 here !!!!! After tested, if too lower to make memory overflow!!
    elif args.m == 'cs':
        us.cosine(0.67)
    elif args.m == 'dcs':
        us.discrete_cosine(0.66)

if __name__ == "__main__":
    main()  
    '''
    @ Windows test:

    #eg: python main.py -d D:/user_movie_rating.npy  -s 2020 -m cs

    @ Mac/Linux test: (cmd python3-->python)
    $ python3 main.py -d /Users/user_movie_rating.npy  -s 2020 -m js
    $ python3 main.py -d /Users/user_movie_rating.npy  -s 2020 -m cs
    $ python3 main.py -d /Users/user_movie_rating.npy  -s 2020 -m dcs

    '''

