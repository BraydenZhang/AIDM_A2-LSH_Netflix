# Requirements
- numpy
- python >=3.5

# Modules
- LSH_MinHash.py 
**Minhash to realized LSH**
- LSH_RP.py
**Random projection to realized LSH**
- User_Search_Interface.py
**JS/CS/DCS to realized the similar user pairs search**
- utils.py
- storage.py
Storage.py referenced from github: https://github.com/JintuZheng/Teach-You-Build-Hash-Storage

# Run
@ Windows test:

  2020 is the seed for Online test

#eg: python main.py -d D:/user_movie_rating.npy -s 2020 -m cs

  

@ Mac/Linux test: (cmd python3-->python)

$ python3 main.py -d /Users/user_movie_rating.npy -s 2020 -m js

$ python3 main.py -d /Users/user_movie_rating.npy -s 2020 -m cs

$ python3 main.py -d /Users/user_movie_rating.npy -s 2020 -m dcs
