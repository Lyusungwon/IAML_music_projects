import numpy as np
import editdistance
from abc_io import ABC_READER



def get_nearest_songs(target_music, train_songs, k=5):
    dist = []
    for song in train_songs:
        dist.append(editdistance.eval(target_music, song[:80]))
    dist_sort = np.argsort(dist)[:k]

    for idx in dist_sort:
        print(train_songs[idx], dist[idx])


if __name__ == '__main__':
    reader = ABC_READER(80, 'abc_all_parsed.txt')
    reader.preprocess_abc()

    # get all original songs
    train_songs = reader.get_abc_lists()

    # load generated song
    trans_song = reader.trans_to_abc(song)
    get_nearest_songs(trans_song, train_songs, k=5)
