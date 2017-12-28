import numpy as np
import pandas as pd


def parsing(raw):
    raws = list(raw)
    con = ['!', '!']
    for i in raws:
        last = con[-1]
        seclast = con[-2]
        if last in ['/']:
            i = con.pop() + i
            i = con.pop() + i
        elif last in ['^', '=', '_', '^^', '__']:
            i = con.pop() + i
        elif i in [",", "'", '2', '3', '4', '5', '6', '7', '8', '9']:
            i = con.pop() + i
        con.append(i)
    con.remove(con[0])
    con.remove(con[0])
    return con


class ABC_READER():

    def __init__(self, seq_length, origin_path):
        # HEADER AND SCALE MODE
        self.window_length = seq_length
        self.origin_path = origin_path
        self.seq_length = seq_length
        self.preprocess_abc()

    def get_abc_lists(self):
        # Unnecessary headers will be removed
        # remove blank and whitespace
        clean_lines = []
        with open(self.origin_path, 'r') as abcfile:
            lines = abcfile.readlines()

            clean_lines = [l.strip().replace(" ", "") + '\n' for l in lines if l.strip()]

        abc_list = []

        abc = ''
        for line in clean_lines:
            if line.startswith("T:"):

                abc_list.append(abc)
                abc = ''

            else:
                abc += line

        abc_list.append(abc)
        abc_list.remove(abc_list[0])

        return abc_list

    def preprocess_abc(self):
        abc_list = self.get_abc_lists()

        all_tokens = set()
        abc_list_listed = []
        for abc in abc_list:
            song_parsed = []

            sp_list = abc.split("\n")
            all_tokens.add("\n")
            for elem in sp_list:
                if ('M:' in elem) or ('K:' in elem):
                    all_tokens.add(elem)
                    song_parsed += [elem]
                    song_parsed += ['\n']
                else:
                    chars = parsing(elem)
                    all_tokens = all_tokens.union(chars)
                    song_parsed += chars

            abc_list_listed.append(song_parsed)

        # create vocabulary and assign ids
        note_info = pd.DataFrame(data=np.array(sorted(all_tokens)), columns=['note'])
        self.note_info_dict = note_info['note'].to_dict()
        self.note_info_dict_swap = dict((y, x) for x, y in self.note_info_dict.items())
        # trans token to ids
        abc_list_translated = []

        for abc in abc_list_listed:
            song_trans = []
            for elem in abc:
                id = self.note_info_dict_swap.get(elem)
                song_trans += [id]
            bar_info = song_trans[:4]
            step = self.window_length - 4
            n = (len(song_trans) - 4) // step
            for j in range(n):
                abc_list_translated.append(bar_info + song_trans[j * step + 4:(j + 1) * step + 4])
#             if len(song_trans) > self.window_length:
#                 abc_list_translated.append(song_trans[:self.window_length])

        abc_list_translated = np.array(abc_list_translated)
        abc_size = len(abc_list_translated)
        np.random.shuffle(abc_list_translated)

        train_index = (int)(abc_size * 0.8)

        tr, test = abc_list_translated[:train_index], abc_list_translated[train_index:]

        self.trans_abc_train, self.trans_abc_test = tr.tolist(), test.tolist()

    def trans_trans_songs_to_raw(self, trans_list):
        raw_list = []
        for trans in trans_list:

            abc = self.trans_to_abc(trans)

            raw_list.append(np.asarray(abc))

        return raw_list

    def trans_to_abc(self, trans_note):

        result = []

        for entry in trans_note:
            result.append(self.note_info_dict.get(entry))

        return result

if __name__ == "__main__":
    reader = ABC_READER(80, 'abc_all_parsed.txt')
    reader.preprocess_abc()
