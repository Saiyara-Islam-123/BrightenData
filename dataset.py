from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import torch


def get_dict():

    df = pd.read_csv("passivefeatures_phq9.csv")
    df = df[~df['week'].isin([1, 3])].reset_index(drop=True)
    d = {}
    for i in range(len(df["week"])):
        participant_id = df["participant_id"][i]
        if participant_id not in d:
            d[participant_id] = []
        row = df.loc[i].copy(deep=True)
        row = row.drop(labels=['ROW_ID', 'ROW_VERSION', "day", "participant_id"])

        d[participant_id].append(row.to_dict())

    return d

def create_indicator_row(week):
    d = {"week" : week,
         "aggregate_communication" : -1,
         "call_count" : -1,
         "call_duration" : -1,
         "interaction_diversity" : -1,
         "missed_interactions" : -1,
         "mobility": -1,
         "mobility_radius" : -1,
         "sms_count" : -1,
         "sms_length" : -1,
         "unreturned_calls" : -1,
         "sum_phq9" : -1,

         }
    return d

def get_weeks(rows_list):
    weeks = []
    for row in rows_list:
        weeks.append(row["week"])

    return weeks

def add_indicator_rows(rows_list):

    all_weeks = get_weeks(rows_list)
    for i in range(2, 14, 2):
        week = i
        index = int(week/2)-1
        if week not in all_weeks:
            indicator_row = create_indicator_row( week)
            rows_list.insert(index,indicator_row)

    return rows_list

def extract_seqs(participant_dict):
    features = ["aggregate_communication",
         "call_count",
         "call_duration",
         "interaction_diversity",
         "missed_interactions",
         "mobility",
         "mobility_radius",
         "sms_count",
         "sms_length",
         "unreturned_calls",]
    all_x = []
    all_y = []
    for i in range(len(participant_dict)):
        fs = []
        ps = []
        for f in features:
            fs.append(participant_dict[i][f])
        ps.append(participant_dict[i]["sum_phq9"]/27)
        all_x.append(fs)
        all_y.append(ps)

    return np.array(all_x), np.array(all_y)




def extract_all_seq(dicts):
    X_seq = []
    y_seq = []
    l = [X_seq, y_seq]

    for participant_id in dicts:
        participant_dict = add_indicator_rows(dicts[participant_id])
        mat_X, mat_y = extract_seqs(participant_dict)


        X_seq.append(mat_X[0 : 2, :])
        y_seq.append(mat_y[0 : 2, :])
        X_seq.append(mat_X[2:4, :])
        y_seq.append(mat_y[2:4, :])
        X_seq.append(mat_X[4:, :])
        y_seq.append(mat_y[4:, :])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    return X_seq, y_seq

class BrightenDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X = X_seq
        self.y = y_seq

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):

        return self.X[index], self.y[index]

def load_data(batch_size):
    d = get_dict()
    X_seq, y_seq = extract_all_seq(d)
    dataset = BrightenDataset(X_seq, y_seq)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

if __name__ == '__main__':
    d = get_dict()
    X, y = extract_all_seq(d)
    train, _ = load_data(X_seq=X, y_seq=y)
    for x_seq, y_seq in train:
        print(x_seq)
        print(y_seq)
        break