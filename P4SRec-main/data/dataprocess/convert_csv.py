import gzip
import pandas as pd
import tqdm

def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].map(column_dict)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def amazon(total_inter, file_in, file_out):
    names = ['user', 'item', 'rating', 'time']
    rows_list = []
    pbar = tqdm.tqdm(total=total_inter)
    for line in parse(file_in):
        rev = line['reviewerID']
        asin = line['asin']
        ratings = line['overall']
        time = line['unixReviewTime']
        value = {'user': rev, 'item': asin, 'rating': ratings, 'time': time}
        rows_list.append(value)
        pbar.update(1)

    df = pd.DataFrame(rows_list, columns=names)
    df, user_mapping = convert_unique_idx(df, 'user')
    df, item_mapping = convert_unique_idx(df, 'item')
    items = list(df['item'])
    if 0 in items:
        df['item'] = df['item'].apply(lambda x: x + 1)
    users = list(df['user'])
    if 0 in users:
        df['user'] = df['user'].apply(lambda x: x + 1)

    df.to_csv(file_out, index=False, header=False)
    print(df.shape)

total = 167597  # Beauty:198,502  Sports:296,337   toy:167597
data_name = 'reviews_Toys_and_Games_5.json.gz'  # XXX.json.gz
file_name_out = 'Toys_and_Games.csv'
amazon(total, data_name, file_name_out)