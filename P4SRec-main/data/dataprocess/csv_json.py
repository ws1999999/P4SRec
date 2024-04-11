import json
import pandas as pd


def csv_json(file_in, file_out):
    # 读取 Beauty.csv 文件
    sorted_data = pd.read_csv(file_in, header=None)

    # 按照第一列进行排序
    new_data = sorted_data.sort_values(by=0)

    # 按照第一列（ID）进行分组，并对每个组内的数据按照第四列排序
    data = new_data.groupby(0).apply(lambda x: x.sort_values(by=[3, 1, 2]))

    # 重置索引
    data.reset_index(drop=True, inplace=True)

    sorted_data = data.iloc[:, :-1]

    # 创建唯一ID字典
    unique_ids = {}
    current_id = 1

    # 用唯一ID替换第二列的值
    for i, row in sorted_data.iterrows():
        value = row[1]
        if value not in unique_ids:
            unique_ids[value] = current_id
            current_id += 1

        sorted_data.at[i, 1] = unique_ids[value]

    # 将最后一列转换为整型
    sorted_data[2] = sorted_data[2].astype(int)

    data = {}

    for index, row in sorted_data.iterrows():
        key = str(row[0])
        value = row[2]
        if key not in data:
            data[key] = {}
        data[key][str(row[1])] = int(value)

    # 将字典转换为JSON字符串
    json_str = json.dumps(data)

    # 保存到新文件
    with open(file_out, 'w') as jsonfile:
        jsonfile.write(json_str)


file_in = 'Toys_and_Games.csv'
file_out = '../Toys.json'
csv_json(file_in, file_out)