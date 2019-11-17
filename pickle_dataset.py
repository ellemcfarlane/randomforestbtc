import pickle
import pandas as pd

def pickle_frame(data_path, obj_path):
    with open(data_path, 'r') as file:
        data_frame = pd.read_csv(file)
        with open(obj_path, 'wb') as inp:
            pickle.dump(data_frame, inp)

def save_pickle(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

pickle_frame('raw_csvs/petrol_consumption.csv', 'data_frames/petrol_df.pkl')
# pickle_frame('df_final.csv', 'btc_df.pkl')

def load_data_frame(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# dataset = load_data_frame('data_frames/petrol_df.pkl').to_dict('records')
# save_pickle(dataset, 'points/petrol_points.pkl')
# dataset = load_data_frame('btc_df.pkl')
# # print(dataset.head())
#
# dict_list = dataset.to_dict('records')
# # print(dict_list)
# save_pickle(dict_list, 'btc_points.pkl')
#
# subsample = dict_list[:6]
#
# save_pickle(subsample, 'subsample.pkl')
