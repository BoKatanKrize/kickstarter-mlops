import re
import glob
import pickle
import pandas as pd


def save_data(df, info_data,
              year, month,
              is_split=False):
    if is_split:
        fnames = []
        for key, value in df.items():
            if value is not None:
                fname = f'{info_data["prefix_name"]}_{key}_{month}-{year}.parquet'
                value.to_parquet(f'{info_data["path_local_out"]}/{fname}')
                fnames.append(fname)
        info_data["fnames"] = fnames
    else:
        fname = f'{info_data["prefix_name"]}_{month}-{year}.parquet'
        df.to_parquet(f'{info_data["path_local_out"]}/{fname}')
        info_data['fnames'] = [fname]

    return info_data


def save_pipe(pipe, info_pipe, suffix=None):
    pre = info_pipe["prefix_name"]
    pipe_name = f'{pre}.pkl' if suffix is None else f'{pre}_{suffix}.pkl'
    output_file_path = f'{info_pipe["path_local_out"]}/{pipe_name}'
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(pipe, output_file)
    if info_pipe["fnames"] is None:
        info_pipe["fnames"] = [pipe_name]
    else:
        info_pipe["fnames"].append(pipe_name)
    return info_pipe


def load_data(info_data, is_split=False):
    fnames = glob.glob(f'{info_data["path_local_in"]}/*.parquet')
    keys = ['full', 'train', 'val', 'test']
    ddf = {key: None for key in keys}
    if is_split:
        for fname in fnames:
            for key in keys:
                if key in fname:
                    ddf[key] = pd.read_parquet(fname,
                                               engine='pyarrow')
    else:
        fname = fnames[0]
        ddf['full'] = pd.read_parquet(fname, engine='pyarrow')

    # Extract month and year using regular expressions
    match = re.search(r'(\d{2})-(\d{4})\.parquet', fname)
    month = match.group(1)
    year = match.group(2)

    return ddf, year, month
