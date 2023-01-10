import pandas as pd


def create_mirror_df(df_to_mirror):
    mirror_df_list = []
    for side_to_copy in ['left_', 'right_']:
        side_to_write = 'left_' if side_to_copy == 'right_' else 'right_'
        turn_sample = df_to_mirror.copy()
        for col_to_copy in turn_sample.columns[turn_sample.columns.str.startswith(side_to_copy)]:
            col_name = col_to_copy.replace(side_to_copy, '')
            turn_sample[side_to_write + col_name] = turn_sample[col_to_copy]
        turn_sample['mirror_side'] = side_to_copy
        mirror_df_list.append(turn_sample)
    return pd.concat(mirror_df_list)