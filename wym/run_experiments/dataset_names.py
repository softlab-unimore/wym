import pandas as pd

sorted_dataset_names = [
    'BeerAdvo-RateBeer',
    'fodors-zagats',
    'iTunes-Amazon',
    'dirty_itunes_amazon',
    'DBLP-Scholar',
    'dirty_dblp_scholar',
    'walmart-amazon',
    'dirty_walmart_amazon',
    'DBLP-ACM',
    'dirty_dblp_acm',
    'Amazon-Google',
    'Abt-Buy',
]

sorted_codes = ['S-DG',
                'S-DA',
                'S-AG',
                'S-WA',
                'S-BR',
                'S-IA',
                'S-FZ',
                'T-AB',
                'D-IA',
                'D-DA',
                'D-DG',
                'D-WA']

# time_df['df_code'] = pd.Categorical(time_df['df_code'], categories=sorted_codes, ordered=True)
# or
# time_df['df_code'] = time_df['df_code'].astype('category', categories=sorted_codes, ordered=True)
# time_df.sort_values('df_code')

dataset_code_dict = {'BeerAdvo-RateBeer': 'S-BR',
                     'itunes-amazon': 'S-IA', 'iTunes-Amazon': 'S-IA',
                     'fodors-zagats': 'S-FZ',
                     'dblp_acm': 'S-DA', 'DBLP-ACM': 'S-DA',
                     'DBLP-GoogleScholar': 'S-DG', 'DBLP-Scholar': 'S-DG',
                     'amazon-google': 'S-AG', 'Amazon-Google': 'S-AG',
                     'walmart-amazon': 'S-WA',
                     'Abt-Buy': 'T-AB',
                     'dirty_itunes_amazon': 'D-IA',
                     'dirty_dblp_acm': 'D-DA',
                     'dirty_dblp_scholar': 'D-DG',
                     'dirty_walmart_amazon': 'D-WA'}

conf_code_map = {'all': 'all',
                 'R_L+Rafter': 'X_Y+Xafter', 'L_R+Lafter': 'X_Y+Xafter',
                 'R_R+Lafter': 'X_X+Yafter', 'L_L+Rafter': 'X_X+Yafter',
                 'R_L+Rbefore': 'X_Y+Xbefore', 'L_R+Lbefore': 'X_Y+Xbefore',
                 'R_L+RafterNOV': 'X_Y+XafterNOV', 'L_R+LafterNOV': 'X_Y+XafterNOV',
                 'R_L+RbeforeNOV': 'X_Y+XbeforeNOV', 'L_R+LbeforeNOV': 'X_Y+XbeforeNOV',
                 'R_R+LafterNOV': 'X_X+YafterNOV', 'L_L+RafterNOV': 'X_X+YafterNOV',
                 'left': 'X_Y', 'right': 'X_Y',
                 'leftCopy': 'X_YCopy', 'rightCopy': 'X_YCopy',
                 'mojito_copy_R': 'mojito_copy', 'mojito_copy_L': 'mojito_copy',
                 'mojito_drop': 'mojito_drop',
                 'LIME': 'LIME',
                 'MOJITO': 'MOJITO',
                 'LEMON': 'LEMON',
                 }
