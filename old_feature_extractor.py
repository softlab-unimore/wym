class FeatureExtractor():
    def __init__(self, net_paired, loader_paired, net_unpaired, loader_unpaired, df, word_vectors,
                 exclude_attrs=['id', 'left_id', 'right_id', 'label'], n_proc=2):
        self.n_proc = n_proc
        self.word_vectors = word_vectors
        self.net_paired = net_paired
        self.loader_paired = loader_paired
        self.net_unpaired = net_unpaired
        self.loader_unpaired = loader_unpaired
        self.cols = [x[5:] for x in df.columns if x not in exclude_attrs and x.startswith('left_')]

    def process(self, df):
        paired, unpaired = self.generate_paired_unpaired(df)
        return self.extract_features(paired, unpaired, df)

    def generate_paired_unpaired(self, df):
        common_words_df, non_common_words_df = self.split_paired_unpaired(df)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = self.loader_paired.preprocess(common_words_df, self.word_vectors)
        grouped = self.loader_paired.aggregated.copy()
        grouped['pred'] = self.net_paired(X.to(device)).cpu().detach().numpy()
        tmp = common_words_df.copy()
        merge_cols = ['attribute', 'left_word', 'right_word']
        paired_words = tmp.merge(grouped[merge_cols + ['pred']], on=merge_cols, suffixes=('', ''))
        self.paired_raw = paired_words

        X = self.loader_unpaired.preprocess(non_common_words_df, self.word_vectors)
        grouped = self.loader_unpaired.aggregated.copy()
        grouped['pred'] = self.net_unpaired(X.to(device)).cpu().detach().numpy()
        tmp = non_common_words_df.copy()
        merge_cols = ['attribute', 'word']
        unpaired_words = tmp.merge(grouped[merge_cols + ['pred']], on=merge_cols, suffixes=('', ''))
        self.unpaired_raw = unpaired_words
        return paired_words, unpaired_words

    def split_paired_unpaired(self, df):
        tk_divider = Tokens_divider(self.cols, self.word_vectors)
        pool = Pool(self.n_proc)
        tmp = pool.map(tk_divider.tokens_in_vocab_division, df.iterrows())
        pool.close()
        pool.join()
        tuples_common, tuples_non_common = [], []
        for x, y in tmp:
            tuples_common += x
            tuples_non_common += y
        common_words_df = pd.DataFrame(tuples_common)
        non_common_words_df = pd.DataFrame(tuples_non_common)
        return common_words_df, non_common_words_df

    def extract_features(self, com_df, non_com_df, df):
        functions = ['mean', 'sum', 'count']
        paired_stat = com_df.groupby(['id'])['pred'].agg(functions)
        tmp = non_com_df.copy()
        tmp['pred'] = 1 - tmp['pred']
        stat = (tmp.groupby(['id', 'side'])['pred']).agg(functions)
        unpaired_stat = stat.unstack(1)
        unpaired_stat.columns = ['_'.join(col) for col in unpaired_stat.columns]
        unpaired_stat = unpaired_stat.fillna(0)
        left_minor = unpaired_stat['count_left'] < unpaired_stat['count_right']
        tmp = unpaired_stat[left_minor][pd.Index(functions) + '_left']
        tmp.columns = functions
        tmp2 = unpaired_stat[~left_minor][pd.Index(functions) + '_right']
        tmp2.columns = functions
        unpaired_stat = pd.concat([tmp, tmp2])
        paired_stat.columns += '_paired'
        unpaired_stat.columns += '_unpaired'

        stat = paired_stat.merge(unpaired_stat, on='id', how='outer').merge(df[['id', 'label']], on='id',
                                                                            how='outer').fillna(0)
        stat['mean_diff'] = stat['mean_paired'] - stat['mean_unpaired']
        stat['sum_diff'] = stat['sum_paired'] - stat['sum_unpaired']
        stat['overlap'] = stat['count_paired'] / (stat['count_paired'] + stat['count_unpaired'])
        stat = stat.fillna(0)
        return stat


class FeatureExtractorOOV():
    def __init__(self, cols, word_vectors, n_process=2):
        self.word_vectors = word_vectors
        self.n_process = n_process
        self.cols = cols
        self.lp = 'left_'
        self.rp = 'right_'
        self.all_cols = [self.lp + col for col in cols] + [self.rp + col for col in cols]

    def process(self, non_in_vocab_df):
        com_df, non_com_df = self.process_oov(non_in_vocab_df)
        pair_to_add = self.check_codes(non_com_df)
        com_df = pd.concat([com_df, pd.DataFrame(pair_to_add)])
        self.paired_raw = com_df
        self.unpaired_raw = non_com_df
        oov_stat = self.extract_features(com_df, non_com_df, non_in_vocab_df)
        return oov_stat

    def process_oov(self, df):
        pool = Pool(self.n_process)
        tmp_list = pool.map(self.tokens_in_vocab_division, df.iterrows())
        pool.close()
        pool.join()
        tmp1, tmp2 = [], []
        for x, y in tmp_list:
            tmp1 += x
            tmp2 += y
        com_df = pd.DataFrame(tmp1)
        non_com_df = pd.DataFrame(tmp2)
        com_df['is_code'] = com_df['left_word'].apply(self.is_code)
        non_com_df['is_code'] = non_com_df['word'].apply(self.is_code)
        return com_df, non_com_df

    def tokens_in_vocab_division(self, item):
        index, el = item
        label = el['label']
        el_words = {}
        for col in self.cols:
            if el.isnull()[[self.lp + col, self.rp + col]].any():  # both description must be non Nan
                el_words[self.lp + col] = []
                el_words[self.rp + col] = []
            else:
                el_words[self.lp + col] = str(el[self.lp + col]).split()
                el_words[self.rp + col] = str(el[self.rp + col]).split()
        common, non_common = self.split_paired_unpaired_tokens(el_words)
        return self.to_list_of_dict(common, non_common, index, label)

    def split_paired_unpaired_tokens(self, el_words):
        common_words = {col: [] for col in self.cols}
        non_common_words = {col: [] for col in self.all_cols}
        for col in self.cols:
            l_words, r_words = el_words[self.lp + col], el_words[self.rp + col]
            paired = []
            for word in l_words:
                if word in r_words:
                    common_words[col] += [(word, word)]
                    paired.append(word)
                else:
                    non_common_words[self.lp + col] += [word]
            for word in np.setdiff1d(r_words, paired):
                non_common_words[self.rp + col] += [word]
        return common_words, non_common_words

    def to_list_of_dict(self, common_words, non_common_words, index, label):
        tuples_common = []
        tmp_dict = {}
        tmp_dict.update(id=index, label=label)
        for col in self.cols:
            tmp_dict.update(attribute=col)
            for pair in common_words[col]:
                tmp_dict.update(left_word=pair[0], right_word=pair[1])
                tuples_common.append(tmp_dict.copy())

        tuples_non_common = []
        tmp_dict = {}
        tmp_dict.update(id=index, label=label)
        for col in self.cols:
            tmp_dict.update(attribute=col)
            tmp_dict.update(side='left')
            prefix = self.lp
            for word in non_common_words[prefix + col]:
                tmp_dict.update(word=word)
                tuples_non_common.append(tmp_dict.copy())

            tmp_dict.update(side='right')
            prefix = self.rp
            for word in non_common_words[prefix + col]:
                tmp_dict.update(word=word)
                tuples_non_common.append(tmp_dict.copy())
        return tuples_common, tuples_non_common

    def is_code(self, word):
        tmp_word = word.replace('-', '')
        if len(tmp_word) < 4:
            return False
        tmp = re.search(r'(\d+[A-Za-z]+)|([A-Za-z]+\d+)', tmp_word)
        if tmp is None:
            return False
        tmp_words = re.findall(r'(?P<letters>[A-Za-z]+)', tmp_word)
        if len(tmp_words) > 1:
            return True
        oov = False
        if not self.word_vectors.__contains__(tmp_words[0].lower()):
            return True
        return False

    def check_codes(self, df):
        all_attr = df
        for attr in all_attr.attribute.unique():
            x = all_attr[all_attr.attribute == attr]
            id_code_l = x[(x.is_code == True) & (x.side == 'left')].id.values
            id_code_r = x[(x.is_code == True) & (x.side == 'right')].id.values
            pair_to_add = []
            for id in np.intersect1d(id_code_l, id_code_r):
                df_id = x[(x.id == id) & (x.is_code == True)]
                l_codes = df_id[(df_id.side == 'left')].word.values
                r_codes = df_id[(df_id.side == 'right')].word.values
                l_codes_replaced = [code.replace('-', '') for code in l_codes]
                r_codes_replaced = [code.replace('-', '') for code in r_codes]
                for r_idx, r_code in enumerate(r_codes_replaced):
                    conatined = False
                    for l_idx, l_code in enumerate(l_codes_replaced):
                        if len(r_code) < len(l_code):
                            contained = r_code in l_code
                        else:
                            contained = l_code in r_code
                        if contained:
                            tmp_dict = df_id[df_id.word == l_codes[l_idx]].iloc[0].to_dict()
                            tmp_dict.pop('side')
                            tmp_dict.pop('word')
                            tmp_dict['left_word'] = l_code
                            tmp_dict['right_word'] = r_code
                            pair_to_add.append(tmp_dict.copy())
                            l_codes_replaced.remove(l_code)
                            all_attr.drop(df_id[df_id.word == l_codes[l_idx]].index, inplace=True)
                            all_attr.drop(df_id[df_id.word == r_codes[r_idx]].index, inplace=True)
                            break
        return pair_to_add

    def extract_features(self, com_df, non_com_df, df):
        com_features = com_df.groupby(['id']).agg(
            {'is_code': ['count', ('n_code', lambda x: x[x == True].count())]}).droplevel(0, 1)
        tmp = non_com_df.groupby(['id', 'side']).agg(
            {'is_code': ['count', ('n_code', lambda x: x[x == True].count())]}).unstack(1).droplevel(0, 1)
        tmp.columns = [x + '_' + y for x, y in tmp.columns]
        non_com_features = tmp

        com_features.columns += '_paired'
        non_com_features.columns += '_unpaired'
        features = non_com_features.add_prefix('unpaired_').merge(
            com_features.add_prefix('paired_'), on='id', how='outer').merge(
            df.set_index('id')[['label']], on='id', how='outer').fillna(0).astype(int)
        features.add_prefix('oov_')
        return features