import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from aliasmatching.utils import *


class BirdMatching:
    """
      Adaptation of the approach from the 'Mining Email Social Networks'. Algorithm measures pair-wise similarity
      between all participants and splits them into clusters with Agglomerative clustering. All distances are values
      from 0 to 1. For each measure <measure>_coef (should be from 0 to 1) adjusts influence of the measure:
      adjusted_measure = 1 - (1 - measure) * <measure>_coef

      :param users: dataframe with that contains columns 'name', 'login', and 'email' with corresponding information in it. If 'full_id' column is present, will be treated as an id of each user, otherwise custom id will be constructed and added to the DataFrame
      :param distance_threshold: distance parameter for clustering
      :param name_coef: weight for name similarity
      :param email_name_coef: weight for name-email similarity
      :param email_score_coef: weight for e-mail similarity
      :param login_score_coef: weight for login similarity
      :param login_email_coef: weight for login-email similarity
      :param login_name_coef: weight for login-name similarity

      :return: users dataframe with 'cluster'. Users that are deemed as one have same number in 'cluster' column
    """

    def __init__(self,
                 distance_threshold=0.1,
                 name_coef=1,
                 email_name_coef=1,
                 email_coef=1,
                 login_coef=1,
                 login_email_coef=0,
                 login_name_coef=0):
        self.distance_threshold = distance_threshold
        self.score_config = {'name_coef': name_coef,
                             'email_name_coef': email_name_coef,
                             'email_coef': email_coef,
                             'login_coef': login_coef,
                             'login_email_coef': login_email_coef,
                             'login_name_coef': login_name_coef}

    def distance(self, u1, u2):
        if u1['name'] is np.nan or u2['name'] is np.nan:
            name_score = 1
        else:

            name_score = name_distance(u1, u2)
            name_score = adjust_score(name_score, self.score_config['name_coef'])

            email_name_score = name_email_distance(u1, u2)
            email_name_score = adjust_score(email_name_score, self.score_config['email_name_coef'])

            name_score = min(name_score, email_name_score)

            # handle score

        email_score = email_distance(u1, u2)
        email_score = adjust_score(email_score, self.score_config['email_coef'])

        login_score = login_distance(u1, u2)
        login_score = adjust_score(login_score, self.score_config['login_coef'])

        login_email_score = login_email_distance(u1, u2)
        login_email_score = adjust_score(login_email_score, self.score_config['login_email_coef'])

        login_name_score = login_name_distance(u1, u2)
        login_name_score = adjust_score(login_name_score, self.score_config['login_name_coef'])

        handle_score = min(email_score, login_score, login_email_score, login_name_score)

        return min(name_score, handle_score)

    def get_clusters(self,
                     users):
        users_og = users
        users = users.copy()
        users = users.drop_duplicates().reset_index().drop('index', axis=1)
        users = users.fillna('')

        if 'full_id' not in users:
            users['full_id'] = users.apply(lambda x: f"{x['name']}:{x['email']}:{x['login']}", axis=1)
            users_og['full_id'] = users['full_id']
        users.name = users.name.apply(name_preprocess)

        users['email_base'] = users.email.apply(email_base)
        users['first_name'] = users.name.apply(first_name)
        users['last_name'] = users.name.apply(last_name)

        sim_matrix = get_sim_matrix(users,
                                    self.distance)

        agg = AgglomerativeClustering(n_clusters=None,
                                      distance_threshold=self.distance_threshold,
                                      affinity='precomputed',
                                      linkage='complete').fit(sim_matrix)

        users['cluster'] = agg.labels_
        df_cs = users[['cluster', 'name']].groupby('cluster').count().reset_index().rename({'name': 'cluster_size'},
                                                                                           axis=1)

        users = users.join(df_cs, on='cluster', rsuffix='_r').drop('cluster_r', axis=1)
        users = users.sort_values(['cluster_size', 'cluster'], ascending=False).reset_index().drop('index', axis=1)

        users['cluster2'] = -users.index - 1

        users.loc[users['cluster'].isna(), 'cluster'] = users['cluster2'][users['cluster'].isna()]
        users['id'] = pd.factorize(users['cluster'])[0]

        key2id = {x['full_id']: x['id'] for _, x in users.iterrows()}
        key2id = defaultdict(lambda: np.nan, key2id)

        return key2id

    def process(self, users):

        key2id = self.get_clusters(users)

        users['cluster'] = users['full_id'].apply(lambda x: key2id[x])
        return users