import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from Levenshtein import distance as LevDist
import string
from tqdm import tqdm
from collections import defaultdict


def remove_punctuation(name):
    """
    removes punctuation
    """
    return name.translate(str.maketrans('', '', string.punctuation)).lower()


# prefixes, postfixes, and titles to remove from name
fixes = ['jr', 'sr', 'dr', 'mr', 'mrs']
titles = ['admin', 'support']
ban_words = fixes + titles


def remove_ban_words(name):
    """
    removes banned words from name
    """
    name_parts = []
    for p in name.split():
        if p not in ban_words:
            name_parts.append(p)
    return " ".join(name_parts)


def name_preprocess(name):
    """
    removes punctuation and banned words from the name
    """
    return remove_ban_words(remove_punctuation(name))


def first_name(name):
    """
    returns first name
    """
    if name == '':
        return name
    return name.split()[0]


def last_name(name):
    """
    return last name
    """
    if name == '':
        return ''
    name_parts = name.split()
    if len(name_parts) == 1:
        return ''
    return name_parts[-1]


def shorten_email(email):
    """
    returns shorten form of the e-mail (everything before @)
    """
    if email == '':
        return ''
    return email.split('@')[0]


def get_norm_levdist(str1, str2):
    """
    Normalised Levenshtein distance between two strings
    """
    ld = LevDist(str1, str2)
    ml = max(len(str1), len(str2))

    if ml == 0:
        return 0

    score = ld / ml

    return score


def name_handle_dist(name, handle):
    """
    checks if first and second name are in the handle (e-mail or login)
    """

    fn, ln = name
    if len(fn) > 1 and len(ln) > 1:
        if fn in handle and ln in handle:
            return 0

    # if len(fn) > 0 and len(ln) > 0:
    #     names_with_initials = [fn[0] + ln, fn + ln[0], ln + fn[0], ln[0] + fn]
    #     for nwi in names_with_initials:
    #         if len(nwi) > 2 and nwi in handle:
    #             return 0
    return 1


def adjust_score(score, weight):
    if weight < 0:
        weight = 0
    if weight > 1:
        weight = 1
    return 1 - (1 - score) * weight


def sim_users(u1, u2,
              full_name_coef=1,
              part_name_coef=1,
              email_name_coef=1,
              email_coef=1,
              login_coef=1,
              login_email_coef=0,
              login_name_coef=0):
    """
    similarity of two users ids based on their name, e-mail, and login
    """
    # name scores
    if u1['name'] is np.nan or u2['name'] is np.nan:
        name_score = 1
    else:
        full_name_score = get_norm_levdist(u1['name'], u2['name'])
        full_name_score = adjust_score(full_name_score, full_name_coef)

        part_name_score = get_norm_levdist(u1['first_name'], u2['first_name']) + get_norm_levdist(u1['last_name'],
                                                                                                  u2['last_name'])
        part_name_score /= 2
        part_name_score = adjust_score(part_name_score, part_name_coef)

        email_name_score = max(name_handle_dist((u1['first_name'], u1['last_name']),
                                                u2['email']),
                               name_handle_dist((u2['first_name'], u2['last_name']),
                                                u1['email'])
                               )

        email_name_score = adjust_score(email_name_score, email_name_coef)

        name_score = min(full_name_score, part_name_score, email_name_score)

    # handle score

    email_score = 1
    if not u1['short_email'] is np.nan and not u2['short_email'] is np.nan:
        if len(u1['short_email']) > 2 and len(u2['short_email']) > 2:
            email_score = get_norm_levdist(u1['short_email'], u2['short_email'])
    email_score = adjust_score(email_score, email_coef)

    login_score = 1
    if not u1['login'] is np.nan and not u2['login'] is np.nan:
        if len(u1['login']) > 2 and len(u2['login']) > 2:
            login_score = get_norm_levdist(u1['login'], u2['login'])
    login_score = adjust_score(login_score, login_coef)

    login_email_score = 1
    if not u1['login'] is np.nan and not u2['short_email'] is np.nan:
        if len(u1['login']) > 2 and len(u2['short_email']) > 2:
            login_email_score = get_norm_levdist(u1['login'], u2['short_email'])

    if not u1['short_email'] is np.nan and not u2['login'] is np.nan:
        if len(u1['short_email']) > 2 and len(u2['login']) > 2:
            login_email_score = min(login_email_score, get_norm_levdist(u1['short_email'], u2['login']))
    login_email_score = adjust_score(login_email_score, login_email_coef)

    login_name_score = max(name_handle_dist((u1['first_name'], u1['last_name']),
                                            u2['login']),
                           name_handle_dist((u2['first_name'], u2['last_name']),
                                            u1['login'])
                           )
    login_name_score = adjust_score(login_name_score, login_name_coef)

    handle_score = min(email_score, login_score, login_email_score, login_name_score)

    return min(name_score, handle_score)


def get_sim_matrix(users,
                   full_name_coef=1,
                   part_name_coef=1,
                   email_name_coef=1,
                   email_score_coef=1,
                   login_score_coef=1,
                   login_email_coef=0,
                   login_name_coef=0):
    """
    calculates name similarity matrix between users
    """
    sim_matrix = np.zeros((len(users), len(users)))
    for i1, row1 in tqdm(users.iterrows()):
        def score(i2, row2):
            if i1 < i2:
                return sim_users(row1, row2, full_name_coef, part_name_coef, email_name_coef, email_score_coef,
                                 login_score_coef, login_email_coef, login_name_coef)
            if i1 == i2:
                return 0
            return 0

        sim_matrix[i1] = [score(*p) for p in users.iterrows()]
    sim_matrix = sim_matrix + sim_matrix.T

    return sim_matrix


def get_clusters(users,
                 distance_threshold=0.1,
                 full_name_coef=1,
                 part_name_coef=1,
                 email_name_coef=1,
                 email_score_coef=1,
                 login_score_coef=1,
                 login_email_coef=0,
                 login_name_coef=0):
    """
    Adaptation of the approach from the 'Mining Email Social Networks'. Algorithm measures pair-wise similarity
    between all participants and splits them into clusters with Agglomerative clustering.

    :param users: dataframe with users names, e-mails, and logins
    :param distance_threshold: distance parameter for clustering
    :param full_name_coef:
    :param part_name_coef:
    :param email_name_coef:
    :param email_score_coef:
    :param login_score_coef:
    :param login_email_coef:
    :param login_name_coef:
    :return: dict which provides cluster id for each user
    """
    users = users.copy()
    users = users.drop_duplicates().reset_index().drop('index', axis=1)
    users = users.fillna('')
    users.name = users.name.apply(name_preprocess)

    users['short_email'] = users.email.apply(shorten_email)
    users['first_name'] = users.name.apply(first_name)
    users['last_name'] = users.name.apply(last_name)

    sim_matrix = get_sim_matrix(users,
                                full_name_coef,
                                part_name_coef,
                                email_name_coef,
                                email_score_coef,
                                login_score_coef,
                                login_email_coef,
                                login_name_coef)
    agg = AgglomerativeClustering(n_clusters=None,
                                  distance_threshold=distance_threshold,
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
