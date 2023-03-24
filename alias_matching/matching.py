from alias_matching.utils import get_clusters


def bird_matching(users,
                  distance_threshold=0.1,
                  name_coef=1,
                  email_name_coef=1,
                  email_score_coef=1,
                  login_score_coef=1,
                  login_email_coef=0,
                  login_name_coef=0):
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

    if 'full_id' not in users:
        users['full_id'] = users.apply(lambda x: f"{x['name']}:{x['email']}:{x['login']}", axis=1)

    key2id = get_clusters(users,
                          distance_threshold,
                          name_coef,
                          email_name_coef,
                          email_score_coef,
                          login_score_coef,
                          login_email_coef,
                          login_name_coef)

    users['cluster'] = users['full_id'].apply(lambda x: key2id[x])
    return users
