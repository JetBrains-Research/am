from .utils import get_clusters


def alias_matching(users, threshold=0.1,
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
       :param login_name_coef:/
       :return: dict which provides cluster id for each user
   """

    if 'full_id' not in users:
        users['full_id'] = users.apply(lambda x: f"{x['name']}:{x['email']}:{x['login']}", axis=1)

    key2id = get_clusters(users,
                          threshold,
                          full_name_coef,
                          part_name_coef,
                          email_name_coef,
                          email_score_coef,
                          login_score_coef,
                          login_email_coef,
                          login_name_coef)

    users['cluster'] = users['full_id'].apply(lambda x: key2id[x])
    return users
