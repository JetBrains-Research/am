import numpy as np

from Levenshtein import distance as LevDist
import string
from tqdm import tqdm


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


def email_base(email):
    """
    returns the base of e-mail (everything before @)
    """
    if email == '':
        return ''
    return email.split('@')[0]


def na_handler(function):
    def wrapper(*args):
        for arg in args:
            if arg == '':
                return 1
        return function(*args)

    return wrapper


@na_handler
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


@na_handler
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


def name_distance(u1, u2):
    full_name_score = get_norm_levdist(u1['name'], u2['name'])

    part_name_score = get_norm_levdist(u1['first_name'], u2['first_name']) + get_norm_levdist(u1['last_name'],
                                                                                              u2['last_name'])
    part_name_score /= 2

    return min(full_name_score, part_name_score)


def name_email_distance(u1, u2):
    score = max(name_handle_dist((u1['first_name'], u1['last_name']),
                                 u2['email']),
                name_handle_dist((u2['first_name'], u2['last_name']),
                                 u1['email'])
                )

    return score


def login_name_distance(u1, u2):
    score = max(name_handle_dist((u1['first_name'], u1['last_name']),
                                 u2['login']),
                name_handle_dist((u2['first_name'], u2['last_name']),
                                 u1['login'])
                )
    return score


def login_email_distance(u1, u2):
    score = 1
    if not u1['login'] is np.nan and not u2['email_base'] is np.nan:
        if len(u1['login']) > 2 and len(u2['email_base']) > 2:
            score = get_norm_levdist(u1['login'], u2['email_base'])
    if not u1['email_base'] is np.nan and not u2['login'] is np.nan:
        if len(u1['email_base']) > 2 and len(u2['login']) > 2:
            score = min(score, get_norm_levdist(u1['email_base'], u2['login']))
    return score


def login_distance(u1, u2):
    score = 1
    if not u1['login'] is np.nan and not u2['login'] is np.nan:
        if len(u1['login']) > 2 and len(u2['login']) > 2:
            score = get_norm_levdist(u1['login'], u2['login'])
    return score


def email_distance(u1, u2):
    score = 1
    if not u1['email_base'] is np.nan and not u2['email_base'] is np.nan:
        if len(u1['email_base']) > 2 and len(u2['email_base']) > 2:
            score = get_norm_levdist(u1['email_base'], u2['email_base'])
    return score


def get_sim_matrix(users,
                   sim_measure):
    """
    calculates name similarity matrix between users
    """
    sim_matrix = np.zeros((len(users), len(users)))
    for i1, row1 in tqdm(users.iterrows()):
        sim_matrix[i1] = [sim_measure(row1, row2) if i1 < i2 else 0 for (i2, row2) in users.iterrows()]
    sim_matrix = sim_matrix + sim_matrix.T

    return sim_matrix
