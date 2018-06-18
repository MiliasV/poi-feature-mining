import postgis_functions
import combine_google_fsq_for_gt
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
import Levenshtein
import pois_storing_functions
import psycopg2


def get_str_similarity(source, target, label):
    try:
        source = source.lower()
        target = target.lower()
        long_sub = combine_google_fsq_for_gt.longest_substring(source, target)
        similarities = {
            label + "rosimilarity": combine_google_fsq_for_gt.get_ro_similarity(source, target),
            label + "dlevensimilarity": 1 - normalized_damerau_levenshtein_distance(source, target),
            label + "levensimilarity": Levenshtein.ratio(source, target),
            label + "phoneticsimilarity": combine_google_fsq_for_gt.get_levenshtein_phonetic_similarity(source, target),
        }
        if (len(long_sub)):
            similarities[label + "lenlongsubstring"] = len(long_sub)/len(source)
        else:
            similarities[label + "lenlongsubstring"] = 0
    except:
        similarities = {
            label + "rosimilarity": None,
            label + "dlevensimilarity": None,
            label + "levensimilarity": None,
            label + "phoneticsimilarity": None,
            label + "lenlongsubstring": None,
        }
    res = {k: v for k, v in similarities.items()}
    return res


def matched(fpoint, gpoint):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT * FROM gt_matching_google_fsq_ams_table WHERE fsqid='{fid}' AND googleid='{gid}'".
              format(fid=fpoint["id"], gid=gpoint["id"]))
    return c.fetchone()


def get_similarities_dict(fpoint, gpoint):
    res = {}
    for label in ["name", "website", "phone", "street", "streetnum"]:
        if label == "street" or label == "streetnum":
            res.update(get_str_similarity(fpoint[label], gpoint[label + "lng"], label))
        else:
            res.update(get_str_similarity(fpoint[label], gpoint[label], label))
    return res


def add_similarities_to_db(session, STable, sim_dict):
    try:
        session.add(STable(**sim_dict))
        session.commit()
        #print("~~ Similarities INSERTED!")
    except Exception as err:
        # errcount+=1
        #print("~~ similarities NOT INSERTED")
        print(err)
        session.rollback()


if __name__ == '__main__':
    fpoints = postgis_functions.get_pois_for_matching("gt_fsq_ams_matched_table", 0)
    session, STable = pois_storing_functions.setup_db("similarities_ams_table", "", "similarities")
    rad = 300
    point = 0
    for fpoint in fpoints:
        print("Point: ", point)
        point+=1
        count_matched = 0
        count_nmatched = 0
        google_closest_points = postgis_functions.get_matching_attr_from_pois_within_radius\
             ("google_ams_whole_clipped_40", fpoint["lat"], fpoint["lng"], rad)
        for gpoint in google_closest_points:
            sim_dict = get_similarities_dict(fpoint, gpoint)
            sim_dict["id"] = fpoint["id"] + "_" + gpoint["id"]
            if matched(fpoint, gpoint):
                count_matched+=1
                sim_dict["match"] = 1
                add_similarities_to_db(session, STable, sim_dict)
                if count_nmatched>=10:
                    break
            elif count_nmatched<10:
                count_nmatched+=1
                sim_dict["match"] = 0
                add_similarities_to_db(session, STable, sim_dict)
        print("MATCHED: {match} \nNON MATCHED: {nmatched}".format(match=count_matched, nmatched=count_nmatched))


