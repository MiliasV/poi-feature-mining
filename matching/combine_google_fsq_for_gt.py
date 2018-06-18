import sys
sys.path.append("..")
import postgis_functions
from difflib import SequenceMatcher
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
import fuzzy
import Levenshtein
from nltk.corpus import wordnet
from transliterate import translit
import pois_storing_functions
from urllib.parse import urlparse

errcount = 0

# Match Google & Foursquare POIs for creating ground truth
# (Strict matching rules are used)


def longest_substring(str1, str2):
    # source: https://www.geeksforgeeks.org/sequencematcher-in-python-for-longest-common-substring/
    # initialize SequenceMatcher object with
    # input string
    seq_match = SequenceMatcher(None, str1, str2)

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)
    match = seq_match.find_longest_match(0, len(str1), 0, len(str2))

    # print longest substring
    if (match.size != 0):
        return str1[match.a: match.a + match.size]
    else:
        return ""


def get_ro_similarity(a, b):
    # based on Ratcliff and Obershelp's algorithm
    return SequenceMatcher(None, a, b).ratio()


def get_name_in_latin_alph(name):
    try:
        return translit(name, reversed=True)
    except:
        return name


def get_levenshtein_phonetic_similarity(osm_name, source_name):
    dmeta = fuzzy.DMetaphone()
    try:
        dmeta_osm = dmeta(osm_name)[0]#.decode("utf-8")
        dmeta_source = dmeta(source_name)[0]#.decode("utf-8")
        return Levenshtein.ratio(dmeta_osm, dmeta_source)

    except Exception as err:
        return None


def get_str_similarity(source, target):
    source = source.lower()
    target = target.lower()
    long_sub = longest_substring(source, target)
    #"long_substring": longest_substring(source, target),
    similarities = {
        "ro_similarity": get_ro_similarity(source, target),
        "dleven_similarity": 1 - normalized_damerau_levenshtein_distance(source, target),
        "leven_similarity": Levenshtein.ratio(source, target),
        "phonetic_similarity": get_levenshtein_phonetic_similarity(source, target),
    }
    if (len(long_sub)):
        similarities["len_long_substring"] = len(long_sub)/len(source)
    else:
        similarities["len_long_substring"] = 0
    res = {k: v for k, v in similarities.items() if v is not None}
    avg_sim = sum(res.values())/len(res.values())
    return avg_sim


def get_gf_similarity_score(google_info, fsq_info):
    # calculate the google-foursquare similarity score
    # (and not the girlfriend similarity score! :p )
    print("ok")


def match_by_phone(fpoint, gpoint, thres):
    try:
        if get_str_similarity(fpoint["phone"], gpoint["phone"])>=thres:
            return True
    except:
        pass
    return False


def match_by_addr(fpoint, gpoint):
    try:
        if fpoint["street"] and fpoint["streetnum"] and \
                fpoint["street"] == gpoint["streetlng"] and fpoint["streetnum"] == gpoint["streetnumlng"]:
            return True
    except:
        pass
    return False


def match_by_name(fpoint, gpoint, thres):
    if get_str_similarity(fpoint["name"], gpoint["name"])>=thres:
        return True
    return False


def match_by_street(fpoint, gpoint, thres):
    try:
    #if "street" in fpoint and "streetlng" in gpoint and fpoint["street"] and gpoint["streetlng"]:
        if get_str_similarity(fpoint["street"], gpoint["streetlng"])>=thres:
            return True
    except:
        pass
    return False


def match_by_website(fpoint, gpoint, thres):
    try:
        fweb = urlparse(fpoint["website"])
        gweb = urlparse(gpoint["website"])
        fweb = fweb["netloc"]
        gweb = gweb["netloc"]
        if Levenshtein.ratio(fweb, gweb)>=thres:
            return True
    except:
        pass
    return False


def print_matched(fpoint, gpoint, attr):
    print("~~~~ MATCH ~~~~ ", attr)
    print("~FSQ: ", fpoint)
    print("~GOO: ", gpoint)


def add_matched_to_db(session, FTable, GTable, MTable, fpoint, gpoint, reason):
    match = {"id": fpoint["id"] + "_" + gpoint["id"], "fsqid": fpoint["id"],
             "googleid": gpoint["id"], "reason": reason}
    global errcount
    try:
        session.add(FTable(**fpoint))
        session.add(GTable(**gpoint))
        session.add(MTable(**match))
        session.commit()
        print("~~ MATCH INSERTED!")
        # print(fpoint)
        # print(gpoint)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    except Exception as err:
        errcount+=1
        print("~~ MATCH NOT INSERTED", reason, fpoint["point"])
        print(fpoint)
        print(gpoint)
        print(err)
        session.rollback()


if __name__ == "__main__":
    # choose radius
    #for rad in [275]:
    # a = "Wildkamperen"
    # print(get_str_similarity(a, "MELEDI Amsterdam"))
    # print(b)
    fpoints = postgis_functions.get_pois_for_matching("fsq_ams_places", 0)
    # Create matched tables
    session, FTable = pois_storing_functions.setup_db("axp_fsq_ams_matched_table", "", "fsq_matched")
    session.close()
    session, GTable = pois_storing_functions.setup_db("axp_google_ams_matched_table", "", "google_matched")
    session.close()
    session, MTable = pois_storing_functions.setup_db("axp_matching_google_fsq_ams_table", "", "matching_table")
    rad = 300
    ###############
    # FOR EACH POI#
    ###############
    countnamestreet=0
    countwebstreet=0
    countphone=0
    countaddr=0
    point = 0
    countwebname = 0
    countnamedist = 0
    matching_score = 0
    count = 0
    thres = 7
    for fpoint in fpoints:
        score = 0
        # if count==:
        #     print(fpoint["originalpointindex"])
        #     break
        print("#################################################################################")
        print("POINT: ", point)
        print("COUNT (addr): ", countaddr)
        print("COUNT (Phone): ", countphone)
        print("COUNT (Namestreet): ", countnamestreet)
        print("COUNT (webstreet): ", countwebstreet)
        print("COUNT (webname): ", countwebname)
        print("COUNT (namedist): ", countnamedist)
        print("TOTAL: ", count)#countaddr +countphone +countnamestreet + countwebstreet + countwebname + countnamedist)
        print("@@@@@@@@@@@@@@@@@@@@@")
        point+=1
        # Gather places within radius
        google_closest_points = postgis_functions.get_matching_attr_from_pois_within_radius\
             ("google_ams_whole_clipped_40", fpoint["lat"], fpoint["lng"], rad)
        for gpoint in google_closest_points:
            # if match_by_website(fpoint, gpoint, 0.8) and match_by_name(fpoint, gpoint, 0.6):
            #     if score< 2:
            #         fmatched, gmatched, reason, score = fpoint.copy(), gpoint.copy(), "web_name",  2
            #         countwebname+=1
            # match by website and street
            if match_by_website(fpoint, gpoint, 0.8) and match_by_street(fpoint, gpoint, 0.8):
                if score< 3:
                    fmatched, gmatched, reason, score = fpoint.copy(), gpoint.copy(), "web_street",  3
                    countwebstreet+=1
                # Match by phone  (if it exists) (total match!)
            if match_by_phone(fpoint, gpoint, 1) and match_by_name(fpoint, gpoint, 0.5):
                if score < 6:
                    fmatched, gmatched, reason, score = fpoint.copy(), gpoint.copy(), "phone_name", 6
                    countphone += 1
            # match by addr total and name 0.3
            if match_by_addr(fpoint, gpoint) and match_by_name(fpoint, gpoint, 0.7):
                if score < 5:
                    fmatched, gmatched, reason, score = fpoint.copy(), gpoint.copy(), "addr_name",  7
                    countaddr+=1
            # match by name and street
            if match_by_name(fpoint, gpoint, 0.7) and match_by_street(fpoint, gpoint, 0.8):
                if score < 7:
                    fmatched, gmatched, reason, score = fpoint.copy(), gpoint.copy(), "street_name",  7
                    countnamestreet+=1
            if match_by_name(fpoint, gpoint, 0.7) and postgis_functions.get_distance(fpoint, gpoint)<=40:
                if score < 7:
                    fmatched, gmatched, reason, score = fpoint.copy(), gpoint.copy(), "dist_name",  8
                    countnamedist+=1
        if score > 0:
            count+=1
            fmatched["point"] = point
            gmatched["point"] = point
            add_matched_to_db(session, FTable, GTable, MTable, fmatched, gmatched, reason)

    print("RAD = ", rad)
    print("Total Points Matched = ", count , " / ", point) # countnamestreet + countwebstreet + countphone + countaddr + countwebname + countnamedist)
    print("ERRORS: ", errcount)
