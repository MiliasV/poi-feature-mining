import foursquare as fs
import sys
sys.path.append("..")
import foursquare_my.fsq_config as fsq_config
import foursquare_my.get_fsq_data as get_fsq_data
import get_map_points_to_search
import pprint
import pois_storing_functions
import simplejson as json
import sqlite3
import logging_functions


def setup():
    CLIENT_ID = fsq_config.cl_id
    CLIENT_SECRET = fsq_config.cl_sec
    client = fs.Foursquare(client_id=CLIENT_ID,
                           client_secret=CLIENT_SECRET)
    return client


def get_value_if_exist(dict, value):
    if value in dict:
        return dict[value]
    else:
        return None


def extract_data_from_json(fsq_json, categories_tree):
    res = {}
    res["lat"] = fsq_json["location"]["lat"]
    res["lng"] = fsq_json["location"]["lng"]
    res["id"] = fsq_json["id"]
    res["name"] = fsq_json["name"]
    if get_value_if_exist(fsq_json, "popular"):
        res["poptimes"] = True
    else:
        res["poptimes"] = False
    res["website"] = get_value_if_exist(fsq_json, "url")
    res["street"], res["streetnum"] = get_fsq_data.get_addr_from_venue(fsq_json)
    if "stats" in fsq_json:
        res["checkins"] = get_value_if_exist(fsq_json["stats"], "checkinsCount")
        res["userscount"] = get_value_if_exist(fsq_json["stats"], "usersCount")
        res["tipcount"] = get_value_if_exist(fsq_json["stats"], "tipCount")
        res["visitscount"] = get_value_if_exist(fsq_json["stats"], "visitsCount")
    res["rating"] = get_value_if_exist(fsq_json, "rating")
    if "price" in fsq_json:
        res["price"] = get_value_if_exist(fsq_json["price"], "message")
    if "likes" in fsq_json:
        res["likescount"] = get_value_if_exist(fsq_json["likes"], "count")
    if "photos" in fsq_json:
        res["photoscount"] = get_value_if_exist(fsq_json["photos"], "count")
    if "contact" in fsq_json:
        res["facebook"] = get_value_if_exist(fsq_json["contact"], "facebookName")
        res["twitter"] = get_value_if_exist(fsq_json["contact"], "twitter")
        res["phone"] = get_value_if_exist(fsq_json["contact"], "formattedPhone")
    if "categories" in fsq_json:
        res["type1"] = fsq_json["categories"][0]["name"]
        for categ in fsq_json["categories"]:
            if categ["primary"] == True:
                res["type1"] = categ["name"]
                break
    res["type2"], res["type3"], res["type4"], res["type5"] = None, None, None, None
    categ_hier = get_fsq_data.find_key_path_from_value(categories_tree["categories"], res["type1"])
    for i in range(len(categ_hier)-1):
        res["type" + str(i+2)] = categ_hier[i+1]

    return res


def insert_data(session, FTable, search_lat, search_lng, fsq_json, ogc_fid,
                count_places, count_duplicates, categories_tree):
    res = extract_data_from_json(fsq_json, categories_tree)
    res["json"] = json.dumps(fsq_json)
    res["geom"] = 'POINT({} {})'.format(res["lng"], res["lat"])
    res["searchlat"] = search_lat
    res["searchlng"] = search_lng
    res["originalpointindex"] = ogc_fid
    try:
        session.add(
            FTable(**res))
        session.commit()
        count_places += 1
        print("~~ ", fsq_json["name"], " INSERTED!")

    except Exception as err:
        session.rollback()
        count_duplicates += 1
        print("# NOT INSERTED: ", err)
    return count_places, count_duplicates


def config_parameters_for_searching(source, city):
    if city == "ams":
        center_db = "/home/bill/Desktop/thesis/maps/amsterdam/" + city + "_whole_clipped_with_lat_lon.sqlite"
        table = city + "_whole_clipped_with_lat_lon.sqlite"
    elif city=="ath":
        center_db = "/home/bill/Desktop/thesis/maps/athens/" + city + "_whole_clipped_with_lat_lon.sqlite"
        table = "ath_center_export_lat_lon"

    rad = 28
    # databases
    logfile = "/home/bill/Desktop/thesis/logfiles/" + source + "_" + city + "_whole_clipped_point_iterations.txt"
    errorfile = "/home/bill/Desktop/thesis/logfiles/" + source + "_" + city + "whole_clipped_errors.txt"

    # connect to osm db
    conn = sqlite3.connect(center_db)
    c = conn.cursor()
    last_searched_id = get_map_points_to_search.get_last_id_from_logfile(logfile)
    # ams_whole_clipped_with_lat_lon ams table
    c.execute("SELECT ogc_fid, lat, Lon "
              "FROM {table} "
              "WHERE ogc_fid>={last_id}-1 "
              "ORDER BY ogc_fid ASC".format(last_id=last_searched_id, table=table))
    return c, rad, logfile, errorfile


if __name__ == '__main__':
    city = "ath"
    cl = setup()
    c, rad, logfile, errorfile = config_parameters_for_searching("fsq",city)
    last_searched_id = pois_storing_functions.get_last_id_from_logfile(logfile)
    # setup table and session
    # define which table (fsq table, count table)
    pp = pprint.PrettyPrinter(indent=4)
    # get hierarchy categories tree
    categories_tree = cl.venues.categories()
    #print(get_fsq_data.find_key_path_from_value(categories_tree["categories"], "Cantonese Restaurant"))
    # For each point --> search nearby in google
    for ogc_fid, point_lat, point_lng in c:
        count_places = 0
        count_duplicates = 0
        print("POINT: ", ogc_fid, point_lat, point_lng)
        # keep the id of the last searched point
        get_map_points_to_search.log_last_searched_point(logfile, ogc_fid)
        ll = str(point_lat) + ", " + str(point_lng)
        # define which table
        session, FTable, CTable = pois_storing_functions.setup_db("fsq_" + city + "_whole__clipped_40",
                                                                  "fsq_" + city + "_whole_clipped_40_count","FSQ")
        # specify which categories to search for!
        # food = 4d4b7105d754a06374d81259
        # arts and entertainment = 4d4b7104d754a06370d81259
        # college & uni = 4d4b7105d754a06372d81259
        # nightlife spot = 4d4b7105d754a06376d81259
        # outdoor & recreation = 4d4b7105d754a06377d81259
        # food and drink shop = 4bf58dd8d48988d1f9941735
        # outdoors and recreation = 4d4b7105d754a06377d81259
        # Bank = 4bf58dd8d48988d10a951735
        # clοthing store = 4bf58dd8d48988d103951735
        # drugstore = 5745c2e4498e11e7bccabdbd
        # Bus station = 4bf58dd8d48988d1fe931735
        # Bus stop = 52f2ab2ebcbc57f1066b8b4f
        # hotel = 4bf58dd8d48988d1fa931735
        # train station = 4bf58dd8d48988d129951735
        # tram station = 52f2ab2ebcbc57f1066b8b51

        categories = "4d4b7105d754a06374d81259," \
                     "4d4b7104d754a06370d81259," \
                     "4d4b7105d754a06372d81259," \
                     "4d4b7105d754a06376d81259," \
                     "4d4b7105d754a06377d81259,"\
                     "4d4b7105d754a06377d81259,"\
                     "4bf58dd8d48988d1f9941735,"\
                     "4bf58dd8d48988d10a951735," \
                     "4bf58dd8d48988d103951735," \
                     "5745c2e4498e11e7bccabdbd," \
                     "4bf58dd8d48988d1fe931735," \
                     "52f2ab2ebcbc57f1066b8b4f," \
                     "4bf58dd8d48988d1fa931735," \
                     "4bf58dd8d48988d129951735," \
                     "52f2ab2ebcbc57f1066b8b51"
        venues = get_fsq_data.get_venues_by_ll(cl, ll, rad, categories)
        for venue in venues["venues"]:
            fsq_json = get_fsq_data.get_venue_details(cl, venue["id"])
            #count_places, count_duplicates =\
            lat = fsq_json["location"]["lat"]
            lng = fsq_json["location"]["lng"]
            count_places, count_duplicates = insert_data(session, FTable, point_lat, point_lng, fsq_json, ogc_fid,
                            count_places, count_duplicates, categories_tree)
            #pois_storing_functions.insert_count_data(session, CTable, ogc_fid, count_places, count_duplicates)