import postgis_functions
import pprint
import ast
import json
import pois_storing_functions


def gf_has_attribute(point, attribute):
    fhasattr = 0
    ghasattr = 0
    if point["g" + attribute]:
        ghasattr = 1
    if point["f" + attribute]:
        fhasattr = 1
    return ghasattr, fhasattr


def has_attribute(point, attribute):
    hasattr = 0
    if point[attribute]:
        hasattr = 1
    return hasattr


def is_num_in_the_middle(a, b, c):
    if a < b < c:
        return True
    else:
        return False


def get_popular_time(poptimes):
    # define timeslots
    morning = (5, 12)
    afternoon = (13, 17)
    evening = (18, 21)
    night_1 = (22, 24)
    night_2 = (0, 4)
    # find most popular timeslot
    mor = sum(poptimes[morning[0]:morning[1]])
    after = sum(poptimes[afternoon[0]:afternoon[1]])
    even = sum(poptimes[evening[0]:evening[1]])
    night = sum(poptimes[night_1[0]:night_1[1]]) + sum(poptimes[night_2[0]:night_2[1]])
    poptime = max(mor, after, even, night)
    if poptime == mor:
        return "morning"
    elif poptime == after:
        return "afternoon"
    elif poptime == even:
        return "evening"
    elif poptime == night:
        return "night"


def find_most_popular_timeslot(poptimes):
    pop_time_per_day = {}
    if poptimes:
        poptimes = ast.literal_eval(poptimes)
        for day in poptimes:
            most_popular = get_popular_time(day["data"])
            pop_time_per_day[day["name"]] = most_popular
    return pop_time_per_day


def get_opening_hours_from_json(ohjson, gfdata):
    if "opening_hours" in ohjson:
        for day in ohjson["opening_hours"]["periods"]:
            if "close" in day:
                gfdata["day" + str(day["close"]["day"]) + "close"] = day["close"]["time"]
            if "open" in day:
                gfdata["day" + str(day["open"]["day"]) + "open"] = day["open"]["time"]
    return gfdata


def get_nearby_places(d, types, rad):
    if types:
        for t in types:
            d[t["type"] + "_" + rad] = t["count"]
    return d


if __name__ == '__main__':
    # initializations
    count = 0
    city = "ath"
    #format_str = "%Y-%m-%d %H:%M:%S"
    #session, GFTable = pois_storing_functions.setup_db("matched_gf_features_ams", "notused", "gf_features")
    gtab = "matched_places_google_" + city
    ftab = "matched_places_fsq_" + city
    # get places
    gfpoints = postgis_functions.get_google_fsq_features(gtab, ftab)
    session, GFTable = pois_storing_functions.setup_db("matched_places_gf_features_" + city, "notused", "gf")

    #fpoints = postgis_functions.get_rows_from_id_not_in_table("matched_fsq_ams", "matched_text_features_ams", "id")

    # selecting what to store in table from the general information from google and fsq
    for gf in gfpoints:
        gfdata = {}
        # for using the function id is fsq id
        gf["id"] = gf["fid"]
        types_100 = postgis_functions.get_place_types_in_radius(gf, "matched_places_fsq_" + city, 100)
        types_1000 = postgis_functions.get_place_types_in_radius(gf, "matched_places_fsq_" + city, 1000)
        types_5000 = postgis_functions.get_place_types_in_radius(gf, "matched_places_fsq_" + city, 5000)
        gfdata = get_nearby_places(gfdata, types_100, "100")
        gfdata = get_nearby_places(gfdata, types_1000, "1000")
        gfdata = get_nearby_places(gfdata, types_5000, "5000")
        # for k in gfdata:
        #     print('Column("' + k +'", Numeric),')
        # print(a)
        gfdata["id"] = gf["gid"] + "_" + gf["fid"]
        gfdata["name"] = gf["gname"]
        gfdata["type"] = gf["type"]
        gfdata["point"] = gf["point"]
        gfdata["placesid"] = gf["gid"]
        gfdata["fid"] = gf["fid"]
        gfdata["ghasweb"], gfdata["fhasweb"] = gf_has_attribute(gf, "website")
        gfdata["ghasphone"], gfdata["fhasphone"] = gf_has_attribute(gf, "phone")
        gfdata["grscount"] = gf["grscount"]
        gfdata["grating"] = gf["grating"]
        gfdata["frating"] = gf["frating"]
        gfdata["ftipcount"] = gf["ftipcount"]
        gfdata["fprice"] = gf["fprice"]
        gfdata["fphotoscount"] = gf["fphotoscount"]
        gfdata["flikescount"] = gf["flikescount"]
        gfdata["fhasfacebook"] = has_attribute(gf, "ffacebook")
        gfdata["fhastwitter"] = has_attribute(gf, "ftwitter")
        gfdata["ghaspoptimes"] = has_attribute(gf, "gpoptimes")
        gfdata.update(find_most_popular_timeslot(gf["gpoptimes"]))
        gfdata = get_opening_hours_from_json(json.loads(gf["gjson"]), gfdata)
        # for key in gfdata:
        #     print("Column(" + '"' + key + '"' + ", Numeric),")
        print("############################################################")
        try:
            session.add(GFTable(**gfdata))
            session.commit()
            print(gfdata["name"], " INSERTED!")
        except Exception as err:
            session.rollback()
            print("# NOT INSERTED: ", err)
        print("############################################################")