# coding=utf-8

import sqlite3
import psycopg2
import psycopg2.extras



def update_column_to_table_by_key(tab, value, column, key_col, key):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("UPDATE {table} "
              "SET {col}='{v}' "
              "WHERE {keycol} ='{key}'".format(table=tab, v=value, col=column, keycol=key_col, key=key))
    conn.commit()


def add_type_to_table(tab, colid, value, fid):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("UPDATE {table} "
              "SET type='{v}' "
              "WHERE {col} ='{fid}'".format(table=tab, col=colid, v=value, fid=fid))
    conn.commit()


def get_matched_placesid_from_fsqid(fsqid, ftable, gtable):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT google.id from {gtable} as google "
              "INNER JOIN {ftable} as fsq "
              "on fsq.point=google.point "
              "WHERE fsq.id='{fid}'".format(gtable=gtable, ftable=ftable, fid=fsqid))
    return c.fetchall()[0]["id"]


def get_feature_not_in_table(table, table_target, feature):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT * from {table} "
              "WHERE {feature} not in ( "
              "SELECT {feature} from  {table_target} )".format(table=table, feature=feature, table_target=table_target))
    return c


def get_text_from_tweets(table):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT text from {table} ".format(table=table))
    return c.fetchall()


def add_processed_text_to_table(data, col, table, tweetid):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    print("UPDATING...")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        c.execute("UPDATE {tab} "
              "SET {col}='{v}' "
              "WHERE id ='{tid}'".format(tab=table, col=col, v=data, tid=tweetid))
        conn.commit()
    except:
        print("Not UPDATED")

def get_tweets_per_lang_from_fsqid(tab, fsqid, lang):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT * from {table} "
              "WHERE fsqid = '{fid}'"
              "AND lang = '{lang}'".format(table=tab, lang=lang, fid=fsqid))
    return c


def update_tweets_language(table, data, tweetid):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    print("UPDATING...")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("UPDATE {tab} "
              "SET lang='{v}' "
              "WHERE id ='{tid}'".format(tab=table, v=data, tid=tweetid))
    conn.commit()


def get_col_from_feature_per_lang(tab, col,  feature, val, lang):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT {c} from {table} "
              "WHERE {f} = '{fid}'"
              "AND lang = '{lang}'".format(table=tab, c=col, f=feature, lang=lang, fid=val))
    return c.fetchall()


def get_lda_text_per_lang(tab, lang):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT processedtextlda from {table} "
              "WHERE lang = '{lang}'".format(table=tab, lang=lang))
    return c.fetchall()


def get_tweets_lda_text_per_lang_from_fsqid(tab, fsqid, lang):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT processedtextlda from {table} "
              "WHERE fsqid = '{fid}'"
              "AND lang = '{lang}'".format(table=tab, lang=lang, fid=fsqid))
    return c.fetchall()


def get_tweets_from_fsqid(tab, fsqid):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT * from {table} "
              "WHERE fsqid = '{fid}'".format(table=tab, fid=fsqid))
    return c


def get_rows_from_table_where_col_is_null(tab, col):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT * from {table} "
              "WHERE {col} is null".format(table=tab, col=col))
    return c.fetchall()


def get_rows_from_table(tab):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT * from {table}".format(table=tab))
    return c.fetchall()


def get_daytime_by_id(tab, getid):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT createdat from {table} "
              "WHERE fsqid='{getid}'"
              "ORDER BY createdat".format(table=tab, getid=getid))
    return c.fetchall()


def get_id_not_in_table(table_source, table_target, col):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT id from {table_s} "
              "WHERE id NOT IN ("
              "SELECT {id_t} from {table_t})".format(table_s=table_source, table_t=table_target, id_t=col))
    return c.fetchall()


def get_rows_from_id_not_in_table(table_source, table_target, col):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT * from {table_s} "
              "WHERE id NOT IN ("
              "SELECT {id_t} from {table_t})".format(table_s=table_source, table_t=table_target, id_t=col))
    return c.fetchall()


def get_row_by_id(tab, getid):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT * from {table} "
              "WHERE id='{getid}'".format(table=tab, getid=getid))
    poi = c.fetchall()[0]
    res = {k: v if v is not None else "" for k, v in poi.items()}
    return res


def get_distance(fpoint, gpoint):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT ST_DISTANCE_SPHERE(ST_MAKEPOINT({flat},{flng}), ST_MAKEPOINT({glat},{glng}))"\
              .format(flat=fpoint["lat"], flng=fpoint["lng"], glat=gpoint["lat"], glng=gpoint["lng"]))
    return c.fetchall()[0]["st_distance_sphere"]


def get_place_types_in_radius(fpoint, table, rad):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT type, COUNT(*) FROM {tab} "
              "WHERE (ST_DWithin(geom::geography, ST_MakePoint({lng},{lat})::geography,{radius}) "
              "AND id <> '{id}') "
              "GROUP BY type "
              "".format(radius=rad, tab=table, lat=fpoint["lat"], lng=fpoint["lng"], id=fpoint["id"]))
    return c.fetchall()


def get_points_from_db(tab, last_searched):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor()
    c.execute("SELECT rn, id, originalpointindex, lat, lng "
              "FROM "
              "(SELECT row_number() OVER (ORDER BY originalpointindex NULLS LAST) AS rn, "
              "id, originalpointindex, lat, lng "
              " FROM {table} ) AS rowselection "
              "WHERE rn>={last_row}".format(table=tab, last_row=last_searched))
    return c


def get_google_fsq_features(gtab, ftab):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    c.execute("SELECT g.id as gid, f.lat as lat, f.lng as lng, f.id as fid, g.point, g.name as gname, f.name as fname, "
              "f.type, g.website as gwebsite, f.website as fwebsite, g.rscount as  grscount,"
              " g.phone as gphone, f.phone as fphone, g.rating as grating, f.rating as frating, "
              "g.poptimes as gpoptimes, f.tipcount as ftipcount, f.price as fprice, "
              "f.likescount as flikescount, f.photoscount as fphotoscount, f.facebook as ffacebook, "
              "f.twitter as ftwitter, g.json as gjson "
              "FROM {google} as g "
              "INNER JOIN "
              "{fsq} as f "
              "ON g.point = f.point".format(google=gtab, fsq=ftab))
    return c


def get_row_from_feature_and_lang(tab, feature, value, lang):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    c.execute("SELECT * "
              "FROM {table} "
              "WHERE {f}='{v}'AND lang='{l}'".format(table=tab, f=feature, v=value, l=lang))
    return c.fetchall()

def get_pois_from_fsq_db(tab, last_searched):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    c.execute("SELECT * "
              "FROM  {table} "
              "WHERE point>={last_point} "
              "ORDER BY point".format(table=tab, last_point=last_searched))
    return c

def get_photos_from_id(tab, id_list):
    conn = psycopg2.connect("dbname='pois' user='postgres' host='localhost' password='postgres'")
    imgs = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    imgs.execute("SELECT  id, point, placesid, head, panosid, year, month, lat, lng, geom, path  "
                 "FROM {tab_s} "
                 "WHERE id not in {id_l})".format(tab_s=tab, id_l=id_list))
    return imgs

def get_photos_for_od(table_source, table_dest):
    conn = psycopg2.connect("dbname='pois' user='postgres' host='localhost' password='postgres'")
    imgs = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    imgs.execute("SELECT  id, point, placesid, head, panosid, year, month, lat, lng, geom, path  "
                 "FROM {tab_s} "
                 "WHERE placesid not in ("
                 "SELECT  placesid from {tab_d})".format(tab_s=table_source, tab_d=table_dest))
    return imgs


def get_photos_from_db(table, last_inserted):
    conn = psycopg2.connect("dbname='pois' user='postgres' host='localhost' password='postgres'")
    imgs = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    imgs.execute("SELECT  id, point, placesid, head, panosid, year, month, lat, lng, geom, path  "
                 "FROM {tab} "
                 "WHERE point>={last_point} "
                 "ORDER BY point".format(tab=table, last_point=last_inserted))
    return imgs


def get_match_by_id(table, pid):
    conn = psycopg2.connect("dbname='pois' user='postgres' host='localhost' password='postgres'")
    c = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    c.execute("SELECT * FROM {tab} "
                 "WHERE id='{placeid}'".format(tab=table, placeid=pid))
    return c.fetchall()


def get_matched_poi(table, point):
    conn = psycopg2.connect("dbname='pois' user='postgres' host='localhost' password='postgres'")
    poi = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    poi.execute("SELECT * FROM {tab} "
                 "WHERE point='{point}'".format(tab=table, point=point))
    poi = poi.fetchall()[0]
    res = {k: v if v is not None else "" for k, v in poi.items()}
    return res


def get_type_of_place(place_dict):

    # type1
    if "Bar" in place_dict["type1"]:
        return "bar"
    elif "Restaurant" in place_dict["type1"]:
        return "restaurant"
    elif "Café" in place_dict["type1"] or "Cafe" in place_dict["type1"]:
        return "cafe"
    elif "Clothing Store" in place_dict["type1"]:
        return "clothing_store"
    elif "Gym" in place_dict["type1"]:
        return "gym"
    elif "Art Gallery" in place_dict["type1"]:
        return "art_gallery"
    elif "Hotel" in place_dict["type1"]:
        return "hotel"
    elif "Coffee Shop" in place_dict["type1"]:
        return "coffee_shop"
    elif "Nightclub" in place_dict["type1"]:
        return "nightclub"
    elif "Stripclub" in place_dict["type1"]:
        return "stripclub"
    elif "Clothing Store" in place_dict["type1"]:
        return "clothing_store"
    # type 2
    elif "Restaurant" in place_dict["type2"]:
        return "restaurant"
    elif (("Food" in place_dict["type2"] or "Food" in place_dict["type3"]) and
          ("Shop" in place_dict["type1"] or "Store" in place_dict["type1"] or
           "Shop" in place_dict["type2"] or "Store" in place_dict["type2"])):
        return "food_drink_shop"
    elif "Bar" in place_dict["type2"]:
        return "bar"
    elif "Clothing Store" in place_dict["type2"]:
        return "clothing_store"
    elif "Café" in place_dict["type2"] or "Cafe" in place_dict["type2"]:
        return "cafe"
    elif "Hotel" in place_dict["type2"]:
        return "hotel"
    elif "Gym" in place_dict["type2"]:
        return "gym"
    elif ("College" in place_dict["type2"] or
        "College" in place_dict["type3"]):
        return "college_and_university"

    else:
        return False


def get_ll_from_geom(c, table):
    c.execute("SELECT lat, lng FROM {the_table}".format(the_table=table))
    return c


def get_row_and_ll_of_pois(c, table):
    c.execute("SELECT id, ST_x(geom), ST_y(geom) FROM {the_table}".format(the_table=table))
    return c.fetchall()


def get_row_from_index(c, table, index):
    c.execute("SELECT * FROM {the_table} WHERE id = '{idx}'".format(the_table=table, idx=index))
    return c.fetchall()


def get_pois_for_matching(tab, opi):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT * FROM "
              "(SELECT * FROM   {table} ORDER BY originalpointindex) AS ordered "
              "WHERE originalpointindex>={opi}".format(table=tab, opi=opi))
    return c


def get_matching_attr_from_pois_within_radius(table, lat, lng, rad):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT * "
              "FROM {geotable} "
              "WHERE ST_DWithin(geom::geography, ST_MakePoint({lng},{lat})::geography,{rad})"
              "ORDER BY originalpointindex"
              .format(geotable=table, lng=lng, lat=lat, rad=rad))
    return c


def get_pois_from_name(table, poi_name):
    if "'" in poi_name:
        poi_name = poi_name.replace("'", "''")
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT id, name, type1, type2, type3, type4, "
              "website, streetlng, streetnumlng, phone "
              "FROM {geotable} "
              "WHERE name ILIKE '%{poi_name}%'"
              .format(geotable=table, poi_name=poi_name))
    return c


def connect_to_db():
    # connect to osm db
    conn = psycopg2.connect("dbname='pois' user='postgres' host='localhost' password='postgres'")
    c = conn.cursor()
    return c


if __name__ == "__main__":
    # connect to osm db
    conn = psycopg2.connect("dbname='pois' user='postgres' host='localhost' password='postgres'")
    c = conn.cursor()
    long_lat = get_ll_from_geom(c, "google_ams_center_40")
    #ll = '(4.878360100000001, 52.3825754)'
    for row in long_lat:
        print(type(row))
        within_rad = get_pois_within_radius(c, "google_ams_center_40", row, 20)
        for row in within_rad:
            print(row)
        break