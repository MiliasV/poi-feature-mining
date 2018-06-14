import sqlite3
import psycopg2
import psycopg2.extras


def get_distance(fpoint, gpoint):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT ST_DISTANCE_SPHERE(ST_MAKEPOINT({flat},{flng}), ST_MAKEPOINT({glat},{glng}))"\
              .format(flat=fpoint["lat"], flng=fpoint["lng"], glat=gpoint["lat"], glng=gpoint["lng"]))
    return c.fetchall()[0]["st_distance_sphere"]


def get_points_from_db(tab, last_searched):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor()
    c.execute("SELECT rn, id, originalpointindex, lat, lng "
              "FROM "
              "(SELECT row_number() OVER (ORDER BY originalpointindex NULLS LAST) AS rn,  "
              "id, originalpointindex, lat, lng "
              " FROM {table} ) AS rowselection "
              "WHERE rn>={last_row}".format(table=tab, last_row=last_searched))
    return c

#
# def get_attr_for_matching_from_index(c, table):
#     c.execute("SELECT ST_x(geom), ST_y(geom) FROM {the_table}".format(the_table=table))
#     return c.fetchall()


def get_ll_from_geom(c, table):
    c.execute("SELECT lat, lng FROM {the_table}".format(the_table=table))
    return c


def get_row_and_ll_of_pois(c, table):
    c.execute("SELECT id, ST_x(geom), ST_y(geom) FROM {the_table}".format(the_table=table))
    return c.fetchall()


def get_row_from_index(c, table, index):
    c.execute("SELECT * FROM {the_table} WHERE id = '{idx}'".format(the_table=table, idx=index))
    return c.fetchall()


def get_pois_for_matching(tab, last_searched):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT rn, id, originalpointindex, name, type1, type2, type3, type4, "
              "website, street, streetnum, phone, lat, lng "
              "FROM "
              "(SELECT row_number() OVER (ORDER BY originalpointindex NULLS LAST) AS rn,  "
              "id, originalpointindex, lat, lng, name, type1, type2, type3, type4, "
              "website, streetnum, street, phone  "
              " FROM {table} ) AS rowselection "
              "WHERE rn>={last_row}".format(table=tab, last_row=last_searched))
    return c


def get_matching_attr_from_pois_within_radius(table, lng, lat, rad):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    c.execute("SELECT id, name, type1, type2, type3, type4, lat, lng, "
              "website, streetlng, streetnumlng, phone "
              "FROM {geotable} "
              "WHERE ST_DWithin(geom::geography, ST_MakePoint({lat},{lng})::geography,{rad})"
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