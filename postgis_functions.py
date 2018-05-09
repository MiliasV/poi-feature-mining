import sqlite3
import psycopg2


def get_ll_from_geom(c, table):
    c.execute("SELECT ST_x(geom), ST_y(geom) FROM {the_table}".format(the_table=table))
    return c


def get_row_and_ll_of_pois(c, table):
    c.execute("SELECT *, ST_x(geom), ST_y(geom) FROM {the_table}".format(the_table=table))
    return c.fetchall()


def get_pois_within_radius(c, table, ll, radius):
    c.execute("SELECT * FROM {geotable} WHERE ST_DWithin(geom::geography, ST_MakePoint{ll}, {rad})"
              .format(geotable=table, ll=ll, rad=radius))
    return c.fetchall()


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