import sqlite3


def get_last_id_from_logfile(logfile):
    with open(logfile, "r") as f:
        lines = f.readlines()
        f.close()
        return int(lines[-1])


def config_parameters_for_searching(source, city):
    # city = "ath"
    rad = 28
    # databases
    logfile = "/home/bill/Desktop/thesis/logfiles/" + source + "_" + city + "_whole_clipped_point_iterations.txt"
    errorfile = "/home/bill/Desktop/thesis/logfiles/" + source + "_" + city + "whole_clipped_errors.txt"

    center_db = "/home/bill/Desktop/thesis/maps/athens/" + city + "_whole_clipped_with_lat_lon.sqlite"
    # connect to osm db
    conn = sqlite3.connect(center_db)
    c = conn.cursor()
    last_searched_id = get_last_id_from_logfile(logfile)
    # ams_whole_clipped_with_lat_lon ams table
    c.execute("SELECT ogc_fid, lat, Lon "
              "FROM ath_center_export_lat_lon "
              "WHERE ogc_fid>={last_id}-1 "
              "ORDER BY ogc_fid ASC".format(last_id=last_searched_id))
    return c, rad, logfile, errorfile


def log_last_searched_point(logfile, ogc_fid):
    with open(logfile, "w") as text_file:
        text_file.write("Last searched point \n" + str(ogc_fid))
        text_file.close()

