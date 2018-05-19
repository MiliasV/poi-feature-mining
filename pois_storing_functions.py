from sqlalchemy.ext.automap import automap_base
from geoalchemy2 import Geometry
from sqlalchemy import *
from sqlalchemy.orm import Session


def create_fsq_pois_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("originalpointindex", Numeric),
              Column("name", String),
              Column("type1", String),
              Column("type2", String),
              Column("type3", String),
              Column("type4", String),
              Column("type5", String),
              Column("street", String),
              Column("streetnum", String),
              Column("checkins", Numeric),
              Column("userscount", Numeric),
              Column("tipcount", Numeric),
              Column("visitscount", Numeric),
              Column("rating", Numeric),
              Column("price", String),
              Column("likescount", Numeric),
              Column("photoscount", Numeric),
              Column("website", String),
              Column("facebook", String),
              Column("twitter", String),
              Column("phone", String),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("searchlat", Numeric),
              Column("searchlng", Numeric),
              Column("poptimes", String),
              Column("geom", Geometry('Point')),
              Column("json", String))
        metadata.create_all()


def create_google_pois_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("originalpointindex", Numeric),
              Column("name", String),
              Column("type1", String),
              Column("type2", String),
              Column("type3", String),
              Column("type4", String),
              Column("type5", String),
              Column("streetlng", String),
              Column("streetsrt", String),
              Column("streetnumlng", String),
              Column("streetnumsrt", String),
              Column("website", String),
              Column("rscount", Numeric),
              Column("phcount", Numeric),
              Column("phone", String),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("searchlat", Numeric),
              Column("searchlng", Numeric),
              Column("rating", String),
              Column("poptimes", String),
              Column("geom", Geometry('Point')),
              Column("json", String))
        metadata.create_all()


def create_gsv_pois_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("placesid", Numeric),
              Column("panosid", String),
              Column("head", String),
              Column("year", String),
              Column("month", String),
              Column("path", String),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("searchlat", Numeric),
              Column("searchlng", Numeric),
              Column("geom", Geometry('Point')))
        metadata.create_all()


def create_count_per_poi_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", Numeric, primary_key=True, nullable=False),
              Column("countplaces", Numeric),
              Column("countduplicates", Numeric)
              )
        metadata.create_all()


class ALTable(object):
    pass


def setup_db(pois_table_name, count_table_name, source):
    Base = automap_base()
    # Connect to the database
    #db = create_engine("sqlite:///../../../databases/google_places.db")
    db = create_engine('postgresql://postgres:postgres@localhost/pois')
    # create object to manage table definitions
    metadata = MetaData(db)
    # create table if it doesn't exist - also define the table.
    # change it in GTable as well!
    if source == "google":
        create_google_pois_table(db, pois_table_name, metadata)
    else:
        create_fsq_pois_table(db, pois_table_name, metadata)
    create_count_per_poi_table(db, count_table_name,metadata)
    # reflect the tables
    Base.prepare(db, reflect=True)
    STable = getattr(Base.classes, pois_table_name)
    CTable = getattr(Base.classes, count_table_name)
    #data_table = Table('google_ams_centroids_40', metadata, autoload=True)
    # create a Session
    session = Session(db)
    return session, STable, CTable


def get_last_id_from_logfile(logfile):
    with open(logfile, "r") as f:
        lines = f.readlines()
        f.close()
        return int(lines[-1])


def insert_count_data(session, CTable, poi_number, count_places, count_dupl):
    try:
        session.add(CTable(id=poi_number, countplaces=count_places, countduplicates=count_dupl))
        session.commit()
    except Exception as err:
        session.rollback()
        print("# [COUNT] NOT INSERTED: ", err)