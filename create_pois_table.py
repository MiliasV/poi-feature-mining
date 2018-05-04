from sqlalchemy.ext.automap import automap_base
from geoalchemy2 import Geometry
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker, mapper, Session


def create_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("name", String),
              Column("type1", String),
              Column("type2", String),
              Column("type3", String),
              Column("type4", String),
              Column("type5", String),
              Column("rscount", Numeric),
              Column("phcount", Numeric),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("searchlat", Numeric),
              Column("searchlng", Numeric),
              Column("poptimes", String),
              Column("geom", Geometry('Point')),
              Column("json", String))
        metadata.create_all()


class ALTable(object):
    pass


def setup_db():
    Base = automap_base()
    # Connect to the database
    #db = create_engine("sqlite:///../../../databases/google_places.db")
    db = create_engine('postgresql://postgres:postgres@localhost/pois')
    # create object to manage table definitions
    metadata = MetaData(db)
    # create table if it doesn't exist - also define the table.
    # change it in GTable as well!
    create_table(db, "google_ams_center_40", metadata)
    # reflect the tables
    Base.prepare(db, reflect=True)
    GTable = Base.classes.google_ams_center_40
    #data_table = Table('google_ams_centroids_40', metadata, autoload=True)
    # create a Session
    session = Session(db)
    return session, GTable