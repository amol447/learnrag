from sqlalchemy import (
    Column,
    Integer,
    String,
    create_engine,
    ForeignKey,
    Table,
    MetaData,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

table_space = "alternate"


def get_engine():
    user = os.getenv("PGUSER", "learnrag")
    password = os.getenv("PGPASS", "learnrag")
    return create_engine(f"postgresql://{user}:{password}@localhost/img_project")


metadata = MetaData()
image_data = Table(
    "image_data",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("path", String, unique=True, nullable=False),
    Column("overall_embedding", Vector(128), nullable=False),
    Column("image_type", String, nullable=False),
    postgresql_tablespace=table_space,
)


face_data = Table(
    "face_data",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("image_id", Integer, ForeignKey("image_data.id"), nullable=False),
    Column("face_embedding", Vector(128), nullable=True),
    Column("face_box", ARRAY(Integer), nullable=False),
    Column("face_tag_id", Integer, ForeignKey("person.id"), nullable=True),
    postgresql_tablespace=table_space,
)
person = Table(
    "person",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("first_name", String, nullable=False),
    Column("last_name", String, nullable=False),
    postgresql_tablespace=table_space,
)
# Example engine and session creation
engine = get_engine()
metadata.create_all(engine)
# Session = sessionmaker(bind=engine)
# session = Session()
