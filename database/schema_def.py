from sqlalchemy import Column, Integer, String, create_engine, ForeignKey
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os


def get_engine():
    user = os.getenv("PGUSER", "learnrag")
    password = os.getenv("PGPASS", "learnrag")
    return create_engine(f"postgresql://{user}:{password}@localhost/img_project")


Base = declarative_base()


class ImageData(Base):
    __tablename__ = "image_data"
    __table_args__ = {"postgresql_tablespace": "alternate"}
    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String, unique=True, nullable=False)
    overall_embedding = Column(Vector(128), nullable=False)
    image_type = Column(String, nullable=False)


class Person(Base):
    __tablename__ = "person"
    __table_args__ = {"postgresql_tablespace": "alternate"}
    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)


class FaceData(Base):
    __tablename__ = "face_data"
    __table_args__ = {"postgresql_tablespace": "alternate"}
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("image_data.id"), nullable=False)
    face_embedding = Column(Vector(128), nullable=False)
    face_box = Column(ARRAY(Integer), nullable=False)
    face_tag_id = Column(Integer, ForeignKey("person.id"), nullable=True)


# Example engine and session creation
engine = get_engine()
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
