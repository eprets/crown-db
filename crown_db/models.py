from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Image(Base):
    __tablename__ = "images"

    image_id = Column(String, primary_key=True)
    original_name = Column(String, nullable=True)
    path = Column(String, unique=True, nullable=False)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    flight_altitude = Column(Float, nullable=True)
    timestamp = Column(String, nullable=True)
    camera_model = Column(String, nullable=True)
    created_at = Column(String, default=lambda: datetime.now().isoformat())

    annotations = relationship("Annotation", back_populates="image")
    observations = relationship("Observation", back_populates="image")


class Tree(Base):
    __tablename__ = "trees"

    tree_id = Column(String, primary_key=True)
    tree_type = Column(String, nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    height_est = Column(Float, nullable=True)
    created_at = Column(String, default=lambda: datetime.now().isoformat())

    annotations = relationship("Annotation", back_populates="tree")
    observations = relationship("Observation", back_populates="tree")
    levels = relationship("Level", back_populates="tree")


class Annotation(Base):
    __tablename__ = "annotations"

    annotation_id = Column(String, primary_key=True)
    image_id = Column(String, ForeignKey("images.image_id"), nullable=False)
    tree_id = Column(String, ForeignKey("trees.tree_id"), nullable=False)
    tree_type = Column(String, nullable=True)

    x0 = Column(Float, nullable=False)
    y0 = Column(Float, nullable=False)
    a = Column(Float, nullable=False)
    b = Column(Float, nullable=False)
    theta = Column(Float, nullable=False)

    quality = Column(Float, nullable=True)
    created_at = Column(String, default=lambda: datetime.now().isoformat())

    __table_args__ = (
        UniqueConstraint("image_id", "tree_id", name="uq_annotation_image_tree"),
    )

    image = relationship("Image", back_populates="annotations")
    tree = relationship("Tree", back_populates="annotations")
    observation = relationship("Observation", back_populates="annotation", uselist=False)


class Observation(Base):
    __tablename__ = "observations"

    obs_id = Column(String, primary_key=True)
    annotation_id = Column(String, ForeignKey("annotations.annotation_id"), nullable=False, unique=True)
    image_id = Column(String, ForeignKey("images.image_id"), nullable=False)
    tree_id = Column(String, ForeignKey("trees.tree_id"), nullable=False)

    roi_raw_path = Column(String, nullable=False)
    obs_height = Column(Float, nullable=True)
    features_json = Column(Text, nullable=True)

    created_at = Column(String, default=lambda: datetime.now().isoformat())

    annotation = relationship("Annotation", back_populates="observation")
    image = relationship("Image", back_populates="observations")
    tree = relationship("Tree", back_populates="observations")
    level = relationship("Level", back_populates="observation", uselist=False)


class Level(Base):
    __tablename__ = "levels"

    level_id = Column(String, primary_key=True)
    tree_id = Column(String, ForeignKey("trees.tree_id"), nullable=False)
    h_level = Column(Integer, nullable=False)
    source_obs_id = Column(String, ForeignKey("observations.obs_id"), nullable=True)

    data_type = Column(String, nullable=False, default="REAL")   # REAL / SYNTH
    mapping_error = Column(Float, nullable=True)

    roi_norm_path = Column(String, nullable=True)
    roi_mask_norm_path = Column(String, nullable=True)
    ellipse_norm_json = Column(Text, nullable=True)
    features_json = Column(Text, nullable=True)

    synth_method = Column(String, nullable=True)     # linear_blend / pix2pix
    synth_src_h = Column(Integer, nullable=True)

    created_at = Column(String, default=lambda: datetime.now().isoformat())

    __table_args__ = (
        UniqueConstraint("tree_id", "h_level", name="uq_tree_level"),
    )

    tree = relationship("Tree", back_populates="levels")
    observation = relationship("Observation", back_populates="level")


class Mask(Base):
    __tablename__ = "masks"

    mask_id = Column(String, primary_key=True)
    annotation_id = Column(String, ForeignKey("annotations.annotation_id"), nullable=False, unique=True)
    mask_path = Column(String, nullable=False)
    created_at = Column(String, default=lambda: datetime.now().isoformat())

    annotation = relationship("Annotation")