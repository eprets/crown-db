from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import logging

def get_exif_data(image_path):
    """Извлекает EXIF-данные из изображения и возвращает словарь"""
    exif_dict = {}
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return exif_dict

        for tag_id, value in exif.items():
            tag_name = TAGS.get(tag_id, tag_id)
            # Обработка GPS
            if tag_name == "GPSInfo":
                gps_data = {}
                for gps_tag in value:
                    sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                    gps_data[sub_tag] = value[gps_tag]
                exif_dict["GPSInfo"] = gps_data
            else:
                exif_dict[tag_name] = value

        return exif_dict
    except Exception as e:
        logging.warning(f"Не удалось прочитать EXIF из {image_path}: {e}")
        return {}

def get_gps_coordinates(exif_data):
    """Извлекает широту и долготу из EXIF (если есть)"""
    gps = exif_data.get("GPSInfo")
    if not gps:
        return None, None

    def convert_to_degrees(value):
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)

    lat = gps.get("GPSLatitude")
    lon = gps.get("GPSLongitude")
    lat_ref = gps.get("GPSLatitudeRef", "N")
    lon_ref = gps.get("GPSLongitudeRef", "E")

    if lat and lon:
        lat_deg = convert_to_degrees(lat)
        lon_deg = convert_to_degrees(lon)
        if lat_ref == "S":
            lat_deg = -lat_deg
        if lon_ref == "W":
            lon_deg = -lon_deg
        return lat_deg, lon_deg
    return None, None