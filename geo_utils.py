import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pandas import plotting
import jismesh.utils as ju
import collections
from math import radians, sin, cos, sqrt, asin
from shapely.geometry import Point, Polygon

def dist_on_sphere(pos0, pos1):
    '''
    distance based on Haversine formula
    pos: [latitude, longitude]
    output: distance(km)
    Ref: https://en.wikipedia.org/wiki/Haversine_formula
    '''
    radius = 6378.137
    latang1, lngang1 = pos0
    latang2, lngang2 = pos1
    phi1, phi2 = radians(latang1), radians(latang2)
    lam1, lam2 = radians(lngang1), radians(lngang2)
    term1 = sin((phi2 - phi1) / 2.0) ** 2
    term2 = sin((lam2 - lam1) / 2.0) ** 2
    term2 = cos(phi1) * cos(phi2) * term2
    wrk = sqrt(term1 + term2)
    wrk = 2.0 * radius * asin(wrk)
    return wrk

MESH_INFOS = [
    None,
    {"length": 80000, "parent": 1, "ratio": 1, "lat": 1 / 1.5, "lon": 1},
    {"length": 10000, "parent": 1, "ratio": 8, "lat": 5 / 60, "lon": 7.5 / 60},
    {"length": 1000, "parent": 2, "ratio": 10, "lat": 30 / 3600, "lon": 45 / 3600},
    {"length": 500, "parent": 3, "ratio": 2, "lat": 15 / 3600, "lon": 22.5 / 3600},
    {"length": 250,"parent": 4, "ratio": 2, "lat": 7.5 / 3600, "lon": 11.25 / 3600},
    {"length": 125,"parent": 5, "ratio": 2, "lat": 3.75 / 3600, "lon": 5.625 / 3600},
    {"length": 100,"parent": 3, "ratio": 10, "lat": 3 / 3600, "lon": 4.5 / 3600},
    {"length": 50,"parent": 7, "ratio": 2, "lat": 1.5 / 3600, "lon": 2.25 / 3600},
    {"length": 25,"parent": 8, "ratio": 2, "lat": 0.75 / 3600, "lon": 1.125 / 3600}
]


class MeshCodeUtility:

    @staticmethod
    def get_meshcode(lat, lon, m):
        if m in ["80km", 1]:
            return MeshCodeUtility._handle_80km_mesh(lat, lon)
        elif m in ["10km", 2]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 1, 5, 7.5)
        elif m in ["1km", 3]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 2, 0.5, 0.75)
        elif m in ["500m", 4]:
            return MeshCodeUtility._handle_special_mesh(lat, lon, 3, 0.25, 0.375)
        elif m in ["250m", 5]:
            return MeshCodeUtility._handle_special_mesh(lat, lon, 4, 0.125, 0.1875)
        elif m in ["125m", 6]:
            return MeshCodeUtility._handle_special_mesh(lat, lon, 5, 0.0625, 0.09375)
        elif m in ["100m", 7]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 3, 0.05, 0.075)
        elif m in ["50m", 8]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 7, 0.025, 0.0375)
        elif m in ["25m", 9]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 8, 0.0125, 0.01875)
        else:
            raise ValueError('Invalid mesh degree')

    @staticmethod
    def _handle_80km_mesh(lat, lon):
        lat_base, dest_lat = divmod(lat * 60, 40)
        lon_base = int(lon - 100)
        dest_lon = lon - 100 - lon_base
        return {
            "mesh_code": f"{lat_base:02d}{lon_base:02d}",
            "lat": dest_lat,
            "lon": dest_lon
        }

    @staticmethod
    def _handle_interval_mesh(lat, lon, parent_degree, lat_interval, lon_interval):
        base_data = MeshCodeUtility.get_meshcode(lat, lon, parent_degree)
        left_operator, dest_lat = divmod(base_data["lat"] * 60, lat_interval * 60)
        right_operator, dest_lon = divmod(base_data["lon"] * 60, lon_interval * 60)
        return {
            "mesh_code": f"{base_data['mesh_code']}{left_operator:01d}{right_operator:01d}",
            "lat": dest_lat,
            "lon": dest_lon
        }

    @staticmethod
    def _handle_special_mesh(lat, lon, parent_degree, lat_interval, lon_interval):
        base_data = MeshCodeUtility.get_meshcode(lat, lon, parent_degree)
        left_index, dest_lat = divmod(base_data["lat"] * 60, lat_interval * 60)
        right_index, dest_lon = divmod(base_data["lon"] * 60, lon_interval * 60)
        operator = int(2 * left_index + right_index) + 1
        return {
            "mesh_code": f"{base_data['mesh_code']}{operator:01d}",
            "lat": dest_lat,
            "lon": dest_lon
        }

    @staticmethod
    def get_mesh_coords(meshcode, m):
        coords_data = {
            "mesh_code": meshcode,
            "degree": m,
            "south_west_latlon": MeshCodeUtility.get_mesh_latlon(meshcode, m),
            "center_latlon": MeshCodeUtility.get_mesh_center_latlon(meshcode, m),
            "geometry": MeshCodeUtility.get_mesh_geometry(meshcode, m),
        }
        coords_data["latlons"] = [(lat, lon) for lon, lat in coords_data["geometry"][0][:-1]]
        return coords_data

    # Remaining methods would be similar as provided.

    @staticmethod
    def get_mesh_latlon(meshcode, m):
        if m == 1:
            lat = int(meshcode[0:2]) / 1.5
            lon = int(meshcode[2:4]) + 100
            return lat, lon
        if m in [2, 3, 7, 8, 9]:
            lat, lon = MeshCodeUtility.get_mesh_latlon(meshcode[:-2], MESH_INFOS[m]["parent"])
            lat += int(meshcode[-2]) * MESH_INFOS[m]["lat"]
            lon += int(meshcode[-1]) * MESH_INFOS[m]["lon"]
            return lat, lon
        elif m in [4, 5, 6]:
            lat, lon = MeshCodeUtility.get_mesh_latlon(meshcode[:-1], MESH_INFOS[m]["parent"])
            lat += ((int(meshcode[-1]) - 1) // 2) * MESH_INFOS[m]["lat"]
            lon += ((int(meshcode[-1]) - 1) % 2) * MESH_INFOS[m]["lon"]
            return lat, lon
        else:
            raise ValueError('Invalid mesh degree')
   
    @staticmethod
    def get_mesh_center_latlon(meshcode, m):
        lat, lon = MeshCodeUtility.get_mesh_latlon(meshcode, m)
        lat_center = lat + MESH_INFOS[m]["lat"]/2
        lon_center = lon + MESH_INFOS[m]["lon"]/2
        return lat_center, lon_center
    
    @staticmethod
    def get_mesh_geometry(meshcode, m):
        lat, lon = MeshCodeUtility.get_mesh_latlon(meshcode, m)
        return [[[lon, lat],
                 [lon, lat + MESH_INFOS[m]["lat"]],
                 [lon + MESH_INFOS[m]["lon"], lat + MESH_INFOS[m]["lat"]],
                 [lon + MESH_INFOS[m]["lon"], lat],
                 [lon, lat]]]