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
from functools import lru_cache
import os
import math

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
    {"length": 25,"parent": 7, "ratio": 10, "lat": 0.75 / 3600, "lon": 1.125 / 3600}
]

class MeshCodeUtility:

    @staticmethod
    def get_meshcode(lat, lon, m):
        if m in ["80km", 1]:
            return MeshCodeUtility._handle_80km_mesh(lat, lon)
        elif m in ["10km", 2]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 1, 5, 7.5)
        elif m in ["1km", 3]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 2, 30, 45)
        elif m in ["500m", 4]:
            return MeshCodeUtility._handle_special_mesh(lat, lon, 3, 15, 22.5)
        elif m in ["250m", 5]:
            return MeshCodeUtility._handle_special_mesh(lat, lon, 4, 7.5, 11.25)
        elif m in ["125m", 6]:
            return MeshCodeUtility._handle_special_mesh(lat, lon, 5, 3.75, 5.625)
        elif m in ["100m", 7]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 3, 3, 4.5)
        elif m in ["50m", 8]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 7, 1.5, 2.25)
        elif m in ["25m", 9]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 8, 0.75, 1.125)
        else:
            raise ValueError('Invalid mesh degree')

    @staticmethod
    def _handle_80km_mesh(lat, lon):
        lat_base, dest_lat = divmod(lat * 60, 40)
        lat_base = int(lat_base)
        lon_base = int(lon - 100)
        dest_lon = lon - 100 - lon_base
        return {
            "mesh_code": f"{int(lat_base):02d}{int(lon_base):02d}",
            "lat": dest_lat,
            "lon": dest_lon
        }

    @staticmethod
    def _handle_interval_mesh(lat, lon, parent_degree, lat_interval, lon_interval):
        base_data = MeshCodeUtility.get_meshcode(lat, lon, parent_degree)
        if parent_degree == 1:
            base_data["lon"] *= 60
            
        if parent_degree == 2:
            base_data["lat"] *= 60
            base_data["lon"] *= 60

        left_operator, dest_lat = divmod(
            base_data["lat"], lat_interval)
        right_operator, dest_lon = divmod(
            base_data["lon"], lon_interval)
        return {
            "mesh_code": f"{base_data['mesh_code']}{int(left_operator):01d}{int(right_operator):01d}",
            "lat": dest_lat,
            "lon": dest_lon
        }

    @staticmethod
    def _handle_special_mesh(lat, lon, parent_degree, lat_interval, lon_interval):
        base_data = MeshCodeUtility.get_meshcode(lat, lon, parent_degree)
        left_index, dest_lat = divmod(base_data["lat"], lat_interval)
        right_index, dest_lon = divmod(
            base_data["lon"], lon_interval)
        operator = int(2 * left_index + right_index) + 1
        return {
            "mesh_code": f"{base_data['mesh_code']}{int(operator):01d}",
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
        coords_data["latlons"] = [(lat, lon)
                                  for lon, lat in coords_data["geometry"][0][:-1]]
        return coords_data

    # Remaining methods would be similar as provided.

    @staticmethod
    def get_mesh_latlon(meshcode, m):
        if m == 1:
            lat = int(meshcode[0:2]) / 1.5
            lon = int(meshcode[2:4]) + 100
            return lat, lon
        if m in [2, 3, 7, 8, 9]:
            lat, lon = MeshCodeUtility.get_mesh_latlon(
                meshcode[:-2], MESH_INFOS[m]["parent"])
            lat += int(meshcode[-2]) * MESH_INFOS[m]["lat"]
            lon += int(meshcode[-1]) * MESH_INFOS[m]["lon"]
            return lat, lon
        elif m in [4, 5, 6]:
            lat, lon = MeshCodeUtility.get_mesh_latlon(
                meshcode[:-1], MESH_INFOS[m]["parent"])
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

    
# --------------------------------------------------------------------------------
# The following code is adapted from:
# https://github.com/MIERUNE/japan-mesh-tool/blob/master/src/japanmesh
# Copyright belongs to the original authors and MIERUNE.
# --------------------------------------------------------------------------------

# Range for generating mesh codes
MINIMUM_LON = 122.00
MAXIMUM_LON = 154.00
MINIMUM_LAT = 20.00
MAXIMUM_LAT = 46.00

# Define mesh size in longitude and latitude for each mesh number: (x, y)
FIRST_MESH_SIZE = (1, 2 / 3)

@lru_cache(maxsize=None)
def get_meshsize(meshnum: int) -> [float, float]:
    if meshnum == 1:
        return FIRST_MESH_SIZE

    meshinfo = MESH_INFOS[meshnum]
    parent_meshsize = get_meshsize((meshinfo["parent"]))
    meshsize = [
        parent_meshsize[0] / meshinfo["ratio"],
        parent_meshsize[1] / meshinfo["ratio"],
    ]
    return meshsize

def get_start_offset(meshnum: int, lonlat: list) -> tuple:
    """Calculate the number of meshes to skip from the origin (bottom-left) 
    based on the mesh level and longitude-latitude.
    
    Args:
        meshnum (int): Mesh level.
        lonlat ([float, float]): [Longitude, Latitude].

    Returns:
        tuple: (Offset in the x direction, Offset in the y direction).
    """
    x_mesh_dist, y_mesh_dist = get_meshsize(meshnum)

    x_offset = 0
    while lonlat[0] >= MINIMUM_LON + x_mesh_dist * (x_offset + 1):
        x_offset += 1

    y_offset = 0
    while lonlat[1] >= MINIMUM_LAT + y_mesh_dist * (y_offset + 1):
        y_offset += 1

    return x_offset, y_offset


def get_end_offset(meshnum: int, lonlat: list) -> tuple:
    """Calculate the number of meshes to skip from the endpoint (top-right) 
    based on the mesh level and longitude-latitude.
    
    Args:
        meshnum (int): Mesh level.
        lonlat ([float, float]): [Longitude, Latitude].

    Returns:
        tuple: (Offset in the x direction, Offset in the y direction).
    """
    x_mesh_dist, y_mesh_dist = get_meshsize(meshnum)

    x_offset = 0
    while lonlat[0] <= MAXIMUM_LON - x_mesh_dist * (x_offset + 1):
        x_offset += 1

    y_offset = 0
    while lonlat[1] <= MAXIMUM_LAT - y_mesh_dist * (y_offset + 1):
        y_offset += 1

    return x_offset, y_offset

@lru_cache(maxsize=None)
def get_meshcode(meshnum: int, x: int, y: int) -> str:
    # For secondary mesh codes and beyond, append two digits per code depending on the mesh level.
    # 1-3: Standard Area Mesh
    # 4-6: Divided Area Mesh
    # 7 and above: Others
    ratio = MESH_INFOS[meshnum]["ratio"]
    parent = MESH_INFOS[meshnum]["parent"]
    if meshnum == 1:
        # First-level code: Integer value of latitude multiplied by 1.5 + last two digits of longitude's integer part
        return ""
    elif meshnum == 4 or meshnum == 5 or meshnum == 6:
        return get_meshcode(parent, math.floor(x / ratio), math.floor(y / ratio)) + str((y % ratio) * 2 + (x % ratio) + 1)
    else:
        return get_meshcode(parent, math.floor(x / ratio), math.floor(y / ratio)) + str(y % ratio) + str(x % ratio)

@ lru_cache(maxsize=None)
def get_mesh_vertex(x: int, x_size: float, y: int, y_size: float) -> (float, float):
    return MINIMUM_LON + x * x_size, MINIMUM_LAT + y * y_size
    
@lru_cache(maxsize=None)
def get_base_cellcount(meshnum: int) -> (int, int, int):
    unitcount = math.prod([info["ratio"] for idx, info in enumerate(MESH_INFOS) if 0 < idx <= meshnum])

    if not MINIMUM_LON.is_integer():
        raise Exception(f'Unexpected MINIMUM_LON: {MINIMUM_LON}')
    if not MINIMUM_LAT.is_integer() or not (math.floor(MINIMUM_LAT) * 1.5).is_integer():
        raise Exception(f'Unexpected MINIMUM_LAT: {MINIMUM_LAT}')
    x = (math.floor(MINIMUM_LON) - 100) * unitcount
    y = math.floor(math.floor(MINIMUM_LAT) * 1.5) * unitcount

    return x, y, unitcount    

def get_mesh(meshnum: int, x: int, y: int) -> dict:
    """Return the mesh geometry and mesh code based on mesh level and mesh address.
    
    Args:
        meshnum (int): Mesh level.
        x (int): Mesh address counted from the origin to the right.
        y (int): Mesh address counted from the origin upwards.

    Returns:
        dict: {"geometry":<Mesh Geometry>, "code":<Mesh Code>}.
    """
    x_size, y_size = get_meshsize(meshnum)
    left_lon, bottom_lat = get_mesh_vertex(x, x_size, y, y_size)
    right_lon, top_lat = get_mesh_vertex(x + 1, x_size, y + 1, y_size)

#     base_x, base_y, unitcount = get_base_cellcount(meshnum)
#     x_1st = (x + base_x) // unitcount
#     y_1st = (y + base_y) // unitcount
    x_1st = str(int(left_lon))[1:]
    y_1st = str(int(bottom_lat * 1.5))
    
    code = y_1st + x_1st + get_meshcode(meshnum, x, y)

    return {
        "geometry": [[
            [left_lon, bottom_lat],
            [left_lon, top_lat],
            [right_lon, top_lat],
            [right_lon, bottom_lat],
            [left_lon, bottom_lat]
        ]],
        "code": code
    }

def generate_meshes(meshnum: int, extent=None):
    """Return a list of all mesh information overlapping with a specified area.
    
    Args:
        meshnum (int): Mesh level.
        extent (list, optional): Specifying area with a list of longitude-latitude pairs.

    yield:
        dict: {"geometry":<Mesh Geometry>, "code":<Mesh Code>}.
    """
    
    # Calculate the number of meshes in both x and y directions
    x_size, y_size = get_meshsize(meshnum)
    x_mesh_count = math.ceil((MAXIMUM_LON - MINIMUM_LON) / x_size)
    y_mesh_count = math.ceil((MAXIMUM_LAT - MINIMUM_LAT) / y_size)
    # Calculate the number of meshes to skip, i.e., offset
    start_offset = [0, 0]
    end_offset = [0, 0]
    if extent:
        min_lon = min(extent[0][0], extent[1][0])
        min_lat = min(extent[0][1], extent[1][1])
        max_lon = max(extent[0][0], extent[1][0])
        max_lat = max(extent[0][1], extent[1][1])

        # Sort by [lower left longitude-latitude, upper right longitude-latitude]
        cleaned_extent = [
            [min_lon, min_lat],
            [max_lon, max_lat]
        ]

        start_offset = get_start_offset(meshnum, cleaned_extent[0])
        end_offset = get_end_offset(meshnum, cleaned_extent[1])
    for y in range(start_offset[1], y_mesh_count - end_offset[1]):
        for x in range(start_offset[0], x_mesh_count - end_offset[0]):
            yield get_mesh(meshnum, x, y)
            
            
            