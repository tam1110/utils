import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pandas import plotting
import folium
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, asin
from shapely.geometry import Point, Polygon
import collections

def plot_PCA(vec: "ndarray(shape=(N,M))", pca: "削減結果", fst: int, scd: int, matched_condition_users: "list"):
    """
    PCAで得られたデータから，指定した主成分のグラフを表示
    """
    # 主成分と第二主成分でプロット
    plt.figure(figsize=(8, 8))

    for x, y, name in zip(vec[:, fst], vec[:, scd], [i for i in range(vec.shape[0])]):
        plt.text(x, y, name, size=10)

    # color
    color = []
    for i in range(vec.shape[0]):
        if i in matched_condition_users:
            color.append("red")
        else:
            color.append("blue")

    plt.scatter(vec[:, fst], vec[:, scd], alpha=0.8, c=color)
    plt.grid()
    plt.xlabel("PC" + str(fst) +
               "(" + str(pca.explained_variance_ratio_[fst]) + ")")
    plt.ylabel("PC" + str(scd) +
               "(" + str(pca.explained_variance_ratio_[scd]) + ")")
    plt.show()


def plot_dim_reduced_vec(vec: "ndarray", for_text=None, specific=None, color=None, dim=2):
    """
    (次元削減後の)2次元ベクトルをプロット
    vec : shape = (N,2)
    for_text : プロット時の表示されるプロットの名前
    specific : 特定のベクトルのみを表示
    color : クラスタリング後など，プロットを色分けしたいときに使用(list)
    """
    # 使用するベクトルを選択
    valid_vec = vec
    if specific != None:
        valid_vec = []

        for user_num in specific:
            valid_vec.append(vec[user_num].tolist())

        valid_vec = np.array(valid_vec)

    fig = plt.figure(figsize=(10, 10))
    ax = None

    if dim == 2:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection="3d")

    # プロットのテキストの割り当て
    if for_text != None:
        if dim == 2:
            for x, y, name in zip(valid_vec[:, 0], valid_vec[:, 1], for_text):
                ax.text(x, y, name, size=10)
        else:
            for x, y, z, name in zip(valid_vec[:, 0], valid_vec[:, 1], valid_vec[:, 2], for_text):
                ax.text(x, y, z, name, size=2)

    # プロットの色の割り当て
    c = color
    if color == None:
        c = [0 for i in range(valid_vec.shape[0])]
        if specific != None:
            c = []

            for i in range(valid_vec.shape[0]):
                if i in specific:
                    c.append(2)
                else:
                    c.append(0)

    if dim == 2:
        ax.scatter(valid_vec[:, 0], valid_vec[:, 1], alpha=0.8, c=c)
    else:
        ax.scatter(valid_vec[:, 0], valid_vec[:, 1],
                   valid_vec[:, 2], alpha=0.8, c=c)

    ax.grid()
    # ax.show()
    

def SP_vis(df: "dataframe", unum: "int", another_data=None) -> "folium map":
    """
    滞在場所の可視化
    df : dataframe
    unum : 表示したいユーザのID
    another_data : 追加して表示したい情報[name, lat, lon]
    """
    m_lat = df[unum]["center_lat"]
    m_lon = df[unum]["center_lon"]
    m_hour = df[unum]["hour"]
    m_day_of_week = df[unum]["day_of_week"]
    m_label = df[unum]["label"]
    m_elapsed_time = df[unum]["elapsed_time"]

    m = folium.Map(location=[34.942367375, 137.15507335], zoom_start=12)

    for i in range(len(m_lat)):
        popup = str(m_label[i]) + ";" + str(m_lat[i]) + "," + str(m_lon[i]) + "\n" + ";" + str(m_day_of_week[i]) + ";" \
            + str(m_hour[i]) + ";" + str(m_elapsed_time[i])
        folium.Marker([m_lat[i], m_lon[i]], popup=popup).add_to(m)

    if another_data != None:
        for i in range(len(another_data)):
            popup = str(another_data[i][2]) + "," + \
                str(another_data[i][0]) + "," + str(another_data[i][1])
            folium.Marker([another_data[i][0], another_data[i][1]],
                          popup=popup, icon=folium.Icon(color='red')).add_to(m)

    return m

def SP_vis2(df: "dataframe", another_data=None) -> "folium map object":
    """
    滞在場所の可視化
    df : 滞在場所をマップ上に表示したいユーザのdataframe
    another_data : 追加情報のマップ上へのプロット(e.g. superMarketの住所など)
    """
    m_lat = df["center_lat"]
    m_lon = df["center_lon"]
    m_hour = df["hour"]
    m_day_of_week = df["day_of_week"]
    m_label = df["label"]
    m_elapsed_time = df["elapsed_time"]

    m = folium.Map(location=[34.942367375, 137.15507335], zoom_start=12)

    for i in range(len(m_lat)):
        popup = str(m_label[i]) + ";" + str(m_lat[i]) + "," + str(m_lon[i]) + "\n" + ";" \
            + str(m_day_of_week[i]) + ";" + \
            str(m_hour[i]) + ";" + str(m_elapsed_time[i])
        folium.Marker([m_lat[i], m_lon[i]], popup=popup).add_to(m)

    if another_data != None:
        for i in range(len(another_data)):
            print(str(another_data[i][2]) + "," + str(another_data[i][3]))
            popup = str(another_data[i][2]) + "," + str(another_data[i][0]) + "," + str(
                another_data[i][1])  # + "," + str(another_data[i][3]) #文字化けする
            folium.Marker([another_data[i][0], another_data[i][1]],
                          popup=popup, icon=folium.Icon(color='red')).add_to(m)

    return m

def draw_bar_for_clustered_mesh(clusterNum,
                                allUsersDataframe,
                                meshClusterDic,
                                numOfDays,
                                eachElementNumInCluster,
                                holiday,
                                validUsers=None,
                                validMesh=None,
                                validCluster=None,
                                figSize=(13, 15),
                                titleSize=10,
                                doAxisDivide=True,
                                maxBottom=None,
                                fileName=None,
                                graphLayout=1):
    """
    clusterNum : クラスタ数
    allUsersDataframe : 全ユーザのデータフレームが入ったリスト
    meshClusterDic : 各メッシュがどのクラスタに所属しているかを記録した辞書
    numOfDays : [平日の日数, 休日の日数]
    eachElementNumInCluster : 各クラスタに所属しているメッシュの個数を記録した辞書
    holiday : 祝日の日付を格納したリスト
    validUsers : 有効なユーザのリスト
    validMesh : 積層グラフに反映されるのに有効なメッシュのリスト
    validClsuter : 表示したいクラスタのリスト
    figSize : グラフの大きさを示すタプル
    titileSize : グラフのタイトルの大きさ
    doAxisDivide : グラフの縦軸を"eachElementNumInCluster"に応じて正規化するかどうかを示すフラグ
    maxBottom : グラフの縦軸の最大値
    fileName : グラフを保存する場合はその名前を入力
    graphLayout : 積層グラフの（平日，休日）のペアを何列作成するか
    """
    dayofweek_dic = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4,
                     "Saturday": 5, "Sunday": 6}
    stay_time_class_num = 14  # 〜29分，30〜59分，60〜120分，．．． x 平日or休日
    time_split_num = 48  # 24時間を何時間区切りにするか（48=30分区切り）
    bar_graph = np.array([[[float(0) for _ in range(time_split_num)]
                         for _ in range(clusterNum)] for _ in range(stay_time_class_num)])

    ### ここから matplotlib へ渡すデータの作成 ###

    for uname, df in enumerate(allUsersDataframe):
        if validUsers != None and uname not in validUsers:
            continue

        mesh_ids = df["mesh_id"]
        start_times = df["start_time"]
        elapsed_times = df["elapsed_time"]
        day_of_weeks = df["day_of_week"]
        dates = df["date"]

        for mesh_id, start_time, elapsed_time_h, day_of_week, date in zip(mesh_ids, start_times, elapsed_times, day_of_weeks, dates):
            if (validMesh != None) and (mesh_id not in validMesh):
                continue

            if mesh_id == -1:
                continue

            which_cluster = meshClusterDic[mesh_id]  # メッシュが所属するクラスタ情報を取得

            # time zone
            time_zone_hour, time_zone_minute = start_time.split(":")
            tmp = (int(time_zone_hour) * 60 + int(time_zone_minute)) / 30
            time_zone_num = int(tmp)

            # elapsed time
            elapsed_time_s = elapsed_time_h * 60
            tmp = (elapsed_time_s) / 30
            itr = round(tmp, 0)

            # day-of-week
            dayofweek_num = dayofweek_dic[day_of_week]

            if datetime.strptime(date, "%Y/%m/%d") in holiday:  # 祝日判定
                dayofweek_num = 7

            # 24時間を30分区切りにした48次元の内，どこをインクリメントするかという情報を"idx"に格納
            idx = []
            idx_for_friday = []
            idx_for_sunday = []
            times_zero = 0

            for j in range(int(itr) + 1):
                tmp = time_zone_num + j

                if dayofweek_num != 3 and dayofweek_num != 6:
                    idx.append(tmp % 48)
                elif dayofweek_num == 3:  # 金曜から土曜への対応
                    if tmp // 48 == 0:
                        idx.append(tmp % 48)
                    else:
                        idx_for_friday.append(tmp % 48)
                elif dayofweek_num == 6:  # 日曜から月曜への対応
                    if tmp // 48 == 0:
                        idx.append(tmp % 48)
                    else:
                        idx_for_sunday.append(tmp % 48)

            for j in idx:
                if 0 <= dayofweek_num and dayofweek_num <= 4:
                    if elapsed_time_s < 30:
                        bar_graph[0][which_cluster][j] += 1
                    elif 30 <= elapsed_time_s and elapsed_time_s < 60:
                        bar_graph[1][which_cluster][j] += 1
                    elif 60 <= elapsed_time_s and elapsed_time_s < 120:
                        bar_graph[2][which_cluster][j] += 1
                    elif 120 <= elapsed_time_s and elapsed_time_s < 240:
                        bar_graph[3][which_cluster][j] += 1
                    elif 240 <= elapsed_time_s and elapsed_time_s < 360:
                        bar_graph[4][which_cluster][j] += 1
                    elif 360 <= elapsed_time_s and elapsed_time_s < 720:
                        bar_graph[5][which_cluster][j] += 1
                    else:
                        bar_graph[6][which_cluster][j] += 1
                elif 5 <= dayofweek_num and dayofweek_num <= 7:
                    if elapsed_time_s < 30:
                        bar_graph[7][which_cluster][j] += 1
                    elif 30 <= elapsed_time_s and elapsed_time_s < 60:
                        bar_graph[8][which_cluster][j] += 1
                    elif 60 <= elapsed_time_s and elapsed_time_s < 120:
                        bar_graph[9][which_cluster][j] += 1
                    elif 120 <= elapsed_time_s and elapsed_time_s < 240:
                        bar_graph[10][which_cluster][j] += 1
                    elif 240 <= elapsed_time_s and elapsed_time_s < 360:
                        bar_graph[11][which_cluster][j] += 1
                    elif 360 <= elapsed_time_s and elapsed_time_s < 720:
                        bar_graph[12][which_cluster][j] += 1
                    else:
                        bar_graph[13][which_cluster][j] += 1

            for j in idx_for_friday:
                if elapsed_time_s < 30:
                    bar_graph[7][which_cluster][j] += 1
                elif 30 <= elapsed_time_s and elapsed_time_s < 60:
                    bar_graph[8][which_cluster][j] += 1
                elif 60 <= elapsed_time_s and elapsed_time_s < 120:
                    bar_graph[9][which_cluster][j] += 1
                elif 120 <= elapsed_time_s and elapsed_time_s < 240:
                    bar_graph[10][which_cluster][j] += 1
                elif 240 <= elapsed_time_s and elapsed_time_s < 360:
                    bar_graph[11][which_cluster][j] += 1
                elif 360 <= elapsed_time_s and elapsed_time_s < 720:
                    bar_graph[12][which_cluster][j] += 1
                else:
                    bar_graph[13][which_cluster][j] += 1

            for j in idx_for_sunday:
                if elapsed_time_s < 30:
                    bar_graph[0][which_cluster][j] += 1
                elif 30 <= elapsed_time_s and elapsed_time_s < 60:
                    bar_graph[1][which_cluster][j] += 1
                elif 60 <= elapsed_time_s and elapsed_time_s < 120:
                    bar_graph[2][which_cluster][j] += 1
                elif 120 <= elapsed_time_s and elapsed_time_s < 240:
                    bar_graph[3][which_cluster][j] += 1
                elif 240 <= elapsed_time_s and elapsed_time_s < 360:
                    bar_graph[4][which_cluster][j] += 1
                elif 360 <= elapsed_time_s and elapsed_time_s < 720:
                    bar_graph[5][which_cluster][j] += 1
                else:
                    bar_graph[6][which_cluster][j] += 1

    ### ここから matplolib の操作 ###

    x_axis = np.array([i for i in range(48)])
    x_axis_for_show = [int(i / 2) if i % 2 == 0 else "" for i in range(48)]
    plt.figure(figsize=figSize)

    bottom_data = []
    break_between_weekday_and_weekend = int(stay_time_class_num / 2)

    graph_num = clusterNum
    if validCluster != None:
        graph_num = len(validCluster)

    itr = 0  # グラフを挿入する位置を決定するためのイテレーション

    for which_cluster in range(clusterNum):
        if validCluster != None and which_cluster not in validCluster:
            continue

        for j in range(stay_time_class_num):
            tmp = bar_graph[j][which_cluster]

            if doAxisDivide and eachElementNumInCluster[which_cluster] != 0:
                if j < break_between_weekday_and_weekend:
                    tmp = (
                        map(lambda x: x / (numOfDays[0] * eachElementNumInCluster[which_cluster]), tmp))
                else:
                    tmp = (
                        map(lambda x: x / (numOfDays[1] * eachElementNumInCluster[which_cluster]), tmp))
            elif not doAxisDivide:
                if j < break_between_weekday_and_weekend:
                    tmp = (map(lambda x: x / numOfDays[0], tmp))
                else:
                    tmp = (map(lambda x: x / numOfDays[1], tmp))
            else:
                pass

            bottom_data.append(list(tmp))

        max_bottom = 0
        if maxBottom == None:
            tmp_sum_weekday = np.sum(
                np.array(bottom_data[:break_between_weekday_and_weekend]), axis=0)
            tmp_sum_weekend = np.sum(
                np.array(bottom_data[break_between_weekday_and_weekend:]), axis=0)
            max_bottom_weekday = max(tmp_sum_weekday)
            max_bottom_weekend = max(tmp_sum_weekend)

            if max_bottom_weekday > max_bottom_weekend:
                max_bottom = max_bottom_weekday
            else:
                max_bottom = max_bottom_weekend
        else:
            max_bottom = maxBottom

        color_list = ["green", "orange", "blue",
                      "violet", "yellow", "tomato", "purple"]
        fontsize_x = 15
        fontsize_y = 20

        # weekday
        plt.subplot(graph_num // graphLayout + 1, graphLayout * 2, itr*2 + 1)
        plt.xticks(x_axis, x_axis_for_show, fontsize=fontsize_x)
        # plt.title("cluster " + str(which_cluster) + " (element count:" + str(eachElementNumInCluster[which_cluster]) + ")", fontsize=titleSize)
        plt.ylim(0, max_bottom)
        plt.yticks(fontsize=fontsize_y)
        plt.legend(("720~", "360~720", "240~360", "120~240",
                   "60~120", "30~60", "~30"), fontsize=8)
        bottom = np.array([0 for _ in range(48)], dtype=np.float)

        for j in range(break_between_weekday_and_weekend):
            plt.bar(x_axis, bottom_data[j], bottom=bottom, color=color_list[j])
            bottom += bottom_data[j]

        # weekend
        plt.subplot(graph_num // graphLayout + 1, graphLayout * 2, itr*2 + 2)
        plt.xticks(x_axis, x_axis_for_show, fontsize=fontsize_x)
        plt.ylim(0, max_bottom)
        plt.yticks(fontsize=fontsize_y)
        bottom_e = np.array([0 for _ in range(48)], dtype=np.float)

        for j in range(break_between_weekday_and_weekend):
            plt.bar(x_axis, bottom_data[j + 7],
                    bottom=bottom_e, color=color_list[j])
            bottom_e += bottom_data[j + 7]

        bottom_data = []  # リセット
        itr += 1

    #plt.rcParams["font.size"] = 9
    plt.tight_layout()

    if fileName != None:
        plt.savefig(fileName)
        plt.close()
    else:
        plt.show()

def showMesh_in_map(lat_list, lon_list,
                    zoom_lat_lon=None,
                    tiles=None,
                    zoom_level=None,
                    color=None,
                    cluster_data=None,
                    valid_mesh=None,
                    which_cluster_show=None):
    # 描写が遅い場合は，tiles='stamentoner'を使用
    args_zoom_lat_lon = [34.99966368, 137.12538141]
    if zoom_lat_lon != None:
        args_zoom_lat_lon = zoom_lat_lon

#     args_tiles = "Stamen Terrain"
#     if tiles != None:
#         args_tiles = tiles

    args_zoom_level = 13
    if zoom_level != None:
        args_zoom_level = zoom_level

    if tiles != None:
        m = folium.Map(args_zoom_lat_lon,
                       zoom_start=args_zoom_level, tiles=tiles)
    else:
        m = folium.Map(args_zoom_lat_lon, zoom_start=args_zoom_level)

    itr = 0

    for i in range(len(lat_list) - 1):
        lat1 = lat_list[i]
        lat2 = lat_list[i + 1]

        for j in range(len(lon_list) - 1):
            mesh_id = i * (len(lon_list) - 1) + j

            if (valid_mesh != None) and (mesh_id not in valid_mesh):
                itr += 1
                continue

            if (which_cluster_show != None) and (cluster_data[itr] not in which_cluster_show):
                itr += 1
                continue

            lon1 = lon_list[j]
            lon2 = lon_list[j + 1]
            polygon = [(lat1, lon1), (lat1, lon2), (lat2, lon2), (lat2, lon1)]
            popup = str(mesh_id) + ":" + str(lat1) + "," + str(lon1)

            if cluster_data != None:
                popup += ":" + str(cluster_data[itr])

            if color != None:
                folium.Polygon(locations=polygon, color=color[cluster_data[itr] % len(
                    color)], fill=True, fill_opacity=0.5, popup=popup).add_to(m)
            else:
                folium.Polygon(locations=polygon, fill=True,
                               fill_opacity=0, popup=popup).add_to(m)

            itr += 1

    return m

def draw_bar_for_clustered_mesh2(clusterNum,
                                 allUsersDataframe,
                                 meshClusterDic,
                                 numOfDays,
                                 eachElementNumInCluster,
                                 holiday,
                                 validUsers=None,
                                 validMesh=None,
                                 validCluster=None,
                                 figSize=(13, 15),
                                 titleSize=10,
                                 doAxisDivide=True,
                                 maxBottom=None,
                                 fileName=None,
                                 graphLayout=1):
    """
    clusterNum : クラスタ数
    allUsersDataframe : 全ユーザのデータフレームが入ったリスト
    meshClusterDic : 各メッシュがどのクラスタに所属しているかを記録した辞書
    numOfDays : [平日の日数, 休日の日数]
    eachElementNumInCluster : 各クラスタに所属しているメッシュの個数を記録した辞書
    holiday : 祝日の日付を格納したリスト
    validUsers : 有効なユーザのリスト
    validMesh : 積層グラフに反映されるのに有効なメッシュのリスト
    validClsuter : 表示したいクラスタのリスト
    figSize : グラフの大きさを示すタプル
    titileSize : グラフのタイトルの大きさ
    doAxisDivide : グラフの縦軸を"eachElementNumInCluster"に応じて正規化するかどうかを示すフラグ
    maxBottom : グラフの縦軸の最大値
    fileName : グラフを保存する場合はその名前を入力
    graphLayout : 積層グラフの（平日，休日）のペアを何列作成するか
    """
    dayofweek_dic = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4,
                     "Saturday": 5, "Sunday": 6}
    stay_time_class_num = 14  # 〜29分，30〜59分，60〜120分，．．． x 平日or休日
    time_split_num = 48  # 24時間を何時間区切りにするか（48=30分区切り）
    bar_graph = np.array([[[float(0) for _ in range(time_split_num)]
                         for _ in range(clusterNum)] for _ in range(stay_time_class_num)])

    ### ここから matplotlib へ渡すデータの作成 ###

    for uname, df in enumerate(allUsersDataframe):
        if validUsers != None and uname not in validUsers:
            continue

        mesh_ids = df["mesh_id"]
        start_times = df["start_dt_jpn"]
        elapsed_times = df["visiting_minutes"]
        # day_of_weeks = df["day_of_week"]
        # dates = df["date"]

        for mesh_id, start_time, elapsed_time_m in zip(mesh_ids, start_times, elapsed_times):
            if (validMesh != None) and (mesh_id not in validMesh):
                continue

            if mesh_id == -1:
                continue
            try:
                start_dt = datetime.strptime(
                    start_time, "%Y-%m-%d %H:%M:%S.%f")
            except:
                start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            # start_dt += timedelta(hours=9)
            start_hour = start_dt.hour
            start_minute = start_dt.minute
            date = start_dt.date()

            which_cluster = meshClusterDic[mesh_id]  # メッシュが所属するクラスタ情報を取得

            # time zone
            # time_zone_hour, time_zone_minute = start_time.split(":")
            tmp = (start_hour * 60 + start_minute) / 30
            time_zone_num = int(tmp)

            # elapsed time
            tmp = (elapsed_time_m) / 30
            itr = round(tmp, 0)

            # day-of-week
            day_of_week = start_dt.strftime('%A')
            dayofweek_num = dayofweek_dic[day_of_week]

            if date in holiday:  # 祝日判定
                dayofweek_num = 7

            # 24時間を30分区切りにした48次元の内，どこをインクリメントするかという情報を"idx"に格納
            idx = []
            idx_for_friday = []
            idx_for_sunday = []
            times_zero = 0

            for j in range(int(itr) + 1):
                tmp = time_zone_num + j

                if dayofweek_num != 3 and dayofweek_num != 6:
                    idx.append(tmp % 48)
                elif dayofweek_num == 3:  # 金曜から土曜への対応
                    if tmp // 48 == 0:
                        idx.append(tmp % 48)
                    else:
                        idx_for_friday.append(tmp % 48)
                elif dayofweek_num == 6:  # 日曜から月曜への対応
                    if tmp // 48 == 0:
                        idx.append(tmp % 48)
                    else:
                        idx_for_sunday.append(tmp % 48)

            for j in idx:
                if 0 <= dayofweek_num and dayofweek_num <= 4:
                    if elapsed_time_m < 30:
                        bar_graph[0][which_cluster][j] += 1
                    elif 30 <= elapsed_time_m and elapsed_time_m < 60:
                        bar_graph[1][which_cluster][j] += 1
                    elif 60 <= elapsed_time_m and elapsed_time_m < 120:
                        bar_graph[2][which_cluster][j] += 1
                    elif 120 <= elapsed_time_m and elapsed_time_m < 240:
                        bar_graph[3][which_cluster][j] += 1
                    elif 240 <= elapsed_time_m and elapsed_time_m < 360:
                        bar_graph[4][which_cluster][j] += 1
                    elif 360 <= elapsed_time_m and elapsed_time_m < 720:
                        bar_graph[5][which_cluster][j] += 1
                    else:
                        bar_graph[6][which_cluster][j] += 1
                elif 5 <= dayofweek_num and dayofweek_num <= 7:
                    if elapsed_time_m < 30:
                        bar_graph[7][which_cluster][j] += 1
                    elif 30 <= elapsed_time_m and elapsed_time_m < 60:
                        bar_graph[8][which_cluster][j] += 1
                    elif 60 <= elapsed_time_m and elapsed_time_m < 120:
                        bar_graph[9][which_cluster][j] += 1
                    elif 120 <= elapsed_time_m and elapsed_time_m < 240:
                        bar_graph[10][which_cluster][j] += 1
                    elif 240 <= elapsed_time_m and elapsed_time_m < 360:
                        bar_graph[11][which_cluster][j] += 1
                    elif 360 <= elapsed_time_m and elapsed_time_m < 720:
                        bar_graph[12][which_cluster][j] += 1
                    else:
                        bar_graph[13][which_cluster][j] += 1

            for j in idx_for_friday:
                if elapsed_time_m < 30:
                    bar_graph[7][which_cluster][j] += 1
                elif 30 <= elapsed_time_m and elapsed_time_m < 60:
                    bar_graph[8][which_cluster][j] += 1
                elif 60 <= elapsed_time_m and elapsed_time_m < 120:
                    bar_graph[9][which_cluster][j] += 1
                elif 120 <= elapsed_time_m and elapsed_time_m < 240:
                    bar_graph[10][which_cluster][j] += 1
                elif 240 <= elapsed_time_m and elapsed_time_m < 360:
                    bar_graph[11][which_cluster][j] += 1
                elif 360 <= elapsed_time_m and elapsed_time_m < 720:
                    bar_graph[12][which_cluster][j] += 1
                else:
                    bar_graph[13][which_cluster][j] += 1

            for j in idx_for_sunday:
                if elapsed_time_m < 30:
                    bar_graph[0][which_cluster][j] += 1
                elif 30 <= elapsed_time_m and elapsed_time_m < 60:
                    bar_graph[1][which_cluster][j] += 1
                elif 60 <= elapsed_time_m and elapsed_time_m < 120:
                    bar_graph[2][which_cluster][j] += 1
                elif 120 <= elapsed_time_m and elapsed_time_m < 240:
                    bar_graph[3][which_cluster][j] += 1
                elif 240 <= elapsed_time_m and elapsed_time_m < 360:
                    bar_graph[4][which_cluster][j] += 1
                elif 360 <= elapsed_time_m and elapsed_time_m < 720:
                    bar_graph[5][which_cluster][j] += 1
                else:
                    bar_graph[6][which_cluster][j] += 1

    ### ここから matplolib の操作 ###

    x_axis = np.array([i for i in range(48)])
    x_axis_for_show = [int(i / 2) if i % 2 == 0 else "" for i in range(48)]
    plt.figure(figsize=figSize)

    bottom_data = []
    break_between_weekday_and_weekend = int(stay_time_class_num / 2)

    graph_num = clusterNum
    if validCluster != None:
        graph_num = len(validCluster)

    itr = 0  # グラフを挿入する位置を決定するためのイテレーション

    for which_cluster in range(clusterNum):
        if validCluster != None and which_cluster not in validCluster:
            continue

        for j in range(stay_time_class_num):
            tmp = bar_graph[j][which_cluster]

            if doAxisDivide and eachElementNumInCluster[which_cluster] != 0:
                if j < break_between_weekday_and_weekend:
                    tmp = (
                        map(lambda x: x / (numOfDays[0] * eachElementNumInCluster[which_cluster]), tmp))
                else:
                    tmp = (
                        map(lambda x: x / (numOfDays[1] * eachElementNumInCluster[which_cluster]), tmp))
            elif not doAxisDivide:
                if j < break_between_weekday_and_weekend:
                    tmp = (map(lambda x: x / numOfDays[0], tmp))
                else:
                    tmp = (map(lambda x: x / numOfDays[1], tmp))
            else:
                pass

            bottom_data.append(list(tmp))

        max_bottom = 0
        if maxBottom == None:
            tmp_sum_weekday = np.sum(
                np.array(bottom_data[:break_between_weekday_and_weekend]), axis=0)
            tmp_sum_weekend = np.sum(
                np.array(bottom_data[break_between_weekday_and_weekend:]), axis=0)
            max_bottom_weekday = max(tmp_sum_weekday)
            max_bottom_weekend = max(tmp_sum_weekend)

            if max_bottom_weekday > max_bottom_weekend:
                max_bottom = max_bottom_weekday
            else:
                max_bottom = max_bottom_weekend
        else:
            max_bottom = maxBottom

        color_list = ["green", "orange", "blue",
                      "violet", "yellow", "tomato", "purple"]
        fontsize_x = 15
        fontsize_y = 20

        # weekday
        plt.subplot(graph_num // graphLayout + 1, graphLayout * 2, itr*2 + 1)
        plt.xticks(x_axis, x_axis_for_show, fontsize=fontsize_x)
        # plt.title("cluster " + str(which_cluster) + " (element count:" + str(eachElementNumInCluster[which_cluster]) + ")", fontsize=titleSize)
        plt.ylim(0, max_bottom)
        plt.yticks(fontsize=fontsize_y)
        plt.legend(("720~", "360~720", "240~360", "120~240",
                   "60~120", "30~60", "~30"), fontsize=8)
        bottom = np.array([0 for _ in range(48)], dtype=np.float)

        for j in range(break_between_weekday_and_weekend):
            plt.bar(x_axis, bottom_data[j], bottom=bottom, color=color_list[j])
            bottom += bottom_data[j]

        # weekend
        plt.subplot(graph_num // graphLayout + 1, graphLayout * 2, itr*2 + 2)
        plt.xticks(x_axis, x_axis_for_show, fontsize=fontsize_x)
        plt.ylim(0, max_bottom)
        plt.yticks(fontsize=fontsize_y)
        bottom_e = np.array([0 for _ in range(48)], dtype=np.float)

        for j in range(break_between_weekday_and_weekend):
            plt.bar(x_axis, bottom_data[j + 7],
                    bottom=bottom_e, color=color_list[j])
            bottom_e += bottom_data[j + 7]

        bottom_data = []  # リセット
        itr += 1

    #plt.rcParams["font.size"] = 9
    plt.tight_layout()

    if fileName != None:
        plt.savefig(fileName)
        plt.close()
    else:
        plt.show()

def mesh_data_to_json(mesh_data, keys = None):
    features = []
    for mid, mesh in enumerate(mesh_data):
        if mesh["stay_count"] >= 1:
            geom = mesh["geometry"]
            tmp = {"type": "Feature", 
                   "geometry": {"type": "Polygon", "coordinates": geom }}
            props = {}
            for k in keys:
                props[k] = str(mesh[k])    
            tmp["properties"] = props
            features.append(tmp)

    data_json = {
        "type": "FeatureCollection", 
        "features": features
    }
    return data_json

def showMesh_in_map(mesh_list, zoom_lat_lon=None, tiles=None, zoom_level=None, color=None, cluster_data=None, valid_mesh=None, which_cluster_show=None):
    # 描写が遅い場合は，tiles='stamentoner'を使用
    args_zoom_lat_lon = [34.99966368, 137.12538141]
    if zoom_lat_lon != None:
        args_zoom_lat_lon = zoom_lat_lon

    args_zoom_level = 13
    if zoom_level != None:
        args_zoom_level = zoom_level
    
    if tiles != None:
        m = folium.Map(args_zoom_lat_lon, zoom_start=args_zoom_level, tiles=tiles)
    else:
        m = folium.Map(args_zoom_lat_lon, zoom_start=args_zoom_level)
        
    itr = 0
    
    for i, mesh in enumerate(mesh_list):
        mesh_id = mesh["mid"]
        polygon = [(x[1], x[0]) for x in mesh["geometry"][0][:-1]]
        popup = str(mesh_id)

        if cluster_data != None:
            popup += ":" + str(cluster_data[itr])

        if color != None:
            if mesh_id in valid_mesh:
                folium.Polygon(locations=polygon, color=color[cluster_data[itr] % len(color)], fill=True, fill_opacity=0.5, popup=popup).add_to(m)
                
#                 else:
#                     folium.Polygon(locations=polygon, color="white", fill=True, fill_opacity=0.5, popup=popup).add_to(m)
        else:
            folium.Polygon(locations=polygon, fill=True, fill_opacity=0, popup=popup).add_to(m)
#         print(itr, cluster_data[itr])
        itr += 1
        
                
    return m