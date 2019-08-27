import numpy as np
import argparse
import tables
import os
#import modin.pandas as pd
import pandas as pd
import json
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.prepared import prep
from shapely.strtree import STRtree
from tqdm import tqdm
import time

class geoms():
    def __init__(self, parking, prop_bounds, drivethru):
        self.parking = parking
        self.prop_bounds = prop_bounds
        self.drivethru = drivethru

    def get_Geoms(self, row):
        box_geom = Polygon([ [row['x1'], row['y1'] ],
                             [row['x2'], row['y1'] ],
                             [row['x2'], row['y2'] ],
                             [row['x1'], row['y2'] ]]   )


        box_centroid = Point([ row['xc'], row['yc'] ] )

        isParked = bool(self.parking.query(box_geom))

        isOnProp = self.prop_bounds.contains(box_centroid)
        isDT = self.drivethru.contains(box_centroid)

        return [int(isParked), int(isOnProp), int(isDT)]

def main(args):
    """Read in data file from locations and write out individual tracks

    """
    #directory = os.fsencode(args.datadir)
    directory = args.datadir
    file_id = 0
    for location in os.listdir(directory):
        # detections = tables.open_file(os.path.join(location, f""))
        os.makedirs(os.path.join(args.output, location), exist_ok=True)

        dfile = os.path.join(args.datadir,location,'data_table.h5')
        try:
            dets = tables.open_file(dfile, 'r').root.detections
            print(f"Opened data file {dfile}")
        except:
            print(f"Could not open file: {dfile}")

        detections = pd.DataFrame.from_records(dets[:])
        dets.close()

        # get Geometries
        with open(os.path.join(args.datadir,location,location+'.json')) as cf:
            config = json.load(cf)

        print("+ Loading geometries...")
        parking = STRtree(MultiPoint(config['parking_centerpoints']))
        prop_bounds = prep(Polygon(config['property_boundary']))
        dt_bounds = prep(Polygon(config['drivethrough_boundary']))
        print("+     done.")

        # Take only every other row. Reduces loss of significance and allows longer in time sequences
        if args.decimate > 1:
            detections = detections[::args.decimate]

        print("+ Calculating geometries and intersections...")
        # get bbox centroids
        detections['xc'] = detections['x1'] + (detections['x2'] - detections['x1'])/2
        detections['yc'] = detections['y1'] + (detections['y2'] - detections['y1'])/2
        # Get the displacements of the paths instead of the absolute positions
        # detections = detections.groupby('ID', sort=False).rolling(2, on=['xc','yc']).mean().reset_index(drop=True)
        # detections = detections.dropna()
        # detections[['xd', 'yd']] = detections[['xc','yc','ID']].groupby('ID').diff()

        # Non-Ray section ###########################
        g = geoms(parking, prop_bounds, dt_bounds)
        results=[]
        detections = detections[::2]
        for index,row in tqdm(detections.iterrows(), total=detections.shape[0]):
            results.append( g.get_Geoms(row) )
        detections[['isParked', 'isOnProp', 'isDT']] = pd.DataFrame(results, index=detections.index)
        ##############################################

        assert args.minmax != args.z

        if args.minmax:
            min_m = np.array([detections['xc'].min(), detections['yc'].min()], dtype=np.float64)
            max_m = np.array([detections['xc'].max(), detections['yc'].max()], dtype=np.float64)

            # Normalize x and y to [0,1]
            detections[['xc', 'yc']] = (detections[['xc','yc']]- min_m) / (max_m - min_m)

        elif args.z:
            mean_c = np.array([detections['xc'].mean(), detections['yc'].mean()], dtype=np.float64)
            std_c = np.array([detections['xc'].std(), detections['yc'].std()], dtype=np.float64)
            #mean_d = np.array([detections['xd'].mean(), detections['yd'].mean()], dtype=np.float64)
            #std_d = np.array([detections['xd'].std(), detections['yd'].std()], dtype=np.float64)

            # Normalize x and y to [0,1]
            detections[['xc', 'yc']] = (detections[['xc','yc']] - mean_c) / std_c
            #detections[['xd', 'yd']] = (detections[['xd','yd']] - mean_d) / std_d

        print("+       done.")


        print("+ Writing out file.")
        #write out individual paths to separate numpy files
        for id,path in detections.groupby('ID', sort=False):
            if len(path) < args.trim +30:
                continue
            path_temp = path.copy()
            path_temp[['xc','yc']] = path_temp[['xc','yc']].rolling(10).mean()
            path_temp[['xd','yd']] = path_temp[['xc','yc']].diff()
            path_temp = path_temp[['xd','xd','isParked','isOnProp','isDT']].dropna(how='any')
            if path_temp.isnull().values.any():
                print("DF has NAN!!!!")
            np.save(os.path.join(args.output, location, f"{int(id)}.npy"), path_temp.to_numpy())



    return

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help="Directory containing subfolders of each location with h5 files.")
    parser.add_argument('--output', type=str, default="preped_data", help="Directory to write data out to.")
    parser.add_argument('--minmax', action="store_true", default=True, help="Min max scale the inputs. (range [0,1])")
    parser.add_argument('-z', action="store_true", default=False, help="Standard scale the inputs. (z-score)")
    parser.add_argument('--decimate', type=int , default=1, help="Decimate the data by a factor of N. i.e. data[::N]")
    parser.add_argument('--trim', type=int, default=100, help="Dont include paths that are less than this amount.")
    args = parser.parse_args()

    if args.z:
        args.minmax = False

    main(args)
