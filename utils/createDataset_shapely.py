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
#import ray
#ray.init()

#@ray.remote(num_cpus=8)
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

        # get Geometries
        with open(os.path.join(args.datadir,location,location+'.json')) as cf:
            config = json.load(cf)

        print("+ Loading geometries...")
        parking = STRtree(MultiPoint(config['parking_centerpoints']))
        prop_bounds = prep(Polygon(config['property_boundary']))
        dt_bounds = prep(Polygon(config['drivethrough_boundary']))
        print("+     done.")

        print("+ Calculating geometries and intersections...")
        # get bbox centroids
        detections['xc'] = detections['x1'] + (detections['x2'] - detections['x1'])/2
        detections['yc'] = detections['y1'] + (detections['y2'] - detections['y1'])/2

        # Non-Ray section ###########################
        g = geoms(parking, prop_bounds, dt_bounds)
        results=[]
        detections = detections[::2]
        for index,row in tqdm(detections.iterrows(), total=detections.shape[0]):
            results.append( g.get_Geoms(row) )
        detections[['isParked', 'isOnProp', 'isDT']] = pd.DataFrame(results, index=detections.index)
        ##############################################

        # Ray version ################################
        # start_time = time.time()
        # g = geoms.remote(parking_geom, prop, dt)
        # results = []
        # for index, row in detections.iterrows():
        #     results.append(g.get_Geoms.remote(row))
        #
        # results = ray.get(results)
        # detections[['isParked', 'isOnProp', 'isDT']] = pd.DataFrame(results, index=detections.index)
        # print(f"+       done in {(time.time() - start_time)/60}")
        ##############################################

        assert args.minmax != args.z

        if args.minmax:
            min_m = np.array([detections['xc'].min(), detections['yc'].min()], dtype=np.float64)
            max_m = np.array([detections['xc'].max(), detections['yc'].max()], dtype=np.float64)

            # Normalize x and y to [0,1]
            detections[['xc', 'yc']] = (detections[['xc','yc']]- min_m) / (max_m - min_m)

        elif args.z:
            mean_m = np.array([detections['xc'].mean(), detections['yc'].mean()], dtype=np.float64)
            std_m = np.array([detections['xc'].std(), detections['yc'].std()], dtype=np.float64)

            # Normalize x and y to [0,1]
            detections[['xc', 'yc']] = (detections[['xc','yc']] - mean_m) / std_m

        print("+       done.")

        print("+ Writing out file.")
        #write out individual paths to separate numpy files
        for path in detections.groupby('ID', sort=False):
            np.save(os.path.join(args.output, location, f"{int(path[0])}.npy"), path[1].to_numpy())
            break



    return

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help="Directory containing subfolders of each location with h5 files.")
    parser.add_argument('--output', type=str, default="preped_data", help="Directory to write data out to.")
    parser.add_argument('--minmax', action="store_true", default=True, help="Min max scale the inputs. (range [0,1])")
    parser.add_argument('-z', action="store_true", default=False, help="Standard scale the inputs. (z-score)")
    args = parser.parse_args()

    if args.z:
        args.minmax = False

    main(args)
