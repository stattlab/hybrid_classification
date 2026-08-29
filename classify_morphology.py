import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import scipy
from scipy.interpolate import griddata

def z_char(file, frames = range(90,100),threshold=2.4):
    '''z_char: computes the average percentage of lipid particles within a specified z-threshold across multiple frames in a GSD trajectory file.
    Args:
        file (str): Path to the GSD trajectory file.
        frames (list): List of frame indices to analyze.
        threshold (float): Z-threshold for counting lipid particles.
    Returns:
        float: Average percentage of lipid particles within the z-threshold across the specified frames.'''
    
    traj = gsd.hoomd.open(file)
    percent_within_threshold = []
    for f in frames:
        frame = traj[f]
        typeids = frame.particles.typeid
        types = frame.particles.types
        lipid_types = ['C1','C4h']
        lipid_ids = [i for i, t in enumerate(types) if t in lipid_types]
        box = frame.configuration.box
        positions = frame.particles.position
        lipid_positions = positions[np.isin(typeids, lipid_ids)]

        num_in_threshold = np.sum(np.abs(lipid_positions[:,2]) < threshold)
        percent_in_threshold = num_in_threshold / len(lipid_positions)
        percent_within_threshold.append(percent_in_threshold)
    return np.mean(percent_within_threshold)
        

def calc_ts(traj):
            '''calc_ts: calculates the average thickness and area per lipid (APL) of a lipid bilayer from a GSD trajectory.
            Args:
                traj (gsd.hoomd.open): GSD trajectory object.
            Returns:
                tuple: A tuple containing:
                    - thickness (float): Average thickness of the lipid bilayer.
                    - std_thickness (float): Standard deviation of the thickness across pixels.
                    - apl (float): Average area per lipid.
                    - std_apl (float): Standard deviation of the area per lipid.'''
            print(len(traj))
            apls = []
            ts = []
            std_ts = []
            for i in range(len(traj)-1, len(traj)):
                frame = traj[i]
                typeids = frame.particles.typeid
                types = frame.particles.types
                lipid_types = ['C1', 'C4h', 'N4a', 'SN4a', 'Q1', 'Q5']
                hydrophobic_thickness_types = ['C1','N4a', 'SN4a','C4']
                lipid_ids = [j for j, t in enumerate(types) if t in lipid_types]
                hydro_ids = [j for j, t in enumerate(types) if t in hydrophobic_thickness_types]
                positions = frame.particles.position
                lipid_positions = positions[np.isin(typeids, hydro_ids)]
                box = frame.configuration.box[0]
                xy = np.linspace(-box/2+1, box/2-1, 30)
                mesh = np.meshgrid(xy, xy)
                grid_x = mesh[0].ravel()
                grid_y = mesh[1].ravel()
                # For each pixel, find the 4 highest and 4 lowest lipid beads in z within a small radius
                pixel_coords = np.column_stack([grid_x, grid_y])
                thickness_map = np.full(pixel_coords.shape[0], np.nan)
                tree = cKDTree(lipid_positions[:, :2])
                # Set a search radius (e.g., 2.0 units)
                search_radius = 2.0
                for idx_pix, pix in enumerate(pixel_coords):
                    idxs = tree.query_ball_point(pix, r=search_radius)
                    if len(idxs) >= 2:
                        z_vals = lipid_positions[idxs, 2]
                        sorted_z = np.sort(z_vals)
                        # Take average of 4 highest and 4 lowest
                        # Take average of highest and lowest bead in z
                        top = sorted_z[-1]
                        top = np.mean(sorted_z[-5:]) if len(sorted_z) >= 5 else sorted_z[-1]
                        bottom = np.mean(sorted_z[:5]) if len(sorted_z) >= 5 else sorted_z[0]
                        thickness_map[idx_pix] = top - bottom
                # Take the mean thickness over all valid pixels
                thickness = np.nanmean(thickness_map)
                std_thickness = np.nanstd(thickness_map)
                box = frame.configuration.box
                area = box[0] * box[1]
                nump_lipids = np.sum(typeids == types.index('Q1'))/2
                apl = area / nump_lipids
                apls.append(apl)
                ts.append(thickness)
                std_ts.append(std_thickness)
            thickness = np.mean(ts)
            apl = np.mean(apls)
            std_thickness = np.mean(std_ts)
            std_apl = np.std(apls)
            print(f"Thick: {thickness} +/- {std_thickness}, APL: {apl} +/- {std_apl}")
            return thickness, std_thickness, apl, std_apl   

def conc_map(file, pixel_size = 1, frames = range(90,100),threshold=4):
    '''conc_map: computes the concentration map of lipids and polymers in a given trajectory file. It uses DBSCAN clustering to identify clusters of lipids and polymers based on their positions in the simulation box.
    Args:
        file (str): Path to the GSD trajectory file.
        pixel_size (float): Size of the pixels for the concentration map.
        frames (list): List of frame indices to analyze.
        threshold (float): Threshold for clustering.
    Returns:
        tuple: A tuple containing:
            - mean_std_lipid (float): Mean standard deviation of lipid concentration across frames.
            - mean_std_polymer (float): Mean standard deviation of polymer concentration across frames.
            - lipid_hist (ndarray): 2D histogram of lipid positions.
            - polymer_hist (ndarray): 2D histogram of polymer positions.
            - labels_lipid (ndarray): Cluster labels for lipids.
            - labels_polymer (ndarray): Cluster labels for polymers.'''
    traj = gsd.hoomd.open(file)
    all_std_polymer = []
    all_std_lipid = []
    for f in frames:
        frame = traj[f]
        typeids = frame.particles.typeid
        types = frame.particles.types
        lipid_types = ['C1','C4h','N4a','SN4a','Q1','Q5']
        polymer_types = ['C4']
        lipid_ids = [i for i, t in enumerate(types) if t in lipid_types]
        polymer_ids = [i for i, t in enumerate(types) if t in polymer_types]
        box = frame.configuration.box
        positions = frame.particles.position
        lipid_positions = positions[np.isin(typeids, lipid_ids)]
        polymer_positions = positions[np.isin(typeids, polymer_ids)]

        x_bins = np.arange(-box[0]/2, box[0]/2, pixel_size)
        y_bins = np.arange(-box[1]/2, box[1]/2, pixel_size)
        lipid_hist, _, _ = np.histogram2d(lipid_positions[:,0], lipid_positions[:,1], bins=[x_bins, y_bins])
        polymer_hist, _, _ = np.histogram2d(polymer_positions[:,0], polymer_positions[:,1], bins=[x_bins, y_bins])
        if lipid_positions.shape[0] > 0:
            dbscan_lipid = DBSCAN(eps=1.1,min_samples=20)
            labels_lipid = dbscan_lipid.fit_predict(lipid_positions[:, :])
        else:
            labels_lipid = np.array([])
        if polymer_positions.shape[0] > 0:
            dbscan_polymer = DBSCAN(eps=1.1,min_samples=20)
            labels_polymer = dbscan_polymer.fit_predict(polymer_positions[:, :])
        else:
            labels_polymer = np.array([])
        std_lipid = np.std(lipid_hist)
        std_polymer = np.std(polymer_hist)
        all_std_lipid.append(std_lipid)
        all_std_polymer.append(std_polymer)
    return np.mean(all_std_lipid), np.mean(all_std_polymer), lipid_hist, polymer_hist, labels_lipid, labels_polymer

file = '../data/1017/dump/1017-S06/traj.gsd'
root = '../data/1017/dump/1017-S06/'

p = z_char(file, frames=range(90,100))
mean_std_lipid, mean_std_polymer, lipid_hist, polymer_hist, labels_lipid, labels_polymer = conc_map(file, pixel_size=1, frames=range(99,100))
num_lipid_clusters = len(set(labels_lipid)) - (1 if -1 in labels_lipid else 0)
num_polymer_clusters = len(set(labels_polymer)) - (1 if -1 in labels_polymer else 0)
num_unindentified_polymer = np.sum(labels_polymer == -1)
num_unindentified_lipid = np.sum(labels_lipid == -1)
# df.at[idx, 'p'] = p
ts, std_ts, apl, std_apl = calc_ts(gsd.hoomd.open(file))

if num_lipid_clusters ==0 or num_polymer_clusters == 0:
    print("Mixing")
elif mean_std_polymer < 8.012:
    if num_lipid_clusters < 1.5:
        print("Lateral Phase Separation")
    else:
        print("Polymer rich") 
else: 
    if p < 0.573:
        print("Lateral Phase Separation")
    else:
        if mean_std_polymer < 12.318:
            print("Lateral Phase Separation")
        else:
            print("Unzipping")