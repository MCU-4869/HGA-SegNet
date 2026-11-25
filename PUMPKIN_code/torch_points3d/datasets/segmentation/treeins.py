import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
from plyfile import PlyData, PlyElement
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil
# PLY reader
from torch_points3d.modules.KPConv.plyutils import read_ply
from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

# @Treeins: a lot of code copied from torch_points3d/datasets/segmentation/npm3d.py, for changes see @Treeins


#@Treeins: two semantic segmentation classes: non-tree and tree
Treeins_NUM_CLASSES = 2

INV_OBJECT_LABEL = {
    0: "non-tree",
    1: "tree",
}


OBJECT_COLOR = np.asarray(
    [
        [179, 116, 81],  # 'non-tree'  ->  brown
        [77, 174, 84],  # 'tree'  ->  bright green
        [0, 0, 0],  # unlabelled .->. black
    ]
)

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

################################### UTILS #######################################
def object_name_to_label(object_class):
    """convert from object name in NPPM3D to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["unclassified"])
    return object_label

def read_treeins_format(train_file, label_out=True, verbose=False, debug=False, return_rgb=False):
    """extract data from a treeins file"""
    raw_path = train_file
    data = read_ply(raw_path)
    xyz = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T
    if not label_out:
        return xyz
    # @Treeins: read in semantic segmentation labels (0: unclassified, 1: non-tree, 2: tree) and change them to
    # (-1: unclassified, 0, non-tree, 1: tree) because unclassified must have label -1
    semantic_labels = data['scalar_seg'].astype(np.int64)
    semantic_labels = semantic_labels - 1  # 0→-1(ignore), 1→0(non-tree), 2→1(tree)
    # @Treeins: The attribute treeID tells us to which tree a point belongs, hence we use it as instance labels
    instance_labels = data['scalar_Pumpkin_stem'].astype(np.int64)  # 0=non-tree(bg), 1-12=tree instances
    #print(np.unique(instance_labels))

    result = [
        torch.from_numpy(xyz),
        torch.from_numpy(semantic_labels),
        torch.from_numpy(instance_labels),
    ]
    if return_rgb:
        # Per-point RGB colours from PLY, normalised [0,255] -> [0,1]
        rgb = np.vstack((data['red'], data['green'], data['blue'])).astype(np.float32).T / 255.0
        result.append(torch.from_numpy(rgb))
    return tuple(result)


def compute_height_above_ground(xyz, semantic_labels, cell_size=2.0):
    """Terrain-normalised height (canopy height) per point — a real "growth" signal.

    The raw ``z`` in the PUMPKIN PLYs is ABSOLUTE elevation (hillside ~190–310m),
    so raw ``z`` (and ``pos_z_rela``, which is relative to a random cylinder
    centre) mixes terrain relief with actual plant height. This estimates the DTM
    (digital terrain model) as the per-cell minimum ``z`` of the *non-tree*
    (ground) points, fills occluded cells by nearest-neighbour, and returns
    ``z - DTM`` — the true height above local ground (canopy height), label-free.

    Args:
        xyz: [N, 3] float32 (x, y, z) in metres.
        semantic_labels: [N] int64, 0 = non-tree/ground, 1 = tree.
        cell_size: ground-grid resolution in metres (2.0 is robust to occlusion).
    Returns:
        height: [N] float32, metres above ground (>= ~0).
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    semantic_labels = np.asarray(semantic_labels)
    ground = semantic_labels == 0
    if ground.sum() < 50:
        # Degenerate: no reliable ground -> fall back to z minus its global min.
        return (xyz[:, 2] - xyz[:, 2].min()).astype(np.float32)

    min_x, max_x = xyz[:, 0].min(), xyz[:, 0].max()
    min_y, max_y = xyz[:, 1].min(), xyz[:, 1].max()
    nx = int(np.ceil((max_x - min_x) / cell_size)) + 1
    ny = int(np.ceil((max_y - min_y) / cell_size)) + 1

    # Per-cell minimum z over GROUND points (np.minimum.at = in-place min reduce).
    grid = np.full((nx, ny), np.inf, dtype=np.float32)
    gx = ((xyz[ground, 0] - min_x) / cell_size).astype(np.int64)
    gy = ((xyz[ground, 1] - min_y) / cell_size).astype(np.int64)
    np.minimum.at(grid, (gx, gy), xyz[ground, 2])

    # Fill cells with no ground return (occlusion) by nearest valid ground.
    valid = ~np.isinf(grid)
    if valid.sum() < grid.size:
        from scipy.interpolate import griddata
        xs = min_x + (np.arange(nx) + 0.5) * cell_size
        ys = min_y + (np.arange(ny) + 0.5) * cell_size
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        grid = griddata(
            (XX[valid], YY[valid]), grid[valid],
            (XX, YY), method="nearest",
        ).astype(np.float32)

    ix = ((xyz[:, 0] - min_x) / cell_size).astype(np.int64)
    iy = ((xyz[:, 1] - min_y) / cell_size).astype(np.int64)
    dtm = grid[ix, iy]
    return (xyz[:, 2] - dtm).astype(np.float32)


def compute_horizontal_contrast(xyz, semantic_labels, cell_size=2.0, window_radius=3):
    """Horizontal height contrast — the de-confounded "growth" feature.

    Absolute canopy height ``h = z - DTM`` mixes two effects: (1) separating
    adjacent *different* trees (useful) and (2) splitting the *same* tree's
    vertical trunk/canopy stacking (spurious — trunk and canopy have very
    different ``h`` at the same ``(x, y)``). Feeding ``h`` directly therefore
    raises recall at the cost of over-splitting (precision).

    This instead returns ``h`` *relative to its horizontal neighbourhood*, a
    pure function of horizontal location (per-cell, via a Canopy Height Model),
    so trunk and canopy of the same tree get the SAME value (no vertical split)
    while adjacent trees of different height get opposite-sign values (kept
    separation). Steps:

      1. ``h = z - DTM`` (reuses :func:`compute_height_above_ground`).
      2. CHM: per-cell MAX ``h`` -> collapses vertical stacking to the canopy top.
      3. Local horizontal mean of the CHM (uniform filter, radius ``window_radius``).
      4. ``contrast = CHM - local_mean`` (signed: tall>0, short<0).

    Args:
        xyz: [N, 3] float32 (x, y, z) in metres.
        semantic_labels: [N] int64, 0 = non-tree/ground, 1 = tree.
        cell_size: ground/CHM grid resolution in metres.
        window_radius: half-width (in cells) of the horizontal mean filter.
    Returns:
        contrast: [N] float32, canopy height relative to horizontal neighbours.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    semantic_labels = np.asarray(semantic_labels)

    # 1. Terrain-normalised canopy height (ground/occlusion handling already here).
    height = compute_height_above_ground(xyz, semantic_labels, cell_size=cell_size)

    min_x, max_x = xyz[:, 0].min(), xyz[:, 0].max()
    min_y, max_y = xyz[:, 1].min(), xyz[:, 1].max()
    nx = int(np.ceil((max_x - min_x) / cell_size)) + 1
    ny = int(np.ceil((max_y - min_y) / cell_size)) + 1
    gx = ((xyz[:, 0] - min_x) / cell_size).astype(np.int64)
    gy = ((xyz[:, 1] - min_y) / cell_size).astype(np.int64)

    # 2. Canopy Height Model: per-cell MAX height (canopy top, ignores trunk below).
    chm = np.full((nx, ny), -np.inf, dtype=np.float32)
    np.maximum.at(chm, (gx, gy), height)

    # Fill cells with no return (gaps/occlusion) by nearest valid canopy height.
    valid = ~np.isinf(chm)
    if valid.sum() < chm.size:
        from scipy.interpolate import griddata
        xs = min_x + (np.arange(nx) + 0.5) * cell_size
        ys = min_y + (np.arange(ny) + 0.5) * cell_size
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        chm = griddata(
            (XX[valid], YY[valid]), chm[valid],
            (XX, YY), method="nearest",
        ).astype(np.float32)

    # 3. Local horizontal mean of the CHM (blurred canopy surface).
    from scipy.ndimage import uniform_filter
    chm_local_mean = uniform_filter(
        chm, size=2 * window_radius + 1, mode="nearest",
    ).astype(np.float32)

    # 4. Signed contrast: this location's canopy height vs its neighbours.
    contrast_grid = chm - chm_local_mean
    return contrast_grid[gx, gy].astype(np.float32)


def to_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    colors = OBJECT_COLOR[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, "Treeins")
    PlyData([el], byte_order=">").write(file)
    
def to_eval_ply(pos, pre_label, gt, file):
    assert len(pre_label.shape) == 1
    assert len(gt.shape) == 1
    assert pos.shape[0] == pre_label.shape[0]
    assert pos.shape[0] == gt.shape[0]
    pos = np.asarray(pos)
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("preds", "u16"), ("gt", "u16")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["preds"] = np.asarray(pre_label)
    ply_array["gt"] = np.asarray(gt)
    PlyData.write(file)
    
def to_ins_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    max_instance = np.max(np.asarray(label)).astype(np.int32)+1
    rd_colors = np.random.randint(255, size=(max_instance,3), dtype=np.uint8)
    colors = rd_colors[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    PlyData.write(file)



################################### Used for fused NPM3D radius sphere ###################################


class TreeinsOriginalFused(InMemoryDataset):
    """ Original Treeins dataset. Each area is loaded individually and can be processed using a pre_collate transform.
    This transform can be used for example to fuse the area into a single space and split it into 
    spheres or smaller regions. If no fusion is applied, each element in the dataset is a single area by default.

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    forest regions: list of str
        @Treeins: specifies from which forest region(s) data files should be used for training and validation, [] means taking data files from all forest regions
    test_area: list
        @Treeins: during training/running train.py: [] means taking all specified test files (i.e. all files with name ending in "test" for testing, otherwise list of ints indexing into which of these specified test files to use
        @Treeins: during evaluation/running eval.py: paths to files to test model on
    split: str
        can be one of train, trainval, val or test
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    pre_transform
    transform
    pre_filter
    """

    num_classes = Treeins_NUM_CLASSES

    def __init__(
        self,
        root,
        grid_size,
        forest_regions=[], #@Treeins
        test_area=[],
        split="train",
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        keep_instance=False,
        verbose=False,
        debug=False,
    ):
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.forest_regions = forest_regions
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split
        self.grid_size = grid_size
     
        super(TreeinsOriginalFused, self).__init__(root, transform, pre_transform, pre_filter)
        # @Treeins: case for training/when running train.py
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            
            if split == "train":
                path = self.processed_paths[0]
            elif split == "val":
                path = self.processed_paths[1]
            elif split == "test":
                path = self.processed_paths[2]
            elif split == "trainval":
                path = self.processed_paths[3]
            else:
                raise ValueError((f"Split {split} found, but expected either " "train, val, trainval or test"))
            self._load_data(path)

            if split == "test":
                # @Treeins: load all files specified in the test_area list
                if self.test_area == []:
                    "PROBLEM: In branch split==test even though test_area was not initialized yet"
                self.raw_test_data = [torch.load(self.raw_areas_paths[test_area_i]) for test_area_i in self.test_area]

        # @Treeins: case for evaluation/when running eval.py
        else:
            # @Treeins: process all files at the paths given in test_area list for evaluation
            self.process_test(test_area)
            path = self.processed_path
            self._load_data(path)
            self.raw_test_data = [torch.load(raw_area_path) for raw_area_path in self.raw_areas_paths]

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        """returns list of paths to the .ply raw data files we use"""
        if self.forest_regions == []:  # @Treeins: get all data file names in folder self.raw_dir
            return glob.glob(self.raw_dir + '/**/*.ply', recursive=True)
        else: #@Treeins: get data file names coming from the forest region(s) in self.forest_regions within the folder self.raw_dir
            raw_files_list = []
            for region in self.forest_regions:
                raw_files_list += glob.glob(self.raw_dir + "/" + region + "/*.ply", recursive=False)
            return raw_files_list

    @property
    def processed_dir(self):
        """returns path to the directory which contains the processed data files,
               e.g. path/to/project/OutdoorPanopticSeg_V2/data/treeinsfused/processed_0.2"""
        processed_dir_prefix = 'processed_' + str(self.grid_size) #add grid size to the processed directory's name
        if self.forest_regions != []: #@Treeins: add forest regions to the processed directory's name (if not all forest regions are used)
            processed_dir_prefix += "_" + str(self.forest_regions)

        # @Treeins: case for training/when running train.py
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            return osp.join(self.root, processed_dir_prefix)
        # @Treeins: case for evaluation/when running eval.py
        else:
            return osp.join(self.root, processed_dir_prefix+'_test')

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return os.path.join(self.processed_dir, pre_processed_file_names)

    @property
    def raw_areas_paths(self):
        """returns list of paths to .pt files saved in self.processed_dir and created from the .ply raw data files"""
        # @Treeins: case for training/when running train.py
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            if not hasattr(self, "num_datafiles"):
                self.num_datafiles = len(self.raw_file_names)
            return [os.path.join(self.processed_dir, "raw_area_%i.pt" % i) for i in range(self.num_datafiles)]
        # @Treeins: case for evaluation/when running eval.py
        else:
            return [os.path.join(self.processed_dir, 'raw_area_'+os.path.split(f)[-1].split('.')[0]+'.pt') for f in self.test_area]


    @property
    def processed_file_names(self):
        """return list of paths to all kinds of files in the processed directory"""
        # @Treeins: case for training/when running train.py
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            return (
            ["{}.pt".format(s) for s in ["train", "val", "test", "trainval"]]
            + self.raw_areas_paths
            + [self.pre_processed_path]
        )
        # @Treeins: case for evaluation/when running eval.py
        else:
            return ['processed_'+os.path.split(f)[-1].split('.')[0]+'.pt' for f in self.test_area]

    @property
    def raw_test_data(self):
        return self._raw_test_data

    @raw_test_data.setter
    def raw_test_data(self, value):
        self._raw_test_data = value

    @property
    def num_features(self):
        feats = self[0].x
        if feats is not None:
            return feats.shape[-1]
        return 0
    #def download(self):
    #    super().download()

    def process(self):
        """Takes the given .ply files, processes them and saves the newly created files in self.processed_dir.
        This method is used during training/running train.py."""

        if not os.path.exists(self.pre_processed_path):# @Treeins: if we haven't already processed the raw .ply data files in a previous run with the same grid_size and forest_regions
            input_ply_files = self.raw_file_names

            # Gather data per area
            data_list = [[] for _ in range(len(input_ply_files))] #@Treeins: list of lists which each contains one .ply data file
            for area_num, file_path in enumerate(input_ply_files):
                area_name = os.path.split(file_path)[-1]
                xyz, semantic_labels, instance_labels, rgb = read_treeins_format(
                    file_path, label_out=True, verbose=self.verbose, debug=self.debug, return_rgb=True
                )

                data = Data(pos=xyz, y=semantic_labels, rgb=rgb)
                # Growth feature: terrain-normalised canopy height h=z-DTM.
                # Carried through grid/cylinder sampling like `rgb`.
                height = compute_height_above_ground(xyz.numpy(), semantic_labels.numpy())
                data.height = torch.from_numpy(height).float().view(-1, 1)
                data.validation_set = False
                data.test_set = False

                #@Treeins: list of lists which each contains one .ply data file
                if area_name[-7:-4]=="val":
                    data.validation_set = True
                #@Treeins:  if "test" at end of area_name, i.e. at end of .ply file name, we put data file into test set
                elif area_name[-8:-4]=="test":
                    data.test_set = True
                    self.test_area.append(area_num)

                if self.keep_instance:
                    data.instance_labels = instance_labels

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                data_list[area_num].append(data)
            # ★ 2026-11-18: fuse ONE file group at a time. Fusing the whole list
            # (`cT.PointCloudFusion()(data_list)`) builds a second full copy of
            # every point (~50G on 52 plots) before anything is saved -- that was
            # the actual SIGKILL peak (anon hit 89G with memory.max=90G). One
            # .ply per group means `fusion(group)` just returns that Data object,
            # so no copy is made at all. Files written are identical.
            fusion = cT.PointCloudFusion()
            for i, group in enumerate(data_list):
                area = fusion(group)
                torch.save(area, self.raw_areas_paths[i])
                del area
            import gc
            gc.collect()


            for area_datas in data_list:
                # Apply pre_transform
                if self.pre_transform is not None:
                    area_datas = self.pre_transform(area_datas)
            torch.save(data_list, self.pre_processed_path)
        # if we already processed the raw .ply data files in a previous run with the same grid_size and forest_regions, we can simply load the processed data
        else:
            data_list = torch.load(self.pre_processed_path)

        if self.debug:
            return

        train_data_list = []
        val_data_list = []
        trainval_data_list = []
        test_data_list = []
        #list is a list containing one single data file path
        for list in data_list:
            # data is one single file path
            for data in list:
                validation_set = data.validation_set
                del data.validation_set
                test_set = data.test_set
                del data.test_set
                if validation_set:
                    val_data_list.append(data)
                elif test_set:
                    test_data_list.append(data)
                else:
                    train_data_list.append(data)

        # ★ 2026-11-18: drop the owner list BEFORE the transforms. The split lists
        # still hold the Data objects, but as soon as a split's transform replaces
        # its list, that split's raw objects become unreachable and are freed --
        # otherwise `data_list` keeps every raw point alive through the whole
        # transform phase (that was ~50G + transformed copies, over the 90G cap).
        # Do NOT build the pre-transform trainval list here: it would hold a
        # reference to every raw object and defeat the release.
        del data_list
        import gc
        gc.collect()

        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            train_data_list = self.pre_collate_transform(train_data_list)
            val_data_list = self.pre_collate_transform(val_data_list)
            test_data_list = self.pre_collate_transform(test_data_list)
            # ★ 2026-11-18: trainval is exactly train+val, so reuse the transformed
            # lists instead of running the same per-item transform on the same
            # points a second time. Byte-identical as long as the transform is
            # per-item (SaveOriginalPosId + GridSampling3D are). ⚠️ If
            # PointCloudFusion is ever enabled here it fuses ACROSS items and this
            # shortcut stops being equivalent -- restore the extra call then.
        trainval_data_list = val_data_list + train_data_list

        self._save_data(train_data_list, val_data_list, test_data_list, trainval_data_list)

    def process_test(self, test_area):
        """Takes the .ply files specified in data:fold: [...] in the file conf/eval.yaml as test files, processes them and saves the newly created files in self.processed_dir.
        This method is used during evaluation/running eval.py.
        @Treeins: Method is extended so that we can evaluate on more than one test file."""

        self.processed_path = osp.join(self.processed_dir,'processed_test.pt')

        #if not os.path.exists(self.processed_dir):
        #    os.mkdir(self.processed_dir)

        test_data_list = []
        for i, file_path in enumerate(test_area): #for each .ply test data file's path
            area_name = os.path.split(file_path)[-1] #e.g. SCION_plot_31_annotated_test.ply
            processed_area_path = osp.join(self.processed_dir, self.processed_file_names[i])
            if not os.path.exists(processed_area_path): #if .pt file corresponding to .ply test file at file_path hasn't been created and saved in self.processed_dir yet
                xyz, semantic_labels, instance_labels, rgb = read_treeins_format(
                    file_path, label_out=True, verbose=self.verbose, debug=self.debug, return_rgb=True
                )
                data = Data(pos=xyz, y=semantic_labels, rgb=rgb)
                # Growth feature: terrain-normalised canopy height h=z-DTM.
                height = compute_height_above_ground(xyz.numpy(), semantic_labels.numpy())
                data.height = torch.from_numpy(height).float().view(-1, 1)
                if self.keep_instance:
                    data.instance_labels = instance_labels
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                # @Treeins: important to not just append data, but [data], so that after the cT.PointCloudFusion() transformation below, we still separate the single data files
                test_data_list.append([data])
                torch.save(data, processed_area_path)
            else: #if .pt file corresponding to .ply test file at file_path has already been created and saved in self.processed_dir
                data = torch.load(processed_area_path)
                # @Treeins: important to not just append data, but [data], so that after the cT.PointCloudFusion() transformation below, we still separate the single data files
                test_data_list.append([data])

        # test_data_list is a list of lists where each such list contains one single test data file's data -> [[Data (...)],[Data(...)], ...]
        raw_areas = cT.PointCloudFusion()(test_data_list)
        for i, area in enumerate(raw_areas):#@Treeins: for each batch in raw_areas (where one batch comes exactly from one test data file)
            torch.save(area, self.raw_areas_paths[i])

        if self.debug:
            return


        # @Treeins: take the test data file's data out of the inner list:
        # test_data_list is of format [[Data(...)], [Data(...)], ...] vs. new_test_data_list is of format [Data(...), Data(...), ...]
        new_test_data_list = [listelem[0] for listelem in test_data_list]
        test_data_list = new_test_data_list
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            test_data_list = self.pre_collate_transform(test_data_list)
        torch.save(test_data_list, self.processed_path)


    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(val_data_list), self.processed_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[2])
        torch.save(self.collate(trainval_data_list), self.processed_paths[3])

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)


class TreeinsSphere(TreeinsOriginalFused):
    """ Small variation of TreeinsOriginalFused that allows random sampling of spheres
    within an Area during training and validation.

    ⚠️ `radius` is in the **coordinate units of the point cloud**; it is NOT always metres.
      - FOR-instance: coordinates in metres  ⇒ `radius=8` means **8 m**.
      - PUMPKIN (this project): coordinates in **centimetres** ⇒ `radius=8` means **8 cm**
        (diameter 16 cm, spheres on a 16 cm grid — note the grid step below is `2*radius`,
        which is why the historical "0.16m grid" line was consistent while "8m" was not).
    Verified 2026-10-20 against the raw PLY: one instance's bbox is 44.88 × 39.33 × 93.56
    units, which is 45 × 39 × 94 **cm** — as metres it would be a 94 m tall plant.

    If sample_per_epoch is not specified, spheres
    are taken on a `2*radius` grid.

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 4 that denotes the area used for testing
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, root, sample_per_epoch=100, radius=8, grid_size=0.12, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = cT.GridSampling3D(size=grid_size, mode="last")
        # Frozen sample set (opt-in, default OFF -> behaviour is byte-for-byte the
        # old one). See `set_frozen_samples` for why this exists.
        self._frozen = None
        super().__init__(root, grid_size, *args, **kwargs)

    def set_frozen_samples(self, samples):
        """Replay a pre-drawn sample set instead of drawing a fresh one.

        WHY THIS EXISTS
        ---------------
        With `sample_per_epoch > 0`, `get()` calls `_get_random()` (line ~633),
        which redraws `sample_per_epoch` cylinders EVERY epoch -- random centre,
        randomly chosen from 11.5M candidates weighted by class frequency. So the
        val/test score of every run has historically been computed on a DIFFERENT
        set of cylinders each epoch, and two runs are never scored on the same
        inputs.

        For a comparison table that is fatal: a learned model and a classical
        clustering baseline (K-means / Watershed / DBSCAN) CANNOT be made
        comparable unless both see the identical cylinders. It also adds
        epoch-to-epoch noise on top of the model's own.

        Passing a list of pre-drawn Data objects here makes `get(idx)` return
        `samples[idx]` instead. Transforms still apply exactly as before, because
        this only replaces what `get()` returns inside the normal `__getitem__`
        chain. Pass None to restore the original behaviour.

        Build the list with `freeze_samples.py`.
        """
        self._frozen = samples
        if samples is not None:
            # Keep __len__ in step with the frozen set.
            self._sample_per_epoch = len(samples)

    def __len__(self):
        if self._frozen is not None:
            return len(self._frozen)
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._test_spheres)

    def len(self):
        return len(self)

    def get(self, idx):
        if self._frozen is not None:
            return self._frozen[idx].clone()
        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            return self._test_spheres[idx].clone()

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        # @Treeins: case for training/when running train.py
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            super().process()
        # @Treeins: case for evaluation/when running eval.py
        else:
            super().process_test(self.test_area)

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        sphere_sampler = cT.SphereSampling(self._radius, centre[:3], align_origin=False)
        return sphere_sampler(area_data)

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(train_data_list, self.processed_paths[0])
        torch.save(val_data_list, self.processed_paths[1])
        torch.save(test_data_list, self.processed_paths[2])
        torch.save(trainval_data_list, self.processed_paths[3])

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            #print(self._datas)
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.SphereSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos), leaf_size=10)
                setattr(data, cT.SphereSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            # Filter out ignore label (-1) centers
            self._centres_for_sampling = self._centres_for_sampling[self._centres_for_sampling[:, 4] != -1]
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)


class TreeinsCylinder(TreeinsSphere):
    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        last_area = None
        for attempt in range(50):
            chosen_label = np.random.choice(self._labels, p=self._label_counts)
            valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
            centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
            centre = valid_centres[centre_idx]
            area_data = self._datas[centre[3].int()]
            cylinder_sampler = cT.CylinderSampling(self._radius, centre[:3], align_origin=False)
            cylinder_area = cylinder_sampler(area_data)
            last_area = cylinder_area
            if torch.any(cylinder_area.instance_labels > 0):
                return cylinder_area
        print("WARNING: no instance crop found after 50 attempts")
        return last_area

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.CylinderSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
                setattr(data, cT.CylinderSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            # Filter out ignore label (-1) centers
            self._centres_for_sampling = self._centres_for_sampling[self._centres_for_sampling[:, 4] != -1]
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridCylinderSampling(self._radius, self._radius, center=False)
            self._test_spheres = []
            self._num_spheres = []
            for i, data in enumerate(self._datas): #for each test data file's data
                test_spheres = grid_sampler(data)
                # @Treeins: self._test_spheres is list of grid-cylinder-sampled data from all test data
                # -> data from different data files in the same list without separation
                # -> to recover separation again in the end, we need self._num_spheres
                self._test_spheres = self._test_spheres + test_spheres
                # @Treeins: saves how many cylinders were sampled from each test data file respectively
                # -> like this, we can recover which cylinder comes from which test data file
                self._num_spheres = self._num_spheres + [len(test_spheres)]


class TreeinsFusedDataset(BaseDataset):
    """ Wrapper around NPM3DSphere that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get("sampling_format", "sphere")
        dataset_cls = TreeinsCylinder if sampling_format == "cylinder" else TreeinsSphere

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=3000,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
        )
        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data
        
    @property
    def test_data_spheres(self):
        return self.test_dataset[0]._test_spheres

    @property
    def test_data_num_spheres(self):
        return self.test_dataset[0]._num_spheres

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save NPM3D predictions to disk using NPM3D color scheme

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        #from torch_points3d.metrics.s3dis_tracker import S3DISTracker
        #return S3DISTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
        from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
