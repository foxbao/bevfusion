import os
import time
from .kl_utils import precompute_timestamps,match_multi_sensor_data,generate_token,match_sensor_data
from pathlib import Path
class KL():
    #nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.label_dir=self.dataroot/version/'label'
        self.sample_dir=self.dataroot/version/'sample'
        self.sensor_names={
            'helios_front_left':'helios_front_left',
            'helios_rear_right':'helios_rear_right',
            'bp_front_left':'bp_front_left',
            'bp_front_right':'bp_front_right',
            'bp_rear_left':'bp_rear_left',
            'bp_rear_right':'bp_rear_right'}
        self.camera_names={
            'h100f1a_front_left':'h100f1a_front_left',
            'h100f1a_rear_right':'h100f1a_rear_right',
            'h120ua_front_left':'h120ua_front_left',
            'h120ua_front_mid':'h120ua_front_mid',
            'h120ua_front_right':'h120ua_front_right',
            'h120ua_rear_left':'h120ua_rear_left',
            'h120ua_rear_mid':'h120ua_rear_mid',
            'h120ua_rear_right':'h120ua_rear_right',
        }
        self.samples=[]
        start_time = time.time()
        if verbose:
            print("======\nLoading KL tables for version {}...".format(self.version))
        self.load_data()


        if verbose:
            # for table in self.table_names:
            #     print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))
    
    def load_data(self):
        for s_dir in os.listdir(self.label_dir):
            label_s_path = self.label_dir/s_dir  # label/s7
            sample_s_path = self.sample_dir/s_dir  # sample/s7

            # 检查 label 下的子目录是否存在，并且是一个目录
            if os.path.isdir(label_s_path):
                # 遍历 label/s7 下的场景目录（如 igv_1114_rain_01-century02）
                for scenario_dir in os.listdir(label_s_path):
                    label_scenario_path = label_s_path/scenario_dir  # label/s7/igv_1114_rain_01-century02
                    sample_lidar_path = sample_s_path/scenario_dir/'lidar'  # sample/s7/igv_1114_rain_01-century02/lidar
                    sample_localization_path=sample_s_path/scenario_dir/'localization'
                    sample_camera_path=sample_s_path/scenario_dir/'camera'
                    extrinsics_path = sample_s_path/scenario_dir/'extrinsics.json' if (sample_s_path/scenario_dir/'extrinsics.json').exists() else None
                    intrinsics_path = sample_s_path/scenario_dir/'intrinsics.json' if (sample_s_path/scenario_dir/'intrinsics.json').exists() else None
                    # 检查 label 下的场景目录label/s7/igv_1114_rain_01-century02是否存在，并且是一个目录
                    if os.path.isdir(label_scenario_path):
                        # 对于每个lidar目录，预先计算好时间，便于二分查找加速
                        sensor_files={}
                        sensor_timestamps={}
                        for name in self.sensor_names:
                            # 支持 .bin 和 .pcd 文件
                            extensions = ['*.bin', '*.pcd']
                            cur_files = [f for ext in extensions for f in (sample_lidar_path / name).glob(ext)]
                            cur_timestamps = precompute_timestamps(cur_files)
                            sensor_files[name] = cur_files
                            sensor_timestamps[name] = cur_timestamps
                        
                        camera_files={}
                        camera_timestamps={}
                        for name in self.camera_names:
                            cur_files=list((sample_camera_path/name).glob('*.jpeg'))
                            cur_timestamps=precompute_timestamps(cur_files)
                            camera_files[name]=cur_files
                            camera_timestamps[name]=cur_timestamps
                        
                        localization_files=list((sample_localization_path).glob('*.json'))
                        localization_times=precompute_timestamps(localization_files)
                        # 遍历 label 场景目录下的 JSON 文件
                        for json_file in os.listdir(label_scenario_path):
                            if json_file.endswith('.json'):
                                sample={}
                                # 构建 JSON 文件的完整路径
                                json_path = label_scenario_path/json_file
                                timestamp=os.path.splitext(json_file)[0]
                                # print(timestamp)
                                matched_lidars=match_multi_sensor_data(timestamp, sensor_files,sensor_timestamps)
                                matched_cameras=match_multi_sensor_data(timestamp, camera_files,camera_timestamps)
                                # matched_localization=Path(match_sensor_data(timestamp, localization_files,localization_times))
                                matched_localization=[]
                                sample['token']=generate_token()
                                sample['label']=json_path
                                sample['lidars']=matched_lidars
                                sample['cameras']=matched_cameras
                                sample['localization']=matched_localization
                                sample['timestamp']=timestamp
                                sample['extrinsics_path']=extrinsics_path
                                sample['intrinsics_path']=intrinsics_path
                                self.samples.append(sample)
                                # 构建对应的 bin 文件名
        aaa=1
    
    def get_all_sample(self):
        return self.samples
    


if __name__ == '__main__':
    kl = KL(version='v1.0-trainval', dataroot='../data/kl', verbose=True)