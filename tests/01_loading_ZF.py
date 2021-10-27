from VM2dPLS import io
from datetime import timedelta

if __name__ == '__main__':
    all_measurements = io.load_ZF_data_from_ascii("E:/12_VM2dPLS/03_data/BridgeProfile_sh-l_19.asc",
                                                  "E:/12_VM2dPLS/03_data/BridgeProfile_sh-l_19_meta.txt",
                                                  time_region_of_interest=(timedelta(seconds=120),
                                                                           timedelta(seconds=180)))

    print(1)
