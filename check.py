import pickle

if __name__ == "__main__":
    with open("/home/zhao/workspace/VAD/data/nuscenes/vad_nuscenes_infos_temporal_val.pkl", 'rb') as f:
        pick = pickle.load(f)
    print(pick['infos'][0]['cams'])