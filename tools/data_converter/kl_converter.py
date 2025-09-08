import random
import pickle
from .kl import KL

def split_samples(samples):
    total_files = len(samples)
    train_size = int(total_files * 0.95)
    val_size = int(total_files * 0.05)
    split_samples = {
        'train': samples[:train_size],
        'val': samples[train_size:train_size + val_size],
        'test': samples[train_size + val_size:]
    }
    train_samples=split_samples['train']
    val_scenes=split_samples['val']
    test_scenes=split_samples['test']
    return train_samples,val_scenes,test_scenes
    # return train_samples,val_scenes

def create_kl_infos(version, data_path, save_path,with_cam=False):
    from . import kl_dataset_utils
    kl = KL(version=version, dataroot=data_path, verbose=True)
    samples=kl.get_all_sample()
    random.shuffle(samples)

    train_samples,val_samples,test_samples=split_samples(samples)
    train_samples = {d['token'] for d in train_samples}
    val_samples = {d['token'] for d in val_samples}
    test_samples = {d['token'] for d in test_samples}
    
    train_kl_infos,val_kl_infos,test_kl_infos=kl_dataset_utils.fill_trainval_infos(kl,train_samples,val_samples,test_samples)

    print('train sample: %d, val sample: %d, test sample: %d' % (len(train_kl_infos), len(val_kl_infos),len(test_kl_infos)))
    with open(save_path / f'kl_infos_train.pkl', 'wb') as f:
        pickle.dump(train_kl_infos, f)
    with open(save_path / f'kl_infos_val.pkl', 'wb') as f:
        pickle.dump(val_kl_infos, f)
    with open(save_path / f'kl_infos_test.pkl', 'wb') as f:
        pickle.dump(test_kl_infos, f)
    # with open(save_path /version/ f'kl_infos_train.pkl', 'wb') as f:
    #     pickle.dump(train_kl_infos, f)
    # with open(save_path /version/ f'kl_infos_val.pkl', 'wb') as f:
    #     pickle.dump(val_kl_infos, f)
    # with open(save_path /version/ f'kl_infos_test.pkl', 'wb') as f:
    #     pickle.dump(test_kl_infos, f)