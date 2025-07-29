
import random
import pickle

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
    with open(save_path /version/ f'kl_infos_train.pkl', 'wb') as f:
        pickle.dump(train_kl_infos, f)
    with open(save_path /version/ f'kl_infos_val.pkl', 'wb') as f:
        pickle.dump(val_kl_infos, f)
    with open(save_path /version/ f'kl_infos_test.pkl', 'wb') as f:
        pickle.dump(test_kl_infos, f)