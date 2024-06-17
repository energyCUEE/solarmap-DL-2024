from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from data_provider.dataloader_regression_CUEE_per_day_h5file import  DatasetCUEE 
from torch.utils.data import DataLoader
import pdb

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom, 
    'CUEE_PMAPS':  DatasetCUEE,
    'CUEE_PMAPS_NIGHT':  DatasetCUEE,
    'solarmap_exp': DatasetCUEE,
    'solarmap': DatasetCUEE,
    'true_cloud_relation': DatasetCUEE,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    if (flag == 'test'):
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    
    elif (flag == 'val'):
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        args.batch_size = 1
        freq = args.freq 

    elif flag == 'train' :

        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    else:
        print("flag can only be set to one of these: 'test' / 'val' / 'pred' / 'train'")
        KeyError

    if (args.data == "CUEE_PMAPS") or (args.data == "CUEE_PMAPS_NIGHT") or (args.data == "solarmap_exp") or (args.data == "solarmap") or (args.data == "true_cloud_relation") :
  
        data_set = Data(
            root_path=args.root_path,
            test_data_path=args.test_data_path,
            valid_data_path=args.valid_data_path,
            train_data_path=args.train_data_path, 
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            train_only=train_only,
            tag=args.data,
            option_Ihat1=args.option_Ihat1
        ) 

    else:

        data_set = Data(
            root_path=args.root_path, 
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            train_only=train_only
        ) 
     
    print("After preprocessing .... [%s] : %d" % (flag, len(data_set)))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    print("===============================================================")
    return data_set, data_loader
