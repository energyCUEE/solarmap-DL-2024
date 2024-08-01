from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from data_provider.dataloader_cuee import  DatasetCUEE  
import pdb

data_dict = {
    'true_cloud_relation_08JUL24': DatasetCUEE,
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

    if   (args.data == "true_cloud_relation_08JUL24"):
  
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
            option_Ihat1=args.option_Ihat1,
            is_noscaley= args.is_noscaley,
            is_noscalex= args.is_noscalex
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
