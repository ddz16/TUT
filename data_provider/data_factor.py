from data_provider.data_loader import my_collate_func, Dataset_food, Dataset_toy
from torch.utils.data import DataLoader


def data_provider(args, mode):

    if mode == 'test' or mode == 'val':
        shuffle_flag = False
        batch_size = 1
    elif mode == 'train':
        shuffle_flag = True
        batch_size = args.bz
    
    if args.dataset in ['gtea', '50salads', 'breakfast']:
        data_set = Dataset_food(
            root=args.root_path,
            dataset=args.dataset,
            split=args.split,
            mode=mode,
        )
    elif args.dataset == 'assembly':
        data_set = Dataset_toy(
            root=args.root_path,
            dataset=args.dataset,
            split=args.split,
            mode=mode,
        )
    else:
        raise RuntimeError("dataset name must be in [gtea, 50salads, breakfast, assembly]! please check it!")

    print(mode, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=my_collate_func
    )

    return data_set, data_loader