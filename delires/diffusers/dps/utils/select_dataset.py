def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    # -----------------------------------------
    # common
    # -----------------------------------------
    if dataset_type in ['h']:
        from data.dataset_H import DatasetH as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
