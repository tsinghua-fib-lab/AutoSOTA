

def default_conv_config(in_channels, out_channels, local_grad=True, diversity_mode='feature', consistency_mode='feature', conv_type='standard', sampling_len=20, **kwargs):
    config = {
        'in_channels': in_channels,
        'out_channels': out_channels,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'num_classes': 10,
        'sampling_len': sampling_len,
        'dropout': 0.2,
        'pre_backward': False,
        'diversity_mode': diversity_mode,
        'consistency_mode': consistency_mode,
        'consistency_factor': 0.5,
        'diversity_factor': 0.5,
        'conv_type': conv_type,
        'padding_mode': 'reflect',
        'pooling': 'MaxPool2d',
        'pooling_config': {
            'kernel_size': 2,
            'stride': 2,
        },
        'local_grad': local_grad,
        'projecting': False,
    }
    config.update(kwargs)
    return config

def default_global_config(diversity_mode='feature', consistency_mode='feature', sampling_len=20, dropout=0.2, **kwargs):
    config = {
        'diversity_mode': diversity_mode,
        'consistency_mode': consistency_mode,
        'sampling_len': sampling_len,
        "dropout": dropout,
    }
    config.update(kwargs)
    return config

def default_network_config(**kwargs):
    global_config = default_global_config(**kwargs)
    conv1 = default_conv_config(3, 96, **global_config)
    conv1['kernel_size'] = 5
    conv1['padding'] = 2
    conv1['pooling_config'] = {
        'kernel_size': 4,
        'stride': 2,
        'padding': 1,
    }
    conv1['is_first'] = True
    
    conv2 = default_conv_config(96, 384, **global_config)
    conv2['pooling_config'] = {
        'kernel_size': 4,
        'stride': 2,
        'padding': 1,
    }
    
    conv3 = default_conv_config(384, 1536,  **global_config)
    conv3['pooling'] = 'AvgPool2d'
    conv3['pooling_config'] = {
        'kernel_size': 2,
        'stride': 2,
        'padding': 0,
    }
    conv3['is_last'] = True
    net_config = {
        'conv_config': [conv1, conv2, conv3],
        'linear_drop': 0.5,
        'linear_dims': 24576,
        'num_classes': 10,
    }
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'
    net_config.update(global_config)
    return net_config

