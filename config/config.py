DATA = {
    'num_class': 40,
    'root': "/HDD1/mvpservereight/minhyeok/PointCloud/modelnet40_normal_resampled/"
}

SETTING = {
    'num_point': 1024,
    'normal': False
}

TRAIN = {
    'model': "DGCNN", #POINTNET, DGCNN
    'batch_size': 25,
    'learning_rate': 0.001,
    'decay_rate': 1e-4,
    'epoch': 200,
    'print_freq': 50
}