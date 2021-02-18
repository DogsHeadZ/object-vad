import torch
import yaml
import argparse
#
# class convAE(torch.nn.Module):
#
#     def __init__(self, n_channel =3,  t_length = 5, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1):
#         super(convAE, self).__init__()
#         print(n_channel,t_length, memory_size,feature_dim, key_dim,temp_update)
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--config')
#
# args = parser.parse_args()
# config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
#
# train_dataset_args = config['train_dataset_args']
#
# model = convAE(train_dataset_args['c'], train_dataset_args['t_length'], **config['model_args'])



anomaly_score_total_list = []
anomaly_score_total_list += [1,2]
anomaly_score_total_list += [4,5]
print(anomaly_score_total_list)