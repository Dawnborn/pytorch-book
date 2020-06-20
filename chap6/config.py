'''
在模型定义、数据处理和训练等过程都有很多变量，
这些变量应提供默认值，并统一放置在配置文件中，
这样在后期调试、修改代码或迁移程序时会比较方便，
在这里我们将所有可配置项放在config.py中。

    import models
    from config import DefaultConfig
    
    opt = DefaultConfig()
    lr = opt.lr
    model = getattr(models, opt.model)
    dataset = DogCat(opt.train_data_root)

修改参数
    opt = DefaultConfig()
    new_config = {'lr':0.1,'use_gpu':False}
    opt.parse(new_config)
    opt.lr == 0.1
'''

import warnings
import torch as t

class DefaultConfig(object):
    env = 'default'
    model = 'AlexNet'
    env = 'default' # visdom 环境
    vis_port =8097 # visdom 端口
    model = 'AlexNet' # 使用的模型，名字必须与models/__init__.py中的名字一致
   
    train_data_root = './data/train/' # 训练集存放路径
    test_data_root = './data/test1/' # 测试集存放路径
    load_model_path = None #'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128 # batch size
    use_gpu = False #True # use GPU or not
    num_workers = 4 # how many workers for loading data
    print_freq = 20 # print info every N batch

    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
      
    max_epoch = 10
    lr = 0.1 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数
    
    def _parse(self, kwargs):
        '''
        根据字典kwargs更新config参数
        '''
        # 更新参数
        for k, v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self, k, v)
        
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')
        
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
        
opt = DefaultConfig()