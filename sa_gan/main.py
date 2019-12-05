from parameters import *
from trainer import Trainer
# from tester import Tester
from torch.utils.data import Dataset, DataLoader
from custom_dataset import *
from torch.backends import cudnn
from utils import make_folder

def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader
    dataset = JaffeDataset(data_dir = 'jaffe', labels_path = 'labels.csv')
    data_loader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers= 4)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        trainer = Trainer(data_loader, config)
        trainer.train()
    else:
        tester = Tester(data_loader, config)
        tester.test()

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)
