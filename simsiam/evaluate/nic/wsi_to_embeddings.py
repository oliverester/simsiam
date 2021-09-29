

from simsiam.evaluate.eval_utils import get_embeddings
from simsiam.data_provider import DataProvider
from custom_datasets.wsi_dataset.WSIDatasetFolder import WSIDatasetFolder
from simsiam.pretrain.load_pretrained_model import load_pretrained_model
from simsiam.evaluate.nic.nic_parser import nic_parse_config
import os
import tracking


def wsi_to_embeddings(config_path):
    
    args = nic_parse_config(config_path=config_path,
                               verbose=True)
    tracking.log_config(args.model_path, config_path)
     
    args.pretrained = os.path.join(args.model_path, args.checkpoint)
    encoder_model = load_pretrained_model(args)
    
    data_provider = DataProvider(args)
    wsi_loaders = data_provider.get_wsi_loaders()
    
    for wsi_loader in wsi_loaders:
        
          wsi_embeddings = get_embeddings(wsi_loader, encoder_model, args)
          
          ## assemble wsi-embedding - concider physical order and missing patches due to background
        
        
        
        