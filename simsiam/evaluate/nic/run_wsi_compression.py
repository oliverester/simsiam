
import datetime
from pathlib import Path
from PIL import Image
from simsiam.evaluate.eval_utils import get_embeddings
from simsiam.data_provider import DataProvider
from custom_datasets.wsi_dataset.WSIDatasetFolder import WSIDatasetFolder
from simsiam.pretrain.load_pretrained_model import load_pretrained_model
from simsiam.evaluate.nic.nic_parser import nic_parse_config

import numpy as np
import os
import tracking

NIC_LOG_PATH = Path('logdir/nic')

def run_wsi_compression(config_path):
    
    args = nic_parse_config(config_path=config_path,
                               verbose=True)
    
    run_name = f"nic_run" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now() )
    logpath = NIC_LOG_PATH / run_name    
    logpath.mkdir(parents=True, exist_ok=True)
 
    tracking.log_config(logpath, config_path)
     
    args.pretrained = os.path.join(args.model_path, args.checkpoint)
    encoder_model = load_pretrained_model(args)
    
    data_provider = DataProvider(args)
      
    wsidir = args.test_data
    WSIs = WSIDatasetFolder(root_folder=wsidir).get_WSIs()

    # create folder to store compressed wsis
    (logpath / "compressed_wsis").mkdir()
    
    for n, wsi in enumerate(WSIs):
        print(f"Processing WSI {wsi.name}")
        
        wsi_loader = data_provider.get_wsi_loader(wsi=wsi)

        dim_x = wsi.get_metadata('org_n_tiles_col')
        dim_y = wsi.get_metadata('org_n_tiles_row')

        wsi_embeddings, _ = get_embeddings(wsi_loader, encoder_model, args)
            
        ## assemble wsi-embedding - concider physical order and missing patches due to background
        #construct target array for compressed image
        compression = np.zeros(shape=(dim_x, dim_y, wsi_embeddings[0].shape[0]))
        thumbnail = np.zeros(shape=(dim_x, dim_y))
        
        for i, patch in enumerate(wsi.patches):
            x = patch.x
            y = patch.y
            emb = wsi_embeddings[i]
            compression[x,y,:] = emb
            thumbnail[x,y] = 1
        
        # Creates PIL image
        img = Image.fromarray(np.uint8(thumbnail * 255) , 'L')
        img.save(logpath / "compressed_wsis" / f"thumbnail_{wsi.name}.jpg", subsampling=0, quality=100)
        np.save(logpath / "compressed_wsis" / f"{wsi.name}_compression.npy", compression)
        
    
        