import torch
import torch.nn.functional as F
from dataloader import get_dataloader
from model import SimpleModel
from tqdm import tqdm
import pandas as pd
import os
import argparse

"""
Make out.csv to submit
"""

def test(args, test_dataloader, model):
    model.eval()
    result = []
    for imgs, label in tqdm(test_dataloader, ncols=80):
        imgs = imgs.cuda()
        with torch.no_grad():
            pred = model(imgs)
            pred_id = torch.argmax(pred, 1).cpu()
        result += pred_id.tolist()
    return result        
    
def main(args):
    # Load dataset
    test_dataloader = get_dataloader('test', args.bs, False, args.nw)
    
    # Model
    model = SimpleModel()
    model.cuda()
    ckpt = torch.load(os.path.join(args.checkpoints_dir, 'last_ckpt.pth'))
    model.load_state_dict(ckpt['model_state'])
    
    result = test(args, test_dataloader, model)
    
    # Make csv file
    df = pd.DataFrame({'id': test_dataloader.dataset.ids,
                       'category': result})
    df.to_csv('out.csv', index=False)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments
    parser.add_argument("--bs",
                        default=64,
                        type=int,
                        help="Batch size for (positive) training examples."
    )
    parser.add_argument("--nw",
                        default=2,
                        type=int,
                        help="num worker"
    )
    parser.add_argument("--checkpoints_dir",
                        default="logdir",
                        type=str,
                        help="checkpoints dir"
    )
    args = parser.parse_args()
    
    main(args)