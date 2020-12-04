import torch
import torch.nn.functional as F
from dataloader import get_dataloader
from model import SimpleModel
from tqdm import tqdm
import argparse
import os
import random
import numpy as np

def val(args, val_dataloader, model):
    model.eval()
    result = []
    for imgs, label in tqdm(val_dataloader, ncols=80):
        imgs = imgs.cuda()
        with torch.no_grad():
            pred = model(imgs)
            pred_id = torch.argmax(pred, 1).cpu()
        result += (label == pred_id).long().tolist()
    print("Acc: ", sum(result) / len(result))
          
def train(args, train_dataloader, val_dataloader, 
          model, optimizer):
    for epoch in range(args.max_epoch):
        model.train()
        tr_loss, num_tr_steps = 0, 0
        for imgs, label in tqdm(train_dataloader, ncols=80):
            imgs = imgs.cuda()
            label = label.cuda()
            pred = model(imgs)
            loss = F.cross_entropy(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.cpu().item()
            num_tr_steps += 1
            
        print('===== Epoch %d done.' % (epoch+1))
        print('===== Average training loss', tr_loss / num_tr_steps)
        
        # Evaluation
        val(args, val_dataloader, model)
        
        # Save model
        torch.save({'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state' : optimizer.state_dict()},
                    os.path.join(args.checkpoints_dir, 'last_ckpt.pth'))

def main(args):
    seed = 2020
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Make logdir
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    
    # Load dataset
    train_dataloader = get_dataloader('train', args.bs, True, args.nw)
    val_dataloader = get_dataloader('val', args.bs, False, args.nw)
    
    # Model
    model = SimpleModel()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.wd)

    model.cuda()
    
    train(args, train_dataloader, val_dataloader, 
          model, optimizer)
    
    
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
    parser.add_argument("--lr",
                        default=0.05,
                        type=float,
                        help="learning rate"
    )
    parser.add_argument("--wd",
                        default=5e-4,
                        type=float,
                        help="weight decay"
    )
    parser.add_argument("--max_epoch",
                        default=30,
                        type=int,
                        help="max epoch"
    )
    parser.add_argument("--checkpoints_dir",
                        default="logdir",
                        type=str,
                        help="checkpoints dir"
    )
    args = parser.parse_args()
    
    main(args)