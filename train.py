import torch
import os
import config.config as config
from dataset.ModelNetDataLoader import ModelNetDataLoader
from models.pointnet import PointNet
from models.dgcnn import DGCNN
from loss import loss_for_pointnet, loss_for_dgcnn
import utils
import numpy as np

def train(epoch, trainloader, optimizer, model, device):
    model.train()

    if config.TRAIN['model'] == "POINTNET":
        criterion = loss_for_pointnet().to(device)

    avg_loss = 0
    avg_acc = 0

    for idx, batch in enumerate(trainloader):
        points, target = batch
        points = points.data.numpy()
        points = utils.random_point_dropout(points)
        points[:,:, 0:3] = utils.random_scale_point_cloud(points[:,:, 0:3])
        points[:,:, 0:3] = utils.shift_point_cloud(points[:,:, 0:3])
        points = torch.Tensor(points)
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()

        if config.TRAIN['model'] == "POINTNET":
            pred, trans_feat = model(points)
            loss = criterion(pred, target.long(), trans_feat)
        
        elif config.TRAIN['model'] == "DGCNN":
            pred = model(points)
            loss = loss_for_dgcnn(pred, target.long())

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        
        avg_acc += correct.item() / float(points.size()[0])
        avg_loss += loss.item()

        loss.backward()
        optimizer.step()

        if (idx % config.TRAIN['print_freq'] == 0) and (idx > 0):
            avg_loss = avg_loss / config.TRAIN['print_freq']
            avg_acc = avg_acc / config.TRAIN['print_freq']

            print(
                "Epoch: #{0} Batch: {1}\t"
                "Lr: {lr:.6f}\t"
                "LOSS: {loss:.4f}\t"
                "ACC: {acc:.4f}\t"
                .format(epoch, idx, lr=optimizer.param_groups[-1]['lr'], \
                    loss=avg_loss, acc=avg_acc)
            )

            avg_loss = 0
            avg_acc = 0


def valid(testloader, model, device):
    model.eval()

    mean_correct = []
    class_acc = np.zeros((config.DATA['num_class'], 3))

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            points, target = batch
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)

            if config.TRAIN['model'] == "POINTNET":
                pred, _ = model(points)
            
            elif config.TRAIN['model'] == "DGCNN":
                pred = model(points)
                
            pred_choice = pred.data.max(1)[1]

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
                class_acc[cat,0] += classacc.item()/float(points[target==cat].size()[0])
                class_acc[cat,1] += 1
            
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item()/float(points.size()[0]))

        class_acc[:,2] =  class_acc[:,0] / class_acc[:,1]
        class_acc = np.mean(class_acc[:,2])
        instance_acc = np.mean(mean_correct)

        print("TEST CLASS ACC :", str(class_acc))
        print("TEST instance ACC :", str(instance_acc))
        
        return class_acc

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("Load dataset...")
    DATA_PATH = config.DATA['root']

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=config.SETTING['num_point'], split='train', normal_channel=config.SETTING['normal'])
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=config.SETTING['num_point'], split='test', normal_channel=config.SETTING['normal'])
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=config.TRAIN['batch_size'], shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=config.TRAIN['batch_size'], shuffle=False, num_workers=4)
    print("ok!")

    print("Check device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("ok!")

    print("Load model...")
    if config.TRAIN['model'] == "POINTNET":
        model = PointNet(config.DATA['num_class'], normal_channel=config.SETTING['normal'])
    elif config.TRAIN['model'] == "DGCNN":
        model = DGCNN(config.DATA['num_class'])
    model = model.to(device)
    print("ok!")

    print("Load optimizer...")
    optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.TRAIN['learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=config.TRAIN['decay_rate']
                )
    print("ok!")

    # Starting training 
    print("Starting training... ")

    best = 1
    
    for epoch in range(config.TRAIN['epoch']):
        train(epoch, trainDataLoader, optimizer, model, device)
        acc = valid(testDataLoader, model, device)

        # if rmse < best:
        #     torch.save({
        #         "epoch": epoch, 
        #         "model_state_dict": model.state_dict(),
        #         "optim_state_dict": optimizer.state_dict()
        #     }, "ckpt_{}.pth".format(epoch))

        #     best = rmse


if __name__ == "__main__":
    main()