
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from utils.utils import estimator, get_cosine_schedule_with_warmup, create_model
from utils.ema import EMA
from utils.losses import TruncatedLoss
from utils.init_methods import *
import dataloader.dataloader_cifar as dataloader
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=5, type=float, help='weight for unsupervised loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=600, type=int)
parser.add_argument('--r', default=0.8, type=float, help='noise ratio')
parser.add_argument('--c_smooth', type=float, default=0.9)
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup epochs (unlabeled data based)')
parser.add_argument('--seed', default=1)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument("--proj_output_dim", type=int, default=128)
parser.add_argument("--proj_hidden_dim", type=int, default=2048)
parser.add_argument("--num_prototypes", type=int, default=3000)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
device = torch.device('cuda', args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(args, epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader, scheduler, ema_model, GCEloss, label_bank, feat_bank):
    net.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1

    # update gce loss weights
    if epoch >= 60 and epoch % 10 == 0:
        net.eval()
        for batch_idx, (inputs_x, labels_x, w_x, labeled_index) in enumerate(labeled_trainloader):
            inputs_x, labels_x, w_x = inputs_x.cuda(), labels_x.cuda(), w_x.cuda()
            feats, z, logits, _  = net(inputs_x)
            GCEloss.update_weight(logits, labels_x, labeled_index)
        net.train()

    for batch_idx, (inputs_x, labels_x, w_x, labeled_index) in enumerate(labeled_trainloader):
        try:
            inputs_u_w, inputs_u_s, index = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u_w, inputs_u_s, index = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)
        

        inputs_x, labels_x, w_x = inputs_x.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u_w, inputs_u_s = inputs_u_w.cuda(), inputs_u_s.cuda()

        inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s])
        feats, z, logits, _ = net(inputs)
        logits_x = logits[:batch_size,:]
        logits_w, logits_s = logits[batch_size:,:].chunk(2)
        feats_w, feats_s = feats[batch_size:,:].chunk(2)
        z_w, z_s = z[batch_size:,:].chunk(2)

        # cla loss
        class_loss = F.cross_entropy(logits_x, labels_x)

        # GCE loss
        gce_loss = GCEloss(logits_x, labels_x, labeled_index)

        # # simmatch
        # ori_pseudo_label = torch.softmax(logits_w.detach(), dim=-1)
        # with torch.no_grad():
        #     distance_logits = z_w @ feat_bank.t()
        #     distance_prob = F.softmax(distance_logits / 0.1, dim=1)
        #     bs = distance_prob.size(0)
        #     aggregated_prob = torch.zeros([bs, args.num_class]).cuda()
        #     aggregated_prob = aggregated_prob.scatter_add(1, label_bank.expand([bs, -1]), distance_prob)
        #     pseudo_label = ori_pseudo_label * args.c_smooth + aggregated_prob * (1 - args.c_smooth)

        # doublematch loss
        Ls = torch.mean(
            -torch.sum(F.normalize(z_s, p=2, dim=1) * F.normalize(feats_w.detach(), p=2, dim=1), dim=1) + 1)

        # fixmatch loss
        pseudo_label = torch.softmax(logits_w.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(0.95).float()

        Lp = (F.cross_entropy(logits_s, targets_u,
                              reduction='none') * mask).mean()

        # regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits_w, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        # loss = class_loss + 5 * Ls  + Lp + penalty
        loss = gce_loss + Lp

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema_model.update(net)
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t cla loss: %.2f  Lp loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, class_loss.item(), Lp.item()))
        sys.stdout.flush()

        feat_pre[index] = torch.tensor(torch.argmax(torch.softmax(logits_w, dim=1), dim=1),
                                       dtype=torch.int).detach().clone()

def warmup(epoch, net, optimizer, dataloader, all_loss, warm_up):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    losses = torch.zeros(50000)
    for batch_idx, (inputs, labels, path, clean) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        feat, z, logits, _ = net(inputs, grad=False)
        feat_pre[path] = torch.argmax(torch.softmax(logits, dim=1), dim=1).int().detach().clone()

        loss = CEloss(logits, labels)
        with torch.no_grad():
            loss_b = CE(logits, labels)
        for b in range(inputs.size(0)):
            losses[path[b]] = loss_b[b]
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(logits)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    if epoch == warm_up-1:
        history = torch.stack(all_loss)
        input_loss = history.mean(0)
        input_loss = input_loss.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]

        anchor[prob > 0.9] = True

        return prob
    return 0


def test(epoch, net1, optimizer):
    net1.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, _, logits, _ = net1(inputs)
            _, predicted = torch.max(logits, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100.*correct/total
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print("\n| Test Epoch #%d\t Accuracy: %.2f  lr:%.6f\n" %(epoch, acc, lr))
    test_log.write('Epoch:%d   Accuracy:%.2f   lr:%.6f\n'%(epoch, acc, lr))
    test_log.flush()


def ema_test(epoch, net1, ema):
    net1.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        ema.apply_shadow(net1)
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, _, logits, _ = net1(inputs)
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
        ema.restore(net1)
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t ema_Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write('Epoch:%d   ema_Accuracy:%.2f\n' % (epoch, acc))
    test_log.flush()

def eval_train(model, w, gmm_prob):
    model.eval()
    gmm_prob = torch.tensor(gmm_prob).cuda()

    # update prototype label
    for i in range(args.num_prototypes):
        if len(w[i]) != 0:
            idx = torch.tensor(w[i]).cuda()
            idx2 = anchor[idx] == True
            if len(torch.where(idx2 == True)[0]) == 0:
                proto_label[i] = -1
            else:
                proto_label[i] = torch.mode(feat_pre[idx][idx2])[0]

    with torch.no_grad():
        pre = torch.zeros((1, 50000), dtype=torch.bool).squeeze(0).cuda()
        w_all = torch.randn((1, 50000), dtype=torch.float32).squeeze(0).cuda()
        w_recall = []
        w_acc = []
        for batch_idx, (inputs, targets, index, clean) in enumerate(eval_loader):
            inputs, targets, clean = inputs.cuda(), targets.cuda(), torch.tensor(clean).clone().detach().cuda()
            feat, z, logits, _ = model(inputs)

            # estimate
            w = estimator(index, logits, targets, proto_label, feat_pid, k)
            w[gmm_prob[index] < 0.3] = 0
            t = w == 1
            # t[gmm_prob[index] < 0.3] = False

            if t.sum() !=  0:
                w_recall.append((t[t] == (targets[t] == clean[t])).sum() / t.sum())
                # right = clean[t][t[t] != (targets[t] == clean[t])]
                # wrong = targets[t][t[t] != (targets[t] == clean[t])]
                # for r in range(len(right)):
                #     confuse[right[r]][wrong[r]] += 1
                #     confuse[wrong[r]][right[r]] += 1

            w_acc.append(((t == (targets == clean)) == True).sum() / len(t))
            pre[index] = w == 1
            w_all[index] = w
        print('estimator precision', torch.mean(torch.tensor(w_recall)) * 100)
        print('estimator acc', torch.mean(torch.tensor(w_acc)) * 100)


    return np.array(w_all.cpu()), np.array(pre.cpu())


stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w')
test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')

if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=6,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net = create_model(args, device)
ema = EMA(net, 0.999)
cudnn.benchmark = True

no_decay = ['bias', 'bn']
grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 5e-3},
        {'params': [p for n, p in net.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, args.warmup, 48000)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()


# if args.noise_mode=='asym':
#     conf_penalty = NegEntropy()

# init
k = 3
feat_pid = torch.ones((1, 50000), dtype=torch.int).squeeze(0).cuda()
proto_label = torch.ones((1, args.num_prototypes), dtype=torch.int).squeeze(0).cuda()
feat_pre = torch.zeros((1, 50000), dtype=torch.int).squeeze(0).cuda()
anchor = torch.zeros((1, 50000), dtype=torch.bool).squeeze(0).cuda()


w = [[] for i in range(args.num_prototypes)]
all_loss = []  # save the history of losses from two networks
gmm_prob = []

# init prototype
eval_loader = loader.run('eval_train')

if __name__=='__main__':
    init_w(eval_loader, feat_pid, w, net)

    for epoch in range(args.num_epochs+1):
        lr=args.lr
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        test_loader = loader.run('test')
        eval_loader = loader.run('eval_train')

        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net')
            pt = warmup(epoch, net, optimizer, warmup_trainloader, all_loss, warm_up)
            gmm_prob.append(pt)

        else:
            prob, pred = eval_train(net, w, gmm_prob[-1])

            print('Train Net')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred, prob)

            if epoch == warm_up:
                GCEloss = TruncatedLoss().cuda()
                proto_anchor_label, proto_anchor_feat = init_proto_anchor(args, labeled_trainloader, feat_pre, w, anchor, net)
                ema.register()
            train(args, epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader, scheduler, \
                  ema, GCEloss, proto_anchor_label, proto_anchor_feat)  # train net

        test(epoch, net, optimizer)
        ema_test(epoch, net, ema)


