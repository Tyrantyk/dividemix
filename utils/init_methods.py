import torch


def init_w(eval_loader, feat_pid, w, net):
    for batch_idx, (inputs, targets, index, _) in enumerate(eval_loader):
        inputs = inputs.cuda()
        _, _, _, p = net(inputs)
        pp = torch.softmax(p, dim=1)
        pre = torch.argmax(pp, dim=1)
        for i, (idx, p_t) in enumerate(zip(index, pre)):
            feat_pid[idx] = p_t
            w[p_t].append(idx.detach().clone())


def init_proto_anchor(args, labeled_trainloader, feat_pre, w, anchor, model):
    model.eval()
    proto_anchor_label = []
    proto_anchor_idx = []

    # update prototype label
    for i in range(args.num_prototypes):
        if len(w[i]) != 0:
            idx = torch.tensor(w[i]).cuda()
            idx2 = anchor[idx] == True
            if len(torch.where(idx2 == True)[0]) != 0:
                proto_label = torch.mode(feat_pre[idx][idx2])[0]
                # find label and idx of single anchor in every prototypes
                l = feat_pre[idx][idx2]
                for t in range(len(l)):
                    if l[t] == proto_label:
                        proto_anchor_label.append(l[t])
                        proto_anchor_idx.append(torch.arange(50000)[idx][idx2][t])

    proto_anchor_idx = torch.tensor(proto_anchor_idx, dtype=torch.int).cuda()
    proto_anchor_label = torch.tensor(proto_anchor_label, dtype=torch.int64).cuda()
    proto_anchor_feat = torch.randn((len(proto_anchor_idx), 128)).cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, index, clean) in enumerate(labeled_trainloader):
            inputs, targets, clean = inputs.cuda(), targets.cuda(), torch.tensor(clean).clone().detach().cuda()
            feat, z, logits, _ = model(inputs)

            for i in range(len(targets)):
                if index[i] in torch.tensor(proto_anchor_idx).type_as(index):
                    idx = torch.where(proto_anchor_idx == index[i])[0]
                    proto_anchor_feat[idx] = feat[i]

    return proto_anchor_label, proto_anchor_feat
