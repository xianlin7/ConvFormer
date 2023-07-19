from matplotlib.cbook import simple_linear_interpolation
import numpy as np
import torch

def get_records(ftokens, attmaps, layer=12, num_head=12, smooth=1e-5):
    #num_head = attmaps[0].shape[1]
    #layer = len(attmaps)
    rtoken1, rtoken2, rtoken3 = np.zeros((1, layer+1)), np.zeros((1, layer+1)), np.zeros((1, layer+1))
    rmap1, rmap2, rmap3 = np.zeros((1, layer*num_head)), np.zeros((1, layer*num_head)), np.zeros((1, layer*num_head))
    for i in range(layer+1):
        smi = torch.cosine_similarity(ftokens[layer], ftokens[i], dim=-1)
        smi_mean = torch.mean(smi)
        rtoken1[0, i] = np.array(smi_mean.cpu())
        # ---------------
        topk, _ = torch.sort(smi, descending=False)
        topk = topk[:, :256]
        topk_mean = torch.mean(topk)
        rtoken2[0, i] = np.array(topk_mean.cpu())
        # ---------------
        mid_token = torch.mean(ftokens[i], dim=1) # b n d -> b d
        mid_token = torch.unsqueeze(mid_token, 1) # b 1 d
        smi_iner = torch.cosine_similarity(ftokens[i], mid_token, dim=-1)
        smi_iner = torch.mean(smi_iner)
        rtoken3[0, i] = np.array(smi_iner.cpu())
        # ---------- the characteristic of the attention map -------------
        if i < layer:
            attni = attmaps[i]
            _, num_head, n, _ = attni.shape
            for j in range(num_head):
                attnij = attni[0, j, :, :] 
                log_attnij = torch.log2(attnij + smooth)
                entropy = -1 * torch.sum(attnij * log_attnij, dim=-1) / torch.log2(torch.tensor(n*1.0)) # n
                #print(i, " ", j , " ", i*layer + j)
                rmap1[0, i*num_head + j] = np.array(torch.mean(entropy).cpu())
                # ------------------------
                if i == 0:
                    smi_head2 = torch.cosine_similarity(attni[:, j, :, :], attmaps[i+1][:, j, :, :], dim=-1)
                    smi_head2 = torch.mean(smi_head2)
                    rmap2[0, i*num_head + j] = np.array(smi_head2.cpu())
                elif i < layer - 1:
                    smi_head1 = torch.cosine_similarity(attni[:, j, :, :], attmaps[i-1][:, j, :, :], dim=-1)
                    smi_head1 = torch.mean(smi_head1)
                    smi_head2 = torch.cosine_similarity(attni[:, j, :, :], attmaps[i+1][:, j, :, :], dim=-1)
                    smi_head2 = torch.mean(smi_head2)
                    rmap2[0, i*num_head + j] = np.array(smi_head1.cpu() + smi_head2.cpu())
                else:
                    smi_head1 = torch.cosine_similarity(attni[:, j, :, :], attmaps[i-1][:, j, :, :], dim=-1)
                    smi_head1 = torch.mean(smi_head1)
                    rmap2[0, i*num_head + j] = np.array(smi_head1.cpu())
                # --------------------
                smi_head = torch.cosine_similarity(attni, torch.unsqueeze(attni[:, j, :, :], 1), dim=-1)
                smi_head = torch.mean(smi_head)
                rmap3[0, i*num_head + j] = np.array(smi_head.cpu())
    return rtoken1, rtoken2, rtoken3, rmap1, rmap2, rmap3


def get_records2(ftokens, attmaps, layer=12, num_head=12, smooth=1e-5):
    num_head = attmaps[0].shape[1]
    layer = len(attmaps)
    #print(num_head, layer)
    rtoken1, rtoken2, rtoken3 = np.zeros((1, layer+1)), np.zeros((1, layer+1)), np.zeros((1, layer+1))
    rmap1, rmap2, rmap3 = np.zeros((1, layer*num_head)), np.zeros((1, layer*num_head)), np.zeros((1, layer*num_head))
    for i in range(layer+1):
        #smi = torch.cosine_similarity(ftokens[layer], ftokens[i], dim=-1)
        smi = torch.cosine_similarity(ftokens[0], ftokens[0], dim=-1)
        smi_mean = torch.mean(smi)
        rtoken1[0, i] = np.array(smi_mean.cpu())
        # ---------------
        topk, _ = torch.sort(smi, descending=False)
        topk = topk[:, :256]
        topk_mean = torch.mean(topk)
        rtoken2[0, i] = np.array(topk_mean.cpu())
        # ---------------
        mid_token = torch.mean(ftokens[i], dim=1) # b n d -> b d
        mid_token = torch.unsqueeze(mid_token, 1) # b 1 d
        n = ftokens[i].shape[1]
        sim = 0
        for k in range(n):
            tokenk = ftokens[i]
            tokenk = tokenk[:, k, :] # b n d -> b d
            tokenk = torch.unsqueeze(tokenk, 1) # b 1 d
            smi_iner = torch.cosine_similarity(ftokens[i], tokenk, dim=-1)
            smi_iner = (torch.mean(smi_iner) * n -1)/(n-1)
            sim = sim + smi_iner
        sim = sim/n
        rtoken3[0, i] = np.array(sim.cpu())
        # ---------- the characteristic of the attention map -------------
        if i < layer:
            attni = attmaps[i]
            _, num_head, n, _ = attni.shape
            for j in range(num_head):
                attnij = attni[0, j, :, :] 
                log_attnij = torch.log2(attnij + smooth)
                entropy = -1 * torch.sum(attnij * log_attnij, dim=-1) / torch.log2(torch.tensor(n*1.0)) # n
                #print(i, " ", j , " ", i*layer + j)
                rmap1[0, i*num_head + j] = np.array(torch.mean(entropy).cpu())
                # ------------------------
                if i == 0:
                    smi_head2 = torch.cosine_similarity(attni[:, j, :, :], attmaps[i+1][:, j, :, :], dim=-1)
                    smi_head2 = torch.mean(smi_head2)
                    rmap2[0, i*num_head + j] = np.array(smi_head2.cpu())
                elif i < layer - 1:
                    smi_head1 = torch.cosine_similarity(attni[:, j, :, :], attmaps[i-1][:, j, :, :], dim=-1)
                    smi_head1 = torch.mean(smi_head1)
                    smi_head2 = torch.cosine_similarity(attni[:, j, :, :], attmaps[i+1][:, j, :, :], dim=-1)
                    smi_head2 = torch.mean(smi_head2)
                    rmap2[0, i*num_head + j] = (np.array(smi_head1.cpu() + smi_head2.cpu()))/2
                else:
                    smi_head1 = torch.cosine_similarity(attni[:, j, :, :], attmaps[i-1][:, j, :, :], dim=-1)
                    smi_head1 = torch.mean(smi_head1)
                    rmap2[0, i*num_head + j] = np.array(smi_head1.cpu())
                # --------------------
                smi_head = torch.cosine_similarity(attni, torch.unsqueeze(attni[:, j, :, :], 1), dim=-1)
                smi_head = (torch.mean(smi_head) * num_head -1 )/(num_head-1)
                rmap3[0, i*num_head + j] = np.array(smi_head.cpu())
    return rtoken1, rtoken2, rtoken3, rmap1, rmap2, rmap3