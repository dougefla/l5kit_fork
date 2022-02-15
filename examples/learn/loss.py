from torch import nn

def MSELoss(pred, target_pos, target_aval):
    res = 0
    num =0
    for i in range(len(pred)):
        if target_aval[int(i/2)] ==1 :
            res += (pred[i]-target_pos[i])**2
            num+=1
    if num==0:
        return 0
    else:
        return res/num