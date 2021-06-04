import scipy.io as io
import numpy as np
import os

def GetOverlap(Proposals, GTInsts):
    NumInsts = len(GTInsts)
    NumProposals = len(Proposals)

    overlap = np.zeros((NumInsts, NumProposals))
    for i in range(NumProposals):
        Proposal = Proposals[i]
        for j in range(NumInsts):
            GTInst = GTInsts[j]
            overlap[j, i] = np.sum(np.logical_and(Proposal, GTInst)) / np.sum(np.logical_or(Proposal, GTInst))
    return overlap, NumInsts

def OverlapsToLabels(Scores, Overlaps, Thresh):
    NumInsts = Overlaps.shape[0]
    NumDets = Overlaps.shape[1]
    Covered = np.zeros((NumInsts, 1))
    Labels = np.zeros((NumDets, 1))
    SortIndex = np.argsort(-Scores, axis=0)
    for k in range(len(SortIndex)):
        if (np.sum(Covered) == len(Covered)):
            break
        Idx = np.where(Covered == 0)[0]
        Overlap = np.max(Overlaps[Idx, SortIndex[k]])
        AssignID = np.where(Overlaps[Idx, SortIndex[k]] == Overlap)[0]
        if Overlap > Thresh:
            Labels[SortIndex[k], 0] = 1
            Covered[Idx[AssignID], 0] = 1
    return Labels

def VOCap(rec, prec):
    mrec = np.insert(rec, 0, 0)
    mrec = np.append(mrec, 1)
    mpre = np.insert(prec, 0, 0)
    mpre = np.append(mpre, 0)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = np.maximum(mpre[i], mpre[i+1])
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i])
    return ap

def CalAP(Scores, Labels, NumGTs):
    SortID = np.argsort(-Scores, axis=0)
    tp = Labels[SortID]
    fp = 1 - Labels[SortID]
    tp = tp.cumsum()
    fp = fp.cumsum()
    prec = tp / (tp + fp)
    rec = tp / NumGTs
    ap = VOCap(rec, prec)
    return ap

def GetAP(PredMaskScores, OverlapAll, Threshold, NumInsts):
    NumGTs = 0
    NumImgs = len(PredMaskScores)
    TotalDetectedProps = 0
    for i in range(NumImgs):
        TotalDetectedProps += len(PredMaskScores[i])
    Scores = np.zeros((TotalDetectedProps, 1))
    Labels = np.zeros((TotalDetectedProps, 1))
    AllDets = 0
    for k in range(NumImgs):
        NumDets = len(PredMaskScores[k])
        Scores[AllDets: AllDets+NumDets, 0] = PredMaskScores[k]
        NumGTs += NumInsts[k]
        Overlaps = OverlapAll[k]
        if Overlaps == []:
            continue
        Labels[AllDets: AllDets+NumDets] = OverlapsToLabels(Scores[AllDets: AllDets+NumDets, 0], Overlaps, Threshold)
        AllDets += NumDets
    AP = CalAP(Scores, Labels, NumGTs)
    return AP

def EvalCoSegAP(PredMasks, PredScores, GTInstMasks, Threshold):
    NumImgs = len(PredMasks)
    Overlap = []
    NumInsts = []
    for i in range(NumImgs):
        overlap, numinsts = GetOverlap(PredMasks[i], GTInstMasks[i])
        Overlap.append(overlap)
        NumInsts.append(numinsts)
    APAll = np.zeros((len(Threshold), 1))
    for i in range(len(Threshold)):
        APAll[i] = GetAP(PredScores, Overlap, Threshold[i], NumInsts)
    return APAll