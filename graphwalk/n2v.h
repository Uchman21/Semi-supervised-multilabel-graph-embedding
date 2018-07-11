#ifndef N2V_H
#define N2V_H

#include "stdafx.h"

#include "Snap.h"
#include "biasedrandomwalk.h"

/// Generate samples from Walk on Graph
void graphwalk(PWNet& InNet, double& ParamP, double& ParamQ, int& Dimensions,
 int& WalkLen, int& NumWalks, int& WinSize, int& Iter, bool& Verbose,
 TIntFltVH& EmbeddingsHV, TVVec<TInt, int64>& WalksVV); 

/// Version for unweighted graphs
void graphwalk(PNGraph& InNet, double& ParamP, double& ParamQ, int& Dimensions,
 int& WalkLen, int& NumWalks, int& WinSize, int& Iter, bool& Verbose,
 TIntFltVH& EmbeddingsHV,TVVec<TInt, int64>& WalksVV); 

/// Version for weighted graphs. Edges must have TFlt attribute "weight"
void graphwalk(PNEANet& InNet, double& ParamP, double& ParamQ, int& Dimensions,
 int& WalkLen, int& NumWalks, int& WinSize, int& Iter, bool& Verbose,
 TIntFltVH& EmbeddingsHV,TVVec<TInt, int64>& WalksVV);
#endif //N2V_H
