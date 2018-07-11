#include "stdafx.h"

#include "n2v.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

void ParseArgs(int& argc, char* argv[], TStr& InFile, TStr& OutFile, TStr& LabelFile,
 int& Dimensions, int& WalkLen, int& NumWalks, int& WinSize, int& Iter,
 bool& Verbose, double& ParamP, double& ParamQ, bool& Directed, bool& Weighted, bool& AddLabels) {
  Env = TEnv(argc, argv, TNotify::StdNotify);
  Env.PrepArgs(TStr::Fmt("\nAn algorithmic framework for representational learning on graphs."));
  InFile = Env.GetIfArgPrefixStr("-i:", "graph/karate.edgelist",
   "Input graph path");
  OutFile = Env.GetIfArgPrefixStr("-o:", "emb/karate.walk",
   "Output graph path");
  LabelFile = Env.GetIfArgPrefixStr("-y:", "ml_labels/karate.label",
                                    "Graph label path");
  Dimensions = Env.GetIfArgPrefixInt("-d:", 128,
   "Number of dimensions. Default is 128");
  WalkLen = Env.GetIfArgPrefixInt("-l:", 80,
   "Length of walk per source. Default is 80");
  NumWalks = Env.GetIfArgPrefixInt("-r:", 10,
   "Number of walks per source. Default is 10");
  WinSize = Env.GetIfArgPrefixInt("-k:", 10,
   "Context size for optimization. Default is 10");
  Iter = Env.GetIfArgPrefixInt("-e:", 1,
   "Number of epochs in SGD. Default is 1");
  ParamP = Env.GetIfArgPrefixFlt("-p:", 1,
   "Return hyperparameter. Default is 1");
  ParamQ = Env.GetIfArgPrefixFlt("-q:", 1,
   "Inout hyperparameter. Default is 1");
  Verbose = Env.IsArgStr("-v", "Verbose output.");
  Directed = Env.IsArgStr("-dr", "Graph is directed.");
  Weighted = Env.IsArgStr("-w", "Graph is weighted.");
  AddLabels = Env.IsArgStr("-al", "Graph labels are added.");
}

void PermuteLabels(TStrV& Labels, const int& NumLabels, TVVec<TFlt,int64>& LabelArray, TFlt* ASum){

    for(int i=0; i<Labels.Len(); i++){
        for (int j=i; j<Labels.Len(); j++){
            int I = Labels[i].GetInt();
            int J = Labels[j].GetInt();
            if (LabelArray(I,J) > 0){
                LabelArray.PutXY(I,J,LabelArray.At(I,J) + 1.0 );
                LabelArray.PutXY(J,I,LabelArray.At(J,I) + 1.0 );
                ASum[I] +=1.0;
                ASum[J] +=1.0;
            } else {
                LabelArray.PutXY(I,J,1.0 );
                LabelArray.PutXY(J,I, 1.0 );
                ASum[I] =1.0;
                ASum[J] =1.0;
            }
        }
    }
}

void ReadLabels(TStr& LabelFile, int& NumLabels, THash<TInt,TStrV>& LabelV, PWNet& InNet, bool& Verbose){
    TFIn FIn(LabelFile);
    int64 LineCnt = 0;

    try {
        TStr Ln;
        FIn.GetNextLn(Ln);
        TStr Line, Count;
        Ln.SplitOnCh(Line,'#',Count);
        NumLabels = Count.GetInt();
        TVVec<TFlt,int64> LabelArray (NumLabels,NumLabels);
        TFlt ASum [NumLabels] = {0.0};
        while (!FIn.Eof()) {
            TStr Ln;
            FIn.GetNextLn(Ln);
            TStr NodeId, Labels;
            Ln.SplitOnCh(NodeId,' ',Labels);
            TStrV Tokens;
            Labels.SplitOnAllCh(',',Tokens);
            if(Tokens.Len()<1){ continue; }
            Tokens.Sort();
            LabelV.AddDat(NodeId.GetInt()+NumLabels, Tokens);
            PermuteLabels(Tokens, NumLabels, LabelArray, ASum);
            LineCnt ++;
        }
        for(int i=0; i<NumLabels; i++){
            for (int j=0; j<NumLabels; j++){
                if(LabelArray(i,j) > 0){
                    if (!InNet->IsNode(i)){ InNet->AddNode(i); }
                    if (!InNet->IsNode(j)){ InNet->AddNode(j); }
                    InNet->AddEdge(i,j,LabelArray(i,j)  /= ASum[i]);
                }
            }
        }
        if (Verbose) { printf("Read %lld lines from %s\n", (long long)LineCnt, LabelFile.CStr()); }
    } catch (PExcept Except) {
        if (Verbose) {
            printf("Read %lld lines from %s, then %s\n", (long long)LineCnt, LabelFile.CStr(),
                   Except->GetStr().CStr());
        }
    }
}

void ReadGraph(TStr& InFile, TStr& LabelFile, bool& Directed, bool& Weighted, bool& Verbose, bool& AddLabels, PWNet& InNet) {
    int NumLabels = 0;
    THash<TInt, TStrV> LabelV;
    if (AddLabels == true) {
        ReadLabels(LabelFile,NumLabels,LabelV, InNet, Verbose);
    }
    TFIn FIn(InFile);
    int64 LineCnt = 0;
    bool interconnect = true;
    try {
        while (!FIn.Eof()) {
          TStr Ln;
          FIn.GetNextLn(Ln);
          TStr Line, Comment;
          Ln.SplitOnCh(Line,'#',Comment);
          TStrV Tokens;
          Line.SplitOnWs(Tokens);
          if(Tokens.Len()<2){ continue; }
          int64 SrcNId = Tokens[0].GetInt()+NumLabels;
          int64 DstNId = Tokens[1].GetInt()+NumLabels;
          double Weight = 1.0;
          if (Weighted) { Weight = Tokens[2].GetFlt(); }
          if (!InNet->IsNode(SrcNId)){ InNet->AddNode(SrcNId); }
          if (!InNet->IsNode(DstNId)){ InNet->AddNode(DstNId); }
          InNet->AddEdge(SrcNId,DstNId,Weight);
          if (!Directed){ InNet->AddEdge(DstNId,SrcNId,Weight); }
          if (AddLabels == true) {
              if(LabelV.IsKey(SrcNId)){
                TStrV LabelTokens;
                  LabelTokens = LabelV.GetDat(SrcNId);//.SplitOnAllCh(',',LabelTokens) ;
                for (int i=0; i<LabelTokens.Len(); i++){
                    TInt Label = LabelTokens[i].GetInt();
                    InNet->AddEdge(SrcNId,Label,0);
                    InNet->AddEdge(Label,SrcNId,1);
                }
              }
              if(LabelV.IsKey(DstNId)){
                  TStrV LabelTokens;
                  LabelTokens = LabelV.GetDat(DstNId);//.SplitOnAllCh(',',LabelTokens) ;
                  for (int i=0; i<LabelTokens.Len(); i++){
                      TInt Label = LabelTokens[i].GetInt();
                      InNet->AddEdge(DstNId,Label,0);
                      InNet->AddEdge(Label,DstNId,1);
                  }
              }

          }
          LineCnt++;
        }

        if (AddLabels == true) {
            TInt Key;
            TStrV Data;
            int eps = 1;
            int KeyId = LabelV.FFirstKeyId();
            while (LabelV.FNextKeyId(KeyId)) {
                LabelV.GetKeyDat(KeyId, Key, Data);
//                TStrV LabelTokens;
//                Data.SplitOnAllCh(',', LabelTokens);
                TWNet::TNodeI CurrI = InNet->GetNI(Key);
                double MaxW = 0;
                for (int64 j = 0; j < CurrI.GetOutDeg(); j++) {           //for each node x
                    int64 FId = CurrI.GetNbrNId(j);
                    TFlt Weight;
                    if (!(InNet->GetEDat(CurrI.GetId(), FId, Weight))) { continue; }
                    if (LabelV.IsKey(FId)){
                        int IntrsCount = Data.IntrsLen(LabelV.GetDat(FId));
                        TFlt b = (eps+IntrsCount)/(eps + Data.Len());
                        InNet->SetEDat(Key, FId, Weight*b);
                    }
                    if (MaxW < Weight) {
                        MaxW = Weight;
                    }
                }
                for (int i = 0; i < Data.Len(); i++) {
                    InNet->SetEDat(Key, Data[i].GetInt(), MaxW);
                }
            }
        }
        if (Verbose) { printf("Read %lld lines from %s\n", (long long)LineCnt, InFile.CStr()); }
    } catch (PExcept Except) {
    if (Verbose) {
      printf("Read %lld lines from %s, then %s\n", (long long)LineCnt, InFile.CStr(),
       Except->GetStr().CStr());
    }
    }
}


void WriteOutput(TStr& OutFile, TIntFltVH& EmbeddingsHV) {
  TFOut FOut(OutFile);
  bool First = 1;
  for (int i = EmbeddingsHV.FFirstKeyId(); EmbeddingsHV.FNextKeyId(i);) {
    if (First) {
      FOut.PutInt(EmbeddingsHV.Len());
      FOut.PutCh(' ');
      FOut.PutInt(EmbeddingsHV[i].Len());
      FOut.PutLn();
      First = 0;
    }
    FOut.PutInt(EmbeddingsHV.GetKey(i));
    for (int64 j = 0; j < EmbeddingsHV[i].Len(); j++) {
      FOut.PutCh(' ');
      FOut.PutFlt(EmbeddingsHV[i][j]);
    }
    FOut.PutLn();
  }
}

void SaveWalk(TStr& OutFile, int& NumWalks, int& LenNIdsV, TVVec<TInt, int64>& WalksVV, bool& Verbose) {
    TFOut FOut(OutFile);
    int64 WalksDone = 0;

    int y_size = WalksVV.GetYDim();
    TStr Walks[LenNIdsV];
    TVec<TInt, long int> mValV(y_size);
    int64 AllWalks = WalksVV.GetXDim();
    FOut.PutCh('{');
//    #pragma omp parallel for schedule(dynamic)
    for (int64 i = 0; i < LenNIdsV; i++) {

        for (int64 j = 0; j < NumWalks; j++) {
            if (Verbose && WalksDone % 10000 == 0) {
                printf("\rPrepare Writing  Progress: %.2lf%%", (double) WalksDone * 100 / (double) AllWalks);
                fflush(stdout);
            }
            WalksVV.GetRow(j * LenNIdsV + i, mValV);
            TStr Row = "[" + mValV[0].GetStr();
            for (int indN = 1; indN < mValV.Len(); indN++) { Row += ", " + mValV[indN].GetStr(); }
//            #pragma omp critical
            if (Walks[mValV[0]].Len() > 0) {
                Walks[mValV[0]] += "," + Row + "]";
            } else {
                Walks[mValV[0]] = Row + "]";
            }
            WalksDone++;
        }
    }
    if (Verbose) {
        printf("\n");
    }


    for (int i = 0; i < LenNIdsV; i++) {
        if (Verbose && i % 10000 == 0) {
            printf("\rWriting Progress: %.2lf%%", (double) i * 100 / (double) LenNIdsV);
        }

        FOut.PutCh('"');
        FOut.PutInt(i);
        FOut.PutCh('"');
        FOut.PutStr(": [");
        FOut.PutStr(Walks[i]);
        if (i < LenNIdsV-1) {
            FOut.PutStr("],\n");
        } else {
            FOut.PutStr("]}");
        }
    }

    if (Verbose) {
        printf("\n");
        fflush(stdout);
    }

}

int main(int argc, char* argv[]) {
    TStr InFile,OutFile,LabelFile;
    int Dimensions, WalkLen, NumWalks, WinSize, Iter;
    double ParamP, ParamQ;
    bool Directed, Weighted, Verbose,AddLabels;
    ParseArgs(argc, argv, InFile, OutFile,LabelFile, Dimensions, WalkLen, NumWalks, WinSize,
    Iter, Verbose, ParamP, ParamQ, Directed, Weighted,AddLabels);
    PWNet InNet = PWNet::New();
    TIntFltVH EmbeddingsHV;
    ReadGraph(InFile, LabelFile, Directed, Weighted, Verbose, AddLabels, InNet);
//    ReadGraph(InFile, Directed, Weighted, Verbose, InNet);
//    PreprocessTransitionProbs(InNet, ParamP, ParamQ, Verbose);
    int LenNIdsV = InNet->GetMxNId();
    int64 AllWalks = (int64)NumWalks * LenNIdsV;
    TVVec<TInt, int64> WalksVV(AllWalks,WalkLen);
    graphwalk(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter,
    Verbose, EmbeddingsHV, WalksVV);
    SaveWalk(OutFile, NumWalks, LenNIdsV, WalksVV, Verbose);

//    WriteOutput(OutFile, EmbeddingsHV);
    return 0;
}
