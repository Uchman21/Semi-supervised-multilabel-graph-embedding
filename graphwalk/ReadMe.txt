========================================================================
    GraphWalk
========================================================================

graphwalk is built off node2vec 

The code works under Windows with Visual Studio or Cygwin with GCC,
Mac OS X, Linux and other Unix variants with GCC. Make sure that a
C++ compiler is installed on the system. Visual Studio project files
and makefiles are provided. For makefiles, compile the code with
"make all".

/////////////////////////////////////////////////////////////////////////////

Parameters:
Input graph path (-i:)
Output graph path (-o:)
Graph label path (-y:)
Length of walk per source. Default is 80 (-l:)=10
Number of walks per source. Default is 10 (-r:)=10
Return hyperparameter. Default is 1 (-p:)=4
Inout hyperparameter. Default is 1 (-q:)=1
Verbose output. (-v)=YES
Graph is directed. (-dr)=NO
Graph is weighted. (-w)=YES
Graph labels are added. (-al)=YES


/////////////////////////////////////////////////////////////////////////////

Usage:
./graphwalk -i:graph/karate.edgelist -o:emb/karate.emb -l:3 -p:0.3 -dr -v
