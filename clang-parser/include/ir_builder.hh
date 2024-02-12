#ifndef IR_BUILDER_HH
#define IR_BUILDER_HH


#include <iostream>
#include "ir/gennm_ir.hh"

using namespace clang;
using namespace std;

class IRBuilder: public GenNmASTConsumer{

public:
  explicit IRBuilder(clang::Rewriter &R)
    : GenNmASTConsumer(R){}

  virtual bool HandleTopLevelDecl(DeclGroupRef DR) override;

  vector<GenNmFunction*> getFunctions(){
    return functions;
  }

private:

  vector<GenNmFunction*> functions;
};




#endif
