#ifndef CFG_BUILDER_HH
#define CFG_BUILDER_HH

#include "utils.hh"
#include "ast_visitor_interface.hh"
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "cfg.hh"

using namespace clang;
using namespace std;

class CFGBuilder : public GenNmASTConsumer {
public:
    explicit CFGBuilder(clang::Rewriter &R)
        : GenNmASTConsumer(R){}
    virtual bool HandleTopLevelDecl(DeclGroupRef DR) override;

    vector<BinameProgram*>& getPrograms(){
        return programs;
    }

private:
    vector<BinameProgram*> programs;
};



#endif
