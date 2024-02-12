#ifndef CFG_VISITOR_HH
#define CFG_VISITOR_HH

#include "utils.hh"
#include "cfg.hh"

struct BinameStatementVisitor{
  void visit(BinameStatement* stmt){
    stmt->accept(this);
  }

  virtual void visit(SimpleClangStatement* stmt) = 0;

  virtual void visit(SequentialStatement* stmt) = 0;

  virtual void visit(IfStatement* stmt) = 0;

  virtual void visit(WhileStatement* stmt) = 0;

  virtual void visit(ForStatement* stmt) = 0;

  virtual void visit(SwitchStatement* stmt) = 0;

  virtual void visit(LoopTermStatement* stmt) = 0;

  virtual void visit(ExecTermStatement* stmt) = 0;

  virtual void visit(SimpleClangDecl* stmt) = 0;

  virtual void visit(DoWhileStatement* stmt) = 0;
};



#endif

