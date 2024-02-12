#ifndef GENNM_IR_VISITOR_HH
#define GENNM_IR_VISITOR_HH

#include "dbg_config.hh"
#include "utils.hh"
#include "_ir.hh"


struct GenNmIRVisitor{
  virtual void visit(GenNmIRBase* ir){}

  virtual void visit(GenNmExpression* expr){}

  virtual void visit(GenNmVarExpr* expr) = 0;

  virtual void visit(GenNmLiteralExpr* expr) = 0;

  virtual void visit(GenNmBasicExpr* expr) = 0;

  virtual void visit(GenNmCallExpr* expr) = 0;

  virtual void visit(GenNmStatement* stmt) = 0;

  virtual void visit(GenNmAssignStmt* stmt) = 0;

  virtual void visit(GenNmCallStmt* stmt) = 0;

  virtual void visit(GenNmBranchStmt* stmt) = 0;

  virtual void visit(GenNmReturnStmt* stmt) = 0;
  
};






#endif
