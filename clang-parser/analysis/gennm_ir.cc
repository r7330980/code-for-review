#include "ir/gennm_ir.hh"
#include "ir/gennm_ir_visitor.hh"
#include <iostream>

using namespace std;

void GenNmVarExpr::accept(GenNmIRVisitor *visitor) { visitor->visit(this); }

void GenNmLiteralExpr::accept(GenNmIRVisitor *visitor) { visitor->visit(this); }


void GenNmImplicitReturnVarExpr::accept(GenNmIRVisitor *visitor) {
  visitor->visit(this);
}

void GenNmBasicExpr::accept(GenNmIRVisitor *visitor) { visitor->visit(this); }

void GenNmCallExpr::accept(GenNmIRVisitor *visitor) { visitor->visit(this); }

void GenNmAssignStmt::accept(GenNmIRVisitor *visitor) { visitor->visit(this); }

void GenNmCallStmt::accept(GenNmIRVisitor *visitor) { visitor->visit(this); }

void GenNmBranchStmt::accept(GenNmIRVisitor *visitor) { 
  visitor->visit(this); 
}

void GenNmReturnStmt::accept(GenNmIRVisitor *visitor) { visitor->visit(this); }