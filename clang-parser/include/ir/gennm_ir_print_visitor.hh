#ifndef GENNM_IR_PRINT_VISITOR_HH
#define GENNM_IR_PRINT_VISITOR_HH

#include "dbg_config.hh"
#include "gennm_ir.hh"
#include "gennm_ir_visitor.hh"
#include "utils.hh"

#include <iostream>
#include <typeinfo>

using namespace std;
using namespace clang;

struct GenNmIRPrintVisitor : public GenNmIRVisitor {
#define GENNM_PRINT_VISITOR_OUT cout << getIndent()
  void visit(GenNmIRBase *ir) {
    GENNM_PRINT_VISITOR_OUT << "This is GenNmIRBase. This should NOT happen!"
                           << endl;
    assert(false);
  }

  void visit(GenNmExpression *expr) {
    GENNM_PRINT_VISITOR_OUT << "This is GenNmExpression. This should NOT happen!"
                           << endl;
    assert(false);
  }

  void visit(GenNmStatement *stmt) {
    GENNM_PRINT_VISITOR_OUT << "This is GenNmStatement. This should NOT happen!"
                           << endl;
    assert(false);
  }

  void visit(GenNmVarExpr *expr) {
    GENNM_PRINT_VISITOR_OUT << "GenNmVarExpr: varname: " << expr->varName
                           << ", src: " << expr->getSrcText() << endl;
  }

  void visit(GenNmLiteralExpr *expr) {
    GENNM_PRINT_VISITOR_OUT << "GenNmLiteralExpr: literal: " << expr->literal
                           << ", src: " << expr->getSrcText() << endl;
  }

  void visit(GenNmBasicExpr *expr) {
    GENNM_PRINT_VISITOR_OUT << "GenNmBasicExpr: "
                           << "src: " << expr->getSrcText() << endl;
    incIndent();
    GENNM_PRINT_VISITOR_OUT << "def: " << expr->getDefines().size() << endl;
    incIndent();
    for (auto def : expr->getDefines()) {
      def->accept(this);
    }
    decIndent();
    GENNM_PRINT_VISITOR_OUT << "use: " << expr->getUses().size() << endl;
    incIndent();
    for (auto use : expr->getUses()) {      
      use->accept(this);
    }
    decIndent();
    GENNM_PRINT_VISITOR_OUT << "isDirectUse: " << expr->isDirectUse() << endl;
    decIndent();
  }

  void visit(GenNmCallExpr *expr) {
    GENNM_PRINT_VISITOR_OUT << "GenNmCallExpr: "
                           << "funcID: " << expr->funcID
                           << ",src: " << expr->getSrcText() << endl;
    incIndent();
    GENNM_PRINT_VISITOR_OUT << "args: " << endl;
    incIndent();
    for (auto arg : expr->getArgs()) {
      arg->accept(this);
    }
    decIndent();
    decIndent();
  }

  void visit(GenNmCallStmt *stmt) {
    GENNM_PRINT_VISITOR_OUT << "GenNmCallStmt: " << endl;
    incIndent();
    stmt->getCallExpr()->accept(this);
    decIndent();
  }

  void visit(GenNmAssignStmt *stmt) {
    GENNM_PRINT_VISITOR_OUT << "GenNmAssignStmt: " << endl;
    incIndent();
    GENNM_PRINT_VISITOR_OUT << "LHS: " << endl;
    incIndent();
    stmt->getLHS()->accept(this);
    decIndent();
    GENNM_PRINT_VISITOR_OUT << "RHS: " << endl;
    incIndent();
    stmt->getRHS()->accept(this);
    decIndent();
    decIndent();
  }

  void visit(GenNmReturnStmt *stmt) {
    GENNM_PRINT_VISITOR_OUT << "GenNmReturnStmt: " << endl;
    incIndent();
    GENNM_PRINT_VISITOR_OUT << "retVal: " << endl;
    incIndent();
    stmt->getRetVal()->accept(this);
    decIndent();
    decIndent();
  }

  void visit(GenNmBranchStmt *stmt) {
    GENNM_PRINT_VISITOR_OUT << "GenNmBranchStmt: " << endl;
    incIndent();
    GENNM_PRINT_VISITOR_OUT << "successors: " << endl;
    incIndent();
    for (auto succ : stmt->getSuccessors()) {
      GENNM_PRINT_VISITOR_OUT << succ->getLabel() << endl;
    }
    decIndent();
    decIndent();
  }

  void visit(GenNmBasicBlock *bb) {
    GENNM_PRINT_VISITOR_OUT << "GenNmBasicBlock: " << endl;
    incIndent();
    GENNM_PRINT_VISITOR_OUT << "label: " << bb->getLabel() << endl;
    GENNM_PRINT_VISITOR_OUT << "statements: " << endl;
    incIndent();
    for (auto stmt : bb->getStatements()) {
      stmt->accept(this);
    }
    decIndent();
    decIndent();
  }

  void visit(GenNmFunction *func) {
    GENNM_PRINT_VISITOR_OUT << "GenNmFunction: " << endl;
    incIndent();
    GENNM_PRINT_VISITOR_OUT << "funcID: " << func->getFuncID() << endl;
    GENNM_PRINT_VISITOR_OUT << "params: " << endl;
    incIndent();
    for (auto param : func->getArgs()) {
      param->accept(this);
    }
    decIndent();
    GENNM_PRINT_VISITOR_OUT << "basicBlocks: " << endl;
    incIndent();
    for (auto bb : func->getBasicBlocks()) {
      visit(bb);
    }
    decIndent();
    decIndent();
  }

private:
  int indent = 0;
  void incIndent() { indent++; }

  void decIndent() { indent--; }

  string getIndent() {
    string ret = "";
    for (int i = 0; i < indent; i++) {
      ret += "  ";
    }
    return ret;
  }
};

#endif