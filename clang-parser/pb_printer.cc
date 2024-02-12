#include "pb_printer.hh"
#include "ir/gennm_ir.hh"
#include "gennm_ir.pb.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

class PBPrinter {
public:
  PBPrinter(string fname, GenNmFunction *func) : fname(fname), func(func) {}

  void print() {
    ofstream output(fname, ios::out | ios::trunc | ios::binary);
    if (!output) {
      cerr << "Failed to open output file `" << fname << "`." << endl;
      exit(1);
    }
    auto pbFunc = printFunc(func);
    pbFunc->SerializeToOstream(&output);
    output.close();
  }

private:
  pb::GenNmFunction *printFunc(GenNmFunction *gennmFunc) {
    auto pbFunc = new pb::GenNmFunction();
    pbFunc->set_funcid(gennmFunc->getFuncID());
    for (auto bb : gennmFunc->getBasicBlocks()) {
      auto pbBB = printBB(bb);
      pbFunc->mutable_blocks()->AddAllocated(pbBB);
    }
    for (auto arg : gennmFunc->getArgs()) {
      auto pbArg = printVarExpr(arg);
      pbFunc->mutable_args()->AddAllocated(pbArg);
    }
    return pbFunc;
  }

  pb::GenNmBasicBlock *printBB(GenNmBasicBlock *gennmBB) {
    auto pbBB = new pb::GenNmBasicBlock();
    pbBB->set_blockname(gennmBB->getLabel());
    pbBB->set_isterminate(gennmBB->terminated());
    for (auto stmt : gennmBB->getStatements()) {
      // auto pbStmt = printStmt(stmt);
      auto pbExpr = printExpr(stmt);
      pbBB->mutable_exprs()->AddAllocated(pbExpr);
    }
    for (auto succ : gennmBB->getSuccessors()) {
      pbBB->add_successors(succ->getLabel());
    }
    for (auto pred : gennmBB->getPredecessors()) {
      pbBB->add_predecessors(pred->getLabel());
    }
    return pbBB;
  }

  pb::GenNmExpression *printExpr(GenNmExpression *gennmExpr) {
    auto pbExpr = new pb::GenNmExpression();
    if (auto gennmImplicitReturnVarExpr =
            dynamic_cast<GenNmImplicitReturnVarExpr *>(gennmExpr)) {
      pbExpr->set_allocated_implicitreturnvarexpr(
          printImplicitReturnVarExpr(gennmImplicitReturnVarExpr));
    } else if (auto gennmVarExpr = dynamic_cast<GenNmVarExpr *>(gennmExpr)) {
      pbExpr->set_allocated_varexpr(printVarExpr(gennmVarExpr));
    } else if (auto gennmBasicExpr = dynamic_cast<GenNmBasicExpr *>(gennmExpr)) {
      pbExpr->set_allocated_basicexpr(printBasicExpr(gennmBasicExpr));
    } else if (auto gennmCallExpr = dynamic_cast<GenNmCallExpr *>(gennmExpr)) {
      pbExpr->set_allocated_callstmt(printCallExpr(gennmCallExpr));
    } else if (auto branchExpr = dynamic_cast<GenNmBranchStmt *>(gennmExpr)) {
      pbExpr->set_allocated_branchstmt(printBranchStmt(branchExpr));
    } else if (auto returnExpr = dynamic_cast<GenNmReturnStmt *>(gennmExpr)) {
      pbExpr->set_allocated_returnstmt(printReturnStmt(returnExpr));
    } else {
      cerr << "Unknown expression type." << endl;
      cerr << "Type: " << typeid(gennmExpr).name() << endl;
    }
    pbExpr->set_srctext(gennmExpr->getSrcText());
    return pbExpr;
  }

  pb::GenNmVarExpr *printVarExpr(GenNmVarExpr *gennmVarExpr) {
    auto pbVarExpr = new pb::GenNmVarExpr();
    pbVarExpr->set_varname(gennmVarExpr->varName);
    return pbVarExpr;
  }

  pb::GenNmImplicitReturnVarExpr *printImplicitReturnVarExpr(
      GenNmImplicitReturnVarExpr *gennmImplicitReturnVarExpr) {
    auto pbImplicitReturnVarExpr = new pb::GenNmImplicitReturnVarExpr();
    pbImplicitReturnVarExpr->set_varname(gennmImplicitReturnVarExpr->varName);
    pbImplicitReturnVarExpr->set_funcid(gennmImplicitReturnVarExpr->functionID);
    return pbImplicitReturnVarExpr;
  }

  pb::GenNmBasicExpr *printBasicExpr(GenNmBasicExpr *gennmBasicExpr) {
    auto pbBasicExpr = new pb::GenNmBasicExpr();
    pbBasicExpr->set_isdirectuse(gennmBasicExpr->isDirectUse());
    for (auto def : gennmBasicExpr->getDefines()) {
      auto defExpr = printExpr(def);
      pbBasicExpr->mutable_defs()->AddAllocated(defExpr);
      // *pbBasicExpr->mutable_defs()->Add() = *defExpr;
      // delete defExpr;
    }
    for (auto use : gennmBasicExpr->getUses()) {
      auto useExpr = printExpr(use);
      pbBasicExpr->mutable_uses()->AddAllocated(useExpr);
    }

    return pbBasicExpr;
  }

  pb::GenNmCallStmt *printCallExpr(GenNmCallExpr *gennmCallExpr) {
    auto pbCallStmt = new pb::GenNmCallStmt();
    pbCallStmt->set_funcid(gennmCallExpr->funcID);
    for (auto arg : gennmCallExpr->getArgs()) {
      auto argExpr = printExpr(arg);
      pbCallStmt->mutable_args()->AddAllocated(argExpr);
    }
    return pbCallStmt;
  }

  pb::GenNmCallStmt *printCallStmt(GenNmCallStmt *gennmCallStmt) {
    return printCallExpr(gennmCallStmt->getCallExpr());
  }

  pb::GenNmReturnStmt *printReturnStmt(GenNmReturnStmt *gennmReturnStmt) {
    auto pbReturnStmt = new pb::GenNmReturnStmt();
    auto pbRetVal = printExpr(gennmReturnStmt->getRetVal());
    pbReturnStmt->set_allocated_retval(pbRetVal);
    return pbReturnStmt;
  }

  pb::GenNmBranchStmt *printBranchStmt(GenNmBranchStmt *gennmBranchStmt) {
    auto pbBranchStmt = new pb::GenNmBranchStmt();
    for (auto succ : gennmBranchStmt->getSuccessors()) {
      pbBranchStmt->add_successors(succ->getLabel());
    }
    return pbBranchStmt;
  }

private:
  string fname;
  GenNmFunction *func;
};

void writeToFile(string fname, GenNmFunction *func) {
  PBPrinter printer(fname, func);
  printer.print();
}