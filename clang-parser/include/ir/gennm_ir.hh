#ifndef GENNM_IR_HH
#define GENNM_IR_HH

#include "dbg_config.hh"
#include "utils.hh"
#include <unordered_set>

using namespace std;

#ifdef DBG_IR
#define IR_DBG_OUT                                                             \
  if (true)                                                                    \
  cout << "[IR] "
#define IR_DBG_OUT_PREFIX(prefix)                                              \
  if (true)                                                                    \
  cout << prefix
#else
#define IR_DBG_OUT                                                             \
  if (false)                                                                   \
  cout
#define IR_DBG_OUT_PREFIX(prefix)                                              \
  if (false)                                                                   \
  cout
#endif

struct GenNmIRVisitor;
struct GenNmBasicBlock;
struct GenNmStatement;
struct GenNmExpression;
struct GenNmFunction;

struct GenNmIRBase {
  virtual void accept(GenNmIRVisitor *visitor) = 0;

  string getSrcText() { return srcText; }

protected:
  GenNmIRBase(string srcText) : srcText(srcText) {}

private:
  string srcText;
  // TODO: add source location
};

struct GenNmExpression : public GenNmIRBase {
  virtual void accept(GenNmIRVisitor *visitor) = 0;

  virtual bool isTerminator() { return false; }

protected:
  GenNmExpression(string srcText) : GenNmIRBase(srcText) {}
};

struct GenNmVarExpr : public GenNmExpression {
  GenNmVarExpr(string varName, string srcText)
      : varName(varName), GenNmExpression(srcText) {}

  void accept(GenNmIRVisitor *visitor) override;

  const string varName;
};

struct GenNmImplicitReturnVarExpr : public GenNmVarExpr {
  GenNmImplicitReturnVarExpr(string varName, string functionID, string srcText)
      : GenNmVarExpr(varName, srcText), functionID(functionID) {}

  void accept(GenNmIRVisitor *visitor);

  const string functionID;
};

struct GenNmLiteralExpr : public GenNmExpression {
  GenNmLiteralExpr(string literal, string srcText)
      : literal(literal), GenNmExpression(srcText) {}

  void accept(GenNmIRVisitor *visitor);

  const string literal;
};

struct GenNmBasicExpr : public GenNmExpression {
  GenNmBasicExpr(unordered_set<GenNmVarExpr *> defines,
                unordered_set<GenNmVarExpr *> uses, bool directUse, string srcText)
      : GenNmExpression(srcText), directUse(directUse) {    
    unordered_set<string> defNames;
    vector<GenNmVarExpr *> toRemove;
    for (auto def : defines) {
      if (defNames.count(def->varName)) {
        toRemove.push_back(def);
        continue;
      }
      defNames.insert(def->varName);
      this->defines.insert(def);
    }
    for (auto def : toRemove) {
      delete def;
    }
    toRemove.clear();
    unordered_set<string> useNames;
    for (auto use : uses) {
      if (useNames.count(use->varName)) {
        toRemove.push_back(use);
        continue;
      }
      useNames.insert(use->varName);
      this->uses.insert(use);
    }
    for (auto use : toRemove) {
      delete use;
    }
  }
  void accept(GenNmIRVisitor *visitor);

  bool isDirectUse() { return directUse; }

  unordered_set<GenNmVarExpr *> &getDefines() { return defines; }

  unordered_set<GenNmVarExpr *> &getUses() { return uses; }

private:
  unordered_set<GenNmVarExpr *> defines;
  unordered_set<GenNmVarExpr *> uses;
  bool directUse;
};

struct GenNmCallExpr : public GenNmExpression {
  GenNmCallExpr(string funcID, vector<GenNmExpression *> args, string srcText)
      : funcID(funcID), args(args), GenNmExpression(srcText) {}
  void accept(GenNmIRVisitor *visitor);

  vector<GenNmExpression *> &getArgs() { return args; }

  const string funcID;

private:
  vector<GenNmExpression *> args;
};

struct GenNmStatement : public GenNmExpression {

  virtual void accept(GenNmIRVisitor *visitor) = 0;

protected:
  GenNmStatement(string srcText) : GenNmExpression(srcText) {}
};

struct GenNmAssignStmt : public GenNmStatement {
  GenNmAssignStmt(GenNmVarExpr *lhs, GenNmExpression *rhs, string srcText)
      : lhs(lhs), rhs(rhs), GenNmStatement(srcText) {}
  void accept(GenNmIRVisitor *visitor);

  GenNmVarExpr *getLHS() { return lhs; }

  GenNmExpression *getRHS() { return rhs; }

private:
  GenNmVarExpr *lhs;
  GenNmExpression *rhs;
};

struct GenNmCallStmt : public GenNmStatement {
  GenNmCallStmt(GenNmCallExpr *callExpr, string srcText)
      : callExpr(callExpr), GenNmStatement(srcText) {}
  void accept(GenNmIRVisitor *visitor);

  GenNmCallExpr *getCallExpr() { return callExpr; }

private:
  GenNmCallExpr *callExpr;
};

struct GenNmBranchStmt : public GenNmStatement {
  GenNmBranchStmt(vector<GenNmBasicBlock *> successors, string srcText)
      : successors(successors), GenNmStatement(srcText) {}

  void accept(GenNmIRVisitor *visitor);

  vector<GenNmBasicBlock *> &getSuccessors() { return successors; }

  bool isTerminator() { return true; }

private:
  vector<GenNmBasicBlock *> successors;
};

struct GenNmReturnStmt : public GenNmStatement {
  GenNmReturnStmt(GenNmExpression *retVal, string srcText)
      : GenNmStatement(srcText), retVal(retVal) {}

  void accept(GenNmIRVisitor *visitor);

  bool isTerminator() { return true; }

  GenNmExpression *getRetVal() { return retVal; }

private:
  GenNmExpression *retVal;
};

struct GenNmBasicBlock {
  GenNmBasicBlock(string label) : label(label) {}

  string getLabel() { return label; }

  bool terminated() {
    return statements.size() > 0 && statements.back()->isTerminator();
  }

  void setLabel(string label) { this->label = label; }

  vector<GenNmExpression *> &getStatements() { return statements; }

  vector<GenNmBasicBlock *> &getSuccessors() { return successors; }

  vector<GenNmBasicBlock *> &getPredecessors() { return predecessors; }

private:
  string label;
  vector<GenNmExpression *> statements = {};
  vector<GenNmBasicBlock *> successors = {};
  vector<GenNmBasicBlock *> predecessors = {};
};

struct GenNmFunction {
  GenNmFunction(string funcID, vector<GenNmVarExpr *> args,
               vector<GenNmBasicBlock *> basicBlocks)
      : funcID(funcID), args(args), basicBlocks(basicBlocks) {}

  void normalizeBBLabels() {
    int cnt = 0;
    for (auto bb : basicBlocks) {
      auto oriLabel = bb->getLabel();
      auto newLabel = "bb:%" + to_string(cnt) + "-" + oriLabel;
      bb->setLabel(newLabel);
      cnt++;
    }
  }

  string getFuncID() { return funcID; }

  vector<GenNmVarExpr *> &getArgs() { return args; }

  vector<GenNmBasicBlock *> &getBasicBlocks() { return basicBlocks; }

private:
  string funcID;
  vector<GenNmVarExpr *> args;
  vector<GenNmBasicBlock *> basicBlocks;
};

#endif