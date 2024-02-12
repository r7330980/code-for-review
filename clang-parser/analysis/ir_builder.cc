#include <clang/Analysis/CFG.h>

#include "dbg_config.hh"
#include "ir/gennm_ir.hh"
#include "ir_builder.hh"
#include "utils.hh"
#include <iostream>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#ifdef DBG_IR_BUILDER
#define IR_BUILDER_DBG_OUT                                                     \
  if (true)                                                                    \
  cout
#define IR_BUILDER_DBG_OUT_PREFIX(prefix)                                      \
  if (true)                                                                    \
  cout << prefix
#else
#define IR_BUILDER_DBG_OUT                                                     \
  if (false)                                                                   \
  cout
#define IR_BUILDER_DBG_OUT_PREFIX(prefix)                                      \
  if (false)                                                                   \
  cout
#endif

using namespace clang;
using namespace std;

struct GenNmDefUse {
  unordered_set<GenNmVarExpr *> defs = {}, uses = {};
  bool isDirectUse = false;
};

static inline string getSourceCodeFromRange(const Rewriter &rewriter,
                                            SourceRange range) {
  auto &srcManager = rewriter.getSourceMgr();
  auto beginFileLoc = srcManager.getFileLoc(range.getBegin());
  auto endFileLoc = srcManager.getFileLoc(range.getEnd());
  auto expansionRange = srcManager.getExpansionRange(range);
  auto beginStr = beginFileLoc.printToString(srcManager);
  auto endStr = endFileLoc.printToString(srcManager);
  auto srcStr = rewriter.getRewrittenText(expansionRange);
  srcStr = "(" + beginStr + "~" + endStr + "): " + srcStr;
  return srcStr;
}

class LHSGenNmExprBuilder : public RecursiveASTVisitor<LHSGenNmExprBuilder> {
public:
  LHSGenNmExprBuilder(Rewriter &rewriter) : rewriter(rewriter) {}

  optional<GenNmVarExpr *> parseLHS(Expr *lhs) {
    lhs = lhs->IgnoreParenCasts();
    auto srcStr = getSourceCodeFromRange(rewriter, lhs->getSourceRange());
    switch (lhs->getStmtClass()) {
    case Stmt::DeclRefExprClass: {
      auto declRefExpr = dyn_cast<DeclRefExpr>(lhs);
      auto decl = declRefExpr->getDecl();
      auto declName = decl->getName().str();
      return new GenNmVarExpr(declName, srcStr);
    }
    case Stmt::RecoveryExprClass: {
      auto recoveryExpr = dyn_cast<RecoveryExpr>(lhs);
      auto srcStr =
          getSourceCodeFromRange(rewriter, recoveryExpr->getSourceRange());
      return new GenNmVarExpr(srcStr, srcStr);
    }
    default:
      return nullopt;
    }
  }

private:
  Rewriter &rewriter;
};

class RHSGenNmExprBuilder : public RecursiveASTVisitor<RHSGenNmExprBuilder> {

public:
  RHSGenNmExprBuilder(Rewriter &rewriter) : rewriter(rewriter) {}

  GenNmExpression *parseRHS(Expr *rhs) {
    exprs.clear();
    IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::parseRHS] ")
        << "Parsing RHS: "
        << getSourceCodeFromRange(rewriter, rhs->getSourceRange()) << endl;
    auto gennmDefUse = createGenNmExpression(rhs);
    if (gennmDefUse->uses.size() == 1 && gennmDefUse->defs.size() == 0) {
      auto uses = gennmDefUse->uses;
      auto firstUse = *uses.begin();
      if (auto implRetVarExpr =
              dynamic_cast<GenNmImplicitReturnVarExpr *>(firstUse)) {
        IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::parseRHS] ")
            << "Implicit return variable: " << implRetVarExpr->varName << endl;
        IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::parseRHS] ")
            << "exprs size: " << exprs.size() << endl;
        return implRetVarExpr;
      }
      if (auto varExpr = dynamic_cast<GenNmVarExpr *>(firstUse)) {
        auto varname = varExpr->varName;
        if (isDirectUse(rhs, varname)) {
          IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::parseRHS] ")
              << "Direct use of variable: " << varname << endl;
          IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::parseRHS] ")
              << "exprs size: " << exprs.size() << endl;
          return varExpr;
        }
      }
    }
    bool isDirectUse = gennmDefUse->isDirectUse;
    auto basicExpr = new GenNmBasicExpr(
        gennmDefUse->defs, gennmDefUse->uses, isDirectUse,
        getSourceCodeFromRange(rewriter, rhs->getSourceRange()));
    IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::parseRHS] ")
        << "Basic expression: " << basicExpr->getSrcText() << ", def: [";
    for (auto def : basicExpr->getDefines()) {
      IR_BUILDER_DBG_OUT << def->varName << ", ";
    }
    IR_BUILDER_DBG_OUT << "], use: [";
    for (auto use : basicExpr->getUses()) {
      IR_BUILDER_DBG_OUT << use->varName << ", ";
    }
    IR_BUILDER_DBG_OUT << "]" << endl;

    IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::parseRHS] ")
        << "exprs size: " << exprs.size() << endl;

    return basicExpr;
  }

  vector<GenNmExpression *> &getExprs() { return exprs; }

private:
  optional<GenNmVarExpr *> parseLHS(Expr *lhs) {
    return lhsBuilder.parseLHS(lhs);
  }

  optional<string> parseCallee(Expr *callee) {
    callee = callee->IgnoreParenCasts();
    switch (callee->getStmtClass()) {
    case Stmt::DeclRefExprClass: {
      auto declRefExpr = dyn_cast<DeclRefExpr>(callee);
      auto decl = declRefExpr->getDecl();
      auto declName = decl->getName().str();
      return declName;
    }
    case Stmt::RecoveryExprClass: {
      auto recoveryExpr = dyn_cast<RecoveryExpr>(callee);
      auto srcStr =
          getSourceCodeFromRange(rewriter, recoveryExpr->getSourceRange());
      return srcStr;
    }
    default: {
      IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::parseCallee] ")
          << "Cannot parse callee with className " << callee->getStmtClassName()
          << endl;
      return nullopt;
    }
    }
  }

  bool isDirectUse(Stmt *clangStmt, string varName) {
    if (Expr *expr = dyn_cast<Expr>(clangStmt)) {
      clangStmt = expr->IgnoreParenCasts();
    }
    switch (clangStmt->getStmtClass()) {
    case Stmt::DeclRefExprClass: {
      auto declRefExpr = dyn_cast<DeclRefExpr>(clangStmt);
      auto decl = declRefExpr->getDecl();
      auto declName = decl->getName().str();
      return declName == varName;
    }
    case Stmt::RecoveryExprClass: {
      auto recoveryExpr = dyn_cast<RecoveryExpr>(clangStmt);
      auto srcStr =
          getSourceCodeFromRange(rewriter, recoveryExpr->getSourceRange());
      return srcStr == varName;
    }
    case Stmt::BinaryOperatorClass: {
      auto binaryOp = dyn_cast<BinaryOperator>(clangStmt);
      if (binaryOp->isAssignmentOp()) {
        return true;
      }
      return false;
    }
    case Stmt::CompoundAssignOperatorClass: {
      return false;
    }
    case Stmt::UnaryOperatorClass: {
      auto unaryOp = dyn_cast<UnaryOperator>(clangStmt);
      if (unaryOp->isIncrementDecrementOp()) {
        return true;
      }
      return false;
    }
    case Stmt::StringLiteralClass:
    case Stmt::IntegerLiteralClass:
    case Stmt::FloatingLiteralClass:
    case Stmt::CharacterLiteralClass:
      return false;
    default: {
      for (auto child = clangStmt->child_begin();
           child != clangStmt->child_end(); child++) {
        if (!isDirectUse(*child, varName)) {
          return false;
        }
      }
      return true;
    }
    }
  }

  // intermediate expressions: exprs that may be used by other expressions
  // final expressions: exprs that we don't have to trace their uses
  // for example:
  // c = a = sub_12345(arg1, arg2) + b
  // after parsing `sub_12345(arg1, arg2)`, `arg1` and `arg2` are final exprs
  // the results should look like:
  // in exprs: [GenNmExpr(arg1),
  //            GenNmExpr(arg2),
  //            CallExpr(sub_12345(arg1, arg2)),
  //            {use: [implicit(sub_12345), b], def: [a]},
  //            {use: [a], def: [c]}
  //           ]
  // returns intermediate expressions
  // add final expressions to exprs
  GenNmDefUse *createGenNmExpression(Stmt *clangExpr) {
    if (Expr *expr = dyn_cast<Expr>(clangExpr)) {
      clangExpr = expr->IgnoreParenCasts();
    }
    switch (clangExpr->getStmtClass()) {
    case Stmt::DeclRefExprClass: {
      IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::createGenNmExpression] ")
          << "DeclRefExpr" << endl;
      auto declRefExpr = dyn_cast<DeclRefExpr>(clangExpr);
      auto decl = declRefExpr->getDecl();
      auto declName = decl->getName().str();
      auto varNameExpr = new GenNmVarExpr(
          declName,
          getSourceCodeFromRange(rewriter, clangExpr->getSourceRange()));
      auto defUse = new GenNmDefUse();
      defUse->uses.insert(varNameExpr);
      defUse->isDirectUse = true;
      return defUse;
    }
    case Stmt::RecoveryExprClass: {
      IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::createGenNmExpression] ")
          << "RecoveryExpr" << endl;
      auto recoveryExpr = dyn_cast<RecoveryExpr>(clangExpr);
      auto srcStr =
          getSourceCodeFromRange(rewriter, recoveryExpr->getSourceRange());
      auto varNameExpr = new GenNmVarExpr(srcStr, srcStr);
      auto defUse = new GenNmDefUse();
      defUse->uses.insert(varNameExpr);
      defUse->isDirectUse = true;
      return defUse;
    }
    case Stmt::ArraySubscriptExprClass: {
      IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::createGenNmExpression] ")
          << "ArraySubscriptExpr" << endl;
      auto arraySubscriptExpr = dyn_cast<ArraySubscriptExpr>(clangExpr);
      auto base = arraySubscriptExpr->getBase();
      auto index = arraySubscriptExpr->getIdx();
      auto baseDefUse = createGenNmExpression(base);
      auto indexDefUse = createGenNmExpression(index);
      baseDefUse->defs.insert(indexDefUse->defs.begin(),
                              indexDefUse->defs.end());
      baseDefUse->uses.insert(indexDefUse->uses.begin(),
                              indexDefUse->uses.end());
      delete indexDefUse;
      baseDefUse->isDirectUse = false;
      return baseDefUse;
    }
    case Stmt::BinaryOperatorClass: {
      IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::createGenNmExpression] ")
          << "BinaryOperator" << endl;
      auto binaryOperator = dyn_cast<BinaryOperator>(clangExpr);
      auto lhs = binaryOperator->getLHS();
      auto rhs = binaryOperator->getRHS();
      auto lhsDefUse = createGenNmExpression(lhs);
      auto rhsDefUse = createGenNmExpression(rhs);
      if (binaryOperator->isAssignmentOp()) {
        auto lhsAsLeftValue = parseLHS(lhs);
        if (lhsAsLeftValue.has_value()) {
          rhsDefUse->defs.insert(lhsAsLeftValue.value());
        } else {
          rhsDefUse->defs.insert(lhsDefUse->defs.begin(),
                                 lhsDefUse->defs.end());
          rhsDefUse->uses.insert(lhsDefUse->uses.begin(),
                                 lhsDefUse->uses.end());
        }
      } else {
        rhsDefUse->defs.insert(lhsDefUse->defs.begin(), lhsDefUse->defs.end());
        rhsDefUse->uses.insert(lhsDefUse->uses.begin(), lhsDefUse->uses.end());
        rhsDefUse->isDirectUse = false;
      }
      delete lhsDefUse;
      return rhsDefUse;
    }
    case Stmt::CompoundAssignOperatorClass: {
      IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::createGenNmExpression] ")
          << "CompoundAssignOperator" << endl;
      auto compoundAssignOperator = dyn_cast<CompoundAssignOperator>(clangExpr);
      auto lhs = compoundAssignOperator->getLHS();
      auto rhs = compoundAssignOperator->getRHS();
      auto lhsDefUse = createGenNmExpression(lhs);
      auto rhsDefUse = createGenNmExpression(rhs);
      auto lhsAsLeftValue = parseLHS(lhs);
      if (lhsAsLeftValue.has_value()) {
        lhsDefUse->defs.insert(lhsAsLeftValue.value());
        lhsDefUse->uses.insert(lhsAsLeftValue.value());
      }
      lhsDefUse->defs.insert(rhsDefUse->defs.begin(), rhsDefUse->defs.end());
      lhsDefUse->uses.insert(rhsDefUse->uses.begin(), rhsDefUse->uses.end());
      delete rhsDefUse;
      lhsDefUse->isDirectUse = false;
      return lhsDefUse;
    }
    case Stmt::UnaryOperatorClass: {
      IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::createGenNmExpression] ")
          << "UnaryOperator" << endl;
      auto unaryOperator = dyn_cast<UnaryOperator>(clangExpr);

      auto subExpr = unaryOperator->getSubExpr();
      auto subExprDefUse = createGenNmExpression(subExpr);
      if(unaryOperator->isIncrementDecrementOp()) {
        // subExprDefUse->isDirectUse = true;
      }else{
        subExprDefUse->isDirectUse = false;
      }
      return subExprDefUse;
    }

    case Stmt::CallExprClass: {
      IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::createGenNmExpression] ")
          << "CallExpr" << endl;
      auto callExpr = dyn_cast<CallExpr>(clangExpr);
      auto gennmExprArgs = vector<GenNmExpression *>();
      // construct arg list expressions from the argument list
      // we handle special cases where the argument is a direct use of a
      // variable, or an implicit return variable of a function call
      for (auto arg = callExpr->arg_begin(); arg != callExpr->arg_end();
           arg++) {
        auto srcText =
            getSourceCodeFromRange(rewriter, (*arg)->getSourceRange());
        auto argDefUse = createGenNmExpression(*arg);
        if (argDefUse->uses.size() == 1 && argDefUse->defs.size() == 0) {
          auto uses = argDefUse->uses;
          auto usedVarName = (*uses.begin())->varName;
          if (isDirectUse(*arg, usedVarName)) {
            // a direct use of a variable
            auto varNameExpr = new GenNmVarExpr(usedVarName, srcText);
            gennmExprArgs.push_back(varNameExpr);
            delete argDefUse;
            continue;
          } else if (GenNmImplicitReturnVarExpr *implRetVar =
                         dynamic_cast<GenNmImplicitReturnVarExpr *>(
                             *uses.begin())) {
            // an implicit return variable of a function call
            gennmExprArgs.push_back(implRetVar);
            delete argDefUse;
            continue;
          }
        }
        // in other cases, we create a basic expression
        auto basicExpr =
            new GenNmBasicExpr(argDefUse->defs, argDefUse->uses, false, srcText);
        gennmExprArgs.push_back(basicExpr);
        delete argDefUse;
      }

      auto callee = callExpr->getCallee();
      auto gennmCalleeNameOpt = parseCallee(callee);
      if (!gennmCalleeNameOpt.has_value()) {
        IR_BUILDER_DBG_OUT_PREFIX("[RHSBuilder::callExpr] ")
            << "Cannot parse callee for expr: "
            << getSourceCodeFromRange(rewriter, callee->getSourceRange())
            << endl;
        // if there's no callee name, we don't try to construct a call
        // expression instead, we simply add all arguments to exprs and
        // return
        for (auto expr : gennmExprArgs) {
          exprs.push_back(expr);
        }
        // it's worth noting that variable def/use cannot pass through a
        // function call
        return new GenNmDefUse();
      }
      auto gennmCalleeName = gennmCalleeNameOpt.value();
      auto gennmCallExpr = new GenNmCallExpr(
          gennmCalleeName, gennmExprArgs,
          getSourceCodeFromRange(rewriter, callExpr->getSourceRange()));
      exprs.push_back(gennmCallExpr);
      auto defUse = new GenNmDefUse();
      auto implicitUseRetValue = new GenNmImplicitReturnVarExpr(
          "__gennm_impl_ret_" + gennmCalleeName, gennmCalleeName,
          getSourceCodeFromRange(rewriter, callExpr->getSourceRange()));
      defUse->uses.insert(implicitUseRetValue);
      defUse->isDirectUse = true;
      return defUse;
    }

    default:
      GenNmDefUse *defUse = new GenNmDefUse();
      for (auto child = clangExpr->child_begin();
           child != clangExpr->child_end(); child++) {
        auto childDefUse = createGenNmExpression(*child);
        defUse->defs.insert(childDefUse->defs.begin(), childDefUse->defs.end());
        defUse->uses.insert(childDefUse->uses.begin(), childDefUse->uses.end());
        delete childDefUse;
      }
      return defUse;
    }
  }

  Rewriter &rewriter;
  LHSGenNmExprBuilder lhsBuilder = LHSGenNmExprBuilder(rewriter);
  // Essentially, gennm expressions are "relations", instead of operations.
  // To facilitate future development, we guarantee that the expressions
  // appears in post order. i.e., the expressions defining operands appear
  // before the expressions defining the operator
  vector<GenNmExpression *> exprs = {};
};

class FunctionIRBuilder {

  struct GenNmBreakContinue {
    GenNmBasicBlock *breakBB = nullptr, *continueBB = nullptr;
  };

public:
  explicit FunctionIRBuilder(clang::Rewriter &R) : rewriter(R) {}

  string getIndent() {
    string ret = "";
    for (int i = 0; i < indent; i++) {
      ret += "  ";
    }
    return ret;
  }

  void incIndent() { indent++; }
  void decIndent() { indent--; }

  GenNmFunction *buildFunction(const FunctionDecl &funcDecl) {
    incIndent();
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildFunction] ")
        << "Building function " << funcDecl.getNameAsString() << endl;

    incIndent();
    auto params = funcDecl.parameters();
    vector<GenNmVarExpr *> paramExprs;
    for (auto &param : params) {
      auto paramSrcStr = getSourceCodeFromRange(param->getSourceRange());
      IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildFunction] ")
          << "param: " << paramSrcStr << endl;
      auto paramExpr = new GenNmVarExpr(param->getName().str(), paramSrcStr);
      paramExprs.push_back(paramExpr);
    }
    decIndent();
    vector<GenNmBasicBlock *> bbList = {};
    auto func =
        new GenNmFunction(funcDecl.getNameAsString(), paramExprs, bbList);
    curFunc = func;
    auto bb = getNewBasicBlock(true, "entry");
    auto body = funcDecl.getBody();
    buildStmt(body);
    decIndent();
    return func;
  }

  void buildDeclStmt(DeclStmt *clangStmt) {
    incIndent();
    for (auto decl = clangStmt->decl_begin(); decl != clangStmt->decl_end();
         decl++) {
      auto declStmt = *decl;
      if (!isa<VarDecl>(declStmt)) {
        IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildDeclStmt] ")
            << "decl is not a var decl" << endl;
        continue;
      }
      auto varDecl = dyn_cast<VarDecl>(declStmt);
      auto declSrcStr = getSourceCodeFromRange(varDecl->getSourceRange());
      IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildDeclStmt] ")
          << "decl: " << declSrcStr << endl;
      if (!varDecl->hasInit()) {
        IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildDeclStmt] ")
            << "decl has no init" << endl;
        continue;
      }
      auto init = varDecl->getInit();
      auto rhsExpr = rhsBuilder.parseRHS(init);
      auto exprsFromRhs = rhsBuilder.getExprs();
      curBB->getStatements().insert(curBB->getStatements().end(),
                                    exprsFromRhs.begin(), exprsFromRhs.end());
      auto varName = varDecl->getName().str();
      auto lhs = new GenNmVarExpr(varName, declSrcStr);
      auto assignmentStmt = new GenNmAssignStmt(lhs, rhsExpr, declSrcStr);
      curBB->getStatements().push_back(assignmentStmt);
      IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildDeclStmt] ")
          << "assignmentStmt: lhs: " << assignmentStmt->getLHS()->varName
          << ", rhs: " << assignmentStmt->getRHS()->getSrcText() << endl;
    }
  }

  void buildReturnStmt(ReturnStmt *clangRetStmt) {
    incIndent();
    auto retVal = clangRetStmt->getRetValue();
    auto retValExpr = rhsBuilder.parseRHS(retVal);
    auto exprsFromRhs = rhsBuilder.getExprs();
    curBB->getStatements().insert(curBB->getStatements().end(),
                                  exprsFromRhs.begin(), exprsFromRhs.end());
    auto returnStmt = new GenNmReturnStmt(
        retValExpr, getSourceCodeFromRange(clangRetStmt->getSourceRange()));
    curBB->getStatements().push_back(returnStmt);
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildReturnStmt] ")
        << "returnStmt: " << returnStmt->getSrcText() << endl;
    decIndent();
  }

  void buildIfStmt(IfStmt *clangIfStmt) {
    incIndent();
    auto cond = clangIfStmt->getCond();
    // we don't care the condExpr, simply add it to the current basic block
    auto condExpr = rhsBuilder.parseRHS(cond);
    auto exprsFromRhs = rhsBuilder.getExprs();
    curBB->getStatements().insert(curBB->getStatements().end(),
                                  exprsFromRhs.begin(), exprsFromRhs.end());
    curBB->getStatements().push_back(condExpr);
    auto branchStmt =
        new GenNmBranchStmt({}, getSourceCodeFromRange(cond->getSourceRange()));
    curBB->getStatements().push_back(branchStmt);
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildIfStmt] ")
        << "branchStmt: " << branchStmt->getSrcText() << endl;
    if (auto dbgCondExpr = dynamic_cast<GenNmBasicExpr *>(condExpr)) {
      auto sizeOfDef = dbgCondExpr->getDefines().size();
      auto sizeOfUse = dbgCondExpr->getUses().size();
      IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildIfStmt] ")
          << "condExpr: " << condExpr->getSrcText() << ", def: [";
      for (auto def : dbgCondExpr->getDefines()) {
        IR_BUILDER_DBG_OUT << def->varName << ", ";
      }
      IR_BUILDER_DBG_OUT << "], use: [";
      for (auto use : dbgCondExpr->getUses()) {
        IR_BUILDER_DBG_OUT << use->varName << ", ";
      }
      IR_BUILDER_DBG_OUT << "]" << endl;
    }
    auto thenBlk = getNewBasicBlock(false, "if.then");
    auto elseBlk = getNewBasicBlock(false, "if.else");
    auto mergeBlk = getNewBasicBlock(false, "if.end");
    branchStmt->getSuccessors().push_back(thenBlk);
    branchStmt->getSuccessors().push_back(elseBlk);
    thenBlk->getPredecessors().push_back(curBB);
    elseBlk->getPredecessors().push_back(curBB);
    curBB->getSuccessors().push_back(thenBlk);
    curBB->getSuccessors().push_back(elseBlk);
    curBB = thenBlk;
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildIfStmt] ")
        << "Emiting then blk" << endl;
    buildStmt(clangIfStmt->getThen());
    if (!curBB->terminated()) {
      // add a branch to the merge block
      auto thenBranchStmt = new GenNmBranchStmt({mergeBlk}, "End of if.then");
      curBB->getStatements().push_back(thenBranchStmt);
      curBB->getSuccessors().push_back(mergeBlk);
      mergeBlk->getPredecessors().push_back(curBB);
    }
    curBB = elseBlk;
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildIfStmt] ")
        << "Emiting else blk" << endl;
    if (clangIfStmt->getElse() != nullptr) {
      buildStmt(clangIfStmt->getElse());
    }
    if (!curBB->terminated()) {
      // add a branch to the merge block
      auto elseBranchStmt = new GenNmBranchStmt({mergeBlk}, "End of if.else");
      curBB->getStatements().push_back(elseBranchStmt);
      curBB->getSuccessors().push_back(mergeBlk);
      mergeBlk->getPredecessors().push_back(curBB);
    }
    curBB = mergeBlk;
    decIndent();
  }

  void buildWhileStmt(WhileStmt *clangWhileStmt) {
    incIndent();
    auto condBlk = getNewBasicBlock(true, "while.cond");
    auto cond = clangWhileStmt->getCond();
    // we don't care the condExpr, simply add it to the current basic block
    auto condExpr = rhsBuilder.parseRHS(cond);
    auto exprsFromRhs = rhsBuilder.getExprs();
    curBB->getStatements().insert(curBB->getStatements().end(),
                                  exprsFromRhs.begin(), exprsFromRhs.end());
    curBB->getStatements().push_back(condExpr);
    auto branchStmt =
        new GenNmBranchStmt({}, getSourceCodeFromRange(cond->getSourceRange()));
    curBB->getStatements().push_back(branchStmt);
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildWhileStmt] ")
        << "branchStmt: " << branchStmt->getSrcText() << endl;
    auto bodyBlk = getNewBasicBlock(false, "while.body");
    auto mergeBlk = getNewBasicBlock(false, "while.end");
    branchStmt->getSuccessors().push_back(bodyBlk);
    branchStmt->getSuccessors().push_back(mergeBlk);
    bodyBlk->getPredecessors().push_back(curBB);
    mergeBlk->getPredecessors().push_back(curBB);
    curBB->getSuccessors().push_back(bodyBlk);
    curBB->getSuccessors().push_back(mergeBlk);
    GenNmBreakContinue breakContinue = {};
    breakContinue.continueBB = condBlk;
    breakContinue.breakBB = mergeBlk;
    breakContinueStack.push_back(breakContinue);
    curBB = bodyBlk;
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildWhileStmt] ")
        << "Emiting body blk" << endl;
    buildStmt(clangWhileStmt->getBody());
    if (!curBB->terminated()) {
      // add a branch to the cond block
      auto bodyBranchStmt = new GenNmBranchStmt({condBlk}, "End of while.body");
      curBB->getStatements().push_back(bodyBranchStmt);
      curBB->getSuccessors().push_back(condBlk);
      condBlk->getPredecessors().push_back(curBB);
    }
    breakContinueStack.pop_back();
    curBB = mergeBlk;
  }

  void buildForStmt(Stmt *clangStmt) {
    incIndent();
    auto forStmt = dyn_cast<ForStmt>(clangStmt);
    auto init = forStmt->getInit();
    buildStmt(init);
    auto cond = forStmt->getCond();
    auto inc = forStmt->getInc();
    auto body = forStmt->getBody();
    auto condBlk = getNewBasicBlock(true, "for.cond");
    auto bodyBlk = getNewBasicBlock(false, "for.body");
    auto incBlk = getNewBasicBlock(false, "for.inc");
    auto mergeBlk = getNewBasicBlock(false, "for.end");
    auto branchStmt = new GenNmBranchStmt(
        {bodyBlk, mergeBlk}, getSourceCodeFromRange(cond->getSourceRange()));
    curBB->getStatements().push_back(branchStmt);
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildForStmt] ")
        << "branchStmt: " << branchStmt->getSrcText() << endl;
    bodyBlk->getPredecessors().push_back(curBB);
    mergeBlk->getPredecessors().push_back(curBB);
    curBB->getSuccessors().push_back(bodyBlk);
    curBB->getSuccessors().push_back(mergeBlk);
    GenNmBreakContinue breakContinue = {};
    breakContinue.continueBB = incBlk;
    breakContinue.breakBB = mergeBlk;
    breakContinueStack.push_back(breakContinue);
    curBB = bodyBlk;
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildForStmt] ")
        << "Emiting body blk" << endl;
    buildStmt(body);
    if (!curBB->terminated()) {
      // add a branch to the inc block
      auto bodyBranchStmt = new GenNmBranchStmt({incBlk}, "End of for.body");
      curBB->getStatements().push_back(bodyBranchStmt);
      curBB->getSuccessors().push_back(incBlk);
      incBlk->getPredecessors().push_back(curBB);
    }
    curBB = incBlk;
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildForStmt] ")
        << "Emiting inc blk" << endl;
    buildStmt(inc);
    if (!curBB->terminated()) {
      // add a branch to the cond block
      auto incBranchStmt = new GenNmBranchStmt({condBlk}, "End of for.inc");
      curBB->getStatements().push_back(incBranchStmt);
      curBB->getSuccessors().push_back(condBlk);
      condBlk->getPredecessors().push_back(curBB);
    }
    breakContinueStack.pop_back();
    curBB = mergeBlk;
    decIndent();
  }

  void buildDoStmt(Stmt *clangDoStmt) {
    incIndent();
    auto doStmt = dyn_cast<DoStmt>(clangDoStmt);
    auto cond = doStmt->getCond();
    auto body = doStmt->getBody();
    auto bodyBlk = getNewBasicBlock(true, "do.body");
    auto condBlk = getNewBasicBlock(false, "do.cond");
    auto mergeBlk = getNewBasicBlock(false, "do.end");
    GenNmBreakContinue breakContinue = {};
    breakContinue.continueBB = condBlk;
    breakContinue.breakBB = mergeBlk;
    breakContinueStack.push_back(breakContinue);
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildDoStmt] ")
        << "Emiting body blk" << endl;
    buildStmt(body);
    if (!curBB->terminated()) {
      // add a branch to the cond block
      auto bodyBranchStmt = new GenNmBranchStmt({condBlk}, "End of do.body");
      curBB->getStatements().push_back(bodyBranchStmt);
      curBB->getSuccessors().push_back(condBlk);
      condBlk->getPredecessors().push_back(curBB);
    }
    curBB = condBlk;
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildDoStmt] ")
        << "Emiting cond blk" << endl;
    auto condExpr = rhsBuilder.parseRHS(cond);
    auto exprsFromRhs = rhsBuilder.getExprs();
    curBB->getStatements().insert(curBB->getStatements().end(),
                                  exprsFromRhs.begin(), exprsFromRhs.end());
    curBB->getStatements().push_back(condExpr);
    auto branchStmt = new GenNmBranchStmt(
        {bodyBlk, mergeBlk}, getSourceCodeFromRange(cond->getSourceRange()));
    curBB->getStatements().push_back(branchStmt);
    IR_BUILDER_DBG_OUT_PREFIX("[FIRBuilder:buildDoStmt] ")
        << "branchStmt: " << branchStmt->getSrcText() << endl;
    bodyBlk->getPredecessors().push_back(curBB);
    mergeBlk->getPredecessors().push_back(curBB);
    curBB->getSuccessors().push_back(bodyBlk);
    curBB->getSuccessors().push_back(mergeBlk);
    breakContinueStack.pop_back();
    curBB = mergeBlk;
    decIndent();
  }

  void buildStmt(Stmt *clangStmt) {
    switch (clangStmt->getStmtClass()) {
    case Stmt::DeclStmtClass: {
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "DeclStmt" << endl;
      auto declStmt = dyn_cast<DeclStmt>(clangStmt);
      buildDeclStmt(declStmt);
      break;
    }
    case Stmt::ReturnStmtClass: {
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "ReturnStmt" << endl;
      auto returnStmt = dyn_cast<ReturnStmt>(clangStmt);
      buildReturnStmt(returnStmt);
      break;
    }
    case Stmt::IfStmtClass: {
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "IfStmt" << endl;
      auto ifStmt = dyn_cast<IfStmt>(clangStmt);
      buildIfStmt(ifStmt);
      break;
    }
    case Stmt::WhileStmtClass: {
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "WhileStmt" << endl;
      auto whileStmt = dyn_cast<WhileStmt>(clangStmt);
      buildWhileStmt(whileStmt);
      break;
    }

    case Stmt::ForStmtClass: {
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "ForStmt" << endl;
      auto forStmt = dyn_cast<ForStmt>(clangStmt);
      buildForStmt(forStmt);
      break;
    }
    case Stmt::CompoundStmtClass: {
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "CompoundStmt" << endl;
      auto compoundStmt = dyn_cast<CompoundStmt>(clangStmt);
      for (auto stmt = compoundStmt->body_begin();
           stmt != compoundStmt->body_end(); stmt++) {
        buildStmt(*stmt);
      }
      break;
    }
    case Stmt::BreakStmtClass: {
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "BreakStmt" << endl;
      auto breakStmt = dyn_cast<BreakStmt>(clangStmt);
      auto breakContinue = breakContinueStack.back();
      auto breakBB = breakContinue.breakBB;
      auto breakBranchStmt = new GenNmBranchStmt({breakBB}, "break");
      curBB->getStatements().push_back(breakBranchStmt);
      curBB->getSuccessors().push_back(breakBB);
      breakBB->getPredecessors().push_back(curBB);
      break;
    }
    case Stmt::ContinueStmtClass: {
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "ContinueStmt" << endl;
      auto continueStmt = dyn_cast<ContinueStmt>(clangStmt);
      auto breakContinue = breakContinueStack.back();
      auto continueBB = breakContinue.continueBB;
      auto continueBranchStmt = new GenNmBranchStmt({continueBB}, "continue");
      curBB->getStatements().push_back(continueBranchStmt);
      curBB->getSuccessors().push_back(continueBB);
      continueBB->getPredecessors().push_back(curBB);
      break;
    }
    case Stmt::DoStmtClass: {
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "DoStmt" << endl;
      auto doStmt = dyn_cast<DoStmt>(clangStmt);
      buildDoStmt(doStmt->getBody());
      break;
    }
    case Stmt::LabelStmtClass: {
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "LabelStmt" << endl;
      auto labelStmt = dyn_cast<LabelStmt>(clangStmt);
      auto labelDecl = labelStmt->getDecl();
      auto labelName = labelDecl->getName().str();
      if (label2BB.count(labelName)) {
        auto labelBB = label2BB[labelName];
        curBB->getSuccessors().push_back(labelBB);
        labelBB->getPredecessors().push_back(curBB);
        curBB = labelBB;
      } else {
        auto labelBB = getNewBasicBlock(true, labelName);
        label2BB[labelName] = labelBB;
      }
      auto subStmt = labelStmt->getSubStmt();
      buildStmt(subStmt);
      break;
    }
    case Stmt::GotoStmtClass: {
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "GotoStmt" << endl;
      auto gotoStmt = dyn_cast<GotoStmt>(clangStmt);
      auto labelDecl = gotoStmt->getLabel();
      auto labelName = labelDecl->getName().str();
      GenNmBasicBlock *labelBB = nullptr;
      if (label2BB.count(labelName)) {
        labelBB = label2BB[labelName];
      } else {
        labelBB = getNewBasicBlock(false, labelName);
        label2BB[labelName] = labelBB;
      }
      auto gennmBranchStmt = new GenNmBranchStmt({labelBB}, "goto");
      curBB->getSuccessors().push_back(labelBB);
      labelBB->getPredecessors().push_back(curBB);
      curBB->getStatements().push_back(gennmBranchStmt);
      break;
    }
    case Stmt::SwitchStmtClass: {
      // TODO
      IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ") << "SwitchStmt" << endl;
      break;
    }
    default: {
      // if is an expression, we parse it as an expression
      if (auto expr = dyn_cast<Expr>(clangStmt)) {
        auto exprs = rhsBuilder.parseRHS(expr);
        auto exprsFromRhs = rhsBuilder.getExprs();
        curBB->getStatements().insert(curBB->getStatements().end(),
                                      exprsFromRhs.begin(), exprsFromRhs.end());
        curBB->getStatements().push_back(exprs);
      } else {
        IR_DBG_OUT_PREFIX("[IRBuilder:buildStmt] ")
            << "Unhandled Stmt: " << clangStmt->getStmtClassName() << endl;
      }
      break;
    }
    }
  }
  GenNmBasicBlock *getNewBasicBlock(bool updateCurBB, string label = "") {
    auto bb = new GenNmBasicBlock(label);
    curFunc->getBasicBlocks().push_back(bb);
    if (updateCurBB) {
      if (curBB == nullptr) {
        curBB = bb;
      } else {
        curBB->getSuccessors().push_back(bb);
        bb->getPredecessors().push_back(curBB);
        curBB = bb;
      }
    }
    return bb;
  }

private:
  string getSourceCodeFromRange(SourceRange range) {
    return ::getSourceCodeFromRange(rewriter, range);
  }

  vector<GenNmBreakContinue> breakContinueStack = {};
  int indent = 0;
  Rewriter &rewriter;
  LHSGenNmExprBuilder lhsBuilder = LHSGenNmExprBuilder(rewriter);
  RHSGenNmExprBuilder rhsBuilder = RHSGenNmExprBuilder(rewriter);
  unordered_map<string, GenNmBasicBlock *> label2BB = {};
  GenNmFunction *curFunc = nullptr;
  GenNmBasicBlock *curBB = nullptr;
};

bool IRBuilder::HandleTopLevelDecl(DeclGroupRef DR) {
  for (auto &decl : DR) {
    if (auto funcDecl = dyn_cast<FunctionDecl>(decl)) {
      IR_BUILDER_DBG_OUT_PREFIX("[IRBuilder:HandleTopLevelDecl] ")
          << "FunctionDecl: " << funcDecl->getNameAsString() << endl;
      auto funcIRBuilder = FunctionIRBuilder(rewriter);
      auto func = funcIRBuilder.buildFunction(*funcDecl);
      functions.push_back(func);
    }
  }
  return true;
}