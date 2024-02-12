#include <clang/Analysis/CFG.h>

#include <iostream>

#include "cfg.hh"
#include "cfg_builder.hh"
#include "cfg_print_visitor.hh"
#include "dbg_config.hh"
#include "utils.hh"

using namespace clang;
using namespace std;

#ifdef DBG_CFG_BUILDER
#define CFG_BUILDER_DBG_OUT                                                    \
  if (true)                                                                    \
  cout << "[CFG_BUILDER] "
#else
#define CFG_BUILDER_DBG_OUT                                                    \
  if (false)                                                                   \
  cout
#endif

/// Note: all functions beginning with 'build' allocates dynamic memory!
class CFGBuilderVisitor : public RecursiveASTVisitor<CFGBuilderVisitor> {
public:
  CFGBuilderVisitor(Rewriter &R) : rewriter(R) {}

  BinameProgram *buildCFG(Stmt *stmt, FunctionDecl* fd) {
    CFG_BUILDER_DBG_OUT << "buildCFG" << endl;
    auto seqStmt = buildSequentialStatement(stmt);
    cfg = new BinameProgram(seqStmt, fd);
    return cfg;
  }

  bool TraverseIfStmt(IfStmt *stmt) {
    if (seqStmtStack.empty()) {
      CFG_BUILDER_DBG_OUT << "TraverseIfStmt: empty stack" << endl;
      return true;
    }
    CFG_BUILDER_DBG_OUT << "TraverseIfStmt" << endl;
    // get condition
    CFG_BUILDER_DBG_OUT << "Construct condition statement" << endl;
    auto cond = buildSimpleClangStatement(stmt->getCond());
    // get then statement
    CFG_BUILDER_DBG_OUT << "Construct then statement" << endl;
    auto thenStmt = buildSequentialStatement(stmt->getThen());

    BinameStatement *elseStmt = nullptr;
    // get else statement
    if (stmt->getElse() == nullptr) {
      // no else statement
      CFG_BUILDER_DBG_OUT << "No else statement" << endl;
      elseStmt = new SequentialStatement();
    } else {
      CFG_BUILDER_DBG_OUT << "Construct else statement" << endl;
      elseStmt = buildSequentialStatement(stmt->getElse());
    }
    // build if statement
    CFG_BUILDER_DBG_OUT << "Construct if statement" << endl;
    auto ifStmt = new IfStatement(stmt, cond, thenStmt, elseStmt);
    auto seqStmt = seqStmtStack.back();
    seqStmt->statements.push_back(ifStmt);
    return true;
  }

  bool TraverseDeclStmt(DeclStmt *stmt) {
    CFG_BUILDER_DBG_OUT << "TraverseDeclStmt" << endl;
    pushBackStmt(stmt);
    return true;
  }

  bool TraverseGotoStmt(GotoStmt *stmt) {
    CFG_BUILDER_DBG_OUT << "TraverseGotoStmt" << endl;
    pushBackStmt(stmt);
    return true;
  }

  bool TraverseLabelDecl(LabelDecl *decl) {
    CFG_BUILDER_DBG_OUT << "TraverseLabelDecl" << endl;
    pushBackDecl(decl);
    return true;
  }

  bool TraverseNullStmt(NullStmt *stmt) {
    CFG_BUILDER_DBG_OUT << "TraverseNullStmt" << endl;
    pushBackStmt(stmt);
    return true;
  }

  bool TraverseReturnStmt(ReturnStmt *stmt) {
    CFG_BUILDER_DBG_OUT << "TraverseReturnStmt" << endl;
    ExecTermStatement *retStmt = new ExecTermStatement(stmt);
    if (seqStmtStack.empty()) {
      CFG_BUILDER_DBG_OUT << "TraverseReturnStmt: empty stack" << endl;
      return true;
    }
    auto seqStmt = seqStmtStack.back();
    seqStmt->statements.push_back(retStmt);
    return true;
  }

  bool TraverseSwitchStmt(SwitchStmt *stmt) {
    // TODO
    pushBackStmt(stmt);
    return true;
  }

  bool TraverseWhileStmt(WhileStmt *stmt) {
    if (seqStmtStack.empty()) {
      CFG_BUILDER_DBG_OUT << "TraverseWhileStmt: empty stack" << endl;
      return true;
    }
    CFG_BUILDER_DBG_OUT << "TraverseWhileStmt" << endl;
    // get condition
    CFG_BUILDER_DBG_OUT << "Construct condition statement" << endl;
    auto cond = buildSimpleClangStatement(stmt->getCond());
    // get body
    CFG_BUILDER_DBG_OUT << "Construct body statement" << endl;
    auto body = buildSequentialStatement(stmt->getBody());
    // build while statement
    CFG_BUILDER_DBG_OUT << "Construct while statement" << endl;
    auto whileStmt = new WhileStatement(stmt, cond, body);
    auto seqStmt = seqStmtStack.back();
    seqStmt->statements.push_back(whileStmt);
    return true;
  }
  

  bool TraverseDoStmt(DoStmt *stmt) {
    if (seqStmtStack.empty()) {
      CFG_BUILDER_DBG_OUT << "TraverseDoStmt: empty stack" << endl;
      return true;
    }
    CFG_BUILDER_DBG_OUT << "TraverseDoStmt" << endl;
    // get condition
    CFG_BUILDER_DBG_OUT << "Construct condition statement" << endl;
    auto cond = buildSimpleClangStatement(stmt->getCond());
    // get body
    CFG_BUILDER_DBG_OUT << "Construct body statement" << endl;
    auto body = buildSequentialStatement(stmt->getBody());
    // build while statement
    CFG_BUILDER_DBG_OUT << "Construct do while statement" << endl;
    auto doWhileStmt = new DoWhileStatement(stmt, cond, body);
    auto seqStmt = seqStmtStack.back();
    seqStmt->statements.push_back(doWhileStmt);    
    return true;
  }

  bool TraverseForStmt(ForStmt *stmt) {
    if (seqStmtStack.empty()) {
      CFG_BUILDER_DBG_OUT << "TraverseForStmt: empty stack" << endl;
      return true;
    }
    CFG_BUILDER_DBG_OUT << "TraverseForStmt" << endl;
    // get init
    CFG_BUILDER_DBG_OUT << "Construct init statement" << endl;
    auto init = buildSimpleClangStatement(stmt->getInit());
    // get condition
    CFG_BUILDER_DBG_OUT << "Construct condition statement" << endl;
    auto cond = buildSimpleClangStatement(stmt->getCond());
    // get increment
    CFG_BUILDER_DBG_OUT << "Construct increment statement" << endl;
    auto inc = buildSimpleClangStatement(stmt->getInc());
    // get body
    CFG_BUILDER_DBG_OUT << "Construct body statement" << endl;
    auto body = buildSequentialStatement(stmt->getBody());
    // build for statement
    CFG_BUILDER_DBG_OUT << "Construct for statement" << endl;
    auto forStmt = new ForStatement(stmt, init, cond, inc, body);
    auto seqStmt = seqStmtStack.back();
    seqStmt->statements.push_back(forStmt);
    return true;
  }

  bool TraverseBreakStmt(BreakStmt *stmt) {
    CFG_BUILDER_DBG_OUT << "TraverseBreakStmt" << endl;
    LoopTermStatement *breakStmt = new LoopTermStatement(stmt);
    if (seqStmtStack.empty()) {
      CFG_BUILDER_DBG_OUT << "TraverseBreakStmt: empty stack" << endl;
      return true;
    }
    auto seqStmt = seqStmtStack.back();
    seqStmt->statements.push_back(breakStmt);
    return true;
  }

  bool TraverseContinueStmt(ContinueStmt *stmt) {
    CFG_BUILDER_DBG_OUT << "TraverseContinueStmt" << endl;
    LoopTermStatement *continueStmt = new LoopTermStatement(stmt);
    if (seqStmtStack.empty()) {
      CFG_BUILDER_DBG_OUT << "TraverseContinueStmt: empty stack" << endl;
      return true;
    }
    auto seqStmt = seqStmtStack.back();
    seqStmt->statements.push_back(continueStmt);
    return true;
  }

  bool TraverseConditionalOperator(ConditionalOperator *stmt) {
    CFG_BUILDER_DBG_OUT << "TraverseConditionalOperator" << endl;
    pushBackStmt(stmt);
    return true;
  }

  bool TraverseBinaryOperator(BinaryOperator *stmt) {
    CFG_BUILDER_DBG_OUT << "TraverseBinaryOperator" << endl;
    pushBackStmt(stmt);
    return true;
  }

  bool TraverseUnaryOperator(UnaryOperator *stmt) {
    CFG_BUILDER_DBG_OUT << "TraverseUnaryOperator" << endl;
    pushBackStmt(stmt);
    return true;
  }

  bool TraverseCompoundAssignOperator(CompoundAssignOperator *stmt) {
    CFG_BUILDER_DBG_OUT << "TraverseCompoundAssignOperator" << endl;
    pushBackStmt(stmt);
    return true;
  }

  bool TraverseCallExpr(CallExpr *stmt) {
    CFG_BUILDER_DBG_OUT << "TraverseCallExpr" << endl;
    pushBackStmt(stmt);
    return true;
  }



  // bool VisitStmt(Stmt *stmt) {
  //   if (seqStmtStack.empty()) {
  //     CFG_BUILDER_DBG_OUT << "VisitStmt: empty stack" << endl;
  //     return true;
  //   }
  //   CFG_BUILDER_DBG_OUT << "VisitStmt" << endl;
  //   auto seqStmt = seqStmtStack.back();
  //   seqStmt->statements.push_back(buildSimpleClangStatement(stmt));
  //   return true;
  // }

private:
  Rewriter &rewriter;
  BinameProgram *cfg;
  vector<SequentialStatement *> seqStmtStack;

  void pushBackDecl(Decl *decl) {
    if (seqStmtStack.empty()) {
      CFG_BUILDER_DBG_OUT << "pushBackDecl: empty stack" << endl;
      return;
    }
    auto seqStmt = seqStmtStack.back();
    seqStmt->statements.push_back(buildSimpleClangDecl(decl));
  }

  void pushBackStmt(Stmt *stmt) {
    if (seqStmtStack.empty()) {
      CFG_BUILDER_DBG_OUT << "pushBackStmt: empty stack" << endl;
      return;
    }
    auto seqStmt = seqStmtStack.back();
    seqStmt->statements.push_back(buildSimpleClangStatement(stmt));
  }

  SequentialStatement *buildSequentialStatement(Stmt *stmt) {
    CFG_BUILDER_DBG_OUT << "build SequentialStatement" << endl;
    seqStmtStack.push_back(new SequentialStatement());
    TraverseStmt(stmt);
    auto retSeqStmt = seqStmtStack.back();
    seqStmtStack.pop_back();
    return retSeqStmt;
  }

  SimpleClangDecl *buildSimpleClangDecl(Decl *decl) {
    CFG_BUILDER_DBG_OUT << "build SimpleClangDecl" << endl;
    return new SimpleClangDecl(decl);
  }

  SimpleClangStatement *buildSimpleClangStatement(Stmt *stmt) {
    CFG_BUILDER_DBG_OUT << "build SimpleClangStatement" << endl;
    return new SimpleClangStatement(stmt);
  }
};

bool CFGBuilder::HandleTopLevelDecl(DeclGroupRef DR) {  
  for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; b++) {
    if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*b)) {
      CFG_BUILDER_DBG_OUT << "Found function: " << fd->getNameAsString()
                          << endl;            
      // get function body
      Stmt *funcBody = fd->getBody();
      // build CFG
      CFGBuilderVisitor visitor(rewriter);
      BinameProgram *cfg = visitor.buildCFG(funcBody, fd);
#ifdef DBG_CFG_BUILDER
      // print CFG
      CFG_BUILDER_DBG_OUT
          << "======================= CFG ======================" << endl;
      CFGPrintVisitor printVisitor(rewriter);
      printVisitor.visit(cfg);
#endif
      programs.push_back(cfg);
    }
  }
  return true;
}

// bool CFGBuilder::HandleTopLevelDecl(DeclGroupRef DR) {
//   for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
//     if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*b)) {
//       CFG_BUILDER_DBG_OUT << "Found function: " << fd->getNameAsString()
//                           << endl;
//       // get function body
//       Stmt *funcBody = fd->getBody();
//       // build CFG
//       // get ast context
//       ASTContext &context = fd->getASTContext();
//       CFG *cfg =
//           CFG::buildCFG(fd, funcBody, &context, CFG::BuildOptions()).get();
//       // iterate over all the blocks
//       for (auto it = cfg->begin(); it != cfg->end(); ++it) {
//         CFGBlock *block = *it;
//         CFG_BUILDER_DBG_OUT << "block: " << block->getBlockID() << endl;

//         Stmt *first = nullptr, *last = nullptr;
//         for (auto stmtItr = block->begin(); stmtItr != block->end();
//              ++stmtItr) {
//           // dump to stream
//           // build llvm string stream
//           string dbg;
//           llvm::raw_string_ostream llvmStream(dbg);
//           // dump
//           stmtItr->dumpToStream(llvmStream);
//           // get string
//           string str = llvmStream.str();
//           CFG_BUILDER_DBG_OUT << "stmt: " << str << endl;
//           CFGElement element = *stmtItr;
//           if (Optional<CFGStmt> stmt = element.getAs<CFGStmt>()) {
//             Stmt *s = const_cast<Stmt *>(stmt->getStmt());
//             if (first == nullptr) {
//               first = s;
//             }
//             last = s;
//           }
//         }
//         // if the block is empty, continue
//         if (first == nullptr) {
//           CFG_BUILDER_DBG_OUT << "empty block" << endl;
//           continue;
//         }
//         // get source string from frist to last
//         SourceRange range =
//             SourceRange(first->getBeginLoc(), last->getEndLoc());
//         string text = rewriter.getRewrittenText(range);
//         CFG_BUILDER_DBG_OUT << "text: " << text << endl;

//         // // iterate over all the statements in the block
//         // for (auto it2 = block->begin(); it2 != block->end(); ++it2) {
//         //   CFGElement element = *it2;
//         //   if (Optional<CFGStmt> stmt = element.getAs<CFGStmt>()) {
//         //     Stmt *s = const_cast<Stmt *>(stmt->getStmt());
//         //     CFG_BUILDER_DBG_OUT << "stmt: " << s->getStmtClassName() <<
//         endl;
//         //     // get source string
//         //     SourceRange range = s->getSourceRange();
//         //     string text = rewriter.getRewrittenText(range);
//         //     CFG_BUILDER_DBG_OUT << "text: " << text << endl;
//         //   }
//         // }
//       }
//     }
//   }
//   return true;
// }