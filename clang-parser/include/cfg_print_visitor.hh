#ifndef CFG_PRINT_VISITOR_HH
#define CFG_PRINT_VISITOR_HH

#include "cfg.hh"
#include "cfg_visitor.hh"
#include "dbg_config.hh"
#include "utils.hh"

#include <iostream>

using namespace std;
using namespace clang;

#ifdef DBG_CFG_PRINT_VISITOR
#define CFG_PRINT_VISITOR_DBG_OUT                                              \
  if (true)                                                                    \
  cout << "[CFG_PRINT_VISITOR] "
#else
#define CFG_PRINT_VISITOR_DBG_OUT                                              \
  if (false)                                                                   \
  cout
#endif

struct CFGPrintVisitor : public BinameStatementVisitor {

  CFGPrintVisitor(Rewriter &rewriter) : rewriter(rewriter) {}

  string getIndent() {
    string ret = "";
    for (int i = 0; i < indent; i++) {
      ret += "  ";
    }
    return ret;
  }

  string getSrcStrFromStmt(Stmt *stmt) {
    auto srcRange = stmt->getSourceRange();
    auto beginFileLoc = rewriter.getSourceMgr().getFileLoc(srcRange.getBegin());
    auto endFileLoc = rewriter.getSourceMgr().getFileLoc(srcRange.getEnd());
    auto srcStr =
        rewriter.getRewrittenText(SourceRange(beginFileLoc, endFileLoc));
    // auto srcLoc = srcRange.getBegin();
    // auto srcStr = rewriter.getRewrittenText(SourceRange(
    //     srcLoc, Lexer::getLocForEndOfToken(srcRange.getEnd(), 0,
    //                                        rewriter.getSourceMgr(),
    //                                        rewriter.getLangOpts())));
    return srcStr;
  }

  string getSrcStrFromDecl(Decl *decl) {
    auto srcRange = decl->getSourceRange();
    auto srcLoc = srcRange.getBegin();
    // get file location
    auto beginFileLoc = rewriter.getSourceMgr().getFileLoc(srcLoc);
    auto endFileLoc = rewriter.getSourceMgr().getFileLoc(srcRange.getEnd());

    auto srcStr =
        rewriter.getRewrittenText(SourceRange(beginFileLoc, endFileLoc));
    // Lexer::getLocForEndOfToken(endFileLoc, 0,
    //                                    rewriter.getSourceMgr(),
    //                                    rewriter.getLangOpts())));
    return srcStr;
  }

  void visit(BinameProgram *prog) {
    auto seq = prog->getSequentialStatement();
    seq->accept(this);
  }

  void visit(SimpleClangStatement *stmt) {
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "SimpleClangStatement" << endl;
    indent++;
    auto clangStmtOpt = stmt->getClangStmt();
    if (clangStmtOpt.hasValue()) {
      auto clangStmt = clangStmtOpt.getValue();
      auto srcStr = getSrcStrFromStmt(clangStmt);
      CFG_PRINT_VISITOR_DBG_OUT << getIndent() << srcStr << endl;
    } else {
      CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "nullptr" << endl;
    }
    indent--;
  }

  void visit(SimpleClangDecl *stmt) {
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "SimpleClangDecl" << endl;
    indent++;
    auto clangDeclOpt = stmt->getClangDecl();
    if (clangDeclOpt.hasValue()) {
      auto clangDecl = clangDeclOpt.getValue();
      auto srcStr = getSrcStrFromDecl(clangDecl);
      CFG_PRINT_VISITOR_DBG_OUT << getIndent() << srcStr << endl;
    } else {
      CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "nullptr" << endl;
    }
    indent--;
  }

  void visit(SequentialStatement *stmt) {
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "SequentialStatement" << endl;
    indent++;
    for (auto s : stmt->statements) {
      s->accept(this);
    }
    indent--;
  }

  void visit(IfStatement *stmt) {
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "IfStatement" << endl;
    indent++;
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "cond: ";
    stmt->getCond()->accept(this);
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "then: ";
    stmt->getThenStmt()->accept(this);
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "else: ";
    stmt->getElseStmt()->accept(this);
    indent--;
  }

  void visit(WhileStatement *stmt) {
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "WhileStatement" << endl;
    indent++;
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "cond: ";
    stmt->getCond()->accept(this);
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "body: ";
    stmt->getBody()->accept(this);
    indent--;
  }

  void visit(ForStatement *stmt) {
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "ForStatement" << endl;
    indent++;
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "init: ";
    stmt->getInit()->accept(this);
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "cond: ";
    stmt->getCond()->accept(this);
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "inc: ";
    stmt->getInc()->accept(this);
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "body: ";
    stmt->getBody()->accept(this);
    indent--;
  }

  void visit(SwitchStatement *stmt) {
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "SwitchStatement" << endl;
    indent++;
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "cond: ";
    stmt->getCond()->accept(this);
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "body: ";
    const auto &cases = stmt->getCases();
    for (auto c : cases) {
      CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "case condi: ";
      c.first->accept(this);
      CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "case body: ";
      c.second->accept(this);
    }
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "default: ";
    stmt->getDefaultCase()->accept(this);
    indent--;
  }

  void visit(LoopTermStatement *stmt) {
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "LoopTermStatement" << endl;
    indent++;
    auto clangStmtOpt = stmt->getClangStmt();
    if (clangStmtOpt.hasValue()) {
      auto clangStmt = clangStmtOpt.getValue();
      auto srcStr = getSrcStrFromStmt(clangStmt);
      CFG_PRINT_VISITOR_DBG_OUT << getIndent() << srcStr << endl;
    }
    indent--;
  }

  void visit(ExecTermStatement *stmt) {
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "ExecTermStatement" << endl;
    indent++;
    auto clangStmtOpt = stmt->getClangStmt();
    if (clangStmtOpt.hasValue()) {
      auto clangStmt = clangStmtOpt.getValue();
      auto srcStr = getSrcStrFromStmt(clangStmt);
      CFG_PRINT_VISITOR_DBG_OUT << getIndent() << srcStr << endl;
    }
    indent--;
  }

  void visit(DoWhileStatement *stmt) {
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "DoWhileStatement" << endl;
    indent++;
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "cond: ";
    stmt->getCond()->accept(this);
    CFG_PRINT_VISITOR_DBG_OUT << getIndent() << "body: ";
    stmt->getBody()->accept(this);
    indent--;
  }

private:
  Rewriter &rewriter;
  int indent = 0;
};

#endif