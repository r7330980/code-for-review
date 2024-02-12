#ifndef CFG_NAIVE_SAMPLER_VISITOR_HH
#define CFG_NAIVE_SAMPLER_VISITOR_HH

#include "cfg.hh"
#include "cfg_visitor.hh"
#include "dbg_config.hh"
#include "utils.hh"

#include <clang/Format/Format.h>
#include <iostream>
#include <sstream>
#include <unordered_map>
using namespace std;
using namespace clang;

// #ifdef DBG_CFG_NAIVE_SAMPLER
// #define CFG_NAIVE_SAMPLER_DBG_OUT \
//   if (true) \ cout << "[CFG_NAIVE_SAMPLER] "
// #else
// #define CFG_NAIVE_SAMPLER_DBG_OUT \
//   if (false) \ cout
// #endif

// Note that we assume the following things:
// 1. the rewriter is associated with the related cfg programs
// 2. all cfg nodes are allocated in the heap and will not change their address
// 3. all visitors in this file will not change the cfg nodes

// we do the following things:
// 1. counting possible variants from a given statement
// 2. at each statement, we randomly choose a variant with equal probability

string getSrcStrFromStmt(Rewriter rewriter, Stmt *stmt) {
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

string getSrcStrFromDecl(Rewriter rewriter, Decl *decl) {
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

// first, counting possible variants from a given statement
struct NaiveVariantsCounter : public BinameStatementVisitor {
#ifdef DBG_CFG_NAIVE_SAMPLER
#define CFG_NAIVE_COUNTER_DBG_OUT                                              \
  if (true)                                                                    \
  cout << "[Naive Counter] "
#else
#define CFG_NAIVE_COUNTER_DBG_OUT                                              \
  if (false)                                                                   \
  cout
#endif

  NaiveVariantsCounter(Rewriter &rewriter) : rewriter(rewriter) {}

  void visit(BinameProgram *program) {
    auto seq = program->getSequentialStatement();
    seq->accept(this);
  }

  string getIndent() {
    string ret = "";
    for (int i = 0; i < indent; i++) {
      ret += "  ";
    }
    return ret;
  }

  void visit(SimpleClangStatement *stmt) {
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "SimpleClangStatement" << endl;
    indent++;
    if (stmt->getClangStmt().hasValue()) {
      CFG_NAIVE_COUNTER_DBG_OUT
          << getIndent()
          << getSrcStrFromStmt(rewriter, stmt->getClangStmt().getValue())
          << endl;
    }
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "couting: 1" << endl;
    indent--;
    variantsCount[stmt] = 1;
  }

  void visit(SimpleClangDecl *decl) {
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "SimpleClangDecl" << endl;
    indent++;
    if (decl->getClangDecl().hasValue()) {
      CFG_NAIVE_COUNTER_DBG_OUT
          << getIndent()
          << getSrcStrFromDecl(rewriter, decl->getClangDecl().getValue())
          << endl;
    }
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "couting: 1" << endl;
    indent--;
    variantsCount[decl] = 1;
  }

  void visit(SequentialStatement *stmt) {
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "SequentialStatement" << endl;
    indent++;
    int variants = 1;
    for (auto s : stmt->statements) {
      s->accept(this);
      variants *= variantsCount[s];
    }
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "couting: " << variants << endl;
    indent--;
    variantsCount[stmt] = variants;
  }

  void visit(IfStatement *stmt) {
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "IfStatement" << endl;
    indent++;
    int variants = 0;
    stmt->getThenStmt()->accept(this);
    variants += variantsCount[stmt->getThenStmt()];
    if (stmt->getElseStmt()) {
      stmt->getElseStmt()->accept(this);
      variants += variantsCount[stmt->getElseStmt()];
    }
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "couting: " << variants << endl;
    indent--;
    variantsCount[stmt] = variants;
  }

  void visit(WhileStatement *stmt) {
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "WhileStatement" << endl;
    indent++;
    int variants = 0;
    stmt->getBody()->accept(this);
    variants += variantsCount[stmt->getBody()];
    // skip loop + loop 1 time + loop 2 times
    int adjustedCnt = 1 + variants + variants * variants;
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "couting: " << adjustedCnt
                              << endl;
    indent--;
    variantsCount[stmt] = adjustedCnt;
  }

  void visit(ForStatement *stmt) {
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "ForStatement" << endl;
    indent++;
    int variants = 0;
    stmt->getBody()->accept(this);
    variants += variantsCount[stmt->getBody()];
    // skip loop + loop 1 time + loop 2 times
    int adjustedCnt = 1 + variants + variants * variants;
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "couting: " << adjustedCnt
                              << endl;
    indent--;
    variantsCount[stmt] = adjustedCnt;
  }

  void visit(DoWhileStatement *stmt) {
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "DoWhileStatement" << endl;
    indent++;
    int variants = 0;
    stmt->getBody()->accept(this);
    variants += variantsCount[stmt->getBody()];
    // skip loop + loop 1 time + loop 2 times
    int adjustedCnt = 1 + variants + variants * variants;
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "couting: " << adjustedCnt
                              << endl;
    indent--;
    variantsCount[stmt] = adjustedCnt;
  }

  void visit(SwitchStatement *stmt) {
    // TODO: implement this
    // throw runtime_error("not implemented");
    throw runtime_error("not implemented");
    variantsCount[stmt] = 1;
  }

  void visit(LoopTermStatement *stmt) {
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "LoopTermStatement" << endl;
    indent++;
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "couting: 1" << endl;
    indent--;
    variantsCount[stmt] = 1;
  }

  void visit(ExecTermStatement *stmt) {
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "ExecTermStatement" << endl;
    indent++;
    CFG_NAIVE_COUNTER_DBG_OUT << getIndent() << "couting: 1" << endl;
    indent--;
    variantsCount[stmt] = 1;
  }

  const unordered_map<BinameStatement *, int> &getVariantsCount() const {
    return variantsCount;
  }

private:
  int indent = 0;
  unordered_map<BinameStatement *, int> variantsCount;
  Rewriter &rewriter;
};

double getUniformRandom() {
  int r = rand() & 0xFFFF;
  const int rMax = 0xFFFF;
  return (double)r / (double)rMax;
}

struct NaiveSampler : public BinameStatementVisitor {
#ifdef DBG_CFG_NAIVE_SAMPLER
#define CFG_NAIVE_SAMPLER_DBG_OUT                                              \
  if (true)                                                                    \
  cout << "[Naive Sampler] "
#else
#define CFG_NAIVE_SAMPLER_DBG_OUT                                              \
  if (false)                                                                   \
  cout
#endif

  NaiveSampler(Rewriter &rewriter, const NaiveVariantsCounter &counter)
      : rewriter(rewriter), counter(counter),
        variantsCount(counter.getVariantsCount()) {}

  vector<string> samplePaths(BinameProgram *program, int sampleCnt) {
    vector<string> ret;
    for (int i = 0; i < sampleCnt; i++) {
      ret.push_back(samplePath(program));
    }
    return ret;
  }

  void visit(SimpleClangStatement *stmt) {
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "SimpleClangStatement" << endl;
    indent++;
    if (stmt->getClangStmt().hasValue()) {
      string myStr =
          getSrcStrFromStmt(rewriter, stmt->getClangStmt().getValue());
      CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << myStr << endl;
      if (myStr[myStr.size() - 1] == ';') {
        currentPath << myStr << endl;
      } else {
        currentPath << myStr << ";" << endl;
      }
    }
    indent--;
  }

  void visit(SimpleClangDecl *decl) {
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "SimpleClangDecl" << endl;
    indent++;
    if (decl->getClangDecl().hasValue()) {
      string myStr =
          getSrcStrFromDecl(rewriter, decl->getClangDecl().getValue());
      CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << myStr << endl;
      // if myStr last char is ';', then do not add ';'
      if (myStr[myStr.size() - 1] == ';') {
        currentPath << myStr << endl;
      } else {
        currentPath << myStr << ";" << endl;
      }
    }
    indent--;
  }

  void visit(SequentialStatement *stmt) {
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "SequentialStatement" << endl;
    indent++;
    int variants = variantsCount.at(stmt);
    for (auto s : stmt->statements) {
      s->accept(this);
    }
    indent--;
  }

  void visit(IfStatement *stmt) {
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "IfStatement" << endl;
    indent++;
    int variantsThen = variantsCount.at(stmt->getThenStmt());
    int variantsElse = 1;
    if (stmt->getElseStmt()) {
      variantsElse = variantsCount.at(stmt->getElseStmt());
    }
    int variants = variantsThen + variantsElse;
    double randNum = getUniformRandom();
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "random number: " << randNum
                              << endl;
    // print then ratio and else ratio
    CFG_NAIVE_SAMPLER_DBG_OUT
        << getIndent()
        << "then ratio: " << (double)variantsThen / (double)variants << endl;
    CFG_NAIVE_SAMPLER_DBG_OUT
        << getIndent()
        << "else ratio: " << (double)variantsElse / (double)variants << endl;
    auto condiStr =
        getSrcStrFromStmt(rewriter, stmt->getCond()->getClangStmt().getValue());
    if (randNum < (double)variantsThen / (double)variants) {
      currentPath << "if ( " << condiStr << " ) { " << endl;
      CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "choose then" << endl;
      stmt->getThenStmt()->accept(this);
      currentPath << "}" << endl;
    } else {
      CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "choose else" << endl;
      if (SequentialStatement *elseStr =
              dynamic_cast<SequentialStatement *>(stmt->getElseStmt())) {
        if (elseStr->statements.size() != 0) {
          currentPath << "if ( !(" << condiStr << ") ) { " << endl;
          stmt->getElseStmt()->accept(this);
          currentPath << "}" << endl;
        }
      }
    }
    indent--;
  }

  void visit(WhileStatement *stmt) {
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "WhileStatement" << endl;
    indent++;
    int variants = variantsCount.at(stmt);
    double randNum = getUniformRandom();
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "random number: " << randNum
                              << endl;
    // print loop ratio and skip ratio
    CFG_NAIVE_SAMPLER_DBG_OUT
        << getIndent()
        << "loop ratio: " << (double)variants / (double)(variants) << endl;
    CFG_NAIVE_SAMPLER_DBG_OUT
        << getIndent() << "skip ratio: " << (double)1 / (double)(variants)
        << endl;
    if (randNum < (double)variants / (double)(variants)) {
      currentPath << "while ( "
                  << getSrcStrFromStmt(
                         rewriter, stmt->getCond()->getClangStmt().getValue())
                  << " ) { " << endl;
      CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "choose loop" << endl;
      stmt->getBody()->accept(this);
      currentPath << "}" << endl;
    } else {
      CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "choose skip" << endl;
    }
    indent--;
  }

  void visit(ForStatement *stmt) {
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "ForStatement" << endl;
    indent++;
    int variants = variantsCount.at(stmt);
    double randNum = getUniformRandom();
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "random number: " << randNum
                              << endl;
    // print loop ratio and skip ratio
    CFG_NAIVE_SAMPLER_DBG_OUT
        << getIndent()
        << "loop ratio: " << (double)variants / (double)(variants) << endl;
    CFG_NAIVE_SAMPLER_DBG_OUT
        << getIndent() << "skip ratio: " << (double)1 / (double)(variants)
        << endl;
    if (randNum < (double)variants / (double)(variants)) {
      string initStr = " ";
      if (stmt->getInit()->getClangStmt().hasValue()) {
        initStr = getSrcStrFromStmt(rewriter,
                                    stmt->getInit()->getClangStmt().getValue());
      }
      string incStr = " ";
      if (stmt->getInc()->getClangStmt().hasValue()) {
        incStr = getSrcStrFromStmt(rewriter,
                                   stmt->getInc()->getClangStmt().getValue());
      }
      string condiStr = " ";
      if (stmt->getCond()->getClangStmt().hasValue()) {
        condiStr = getSrcStrFromStmt(
            rewriter, stmt->getCond()->getClangStmt().getValue());
      }
      currentPath << "for ( " << initStr << " ; " << condiStr << " ; " << incStr
                  << " ) { " << endl;

      CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "choose loop" << endl;
      stmt->getBody()->accept(this);
      currentPath << "}" << endl;
    } else {
      CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "choose skip" << endl;
    }
    indent--;
  }

  void visit(DoWhileStatement *stmt) {
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "DoWhileStatement" << endl;
    indent++;
    int variants = variantsCount.at(stmt);
    double randNum = getUniformRandom();
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "random number: " << randNum
                              << endl;
    // print loop ratio and skip ratio
    CFG_NAIVE_SAMPLER_DBG_OUT
        << getIndent()
        << "loop ratio: " << (double)variants / (double)(variants) << endl;
    CFG_NAIVE_SAMPLER_DBG_OUT
        << getIndent() << "skip ratio: " << (double)1 / (double)(variants)
        << endl;
    if (randNum < (double)variants / (double)(variants)) {
      currentPath << "do { " << endl;
      CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "choose loop" << endl;
      stmt->getBody()->accept(this);
      currentPath << "} while ( "
                  << getSrcStrFromStmt(
                         rewriter, stmt->getCond()->getClangStmt().getValue())
                  << " );" << endl;
    } else {
      CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "choose skip" << endl;
    }
    indent--;
  }

  void visit(SwitchStatement *stmt) { throw runtime_error("not implemented"); }

  void visit(LoopTermStatement *stmt) {
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "LoopTermStatement" << endl;
    indent++;
    currentPath << getSrcStrFromStmt(rewriter, stmt->getClangStmt().getValue())
                << ";" << endl;
    indent--;
  }

  void visit(ExecTermStatement *stmt) {
    CFG_NAIVE_SAMPLER_DBG_OUT << getIndent() << "ExecTermStatement" << endl;
    indent++;
    currentPath << getSrcStrFromStmt(rewriter, stmt->getClangStmt().getValue())
                << ";" << endl;
    indent--;
  }

private:
  string getIndent() {
    string ret = "";
    for (int i = 0; i < indent; i++) {
      ret += "  ";
    }
    return ret;
  }

  string samplePath(BinameProgram *program) {
    currentPath.str("");
    indent = 0;
    program->getSequentialStatement()->accept(this);
    // reformat the path
    string pathStr = currentPath.str();
    // auto llvmStyle = format::getLLVMStyle();
    // tooling::Range range(0, pathStr.size());
    // auto reformatPath = format::reformat(llvmStyle, pathStr, range);
    // for (auto c : reformatPath) {
    //   CFG_NAIVE_SAMPLER_DBG_OUT << c.toString();
    // }

    CFG_NAIVE_SAMPLER_DBG_OUT << "========== sampled path: ============== "
                              << endl;
    CFG_NAIVE_SAMPLER_DBG_OUT << pathStr << endl;
    CFG_NAIVE_SAMPLER_DBG_OUT << "=========== end =================" << endl;
    return pathStr;
  }

  stringstream currentPath;
  int indent = 0;
  Rewriter &rewriter;
  const NaiveVariantsCounter &counter;
  const unordered_map<BinameStatement *, int> &variantsCount;
};

#endif