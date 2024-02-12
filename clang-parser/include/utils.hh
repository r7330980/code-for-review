#ifndef UTILS_HH
#define UTILS_HH

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Rewrite/Frontend/Rewriters.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>

#include "ast_visitor_interface.hh"

using namespace clang;
using namespace std;

void consumeAST(CompilerInstance &CI, GenNmASTConsumer &consumer);

void createCompilerInstance(CompilerInstance &theCompiler,
                            const string &filename);

Rewriter createRewriter(CompilerInstance &CI);

/**
 * Had to use this hacking to assign confidence to each identifier.
 * Because we did not distinguish whether a name is for a function or a
 * variable.
 */

enum IdentifierType { VAR, FUNC, UNKNOWN };
// reload enum to <<
static inline ostream &operator<<(ostream &os, IdentifierType type) {
  switch (type) {
  case VAR:
    os << "VAR";
    break;
  case FUNC:
    os << "FUNC";
    break;
  case UNKNOWN:
    os << "UNKNOWN";
    break;
  }
  return os;
}

class IdentifierTypeVisitor
    : public RecursiveASTVisitor<IdentifierTypeVisitor> {
public:
  IdentifierTypeVisitor(const Rewriter &rewriter) : rewriter(rewriter) {}

  bool VisitFunctionDecl(FunctionDecl *funcDecl) {
    if (funcDecl->isThisDeclarationADefinition()) {
      string funcName = funcDecl->getNameInfo().getName().getAsString();
      if (identifierType.count(funcName)) {
        identifierType[funcName] = FUNC;
      }
    }
    return true;
  }
  // parameter decls
  bool VisitParmVarDecl(ParmVarDecl *parmVarDecl) {
    string parmName = parmVarDecl->getNameAsString();
    if (identifierType.count(parmName)) {
      identifierType[parmName] = VAR;
    }
    return true;
  }
  // variable decls
  bool VisitVarDecl(VarDecl *varDecl) {
    string varName = varDecl->getNameAsString();
    if (identifierType.count(varName) && identifierType[varName] == UNKNOWN) {
      identifierType[varName] = VAR;
    }
    return true;
  }
  // call expr
  bool VisitCallExpr(CallExpr *callExpr) {
    auto callee = callExpr->getDirectCallee();
    if (!callee) {
      return true;
    }
    string funcName = callee->getNameAsString();
    if (identifierType.count(funcName) && identifierType[funcName] == UNKNOWN) {
      identifierType[funcName] = FUNC;
    }
    return true;
  }

  // by default, we assume all identifiers are variables
  // visit decl ref
  bool VisitDeclRefExpr(DeclRefExpr *declRefExpr) {
    string identifierName = declRefExpr->getNameInfo().getName().getAsString();
    if (identifierType.count(identifierName) &&
        identifierType[identifierName] == UNKNOWN) {
      identifierType[identifierName] = VAR;
    }
    return true;
  }

  unordered_map<string, IdentifierType>
  calculateIdentifierTypes(vector<string> &identifierNames,
                           FunctionDecl *funcDecl) {
    identifierType.clear();
    for (auto &identifierName : identifierNames) {
      identifierType[identifierName] = UNKNOWN;
    }
    TraverseDecl(funcDecl);
    return identifierType;
  }

private:
  unordered_map<string, IdentifierType> identifierType;
  const Rewriter &rewriter;
};

#endif