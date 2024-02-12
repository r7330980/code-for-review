#ifndef AST_VISITOR_INTERFACE_HH
#define AST_VISITOR_INTERFACE_HH

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

class GenNmASTConsumer : public clang::ASTConsumer {
public:
  explicit GenNmASTConsumer(clang::Rewriter &R) : rewriter(R) {}

protected:
  clang::Rewriter &rewriter;
};

#endif