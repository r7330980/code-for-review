#ifndef CFG_HH
#define CFG_HH
#include "utils.hh"

struct BinameStatementVisitor;

// different types of statements
// simple clang stmt: stmt*
// sequential stmt: vector<stmt*>
// if stmt: if (cond) {then-stmt} else {else-stmt}
// while stmt: while (cond) {body}
// for stmt: for (init; cond; inc) {body}
// switch stmt: switch (cond) {case1: {body1} case2: {body2} ... default:
// {default-body}} loop-termination stmt: break, continue exec-termination stmt:
// return

struct BinameStatement {
public:
  virtual void accept(BinameStatementVisitor *visitor) = 0;

  Optional<Stmt *> getClangStmt() {
    if (clangStmt == nullptr) {
      return None;
    } else {
      return Optional<Stmt *>(clangStmt);
    }
  }

  Optional<Decl *> getClangDecl() {
    if (clangDecl == nullptr) {
      return None;
    } else {
      return Optional<Decl *>(clangDecl);
    }
  }

protected:
  BinameStatement(Stmt *clangStmt) : clangStmt(clangStmt) {}

  BinameStatement(Decl *clangDecl) : clangDecl(clangDecl) {}

private:
  string text;
  Stmt *clangStmt;
  Decl *clangDecl;
};

struct SimpleClangDecl : public BinameStatement {
public:
  SimpleClangDecl(Decl *clangDecl) : BinameStatement(clangDecl) {}

  void accept(BinameStatementVisitor *visitor);
};

struct SimpleClangStatement : public BinameStatement {
public:
  SimpleClangStatement(Stmt *clangStmt) : BinameStatement(clangStmt) {}

  void accept(BinameStatementVisitor *visitor);
};

struct SequentialStatement : public BinameStatement {
  vector<BinameStatement *> statements;
  SequentialStatement() : BinameStatement((Stmt *)nullptr) {}

  void accept(BinameStatementVisitor *visitor);
};

struct IfStatement : public BinameStatement {
public:
  IfStatement(IfStmt *clangIfStmt, BinameStatement *cond,
              BinameStatement *thenStmt, BinameStatement *elseStmt)
      : BinameStatement(clangIfStmt), cond(cond), thenStmt(thenStmt),
        elseStmt(elseStmt) {}

  void accept(BinameStatementVisitor *visitor);

  BinameStatement *getCond() { return cond; }

  BinameStatement *getThenStmt() { return thenStmt; }

  BinameStatement *getElseStmt() { return elseStmt; }

private:
  BinameStatement *cond;
  BinameStatement *thenStmt;
  BinameStatement *elseStmt;
};

struct WhileStatement : public BinameStatement {
public:
  WhileStatement(WhileStmt *clangWhileStmt, BinameStatement *cond,
                 BinameStatement *body)
      : BinameStatement(clangWhileStmt), cond(cond), body(body) {}

  void accept(BinameStatementVisitor *visitor);

  BinameStatement *getCond() { return cond; }

  BinameStatement *getBody() { return body; }

private:
  BinameStatement *cond;
  BinameStatement *body;
};

struct DoWhileStatement : public BinameStatement {
public:
  DoWhileStatement(DoStmt *clangDoWhileStmt, BinameStatement *cond,
                   BinameStatement *body)
      : BinameStatement(clangDoWhileStmt), cond(cond), body(body) {}

  void accept(BinameStatementVisitor *visitor);

  BinameStatement *getCond() { return cond; }

  BinameStatement *getBody() { return body; }

private:
  BinameStatement *cond;
  BinameStatement *body;
};

struct ForStatement : public BinameStatement {
public:
  ForStatement(ForStmt *clangForStmt, BinameStatement *init,
               BinameStatement *cond, BinameStatement *inc,
               BinameStatement *body)
      : BinameStatement(clangForStmt), init(init), cond(cond), inc(inc),
        body(body) {}

  void accept(BinameStatementVisitor *visitor);

  BinameStatement *getInit() { return init; }

  BinameStatement *getCond() { return cond; }

  BinameStatement *getInc() { return inc; }

  BinameStatement *getBody() { return body; }

private:
  BinameStatement *init;
  BinameStatement *cond;
  BinameStatement *inc;
  BinameStatement *body;
};

struct SwitchStatement : public BinameStatement {
public:
  SwitchStatement(SwitchStmt *clangSwitchStmt, BinameStatement *cond,
                  vector<pair<BinameStatement *, BinameStatement *>> cases,
                  BinameStatement *defaultCase)
      : BinameStatement(clangSwitchStmt), cond(cond), cases(cases),
        defaultCase(defaultCase) {}
  void accept(BinameStatementVisitor *visitor);

  BinameStatement *getCond() { return cond; }

  const vector<pair<BinameStatement *, BinameStatement *>> &getCases() const {
    return cases;
  }

  BinameStatement *getDefaultCase() { return defaultCase; }

private:
  BinameStatement *cond;
  vector<pair<BinameStatement *, BinameStatement *>> cases;
  BinameStatement *defaultCase;
};

struct LoopTermStatement : public BinameStatement {
public:
  void accept(BinameStatementVisitor *visitor);
  LoopTermStatement(Stmt *clangStmt) : BinameStatement(clangStmt) {}
};

struct ExecTermStatement : public BinameStatement {
public:
  void accept(BinameStatementVisitor *visitor);
  ExecTermStatement(Stmt *clangStmt) : BinameStatement(clangStmt) {}
};

struct BinameProgram {
public:
  BinameProgram(SequentialStatement *seq, FunctionDecl *fd) : seq(seq), fd(fd) {
    auto returnType = fd->getReturnType();
    vector<pair<string, string>> params;
    for (auto param : fd->parameters()) {
      string type = param->getType().getAsString();
      string name = param->getNameAsString();
      params.push_back(make_pair(type, name));
    }
    funcDeclStr = returnType.getAsString() + " " + fd->getNameAsString() + "(";
    for (int i = 0; i < params.size(); i++) {
      funcDeclStr += params[i].first + " " + params[i].second;
      if (i != params.size() - 1) {
        funcDeclStr += ", ";
      }
    }
    funcDeclStr += ")";
  }  

  SequentialStatement *getSequentialStatement() { return seq; }

  string getName() { return fd->getNameAsString(); }

  string getFuncDeclStr() { return funcDeclStr; }

private:
  string funcDeclStr;
  FunctionDecl *fd;
  SequentialStatement *seq;
};

#endif