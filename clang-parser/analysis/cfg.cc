#include "utils.hh"
#include "cfg.hh"
#include "cfg_visitor.hh"

// implement all `accept` functions
void DoWhileStatement::accept(BinameStatementVisitor *visitor) {
  visitor->visit(this);
}

void SimpleClangDecl::accept(BinameStatementVisitor *visitor) {
  visitor->visit(this);
}

void SimpleClangStatement::accept(BinameStatementVisitor *visitor) {
  visitor->visit(this);
}

void SequentialStatement::accept(BinameStatementVisitor *visitor) {
  visitor->visit(this);
}

void IfStatement::accept(BinameStatementVisitor *visitor) {
  visitor->visit(this);
}

void WhileStatement::accept(BinameStatementVisitor *visitor) {
  visitor->visit(this);
}

void ForStatement::accept(BinameStatementVisitor *visitor) {
  visitor->visit(this);
}

void SwitchStatement::accept(BinameStatementVisitor *visitor) {
  visitor->visit(this);
}

void LoopTermStatement::accept(BinameStatementVisitor *visitor) {
  visitor->visit(this);
}

void ExecTermStatement::accept(BinameStatementVisitor *visitor) {
  visitor->visit(this);
}

