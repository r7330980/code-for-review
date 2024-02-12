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

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#include <boost/program_options.hpp>
#include <filesystem>

// #include "cfg_builder.hh"
// #include "cfg_naive_sampler_visitor.hh"
#include "dbg_config.hh"
#include "utils.hh"
#include "ir_builder.hh"
#include "ir/gennm_ir_print_visitor.hh"
#include "pb_printer.hh"

#ifdef DBG_MAIN
#define MAIN_DBG_OUT                                                           \
  if (true)                                                                    \
  cout
#else
#define MAIN_DBG_OUT                                                           \
  if (false)                                                                   \
  cout
#endif

namespace po = boost::program_options;
namespace fs = std::filesystem;

using namespace clang;
using namespace std;

void parseCliOptions(int argc, char **argv, po::variables_map &vm) {
  // clang-format off
  // Don't touch this!
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")
    ("fin", po::value<string>(),"input file")    
    ("fout", po::value<string>(),"output file")
    ("trans", po::value<vector<string>>(), "transformations to apply")
  ;
  // clang-format on
  // Carry on formatting
  po::positional_options_description p;
  p.add("trans", -1);
  po::store(
      po::command_line_parser(argc, argv).options(desc).positional(p).run(),
      vm);
  po::notify(vm);

  if (vm.count("help") || vm.count("fin") == 0) {
    cout << desc << "\n";
    exit(1);
  }
  // // if does not have dir, use the dir of fin
  // if (vm.count("dir") == 0) {
  //   string filename = vm["fin"].as<string>();
  //   string dir = filename.substr(0, filename.rfind('/'));
  //   vm.insert(make_pair("dir", po::variable_value(dir, false)));
  // }
  // if does not have fout, use the fin + .prop.c
  if (vm.count("fout") == 0) {
    string filename = vm["fin"].as<string>();
    string nameWithoutExtension = filename.substr(0, filename.rfind('.'));
    vm.insert(make_pair(
        "fout", po::variable_value(nameWithoutExtension + ".gennm.pb", false)));
  }
}

void printCliOptions(po::variables_map &vm) {
  cout << "fin: " << vm["fin"].as<string>() << endl;
  // cout << "dir: " << vm["dir"].as<string>() << endl;
  cout << "fout: " << vm["fout"].as<string>() << endl;
  cout << "transformations: " << endl;
  for (auto rule : vm["trans"].as<vector<string>>()) {
    cout << rule << endl;
  }
}


int main(int argc, char **argv) {
  // set random seed
  srand(12345);
  po::variables_map vm;
  parseCliOptions(argc, argv, vm);
  printCliOptions(vm);
  fs::path fin = vm["fin"].as<string>();
  // get filename without extension using fs
  fs::path finWithoutExtension = fin.stem();
  // // copy fin to fout
  // fs::path fout = vm["fout"].as<string>();
  // fs::copy(fin, fout, fs::copy_options::overwrite_existing);

  CompilerInstance theCompiler;
  createCompilerInstance(theCompiler, fin.string());
  Rewriter rewriter = createRewriter(theCompiler);
  IRBuilder irBuilder(rewriter);
  consumeAST(theCompiler, irBuilder);
  auto functions = irBuilder.getFunctions();  
  // GenNmIRPrintVisitor irPrintVisitor;
  assert(functions.size() == 1);
  string fout = vm["fout"].as<string>();
  for(auto func: functions){
    func->normalizeBBLabels();
    // irPrintVisitor.visit(func);
    writeToFile(fout, func);
  }
  
  
  return 0;
}