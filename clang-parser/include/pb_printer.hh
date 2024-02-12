#ifndef PB_PRINTER_HH
#define PB_PRINTER_HH

#include "ir/gennm_ir.hh"
#include <string>


using namespace std;

void writeToFile(string fname, GenNmFunction* func);

#endif