#!/bin/bash

protoc gennm_ir.proto --cpp_out=. --python_out=.
mv gennm_ir.pb.h include/
mv gennm_ir_pb2.py gennm_py_pb_parser/