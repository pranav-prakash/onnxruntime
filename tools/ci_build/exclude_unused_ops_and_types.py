# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import op_registration_utils
import os
import shutil
import sys
import typing

from logger import get_logger

script_path = os.path.dirname(os.path.realpath(__file__))
ort_root = os.path.abspath(os.path.join(script_path, '..', '..', ))
ort_tools_py_path = os.path.abspath(os.path.join(ort_root, 'tools', 'python'))
sys.path.append(ort_tools_py_path)

from util import parse_config  # noqa
from util.ort_format_model.operator_type_usage_processors import OperatorTypeUsageManager  # noqa

log = get_logger("exclude_unused_ops_and_types")


class ExcludeOpsAndTypesRegistrationProcessor(op_registration_utils.RegistrationProcessor):
    def __init__(self, required_ops, op_type_usage_manager, output_file):
        self._required_ops = required_ops
        self._op_types_usage_manager = op_type_usage_manager
        self._output_file = output_file

    def _should_exclude_op(self, domain, operator, start_version, end_version):
        log.info('domain:{} op:{} start:{} end:{}'.format(domain, operator, start_version, end_version))
        if domain not in self._required_ops:
            log.info('domain not found')
            return True

        for opset in self._required_ops[domain]:
            log.info('checking opset {}'.format(opset))
            if opset >= start_version and (end_version is None or opset <= end_version):
                if operator in self._required_ops[domain][opset]:
                    return False  # found a match, do not exclude
                else:
                    log.info('no match in required ops')

        log.info('returning True')
        return True

    def process_registration(self, lines: typing.List[str], constant_for_domain: str, operator: str,
                             start_version: int, end_version: int = None, input_type: str = None):
        # convert from the ORT constant name to the domain string used in the config
        domain = op_registration_utils.map_ort_constant_to_domain(constant_for_domain)
        exclude = False

        if domain:
            # see if entire op is excluded
            exclude = self._should_exclude_op(domain, operator, start_version, end_version)

            # see if a specific typed registration can be excluded
            if not exclude and input_type and self._op_types_usage_manager:
                exclude = not self._op_types_usage_manager.is_typed_registration_needed(domain, operator, input_type)

        if exclude:
            log.info('Disabling {}:{}({}){}'.format(constant_for_domain, operator, start_version,
                                                    '<{}>'.format(input_type) if input_type else ''))
            for line in lines:
                self._output_file.write('// ' + line)

            # edge case of last entry in table where we still need the terminating }; to not be commented out
            if lines[-1].rstrip().endswith('};'):
                self._output_file.write('};\n')
        else:
            for line in lines:
                self._output_file.write(line)

    def process_other_line(self, line):
        self._output_file.write(line)

    def ok(self):
        return True


def _exclude_unused_ops_and_types_in_registrations(required_operators,
                                                   op_type_usage_manager,
                                                   provider_registration_paths):
    '''rewrite provider registration file to exclude unused ops'''

    for kernel_registration_file in provider_registration_paths:
        if not os.path.isfile(kernel_registration_file):
            raise ValueError('Kernel registration file {} does not exist'.format(kernel_registration_file))

        log.info("Processing {}".format(kernel_registration_file))

        backup_path = kernel_registration_file + '~'
        shutil.move(kernel_registration_file, backup_path)

        # read from backup and overwrite original with commented out lines for any kernels that are not required
        with open(kernel_registration_file, 'w') as file_to_write:
            processor = ExcludeOpsAndTypesRegistrationProcessor(required_operators,
                                                                op_type_usage_manager,
                                                                file_to_write)

            op_registration_utils.process_kernel_registration_file(backup_path, processor)

            if not processor.ok():
                # error should have already been logged so just exit
                sys.exit(-1)


def _generate_cpp_defines(ort_root: str, op_type_usage_manager: OperatorTypeUsageManager):

    defines = op_type_usage_manager.get_cpp_defines() if op_type_usage_manager else None
    if not defines:
        return

    # open header file to write
    type_reduction_header_path = os.path.join(ort_root, 'onnxruntime', 'core', 'framework', 'type_reductions.h')
    with open(type_reduction_header_path, 'w') as output:
        output.write('// Copyright (c) Microsoft Corporation. All rights reserved.\n')
        output.write('// Licensed under the MIT License.\n\n')
        output.write('#pragma once\n\n')

        [output.write('{}\n'.format(define)) for define in defines]

    # future: how/where will we write global type limitations?
    # should they come from the ops file or be separate? probably separate - may want to reduce types without
    # reducing operators. this can probably be handled by build.py as we should either use a set of global types
    # OR a set of per-operator types but not both.


def exclude_unused_ops_and_types(config_path, enable_type_reduction=False, use_cuda=True):
    required_ops, op_type_usage_manager = parse_config(config_path, enable_type_reduction)

    registration_files = op_registration_utils.get_kernel_registration_files(ort_root, use_cuda)

    _exclude_unused_ops_and_types_in_registrations(required_ops, op_type_usage_manager, registration_files)

    _generate_cpp_defines(ort_root, op_type_usage_manager)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to exclude unused operator kernels by disabling their registration in ONNX Runtime. "
                    "The types supported by operator kernels may also be reduced if specified in the config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("config_path", type=str,
                        help="Path to configuration file. "
                             "Create with <ORT root>/tools/python/create_reduced_build_config.py and edit if needed. "
                             "See /docs/ONNX_Runtime_Format_Model_Usage.md for more information.")

    args = parser.parse_args()
    config_path = os.path.abspath(args.config_path)

    exclude_unused_ops_and_types(config_path, enable_type_reduction=True, use_cuda=True)