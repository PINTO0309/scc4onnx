#! /usr/bin/env python

import os
import sys
import traceback
from argparse import ArgumentParser
import onnx
import onnx_graphsurgeon as gs
import numpy as np
from typing import Optional

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'


def order_conversion(
    input_op_names_and_order_dims: dict,
    channel_change_inputs: Optional[dict] = None,
    input_onnx_file_path: Optional[str] = '',
    output_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:
    """
    Parameters
    ----------
    input_op_names_and_order_dims: dict
        Specify the name of the input_op to be dimensionally changed and\n
        the order of the dimensions after the change.\n
        The name of the input_op to be dimensionally changed\n
        can be specified multiple times.\n\n
        e.g.\n
        input_op_names_and_order_dims = {\n
            "input_op_name1": [0,3,1,2],\n
            "input_op_name2": [0,2,3,1],\n
            "input_op_name3": [0,3,1,2,4,5],\n
        }

    channel_change_inputs: Optional[dict]
        Change the channel order of RGB and BGR.\n
        If the original model is RGB, it is transposed to BGR.\n
        If the original model is BGR, it is transposed to RGB.\n
        It can be selectively specified from among the OP names\n
        specified in input_op_names_and_order_dims.\n
        OP names not specified in input_op_names_and_order_dims are ignored.\n
        Multiple times can be specified as many times as the number\n
        of OP names specified in input_op_names_and_order_dims.\n
        channel_change_inputs = {"op_name": dimension_number_representing_the_channel}\n
        dimension_number_representing_the_channel must specify\n
        the dimension position after the change in input_op_names_and_order_dims.\n
        For example, dimension_number_representing_the_channel is 1 for NCHW and 3 for NHWC.\n\n
        e.g.\n
        channel_change_inputs = {\n
            "aaa": 1,\n
            "bbb": 3,\n
            "ccc": 2,\n
        }

    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.

    output_onnx_file_path: Optional[str]
        Output onnx file path.\n\
        If output_onnx_file_path is not specified, no .onnx file is output.

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

    Returns
    -------
    channel_converted_graph: onnx.ModelProto
        Channel converted onnx ModelProto
    """

    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)
    graph = gs.import_onnx(onnx_graph)

    # input_op_names_and_order_dims check
    # input_op_names_and_order_dims = {"name": shape, ...}
    inverted_list = {}
    for input_op_name, order_dim in input_op_names_and_order_dims.items():
        for graph_input in graph.inputs:
            if input_op_name == graph_input.name:
                # Length Check
                if len(graph_input.shape) != len(order_dim):
                    print(
                        f'{Color.RED}ERROR:{Color.RESET} '+
                        f'The size of the shape specified in input_op_names_and_order_dims '+
                        f'must match the size of the original model. '+
                        f'input_op_name: {input_op_name}, '+
                        f'shape: {graph_input.shape}, '+
                        f'order_dim: {order_dim}'
                    )
                    sys.exit(1)
                # Check for sequential numbering
                sorted_list = sorted(order_dim)
                for idx, dim in enumerate(sorted_list):
                    if idx != dim:
                        print(
                            f'{Color.RED}ERROR:{Color.RESET} '+
                            f'input_op_names_and_order_dims must be specified with sequential numbers. '+
                            f'input_op_name: {input_op_name}, '+
                            f'shape: {graph_input.shape}, '+
                            f'order_dim: {order_dim}'
                        )
                        sys.exit(1)
                # Generating an inverted list
                #  0 1 2 3      0 1 2 3
                # [0,2,3,1] -> [0,3,1,2]
                # [[0,0],[1,2],[2,3],[3,1]]
                # [[0,0],[3,1],[1,2],[2,3]]
                #
                #  0 1 2 3      0 1 2 3
                # [0,3,1,2] -> [0,2,3,1]
                # [[0,0],[1,3],[2,1],[3,2]]
                # [[0,0],[2,1],[3,2],[1,3]]
                #
                # inverted_list_tmp = [[0,0],[1,2],[2,3],[3,1]]
                # inverted_list = {"input_op_name": inverted_list_tmp}
                # Transpose perm
                #   perm = sorted(inverted_list[input_op_name], key=lambda x: x[1])
                inverted_list_tmp = []
                for idx, dim in enumerate(order_dim):
                    inverted_list_tmp.append([idx, dim])
                inverted_list[input_op_name] = inverted_list_tmp

    # channel_change_input check
    # channel_change_inputs = {"name": dim, ...}
    if channel_change_inputs:
        for input_op_name, input_dim in channel_change_inputs.items():
            for graph_input in graph.inputs:
                if input_op_name == graph_input.name:
                    if graph_input.shape[input_dim] != 3:
                        print(
                            f'{Color.RED}ERROR:{Color.RESET} '+
                            f'When changing the color channel, the dimension must be 3. '+
                            f'input_op_name: {input_op_name}, '+
                            f'shape: {graph_input.shape}, '+
                            f'ndim: {input_dim} ({graph_input.shape[input_dim]})'
                        )
                        sys.exit(1)



    # # Save
    # if output_onnx_file_path:
    #     onnx.save(order_converted_graph, output_onnx_file_path)

    # # 6. Return
    # return order_converted_graph


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '--output_onnx_file_path',
        type=str,
        required=True,
        help='Output onnx file path.'
    )
    parser.add_argument(
        '--input_op_names_and_order_dims',
        nargs=2,
        required=True,
        action='append',
        help=\
            'Specify the name of the input_op to be dimensionally changed and \n'+
            'the order of the dimensions after the change. \n'+
            'The name of the input_op to be dimensionally changed \n'+
            'can be specified multiple times. \n\n'+
            'e.g. \n'+
            '--input_op_names_and_order_dims aaa [0,3,1,2] \n'+
            '--input_op_names_and_order_dims bbb [0,2,3,1] \n'+
            '--input_op_names_and_order_dims ccc [0,3,1,2,4,5]'
    )
    parser.add_argument(
        '--channel_change_inputs',
        nargs=2,
        action='append',
        help=\
            'Change the channel order of RGB and BGR. \n'+
            'If the original model is RGB, it is transposed to BGR. \n'+
            'If the original model is BGR, it is transposed to RGB. \n'+
            'It can be selectively specified from among the OP names '+
            'specified in --input_op_names_and_order_dims. \n'+
            'OP names not specified in --input_op_names_and_order_dims are ignored. \n'+
            'Multiple times can be specified as many times as the number \n'+
            'of OP names specified in --input_op_names_and_order_dims. \n'+
            '--channel_change_inputs op_name dimension_number_representing_the_channel \n'+
            'dimension_number_representing_the_channel must specify \n'+
            'the dimension position before the change in input_op_names_and_order_dims. \n'+
            'For example, dimension_number_representing_the_channel is 1 for NCHW and 3 for NHWC. \n\n'+
            'e.g. \n'+
            '--channel_change_inputs aaa 3 \n'+
            '--channel_change_inputs bbb 1 \n'+
            '--channel_change_inputs ccc 5'
    )
    parser.add_argument(
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    input_onnx_file_path = args.input_onnx_file_path
    output_onnx_file_path = args.output_onnx_file_path
    input_op_names_and_order_dims = args.input_op_names_and_order_dims
    channel_change_inputs = args.channel_change_inputs
    non_verbose = args.non_verbose

    # file existence check
    if not os.path.exists(input_onnx_file_path) or \
        not os.path.isfile(input_onnx_file_path) or \
        not os.path.splitext(input_onnx_file_path)[-1] == '.onnx':

        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The specified file (.onnx) does not exist. or not an onnx file. File: {input_onnx_file_path}'
        )
        sys.exit(1)

    # Load
    onnx_graph = onnx.load(input_onnx_file_path)

    # input_op_names_and_order_dim
    """
    input_op_names_and_order_dims_tmp = {"name": shape}
    """
    input_op_names_and_order_dims_tmp = None
    input_op_names_and_order_dims_tmp = {
        input_op_names_and_order_dim[0]: eval(input_op_names_and_order_dim[1]) for input_op_names_and_order_dim in input_op_names_and_order_dims
    }

    # channel_change_inputs
    """
    channel_change_inputs_tmp = {"name": dim}
    """
    channel_change_inputs_tmp = None
    if channel_change_inputs:
        channel_change_inputs_tmp = {
            channel_change_input[0]: int(channel_change_input[1]) for channel_change_input in channel_change_inputs
        }

    # Generate
    order_converted_graph = order_conversion(
        input_op_names_and_order_dims=input_op_names_and_order_dims_tmp,
        channel_change_inputs=channel_change_inputs_tmp,
        input_onnx_file_path=None,
        output_onnx_file_path=output_onnx_file_path,
        onnx_graph=onnx_graph,
        non_verbose=non_verbose,
    )


if __name__ == '__main__':
    main()