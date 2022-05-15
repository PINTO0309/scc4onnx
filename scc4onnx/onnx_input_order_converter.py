#! /usr/bin/env python

import os
import sys
import ast
import traceback
from argparse import ArgumentParser
import onnx
import onnx_graphsurgeon as gs
import numpy as np
from typing import Optional, List

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


def gen_slice_op(
    previous_node: gs.Node,
    slice_dim: int,
    op_name_suffix_number: int
) -> gs.Node:
    """
    Parameters
    ----------
    previous_node: gs.Node
        Source Node object.

    slice_dim: int
        Dimensions to Slice.

    op_name_suffix_number: str
        Sequential number to be assigned to the end of the \n\
        generated Slice OP name.

    Returns
    -------
    slice: gs.Node
        Slice OP. onnx_graphsurgeon Node.
    """

    slice_output_shape = [dim if idx != slice_dim else 1 for idx, dim in enumerate(previous_node.outputs[0].shape)]
    slice_node_name = f"slice_out_{previous_node.outputs[0].name}_{op_name_suffix_number}"
    slice_out = gs.Variable(
        slice_node_name,
        dtype=previous_node.outputs[0].dtype,
        shape=slice_output_shape
    )
    starts_list = [0 if idx != slice_dim else op_name_suffix_number for idx in range(len(previous_node.outputs[0].shape))]
    ends_list = [2147483647 if idx != slice_dim else op_name_suffix_number+1 for idx, dim in enumerate(previous_node.outputs[0].shape)]
    axes_list = [val for val in range(len(previous_node.outputs[0].shape))]
    slice = gs.Node(
        op="Slice",
        name=f"{slice_node_name}_node",
        inputs=[
            previous_node.outputs[0],
            gs.Constant(
                f"starts_{slice_node_name}",
                np.asarray(starts_list, dtype=np.int64)
            ),
            gs.Constant(
                f"ends_{slice_node_name}",
                np.asarray(ends_list, dtype=np.int64)
            ),
            gs.Constant(
                f"axes_{slice_node_name}",
                np.asarray(axes_list, dtype=np.int64)
            ),
        ],
        outputs=[slice_out]
    )
    return slice


def gen_concat_op(
    input_variable: gs.Variable,
    concat_axis: int,
    concat_op_list: List[gs.Node]
) -> gs.Node:
    """
    Parameters
    ----------
    input_variable: gs.Variable
        Input variable.

    concat_axis: int
        Dimensions to Concat.

    concat_op_list: List[gs.Node]
        OP list of Concat targets.

    Returns
    -------
    concat: gs.Node
        Concat OP. onnx_graphsurgeon Node.
    """

    concat_node_name = f"concat_out_{input_variable.name}"
    concat_out = gs.Variable(
        concat_node_name,
        dtype=input_variable.dtype,
        shape=input_variable.shape
    )
    concat = gs.Node(
        op="Concat",
        name=f"{concat_node_name}_node",
        attrs={"axis": concat_axis},
        inputs=[slice_tmp.outputs[0] for slice_tmp in concat_op_list],
        outputs=[concat_out]
    )
    return concat


def order_conversion(
    input_op_names_and_order_dims: Optional[dict] = None,
    channel_change_inputs: Optional[dict] = None,
    input_onnx_file_path: Optional[str] = '',
    output_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:
    """
    Parameters
    ----------
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

    input_op_names_and_order_dims: Optional[dict]
        Specify the name of the input_op to be dimensionally changed and\n\
        the order of the dimensions after the change.\n\
        The name of the input_op to be dimensionally changed\n\
        can be specified multiple times.\n\n\
        e.g.\n\
        input_op_names_and_order_dims = {\n\
            "input_op_name1": [0,3,1,2],\n\
            "input_op_name2": [0,2,3,1],\n\
            "input_op_name3": [0,3,1,2,4,5],\n\
        }

    channel_change_inputs: Optional[dict]
        Change the channel order of RGB and BGR.\n\
        If the original model is RGB, it is transposed to BGR.\n\
        If the original model is BGR, it is transposed to RGB.\n\
        It can be selectively specified from among the OP names\n\
        specified in input_op_names_and_order_dims.\n\
        OP names not specified in input_op_names_and_order_dims are ignored.\n\
        Multiple times can be specified as many times as the number\n\
        of OP names specified in input_op_names_and_order_dims.\n\
        channel_change_inputs = {"op_name": dimension_number_representing_the_channel}\n\
        dimension_number_representing_the_channel must specify\n\
        the dimension position after the change in input_op_names_and_order_dims.\n\
        For example, dimension_number_representing_the_channel is 1 for NCHW and 3 for NHWC.\n\n\
        e.g.\n\
        channel_change_inputs = {\n\
            "aaa": 1,\n\
            "bbb": 3,\n\
            "ccc": 2,\n\
        }

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

    Returns
    -------
    order_converted_graph: onnx.ModelProto
        Order converted onnx ModelProto
    """

    # Unspecified check for input_onnx_file_path and onnx_graph
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    # Unspecified checks for input_op_names_and_order_dims and channel_change_inputs
    if not input_op_names_and_order_dims and not channel_change_inputs:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'At least one of input_op_names_and_order_dims or channel_change_inputs must be specified.'
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
    new_input_shapes = {}
    if input_op_names_and_order_dims:
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
                    # Transpose perm
                    #   perm = sorted(inverted_list_tmp, key=lambda x: x[1])
                    # inverted_list = {"input_op_name": perm}
                    inverted_list_tmp = []
                    new_input_shapes_tmp = []
                    for idx, dim in enumerate(order_dim):
                        inverted_list_tmp.append([idx, dim])
                        new_input_shapes_tmp.append(graph_input.shape[dim])
                    inverted_list[input_op_name] = [idx[0] for idx in sorted(inverted_list_tmp, key=lambda x: x[1])]
                    new_input_shapes[input_op_name] = new_input_shapes_tmp

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

    transpose_nodes = []
    next_nodes = []
    if input_op_names_and_order_dims:
        for transpose_idx, graph_input in enumerate(graph.inputs):
            if graph_input.name in new_input_shapes:

                # Rewriting of graph input OP variable
                old_graph_input_shape = graph_input.shape
                graph_input.shape = new_input_shapes[graph_input.name]

                # Generate Transpose OP
                transpose_out = gs.Variable(
                    f"transpose_out_{graph_input.name}",
                    dtype=graph_input.dtype,
                    shape=old_graph_input_shape
                )
                transpose = gs.Node(
                    op="Transpose",
                    name=f"input_order_convert_transpose_{transpose_idx}",
                    attrs={"perm": inverted_list[graph_input.name]},
                    inputs=[graph_input],
                    outputs=[transpose_out]
                )

                # Rewrite the input of the next OP to the output of Transpose
                for sub_node in graph.nodes:
                    for idx, sub_node_input in enumerate(sub_node.inputs):
                        if sub_node_input.name == graph_input.name:
                            sub_node.inputs[idx] = transpose.outputs[0]
                            transpose_nodes.append(transpose)
                            next_nodes.append(sub_node)
                            break
                graph.nodes.append(transpose)

    # OP generation for Color channel transposition
    # channel_change_inputs = {"name": dim, ...}
    if channel_change_inputs:
        slice_list_all = []
        concat_list = []
        for input_op_name, input_dim in channel_change_inputs.items():
            if len(transpose_nodes) > 0:
                # Transpose + RGB<->BGR
                for transpose_node, next_node in zip(transpose_nodes, next_nodes):
                    if transpose_node.inputs[0].name == input_op_name:
                        # Generate Slice OP
                        slice_list = []
                        for i in range(3):
                            slice = gen_slice_op(
                                transpose_node,
                                input_dim,
                                i
                            )
                            slice_list.append(slice)
                        slice_list_all.append(slice_list)
                        # Generate Concat OP
                        slice_list.reverse()
                        concat = gen_concat_op(
                            transpose_node.outputs[0],
                            input_dim,
                            slice_list
                        )
                        # Rewrite the input of the next OP to the output of Transpose
                        for idx, next_node_input in enumerate(next_node.inputs):
                            if next_node_input.name == transpose_node.outputs[0].name:
                                next_node.inputs[idx] = concat.outputs[0]
                                break
                        concat_list.append(concat)

            else:
                # RGB<->BGR only
                for graph_node in graph.nodes:
                    for graph_node_input in graph_node.inputs:
                        if graph_node_input.name == input_op_name:
                            # Generate Slice OP
                            slice_list = []
                            for i in range(3):
                                slice = gen_slice_op(
                                    graph_node_input,
                                    input_dim,
                                    i
                                )
                                slice_list.append(slice)
                            slice_list_all.append(slice_list)
                            # Generate Concat OP
                            slice_list.reverse()
                            concat = gen_concat_op(
                                graph_node_input,
                                input_dim,
                                slice_list
                            )
                            # Rewrite the input of the next OP to the output of Transpose
                            for idx, next_node_input in enumerate(graph_node.inputs):
                                if next_node_input.name == graph_node_input.name:
                                    graph_node.inputs[idx] = concat.outputs[0]
                                    break
                            concat_list.append(concat)

        for slice_list in slice_list_all:
            for slice_op in slice_list:
                graph.nodes.append(slice_op)
        for concat_op in concat_list:
            graph.nodes.append(concat_op)


    # Cleanup
    graph.cleanup().toposort()
    order_converted_graph = gs.export_onnx(graph)

    # Optimize
    new_model = None
    try:
        new_model = onnx.shape_inference.infer_shapes(order_converted_graph)
    except Exception as e:
        new_model = order_converted_graph
        if not non_verbose:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                'The input shape of the next OP does not match the output shape. '+
                'Be sure to open the .onnx file to verify the certainty of the geometry.'
            )
            tracetxt = traceback.format_exc().splitlines()[-1]
            print(f'{Color.YELLOW}WARNING:{Color.RESET} {tracetxt}')

    # Save
    if output_onnx_file_path:
        onnx.save(new_model, output_onnx_file_path)

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

    # Return
    return new_model


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
    if input_op_names_and_order_dims:
        input_op_names_and_order_dims_tmp = {
            input_op_names_and_order_dim[0]: ast.literal_eval(input_op_names_and_order_dim[1]) for input_op_names_and_order_dim in input_op_names_and_order_dims
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
