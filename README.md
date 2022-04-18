# scc4onnx
Very simple NCHW and NHWC conversion tool for ONNX. Change to the specified input order for each and every input OP. Also, change the channel order of RGB and BGR. **S**imple **C**hannel **C**onverter for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/scc4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/scc4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/scc4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/scc4onnx?color=2BAF2B)](https://pypi.org/project/scc4onnx/) [![CodeQL](https://github.com/PINTO0309/scc4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/scc4onnx/actions?query=workflow%3ACodeQL)

# Key concept

- [x] Allow the user to specify the name of the input OP to change the input order.
- [x] All number of dimensions can be freely changed, not only 4 dimensions such as NCHW and NHWC.
- [x] Simply rewrite the input order of the input OP to the specified order and extrapolate Transpose after the input OP so that it does not affect the processing of subsequent OPs.
- [x] Allows the user to change the channel order of RGB and BGR by specifying options.

## 1. Setup
### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U scc4onnx
```
### 1-2. Docker
```bash
### docker pull
$ docker pull pinto0309/scc4onnx:latest

### docker build
$ docker build -t pinto0309/scc4onnx:latest .

### docker run
$ docker run --rm -it -v `pwd`:/workdir pinto0309/scc4onnx:latest
$ cd /workdir
```

## 2. CLI Usage
```bash
$ scc4onnx -h

usage:
  scc4onnx [-h]
  --input_onnx_file_path INPUT_ONNX_FILE_PATH
  --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
  [--input_op_names_and_order_dims INPUT_OP_NAME ORDER_DIM]
  [--channel_change_inputs INPUT_OP_NAME DIM]
  [--non_verbose]

optional arguments:
  -h, --help
      show this help message and exit

  --input_onnx_file_path INPUT_ONNX_FILE_PATH
      Input onnx file path.

  --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
      Output onnx file path.

  --input_op_names_and_order_dims INPUT_OP_NAME ORDER_DIM
      Specify the name of the input_op to be dimensionally changed and the order of the
      dimensions after the change.
      The name of the input_op to be dimensionally changed can be specified multiple times.

      e.g.
      --input_op_names_and_order_dims aaa [0,3,1,2] \
      --input_op_names_and_order_dims bbb [0,2,3,1] \
      --input_op_names_and_order_dims ccc [0,3,1,2,4,5]

  --channel_change_inputs INPUT_OP_NAME DIM
      Change the channel order of RGB and BGR. If the original model is RGB,
      it is transposed to BGR.
      If the original model is BGR, it is transposed to RGB. It can be selectively specified
      from among the OP names specified in --input_op_names_and_order_dims.
      OP names not specified in --input_op_names_and_order_dims are ignored.
      Multiple times can be specified as many times as the number of OP names specified
      in --input_op_names_and_order_dims.
      --channel_change_inputs op_name dimension_number_representing_the_channel
      dimension_number_representing_the_channel must specify the dimension position before
      the change in input_op_names_and_order_dims.
      For example, dimension_number_representing_the_channel is 1 for NCHW and 3 for NHWC.

      e.g.
      --channel_change_inputs aaa 3 \
      --channel_change_inputs bbb 1 \
      --channel_change_inputs ccc 5

  --non_verbose
      Do not show all information logs. Only error logs are displayed.
```

## 3. In-script Usage
```python
$ python
>>> from scc4onnx import order_conversion
>>> help(order_conversion)
Help on function order_conversion in module scc4onnx.onnx_input_order_converter:

order_conversion(
  input_op_names_and_order_dims: Union[dict, NoneType] = None,
  channel_change_inputs: Union[dict, NoneType] = None,
  input_onnx_file_path: Union[str, NoneType] = '',
  output_onnx_file_path: Union[str, NoneType] = '',
  onnx_graph: Union[onnx.onnx_ml_pb2.ModelProto, NoneType] = None,
  non_verbose: Union[bool, NoneType] = False
) -> onnx.onnx_ml_pb2.ModelProto

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.
        Either input_onnx_file_path or onnx_graph must be specified.
    
    output_onnx_file_path: Optional[str]
        Output onnx file path.
        If output_onnx_file_path is not specified, no .onnx file is output.
    
    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.
        Either input_onnx_file_path or onnx_graph must be specified.
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.
    
    input_op_names_and_order_dims: Optional[dict]
        Specify the name of the input_op to be dimensionally changed and
        the order of the dimensions after the change.
        The name of the input_op to be dimensionally changed
        can be specified multiple times.
    
        e.g.
        input_op_names_and_order_dims = {
            "input_op_name1": [0,3,1,2],
            "input_op_name2": [0,2,3,1],
            "input_op_name3": [0,3,1,2,4,5],
        }
    
    channel_change_inputs: Optional[dict]
        Change the channel order of RGB and BGR.
        If the original model is RGB, it is transposed to BGR.
        If the original model is BGR, it is transposed to RGB.
        It can be selectively specified from among the OP names
        specified in input_op_names_and_order_dims.
        OP names not specified in input_op_names_and_order_dims are ignored.
        Multiple times can be specified as many times as the number
        of OP names specified in input_op_names_and_order_dims.
        channel_change_inputs = {"op_name": dimension_number_representing_the_channel}
        dimension_number_representing_the_channel must specify
        the dimension position after the change in input_op_names_and_order_dims.
        For example, dimension_number_representing_the_channel is 1 for NCHW and 3 for NHWC.
    
        e.g.
        channel_change_inputs = {
            "aaa": 1,
            "bbb": 3,
            "ccc": 2,
        }
    
    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Default: False
    
    Returns
    -------
    order_converted_graph: onnx.ModelProto
        Order converted onnx ModelProto
```

## 4. CLI Execution
```bash
$ scc4onnx \
--input_onnx_file_path crestereo_next_iter2_240x320.onnx \
--output_onnx_file_path crestereo_next_iter2_240x320_ord.onnx \
--input_op_names_and_order_dims left [0,2,3,1] \
--input_op_names_and_order_dims right [0,2,3,1] \
--channel_change_inputs left 1 \
--channel_change_inputs right 1
```

## 5. In-script Execution
```python
from scc4onnx import order_conversion

order_converted_graph = order_conversion(
    onnx_graph=graph,
    input_op_names_and_order_dims={"left": [0,2,3,1], "right": [0,2,3,1]},
    channel_change_inputs={"left": 1, "right": 1},
    non_verbose=True,
)
```

## 6. Sample
### 6-1. Transpose only
![image](https://user-images.githubusercontent.com/33194443/163724226-e621b32e-1f08-4b86-87ec-bb79fa6a2b80.png)
```bash
$ scc4onnx \
--input_onnx_file_path crestereo_next_iter2_240x320.onnx \
--output_onnx_file_path crestereo_next_iter2_240x320_ord.onnx \
--input_op_names_and_order_dims left [0,2,3,1] \
--input_op_names_and_order_dims right [0,2,3,1]
```
![image](https://user-images.githubusercontent.com/33194443/163724308-3f61282c-6766-4c7a-ac35-3b83a9b0d1bb.png)
![image](https://user-images.githubusercontent.com/33194443/163724341-7fbd3b1c-8321-42ad-abc6-f1db1c10bcf4.png)

### 6-2. Transpose + RGB<->BGR
![image](https://user-images.githubusercontent.com/33194443/163724226-e621b32e-1f08-4b86-87ec-bb79fa6a2b80.png)
```bash
$ scc4onnx \
--input_onnx_file_path crestereo_next_iter2_240x320.onnx \
--output_onnx_file_path crestereo_next_iter2_240x320_ord.onnx \
--input_op_names_and_order_dims left [0,2,3,1] \
--input_op_names_and_order_dims right [0,2,3,1] \
--channel_change_inputs left 1 \
--channel_change_inputs right 1
```
![image](https://user-images.githubusercontent.com/33194443/163744943-ff11a465-47f7-4aa1-9991-986eac318a23.png)

### 6-3. RGB<->BGR only
![image](https://user-images.githubusercontent.com/33194443/163724226-e621b32e-1f08-4b86-87ec-bb79fa6a2b80.png)
```bash
$ scc4onnx \
--input_onnx_file_path crestereo_next_iter2_240x320.onnx \
--output_onnx_file_path crestereo_next_iter2_240x320_ord.onnx \
--channel_change_inputs left 1 \
--channel_change_inputs right 1
```
![image](https://user-images.githubusercontent.com/33194443/163745487-96d2481e-bef9-4c18-8a55-ec0eea7bb684.png)

## 7. Issues
https://github.com/PINTO0309/simple-onnx-processing-tools/issues
