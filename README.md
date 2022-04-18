# [WIP] scc4onnx
Very simple NCHW and NHWC conversion tool for ONNX. Change to the specified input order for each and every input OP. Also, change the channel order of RGB and BGR. **S**imple **C**hannel **C**onverter for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/scc4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/scc4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/scc4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/scc4onnx?color=2BAF2B)](https://pypi.org/project/scc4onnx/) [![CodeQL](https://github.com/PINTO0309/scc4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/scc4onnx/actions?query=workflow%3ACodeQL)

# Key concept

- [x] Allow the user to specify the name of the input OP to change the input order.
- [x] All number of dimensions can be freely changed, not only 4 dimensions such as NCHW and NHWC.
- [x] Simply rewrite the input order of the input OP to the specified order and extrapolate Transpose after the input OP so that it does not affect the processing of subsequent OPs.
- [x] Allows the user to change the channel order of RGB and BGR by specifying options.

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
