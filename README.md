# [WIP] scc4onnx
Very simple NCHW and NHWC conversion tool for ONNX. Change to the specified input order for each and every input OP. Also, change the channel order of RGB and BGR. **S**imple **C**hannel **C**onverter for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/scc4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/scc4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/scc4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/scc4onnx?color=2BAF2B)](https://pypi.org/project/scc4onnx/) [![CodeQL](https://github.com/PINTO0309/scc4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/scc4onnx/actions?query=workflow%3ACodeQL)

# Key concept

- [ ] Allow the user to specify the name of the input OP to change the input order.
- [ ] All number of dimensions can be freely changed, not only 4 dimensions such as NCHW and NHWC.
- [ ] Simply rewrite the input order of the input OP to the specified order and extrapolate Transpose after the input OP so that it does not affect the processing of subsequent OPs.
- [ ] Allows the user to change the channel order of RGB and BGR by specifying options.
