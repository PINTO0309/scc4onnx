
from setuptools import setup, find_packages
from os import path
import re

package_name="scc4onnx"
root_dir = path.abspath(path.dirname(__file__))

with open("README.md") as f:
    long_description = f.read()

with open(path.join(root_dir, package_name, '__init__.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

setup(
    name=package_name,
    version=version,
    description=\
        "Very simple NCHW and NHWC conversion tool for ONNX. "+
        "Change to the specified input order for each and every input OP. "+
        "Also, change the channel order of RGB and BGR. Simple Channel Converter for ONNX. "+
        "Simple Channel Conversion for ONNX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Katsuya Hyodo",
    author_email="rmsdh122@yahoo.co.jp",
    url="https://github.com/PINTO0309/scc4onnx",
    license="MIT License",
    packages=find_packages(),
    platforms=["linux", "unix"],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            "scc4onnx=scc4onnx:main"
        ]
    }
)
