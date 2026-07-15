# 🗜 Model Parameter Compressor

The `appfl.compressor` module can be used for compressing the model parameters or gradients in a lossy manner before the client sends them back to the server for more efficient communication. The server then will decompress the compressed model before the global aggregation.

The `appfl.compressor` currently supports the following lossy compressors. Please refer to their official project/GitHub pages if you want more detailed information of them. Here, we only provide the installation instructions. **Note: SZx need particular permission to access because of the collaboration with a third-party, so we omit its installation here.**

1. [SZ2: Error-bounded Lossy Compressor for HPC Data](https://github.com/szcompressor/SZ)
2. [SZ3: A Modular Error-bounded Lossy Compression Framework for Scientific Datasets](https://github.com/szcompressor/SZ3)
3. [ZFP: Compressed Floating-Point and Integer Arrays](https://pypi.org/project/zfpy/)
4. [SZX: An Ultra-fast Error-bounded Lossy Compressor for Scientific Datasets](https://github.com/szcompressor/SZx)

## Installation
Users can easily install all the above compressors by running the following command.
```bash
appfl-install-compressor
```

## Citation
Please check the following paper for details about how the compressor plays a role in federated learning.

```
@article{wilkins2023efficient,
  title={Efficient Communication in Federated Learning Using Floating-Point Lossy Compression},
  author={Wilkins, Grant and Di, Sheng and Calhoun, Jon C and Kim, Kibaek and Underwood, Robert and Cappello, Franck},
  journal={arXiv preprint arXiv:2312.13461},
  year={2023}
}
```
