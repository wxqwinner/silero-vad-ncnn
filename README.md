# Silero VAD NCNN

Convert the [silero-vad v4](https://github.com/snakers4/silero-vad/tree/v4.0stable) to NCNN, only 16K is supported.

1. vad/collections/resources/silero.jit
[the original jit model](
https://github.com/snakers4/silero-vad/blob/v4.0stable/files/silero_vad.jit)

1. vad/collections/resources/silero_vad.jit
the generated jit model.

1. vad/collections/resources/silero.ncnn.param
the ncnn model.


## Quick Start
- main.py
demo scripts.
- convert.py
Conversion script to generate 'silero.jit', which also supports ncnn model by pnnx, [but there are some issues](https://github.com/pnnx/pnnx/issues/161)
Generate silero.jit and then convert it via the old pnnx version.
[pnnx-20231010-ubuntu.zip](https://github.com/pnnx/pnnx/releases/tag/20231010) work!!!

    ```bash
    ./pnnx silero.jit inputshape=[1,512],[2,1,64],[2,1,64]
    ```

- compare.py
Compare the original jit model with the generated jit model and the ncnn model.
