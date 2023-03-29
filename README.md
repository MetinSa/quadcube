# quadcube

Python library (written in rust) that currently provide one method `pix2vec` for converting between quadcube res15 pixel number to ecliptic unit vectors.

The code is a reimplementation of parts of the COBE DIRBE map binning routines provided by A.J. Banday. See the [COBE Explanatory Supplement](https://lambda.gsfc.nasa.gov/product/cobe/dirbe_exsup.html) for more information.

## Install
`pip install quadcube`

## Example
```python
import quadcube

quadcube.pix2vec(98477345)
# >>> array([[0.37147827],
#           [0.62248052],
#           [0.6888555 ]])

pixels = [315879751, 238749305, 408302824, 290970621, 427780527]
quadcube.pix2vec(pixels)
# >>> array([[ 0.90477257, -0.51544922,  0.91755844,  0.74030402,  0.98561299],
#            [-0.40535959,  0.10338045, -0.39665074, -0.00232418, -0.00949373],
#            [-0.13065298,  0.85066127,  0.02747185, -0.67226822,  0.16875101]])

```
