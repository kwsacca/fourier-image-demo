Fourier Image Demo
=========

|shield1| |shield2| |shield3| |shield4|

.. |shield1| image:: https://img.shields.io/github/release/kwsacca/fourier-image-demo.svg?color=blue
   :width: 20%

.. |shield2| image:: https://img.shields.io/badge/Python-%3E=3.5-blue.svg?color=e6ac00
   :width: 20%

.. |shield3| image:: https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg
   :width: 24%

.. |shield4| image:: https://img.shields.io/github/license/kwsacca/fourier-image-demo.svg?color=blue
   :width: 20%

About
=====

Author: Kevin Sacca ssriceboat@gmail.com

A demonstration of image reconstruction one sinusoid at a time using 2D Fourier
Transformation. (Epilepsy Warning! see below)

.. code:: bash

    python fourier-image-demo.py [-i /path/to/image.png] [-f] [-e]


Supply an image from the CL with -i or use the file browser GUI to select an
image.

The demo begins slow to see a small handful of low-frequency additions, then
exponentially speeds up until reaching the maximum computation speed. To disable
the slow start, use -f for full speed right from the beginning.

Epilepsy warning, use -e to enable epilepsy mode, which prevents the brightest
flashing panes from displaying and refreshing. Pause the demo to see the
disabled panes for as long as it remains paused. (NOTE: The reconstruction of
the input image is always displayed, so depending on your input image, bright
flashing may still occur, choose an image with low brightness and low contrast
to be safer.) Epilepsy mode will disable the -f fullspeed flag.

Nonstandard Dependencies:
   - PyQt
   - cv2
   - PIL
   - ricekey   (a project of mine, pip install ricekey)
   - riceprint (a project of mine, pip install riceprint)

Works on Linux, macOS, Windows.

I don't claim this is the fastest code out there, but it goes about 60 fps and
works on greyscale and color images. I would recommend sticking to 8-bit jpg,
png, and tif image files. This is an old school project I decided to upload.

License
=======

MIT License

Copyright (c) 2015 kwsacca

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

