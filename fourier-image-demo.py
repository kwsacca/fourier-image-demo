## Doxygen header
# @author   Kevin Sacca
# @email    ssriceboat@gmail.com
# @title    Fourier Image Demo
#
# @brief    Image Reconstruction Demonstration using 2D Fourier Transform
#           Decomposition.
#
#           Epilepsy warning! See README and use -e from CL/Terminal/CMD.
#
#           ------Keyboard Controls when running------
#           Pause/Resume with p key
#           Quit with q or Esc keys
#           Toggle Epilepsy Mode with e key (See README for details)
#
#           See README for dependencies. I think all are pip install-able.
#
#           Example command entry in terminal or CMD:
#           python fourier-image-demo.py [-i /path/to/image.png] [-f] [-e]

## Standard library imports
################################################################################
import argparse
import cv2
import glob
from multiprocessing import Pipe
import numpy as np
import os
from PIL import Image
import PyQt5
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import threading
import time


## Custom library imports
################################################################################
from riceprint import pprint, tprint, progressbar
from ricekey import KBHit


## Main script definition
################################################################################
def main(args):
   # Process input image
   src = np.asarray(Image.open(args.input))
   nRows, nCols = src.shape[0], src.shape[1]
   nBands = src.shape[2] if len(src.shape) > 2 else 1

   # 1 - Original Image Bands
   if nBands > 1:
      src = np.flip(src, axis=2) # cv2 flips bands for display
   src = square_image(src)
   src = cv2.resize(src, dsize=(300,300), interpolation=cv2.INTER_CUBIC)
   nRows, nCols = src.shape[0], src.shape[1]
   src = src.reshape((nRows,nCols,nBands))

   # 2 - Fourier Transform - log(mag)
   fft, mag, fft_mag = get_fft(src)

   # Set up empties
   curr_comp = np.zeros(src.shape).astype(np.complex128)
   disp_curr_comp = np.zeros(src.shape).astype(np.uint8)
   sum_comp = np.zeros(src.shape)
   disp_sum_comp = np.zeros(src.shape)
   fft_comp = np.zeros(src.shape)
   disp_fft_comp = np.zeros(src.shape)
   curr_comp_n = np.zeros(src.shape).astype(np.complex128)
   disp_curr_comp_n = np.zeros(src.shape).astype(np.uint8)

   # Create parent (comA) and child (comB) ports of a pipe
   comA, comB = Pipe()

   # Start the keypress monitoring thread
   thread = threading.Thread(target=kbcontrol, args=(comB,))
   thread.start()

   # While the thread is alive, do.
   numFreq = 0
   go = True
   stop = 0
   paused = False
   while go == True:
      if np.max(mag) == 0:
         go = False
         paused = True
         pprint('Complete Image Reconstructed!', 'c')

      while paused == True:
         # If paused, can update frame with components for epilepsy mode
         if freeze == 0:
            # Build mosaic frame
            mosaic = assemble_mosaic(args, src, fft_mag, disp_curr_comp,
                                    disp_sum_comp, disp_fft_comp,
                                    disp_curr_comp_n, force_show=True)

            # Show frame
            cv2.namedWindow('Fourier Transform Display')
            cv2.imshow('Fourier Transform Display', mosaic)
            cv2.waitKey(1)
            freeze = 1

         # Check com for command
         if go == False and stop == 0:
            pprint('Press q or Q to stop and exit.', 'c')
            stop = 1
         if comA.poll():
            msg = comA.recv()
            if msg == 1:
               paused = False
               tprint('')
            elif msg == 0:
               paused = False
               go = False
            elif msg == 2:
               if args.epilepsy == True:
                  args.epilepsy = False
                  pprint('Epilepsy Mode OFF. Press <E> to re-enable.', 'y')
               elif args.epilepsy == False:
                  args.epilepsy = True
                  pprint('Epilepsy Mode ON. Press <E> to disable.', 'dy')
            else:
               tprint('key has no function', 'r')

      if go == True:
         check = 0
         freeze = 0
         # Compute components for each band individually before displaying
         for i in range(nBands):
            # 3 - Current Component (Unscaled)
            tmp_comp = np.zeros((nRows, nCols)).astype(np.complex128)
            my, mx = np.unravel_index(np.argmax(mag[:,:,i]), (nRows, nCols))
            cx = int(np.round(nCols/2 - mx + nCols/2, 0))
            cy = int(np.round(nRows/2 - my + nRows/2, 0))

            # Catch OOB errors
            cy = nRows-1 if cy == nRows else cy
            cx = nCols-1 if cx == nCols else cx
            my = nRows-1 if my == nRows else my
            mx = nCols-1 if mx == nCols else mx

            tmp_comp[my,mx] = fft[my,mx,i]
            tmp_comp[cy,cx] = fft[cy,cx,i]
            curr_comp[:,:,i] = np.fft.ifft2(np.fft.ifftshift(tmp_comp))
            mag[my,mx,i] = 0
            mag[cy,cx,i] = 0

            disp_curr_comp[:,:,i] = (curr_comp[:,:,i].real+128).astype(np.uint8)
            if mx == (nCols/2) and my == (nRows/2):
               x = np.sum(src)/(nRows*nCols)
               disp_curr_comp[:,:,i] = x*np.ones([nRows,nCols]).astype(np.uint8)

            # 4 - Summed Components
            sum_comp[:,:,i] += curr_comp[:,:,i].real
            sum_comp[:,:,i][sum_comp[:,:,i]<0] = 0
            sum_comp[:,:,i][sum_comp[:,:,i]>255] = 255
            disp_sum_comp[:,:,i] = (abs(sum_comp[:,:,i])).astype(np.uint8)

            # 5 - Fourier Coefficients Used - log(mag)
            fft_comp[my,mx,i] = fft_mag[my,mx,i]
            fft_comp[cy,cx,i] = fft_mag[cy,cx,i]
            disp_fft_comp[:,:,i] = fft_comp[:,:,i].astype(np.uint8)

            # 6 - Current Component (Normalized)
            curr_comp_n[:,:,i] = (curr_comp[:,:,i].real /
                                    np.max(abs(curr_comp[:,:,i].real)))*127 +128
            disp_curr_comp_n[:,:,i] = curr_comp_n[:,:,i].real
            disp_curr_comp_n[:,:,i][disp_curr_comp_n[:,:,i] < 0] = 0
            disp_curr_comp_n[:,:,i][disp_curr_comp_n[:,:,i] > 255] = 255

         numFreq += 1
         # Build mosaic frame
         mosaic = assemble_mosaic(args, src, fft_mag, disp_curr_comp,
                                 disp_sum_comp, disp_fft_comp,
                                 disp_curr_comp_n)

         # Show frame
         cv2.namedWindow('Fourier Transform Display')
         cv2.imshow('Fourier Transform Display', mosaic)
         cv2.waitKey(1)

         if args.fullspeed == False:
            if args.epilepsy:
               t = 0.25
            else:
               t = 4 / 2**(numFreq - 1)
            time.sleep(t)

      # Check com for command
      if go == True:
         if comA.poll():
            msg = comA.recv()
            if msg == 1:
               paused = False
               tprint('')
            elif msg == -1:
               paused = True
               tprint('[Paused]', 'dg')
            elif msg == 2:
               if args.epilepsy == True:
                  args.epilepsy = False
                  pprint('Epilepsy Mode OFF. Press <E> to re-enable.', 'y')
               elif args.epilepsy == False:
                  args.epilepsy = True
                  pprint('Epilepsy Mode ON. Press <E> to disable.', 'dy')
            elif msg == 0:
               go = False
            else:
               tprint('key has no function', 'r')

   # Close down
   thread.join()

   return None


## Function definitions
################################################################################
def get_fft(src):
   '''Return 2D FFT of src in log(mag) scale for viewing'''

   if len(src.shape) == 2:
      fft = np.fft.fftshift(np.fft.fft2(src))
      mag = abs(fft)
      fft_mag = (((np.log10(mag)) / (np.max(np.log10(mag))))*255)

   else:
      fft = np.zeros(src.shape).astype(np.complex128)
      mag = np.zeros(src.shape)
      fft_mag = np.zeros(src.shape)
      for i in range(src.shape[2]):
         fft[:,:,i] = np.fft.fftshift(np.fft.fft2(src[:,:,i]))
         mag[:,:,i] = abs(fft[:,:,i])
         fft_mag[:,:,i] = (((np.log10(mag[:,:,i])) /
                           (np.max(np.log10(mag[:,:,i]))))*255)

   return fft, mag, fft_mag


def get_file_gui():
   # Opens GUI to select image
   home = os.path.expanduser('~')
   os.chdir(home)
   app = QApplication(sys.argv)

   tprint('Please select an image. (.jpg, .png, .tif)', 'dg')
   filename, _ = QFileDialog.getOpenFileName()
   filename = str(filename)
   tprint('')

   return filename


def square_image(src):
   ''' Make image a square for display and FFT speed purposes. '''
   nRows, nCols = src.shape[0], src.shape[1]
   if len(src.shape) == 2:
      nBands = 1
   elif len(src.shape) == 3:
      nBands = src.shape[2]

   # Trim Rows
   if nRows > nCols:
      delta = int(nRows - nCols)
      if delta % 2 == 0:
         idx = int(np.ceil(delta / 2))
         if nBands == 1:
            src = src[idx:-idx, :]
         elif nBands > 1:
            src = src[idx:-idx, :, :]
      else:
         idx = int(np.ceil(delta / 2))
         if nBands == 1:
            src = src[idx-1:-idx, :]
         elif nBands > 1:
            src = src[idx-1:-idx, :, :]

   # Trim Cols
   elif nRows < nCols:
      delta = int(nCols - nRows)
      if delta % 2 == 0:
         idx = int(np.ceil(delta / 2))
         if nBands == 1:
            src = src[:, idx:-idx]
         elif nBands > 1:
            src = src[:, idx:-idx, :]
      else:
         idx = int(np.ceil(delta / 2))
         if nBands == 1:
            src = src[:, idx-1:-idx]
         elif nBands > 1:
            src = src[:, idx-1:-idx, :]

   return src


def assemble_mosaic(args, src, fft_mag, disp_curr_comp, disp_sum_comp,
                     disp_fft_comp, disp_curr_comp_n, force_show=False):

   nRows, nCols = src.shape[0], src.shape[1]
   if len(src.shape) == 2:
      nBands = 1
   elif len(src.shape) == 3:
      nBands = src.shape[2]

   mosaic = (np.zeros([2*nRows, 3*nCols, nBands])).astype(np.uint8)

   # Place components into their respective positions
   mosaic[(0):(nRows), (0):(nCols), :] = src.astype(np.uint8)
   mosaic[(0):(nRows), (nCols):(2*nCols), :] = fft_mag.astype(np.uint8)
   if args.epilepsy == True and force_show == False:
      mosaic[(0):(nRows), (2*nCols):(3*nCols), :] = np.zeros(src.shape)
   else:
      mosaic[(0):(nRows), (2*nCols):(3*nCols), :] = disp_curr_comp
   mosaic[(nRows):(2*nRows), (0):(nCols), :] = disp_sum_comp
   mosaic[(nRows):(2*nRows), (nCols):(2*nCols), :] = disp_fft_comp
   if args.epilepsy == True and force_show == False:
      mosaic[(nRows):(2*nRows), (2*nCols):(3*nCols), :] = np.zeros(src.shape)
   else:
      mosaic[(nRows):(2*nRows), (2*nCols):(3*nCols), :] = disp_curr_comp_n

   # Create text boxes
   cv2.rectangle(mosaic, (3,0), (90,13), [0,0,0], -1)
   cv2.rectangle(mosaic, (nCols+3,0), (nCols+205,13), [0,0,0], -1)
   cv2.rectangle(mosaic, (2*nCols+3,0), (2*nCols+130,13), [0,0,0], -1)
   cv2.rectangle(mosaic, (3,nRows), (135,nRows+13), [0,0,0], -1)
   cv2.rectangle(mosaic, (nCols+3,nRows), (nCols+250,nRows+13), [0,0,0], -1)
   cv2.rectangle(mosaic, (2*nCols+3,nRows), (2*nCols+170,nRows+13), [0,0,0], -1)

   # Add text to text boxes
   cv2.putText(mosaic, 'Original Image', (5,10),
           cv2.FONT_HERSHEY_PLAIN, 0.7, [255,255,255])
   cv2.putText(mosaic, '2D Fourier Transform - log(mag)', (nCols+5,10),
           cv2.FONT_HERSHEY_PLAIN, 0.7, [255,255,255])
   if args.epilepsy == True and force_show == False:
      cv2.putText(mosaic, '[DISABLED FOR EPILEPSY]', (2*nCols+5,10),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, [255,255,255])
   else:
      cv2.putText(mosaic, 'Current Component', (2*nCols+5,10),
              cv2.FONT_HERSHEY_PLAIN, 0.7, [255,255,255])
   cv2.putText(mosaic, 'Summed Components', (5,nRows+10),
           cv2.FONT_HERSHEY_PLAIN, 0.7, [255,255,255])
   cv2.putText(mosaic, 'Fourier Components Added - log(mag)',(nCols+5,nRows+10),
           cv2.FONT_HERSHEY_PLAIN, 0.7, [255,255,255])
   if args.epilepsy == True and force_show == False:
      cv2.putText(mosaic, '[DISABLED FOR EPILEPSY]', (2*nCols+5,nRows+10),
              cv2.FONT_HERSHEY_PLAIN, 0.7, [255,255,255])
   else:
      cv2.putText(mosaic, 'Scaled Current Component', (2*nCols+5,nRows+10),
              cv2.FONT_HERSHEY_PLAIN, 0.7, [255,255,255])

   # Draw lines between components
   cv2.line(mosaic, (0,nRows), (3*nCols,nRows), [255,255,255])
   cv2.line(mosaic, (nCols,0), (nCols,2*nRows), [255,255,255])
   cv2.line(mosaic, (2*nCols,0), (2*nCols,2*nRows), [255,255,255])

   return mosaic


def kbcontrol(com):
   """
   Threaded function to enable keyboard events to stop a function. This is an
   example usage of this class that results in the 'q' and 'Esc' keys acting
   as terminating keypresses for a threaded process. This way, when the
   thread running this function returns False, you know outside the thread
   that the user pressed q or Esc and can handle safe shutdown of code
   separately.

   Ord Keymap Lookup Table:
   A - 97      J - 106     S - 115     1 - 49      Esc - 27    =     - 61
   B - 98      K - 107     T - 116     2 - 50      ,   - 44    `     - 96
   C - 99      L - 108     U - 117     3 - 51      .   - 46    \     - 92
   D - 100     M - 109     V - 118     4 - 52      /   - 47    <--   - 127
   E - 101     N - 110     W - 119     5 - 53      ;   - 59    Enter - 10
   F - 102     O - 111     X - 120     6 - 54      '   - 39    Space - 32
   G - 103     P - 112     Y - 121     7 - 55      [   - 91    Tab   - 9
   H - 104     Q - 113     Z - 122     8 - 56      ]   - 93
   I - 105     R - 114     0 - 48      9 - 57      -   - 45
   (You can also chr(ord(key)) to get a string of the character pressed)
   """
   kb = KBHit()
   paused = False
   pprint('Press p to pause/resume. Press q or Esc to stop and exit.', 'g')
   while(True):
      # Check for any keypress
      if kb.kbhit():
         key = kb.getch()
         # If keypress was 'q' or 'Esc', return False to terminate thread
         if ord(key) == 27 or ord(key) == 113:
            pprint('Stopkey pressed. Terminating.')
            tprint('Exiting...')
            kb.set_normal_term()
            com.send(0)
            com.close()
            return False

         # If P is pressed, either pause or resume
         elif ord(key) == 112:
            if paused == False:
               com.send(-1)
               paused = True
            else:
               com.send(1)
               paused = False

         # If E is pressed, toggle Epilepsy Mode
         elif ord(key) == 101:
            com.send(2)

      time.sleep(0.001)


if __name__ == '__main__':
   ap = argparse.ArgumentParser()
   ap.add_argument('-i', '--input', type=str, default=None,
                     help='Path to image file.')
   ap.add_argument('-e', '--epilepsy', action='store_true', default=False,
                     help='If you are prone to epileptic seizures, use -e to \
                           disable the bright flashing panels. \
                           NOTE: Pausing by pressing the P key will show the \
                           components as long as it remains paused. \
                           NOTE: Pressing E will toggle epilepsy mode on and\
                           off while running and unpaused.')
   ap.add_argument('-f', '--fullspeed', action='store_true', default=False,
                     help='If True, time between frames is as low as possible.')
   args = ap.parse_args()

   if args.input is None:
      args.input = get_file_gui()

   # Override fullspeed true if epilepsy flag used
   if args.epilepsy == True:
      args.fullspeed = False
      pprint('WARNING: Epilepsy Mode ON, pause with <P> to see disabled panes.')

   sys.exit(main(args))
