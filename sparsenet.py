import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import fista
import argparse
import os
from PIL import Image
from numpy.fft import fft2, ifft2, fftshift

parser = argparse.ArgumentParser(description='Compute Overcomplete Sparse '
                                 'Dictionary')
parser.add_argument('--imgs', dest='imgs', type=str, default=None,
                    help='Path to folder of images (not prescaled).')
parser.add_argument('--lambdav', type=float, default=0.1,
                    help='Image variance after whitening')
parser.add_argument('--inference', type=str, default='fista',
                    choices=['fista', 'lca'],
                    help='The inference algorithm used')
parser.add_argument('--size', type=int, default=8,
                    help='Patch side length')
parser.add_argument('--suffix', dest='suffix', nargs='+',
                    default=['.jpeg', '.jpg', '.png'],
                    help='List of acceptable suffixes to try load in the image '
                    'folder (should include the ".")')


def sparsenet(IMAGES, patch_dim=8, neurons=128, lambdav=0.1, eta=6.0,
              num_trials=3000, batch_size=100, border=4, inference='lca'):
  """
  IMAGES: list or array of images. First dimension should be the number of
    images. If list, the images can be of different size.
  patch_dim - side length of patch
  lambdav: Sparsity Constraint
  eta: Learning Rate
  num_trials: Learning Iterations
  batch_size: Batch size per iteration
  border: Border when extracting image patches
  Inference: 'lca' or 'fista'
  """
  num_images = len(IMAGES)

  sz = patch_dim
  eta = eta / batch_size

  # Initialize basis functions
  Phi = np.random.randn(sz*sz, neurons)
  Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis=0))))

  I = np.zeros((sz*sz,batch_size))

  for t in range(num_trials):

    # Choose a random image
    imi = np.random.randint(num_images)
    s1, s2 = IMAGES[imi].shape

    for i in range(batch_size):
      r = border + np.random.randint(s1-sz-2*border)
      c = border + np.random.randint(s2-sz-2*border)

      I[:,i] = np.reshape(IMAGES[imi][r:r+sz, c:c+sz], sz*sz, 1)

    # Coefficient Inference
    if inference == 'lca':
      ahat = sparsify(I, Phi, lambdav)
    elif inference == 'fista':
      ahat = fista.fista(I, Phi, lambdav, max_iterations=50)
    else:
      print("Invalid inference option")
      return

    # Calculate Residual Error
    R = I-np.dot(Phi, ahat)

    # Update Basis Functions
    dPhi = eta * (np.dot(R, ahat.T))
    Phi = Phi + dPhi
    Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis=0))))

    # Plot every 100 iterations
    if np.mod(t,100) == 0:
      print("Iteration " + str(t))
      side = int(np.sqrt(neurons))
      image = np.zeros((sz*side+side,sz*side+side))
      for i in range(side):
        for j in range(side):
          patch = np.reshape(Phi[:,i*side+j],(sz,sz))
          patch = patch/np.max(np.abs(patch))
          image[i*sz+i:i*sz+sz+i,j*sz+j:j*sz+sz+j] = patch

      if t == 0:
        plt.ion()
        fig = plt.figure()
        im = plt.imshow(image, cmap=cm.Greys_r, interpolation="nearest")
      else:
        if plt.fignum_exists(fig.number):
          im.set_data(image)
        else:
          print('Figure closed. Exiting')
          return None
      plt.pause(1e-6)
      #  plt.draw()

  return Phi


def sparsify(I, Phi, lambdav, eta=0.1, num_iterations=125):
  """
  LCA Inference.
  I: Image batch (dim x batch)
  Phi: Dictionary (dim x dictionary element)
  lambdav: Sparsity coefficient
  eta: Update rate
  """
  batch_size = np.shape(I)[1]

  (N, M) = np.shape(Phi)
  #  sz = np.sqrt(N)

  b = np.dot(Phi.T, I)
  G = np.dot(Phi.T, Phi) - np.eye(M)

  u = np.zeros((M,batch_size))

  l = 0.5 * np.max(np.abs(b), axis=0)
  a = g(u, l)

  for t in range(num_iterations):
    u = eta * (b - np.dot(G, a)) + (1 - eta) * u
    a = g(u, l)

    l = 0.95 * l
    l[l < lambdav] = lambdav

  return a


# g:  Hard threshold. L0 approximation
def g(u,theta):
  """
  LCA threshold function
  u: coefficients
  theta: threshold value
  """
  a = u
  a[np.abs(a) < theta] = 0
  return a


def preprocess(imgs, filt, lambdav=0.1):
  """ Applies the necessary preprocessing described in the Olshausen Field paper
  to turn natural images into ones acceptable by the algorithm

  Inputs:
    imgs - list(PIL.Image) : Raw images read from disk in PIL format.
    filt - ndarray : Filter parametrized in the Fourier domain.

  Returns:
    list(ndarray) : list of numpy array of images after preprocessing done.
  """
  # The size of the fft
  N = filt.shape[0]
  # The max size of the image. Will be half the size of the FFT filter so that
  # the periodization doesn't affect things.
  M = N // 2
  # Copy the list
  imgs = [i for i in imgs]

  for i in range(len(imgs)):
    im = imgs[i]
    x,y = im.size
    if x > y:
      n = (M, int(M*y/x))
    else:
      n = (int(M*x/y), M)
    im.resize(n, Image.BILINEAR)

    If = fft2(im, (N,N))
    imagew = ifft2(If * filt).real[:y,:x]
    imgs[i] = np.sqrt(lambdav) * imagew / np.std(imagew)

  return imgs


def get_filter():
  """ Gets the denoising/whitening filter used by Olshausen & Field. Returns the
  fourier representation of it.

  Sets the cutoff frequency to be 0.8π
  """
  # Ensure that this large enough to handle all images
  N = 1024
  f0 = 0.4*N
  fx, fy = np.meshgrid(np.arange(-N//2, N//2), np.arange(-N//2, N//2))
  r = np.sqrt(fx**2 + fy**2)

  # Calculate the lowpass filter (for noise removal)
  lpf = np.exp(-(r/f0)**4)

  # Calculate the whitener
  whitener = r

  # Filter
  filt = fftshift(whitener * lpf)
  return filt


if __name__ == '__main__':
  args = parser.parse_args()
  if args.imgs is not None:
    # Check that the folder exists
    if not os.path.isdir(args.imgs):
      print('Provided path not found')
      exit()

    # Check that the suffixes have a dot at the beginning
    args.suffix = [s if s.beginswith('.') else '.'+s for s in args.suffix]

    # Load in all the images in the given path
    imgs = os.listdir(args.imgs)
    imgs = [os.path.join(args.imgs, i) for i in imgs
            if os.path.splitext(i)[1].lower() in args.suffix]
    if len(imgs) == 0:
      print('Could not find any images in the folder which matched '
            'the provided suffixes')
      exit()
    imgs = [Image.open(i).convert('L') for i in imgs]

    filt = get_filter()
    IMAGES = preprocess(imgs, filt, args.lambdav)

  else:
    IMAGES = scipy.io.loadmat('./IMAGES.mat')
    IMAGES = IMAGES['IMAGES']
    IMAGES = np.transpose(IMAGES, (2,0,1))

  sparsenet(IMAGES, patch_dim=args.size, neurons=128, lambdav=args.lambdav,
            eta=6.0, num_trials=3000, batch_size=100, border=4,
            inference=args.inference)
  input('Press any key to exit')
