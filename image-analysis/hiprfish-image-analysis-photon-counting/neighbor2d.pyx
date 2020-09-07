import numpy as np
cimport numpy as np
DTYPE = np.double
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def line_profile_2d_v2(double [:,:] image_padded, int patch_size, int phi_range):
  dim_x = image_padded.shape[0]
  dim_y = image_padded.shape[1]
  cdef int i, j, t
  cdef int li, vli, vlj
  cdef int increment = int((patch_size-1)/2)
  cdef int x_interval, y_interval
  cdef np.ndarray intervals = np.zeros(2, dtype = int)
  cdef np.ndarray line_matrices_np = np.zeros((patch_size, 2, phi_range), dtype = int)
  cdef long [:,:, :] line_matrices = line_matrices_np
  cdef int patch_i, patch_j
  cdef int dim_x_unpad = dim_x - (patch_size-1)
  cdef int dim_y_unpad = dim_y - (patch_size-1)
  lp_np = np.zeros((dim_x_unpad, dim_y_unpad, phi_range, patch_size), dtype = DTYPE)
  voxel_line_profile_np = np.zeros(patch_size)
  lrel_np = np.zeros(patch_size)
  lnorm_np = np.zeros(patch_size)
  image_patch_np = np.zeros((patch_size,patch_size), dtype = np.float64)
  cdef double [:,:,:,:] lp = lp_np
  cdef double [:] voxel_line_profile = voxel_line_profile_np
  cdef double lmin
  cdef double [:] lrel = lrel_np
  cdef double [:] lnorm = lnorm_np
  cdef double [:,:] image_patch = image_patch_np
  for phi in range(phi_range):
      angle_index = phi
      intervals[0] = int(np.round(increment*np.cos(phi*np.pi/phi_range)))
      intervals[1] = int(np.round(increment*np.sin(phi*np.pi/phi_range)))
      max_interval = intervals[np.argmax(np.abs(intervals))]
      interval_signs = np.sign(intervals)
      line_n = int(2*np.abs(max_interval)+1)
      if line_n < patch_size:
        line_diff = int((patch_size - line_n)/2)
        for li in range(line_n):
          h1 = interval_signs[0]*li*(2*np.abs(intervals[0])+1)/line_n
          line_matrices[li+line_diff, 0, angle_index] = int(np.sign(h1)*np.floor(np.abs(h1)) + increment -  intervals[0])
          h2 = interval_signs[1]*li*(2*np.abs(intervals[1])+1)/line_n
          line_matrices[li+line_diff, 1, angle_index] = int(np.sign(h2)*np.floor(np.abs(h2)) + increment -  intervals[1])
        for li in range(line_diff):
          line_matrices[li, :, angle_index] = line_matrices[line_diff, :, angle_index]
        for li in range(line_diff):
          line_matrices[li+line_n+line_diff, :, angle_index] = line_matrices[line_n + line_diff - 1, :, angle_index]
      else:
        for li in range(line_n):
          h1 = interval_signs[0]*li*(2*np.abs(intervals[0])+1)/line_n
          line_matrices[li, 0, angle_index] = int(np.sign(h1)*np.floor(np.abs(h1)) + increment -  intervals[0])
          h2 = interval_signs[1]*li*(2*np.abs(intervals[1])+1)/line_n
          line_matrices[li, 1, angle_index] = int(np.sign(h2)*np.floor(np.abs(h2)) + increment -  intervals[1])
  for i in range(dim_x_unpad):
    for j in range(dim_y_unpad):
      image_patch = image_padded[i:i+patch_size,j:j+patch_size]
      for t in range(phi_range):
        for li in range(patch_size):
          vli = line_matrices[li,0,t]
          vlj = line_matrices[li,1,t]
          lp[i,j,t,li] = image_patch[vli, vlj]
  return(lp_np)
