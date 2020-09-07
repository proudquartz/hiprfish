import numpy as np
cimport numpy as np
from cython.parallel import prange
DTYPE = np.double
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def neighbor_average(float [:,:,:] image_padded, int patch_size):
  dim_x = image_padded.shape[0]
  dim_y = image_padded.shape[1]
  dim_z = image_padded.shape[2]
  cdef int i
  cdef int j
  cdef int k
  cdef int s
  cdef int theta
  cdef int phi
  cdef float average_value
  cdef int patch_i
  cdef int patch_j
  cdef int patch_k
  dim_x_unpad = dim_x - 2*(patch_size-1)
  dim_y_unpad = dim_y - 2*(patch_size-1)
  dim_z_unpad = dim_z - 2*(patch_size-1)
  nei_avg_np = np.zeros((dim_x_unpad, dim_y_unpad, dim_z_unpad, patch_size - 1), dtype = DTYPE)
  cdef float [:,:,:,:] nei_avg = nei_avg_np
  for i in range(dim_x_unpad):
      for j in range(dim_y_unpad):
          for k in range(dim_z_unpad):
              for s in range(1,patch_size):
                average_value = 0
                for patch_i in range(i+10-s, i+11+s):
                  for patch_j in range(j+10-s, j+11+s):
                    for patch_k in range(k+10-s, k+11+s):
                      average_value += image_padded[patch_i, patch_j, patch_k]
                nei_avg[i,j,k,s-1] = average_value/((2*s+1)**3)
  return(nei_avg_np)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def line_profile(np.ndarray[float, ndim = 3] image_padded, int patch_size, int theta_range, int phi_range):
  dim_x = image_padded.shape[0]
  dim_y = image_padded.shape[1]
  dim_z = image_padded.shape[2]
  cdef int i, j, k, t, angle_index, line_diff
  cdef int li, vli, vlj, vlk
  cdef int increment = int((patch_size-1)/2)
  cdef int x_interval, y_interval, z_interval
  cdef np.ndarray intervals = np.zeros(3, dtype = int)
  cdef np.ndarray line_matrices_np = np.zeros((patch_size, 3, (theta_range - 1)*phi_range), dtype = int)
  cdef long [:,:,:] line_matrices = line_matrices_np
  cdef int patch_i, patch_j, patch_k
  cdef int dim_x_unpad = dim_x - 2*(patch_size-1)
  cdef int dim_y_unpad = dim_y - 2*(patch_size-1)
  cdef int dim_z_unpad = dim_z - 2*(patch_size-1)
  lp_np = np.zeros((dim_x_unpad, dim_y_unpad, dim_z_unpad, (theta_range - 1)*phi_range), dtype = DTYPE)
  voxel_line_profile_np = np.zeros(patch_size)
  lrel_np = np.zeros(patch_size)
  lnorm_np = np.zeros(patch_size)
  cdef double [:,:,:,:] lp = lp_np
  cdef double [:] voxel_line_profile = voxel_line_profile_np
  cdef double lmin
  cdef double [:] lrel = lrel_np
  cdef double [:] lnorm = lnorm_np
  for theta in range(1, theta_range):
        for phi in range(phi_range):
            angle_index = (theta - 1)*phi_range + phi
            intervals[0] = int(np.ceil(increment*np.cos(phi*np.pi/9)*np.sin(theta*np.pi/9)))
            intervals[1] = int(np.ceil(increment*np.sin(phi*np.pi/9)*np.sin(theta*np.pi/9)))
            intervals[2] = int(np.ceil(increment*np.cos(theta*np.pi/9)))
            max_interval = intervals[np.argmax(intervals)]
            line_n = 2*max_interval+1
            if line_n < patch_size:
              line_diff = int((patch_size - line_n)/2)
              for li in range(line_diff):
                line_matrices[li, 0, angle_index] = int(np.round(line_diff*(2*intervals[0]+1)/line_n) + increment -  intervals[0])
                line_matrices[li, 1, angle_index] = int(np.round(line_diff*(2*intervals[1]+1)/line_n) + increment -  intervals[1])
                line_matrices[li, 2, angle_index] = int(np.round(line_diff*(2*intervals[2]+1)/line_n) + increment -  intervals[2])
              for li in range(line_n):
                line_matrices[li+line_diff, 0, angle_index] = int(np.round(li*(2*intervals[0]+1)/line_n) + increment -  intervals[0])
                line_matrices[li+line_diff, 1, angle_index] = int(np.round(li*(2*intervals[1]+1)/line_n) + increment -  intervals[1])
                line_matrices[li+line_diff, 2, angle_index] = int(np.round(li*(2*intervals[2]+1)/line_n) + increment -  intervals[2])
              for li in range(line_diff):
                line_matrices[li+line_n, 0, angle_index] = int(np.round(line_n*(2*intervals[0]+1)/line_n) + increment -  intervals[0])
                line_matrices[li+line_n, 1, angle_index] = int(np.round(line_n*(2*intervals[1]+1)/line_n) + increment -  intervals[1])
                line_matrices[li+line_n, 2, angle_index] = int(np.round(line_n*(2*intervals[2]+1)/line_n) + increment -  intervals[2])
            else:
              for li in range(line_n):
                line_matrices[li, 0, angle_index] = int(np.round(li*(2*intervals[0]+1)/line_n) + increment -  intervals[0])
                line_matrices[li, 1, angle_index] = int(np.round(li*(2*intervals[1]+1)/line_n) + increment -  intervals[1])
                line_matrices[li, 2, angle_index] = int(np.round(li*(2*intervals[2]+1)/line_n) + increment -  intervals[2])
  for i in range(dim_x_unpad):
      for j in range(dim_y_unpad):
          for k in range(dim_z_unpad):
              for t in range((theta_range-1)*phi_range):
                for li in range(patch_size):
                  vli = line_matrices[li,0,t]
                  vlj = line_matrices[li,1,t]
                  vlk = line_matrices[li,2,t]
                  print(li, t, vli, vlj, vlk)
                  voxel_line_profile[li] = image_padded[vli, vlj, vlk]
                lmin = min(voxel_line_profile)
                for li in range(patch_size):
                  lrel[li] = voxel_line_profile[li] - lmin
                lmax = max(lrel)
                for li in range(patch_size):
                  lnorm[li] = lrel[li]/lmax
                lp[i,j,k,t] = lnorm[increment]
  return(lp_np)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def line_profile_v2(double [:,:,:] image_padded, int patch_size, int theta_range, int phi_range):
  dim_x = image_padded.shape[0]
  dim_y = image_padded.shape[1]
  dim_z = image_padded.shape[2]
  cdef int i, j, k, t, angle_index, line_n
  cdef int li, vli, vlj, vlk
  cdef int increment = int((patch_size-1)/2)
  cdef int x_interval, y_interval, z_interval
  cdef np.ndarray intervals = np.zeros(3, dtype = int)
  cdef np.ndarray line_matrices_np = np.zeros((patch_size, 3, (theta_range - 1)*phi_range), dtype = int)
  cdef long [:,:,:] line_matrices = line_matrices_np
  cdef int patch_i, patch_j, patch_k
  cdef int dim_x_unpad = dim_x - (patch_size-1)
  cdef int dim_y_unpad = dim_y - (patch_size-1)
  cdef int dim_z_unpad = dim_z - (patch_size-1)
  lp_np = np.zeros((dim_x_unpad, dim_y_unpad, dim_z_unpad, (theta_range - 1)*phi_range, patch_size), dtype = DTYPE)
  voxel_line_profile_np = np.zeros(patch_size)
  lrel_np = np.zeros(patch_size)
  lnorm_np = np.zeros(patch_size)
  image_patch_np = np.zeros((patch_size,patch_size,patch_size), dtype = np.float64)
  cdef double [:,:,:,:,:] lp = lp_np
  cdef double [:] voxel_line_profile = voxel_line_profile_np
  cdef double lmin
  cdef double [:] lrel = lrel_np
  cdef double [:] lnorm = lnorm_np
  cdef double [:,:,:] image_patch = image_patch_np
  for theta in range(1, theta_range):
    for phi in range(phi_range):
      angle_index = (theta - 1)*phi_range + phi
      intervals[0] = int(np.round(increment*np.cos(phi*np.pi/phi_range)*np.sin(theta*np.pi/theta_range)))
      intervals[1] = int(np.round(increment*np.sin(phi*np.pi/phi_range)*np.sin(theta*np.pi/theta_range)))
      intervals[2] = int(np.round(increment*np.cos(theta*np.pi/theta_range)))
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
          h3 = interval_signs[2]*li*(2*np.abs(intervals[2])+1)/line_n
          line_matrices[li+line_diff, 2, angle_index] = int(np.sign(h3)*np.floor(np.abs(h3)) + increment -  intervals[2])
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
          h3 = interval_signs[2]*li*(2*np.abs(intervals[2])+1)/line_n
          line_matrices[li, 2, angle_index] = int(np.sign(h3)*np.floor(np.abs(h3)) + increment -  intervals[2])
  for i in range(dim_x_unpad):
    for j in range(dim_y_unpad):
      for k in range(dim_z_unpad):
        image_patch = image_padded[i:i+patch_size,j:j+patch_size,k:k+patch_size]
        for t in range((theta_range-1)*phi_range):
          for li in range(patch_size):
            vli = line_matrices[li,0,t]
            vlj = line_matrices[li,1,t]
            vlk = line_matrices[li,2,t]
            lp[i,j,k,t,li] = image_patch[vli, vlj, vlk]
  return(lp_np)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def line_profile_memory_efficient_v2(double [:,:,:] image_padded, int patch_size, int theta_range, int phi_range):
  dim_x = image_padded.shape[0]
  dim_y = image_padded.shape[1]
  dim_z = image_padded.shape[2]
  cdef int i, j, k, t, angle_index
  cdef int li, vli, vlj, vlk
  cdef int increment = int((patch_size-1)/2)
  cdef int x_interval, y_interval, z_interval
  cdef np.ndarray intervals = np.zeros(3, dtype = int)
  cdef np.ndarray line_matrices_np = np.zeros((patch_size, 3, (theta_range - 1)*phi_range), dtype = int)
  cdef long [:,:,:] line_matrices = line_matrices_np
  cdef int patch_i, patch_j, patch_k
  cdef int dim_x_unpad = dim_x - (patch_size-1)
  cdef int dim_y_unpad = dim_y - (patch_size-1)
  cdef int dim_z_unpad = dim_z - (patch_size-1)
  lp_np = np.zeros(((theta_range - 1)*phi_range, patch_size), dtype = DTYPE)
  image_enhanced_np = np.zeros((dim_x_unpad, dim_y_unpad, dim_z_unpad, (theta_range - 1)*phi_range), dtype = DTYPE)
  voxel_line_profile_np = np.zeros(patch_size)
  lrel_np = np.zeros(patch_size)
  lnorm_np = np.zeros(patch_size)
  image_patch_np = np.zeros((patch_size,patch_size,patch_size), dtype = np.double)
  lp_norm_np = np.zeros(((theta_range - 1)*phi_range,patch_size), dtype = np.double)
  cdef double [:,:] lp = lp_np
  cdef double [:,:] lp_norm = lp_norm_np
  cdef double [:,:,:,:] image_enhanced = image_enhanced_np
  cdef double [:] voxel_line_profile = voxel_line_profile_np
  cdef double lmin, lp_min, lp_max, lp_range
  cdef double [:] lrel = lrel_np
  cdef double [:] lnorm = lnorm_np
  cdef double [:,:,:] image_patch = image_patch_np
  for theta in range(1, theta_range):
    for phi in range(phi_range):
      angle_index = (theta - 1)*phi_range + phi
      intervals[0] = int(np.round(increment*np.cos(phi*np.pi/phi_range)*np.sin(theta*np.pi/theta_range)))
      intervals[1] = int(np.round(increment*np.sin(phi*np.pi/phi_range)*np.sin(theta*np.pi/theta_range)))
      intervals[2] = int(np.round(increment*np.cos(theta*np.pi/theta_range)))
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
          h3 = interval_signs[2]*li*(2*np.abs(intervals[2])+1)/line_n
          line_matrices[li+line_diff, 2, angle_index] = int(np.sign(h3)*np.floor(np.abs(h3)) + increment -  intervals[2])
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
          h3 = interval_signs[2]*li*(2*np.abs(intervals[2])+1)/line_n
          line_matrices[li, 2, angle_index] = int(np.sign(h3)*np.floor(np.abs(h3)) + increment -  intervals[2])
  for i in range(dim_x_unpad):
    for j in range(dim_y_unpad):
      for k in range(dim_z_unpad):
        image_patch = image_padded[i:i+patch_size,j:j+patch_size,k:k+patch_size]
        for t in range((theta_range-1)*phi_range):
          for li in range(patch_size):
            vli = line_matrices[li,0,t]
            vlj = line_matrices[li,1,t]
            vlk = line_matrices[li,2,t]
            lp[t,li] = image_patch[vli, vlj, vlk]
          lp_min = min(lp[t,:])
          lp_max = max(lp[t,:])
          lp_range = lp_max - lp_min
          lp_range = max(lp_range, 1e-8)
          for li in range(patch_size):
            lp_norm[t,li] = (lp[t,li] - lp_min)/lp_range
          image_enhanced[i,j,k,t] = lp_norm[t,increment]
  return(image_enhanced_np)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def line_profile_memory_efficient_v3(double [:,:,:] image_padded, int patch_size, int theta_range, int phi_range):
  dim_x = image_padded.shape[0]
  dim_y = image_padded.shape[1]
  dim_z = image_padded.shape[2]
  cdef int i, j, k, t, angle_index
  cdef int li, vli, vlj, vlk
  cdef int increment = int((patch_size-1)/2)
  cdef int x_interval, y_interval, z_interval
  cdef double pixel_avg, pixel_lq, pixel_uq
  cdef np.ndarray intervals = np.zeros(3, dtype = int)
  cdef np.ndarray line_matrices_np = np.zeros((patch_size, 3, (theta_range - 1)*phi_range), dtype = int)
  cdef long [:,:,:] line_matrices = line_matrices_np
  cdef int patch_i, patch_j, patch_k
  cdef int dim_x_unpad = dim_x - (patch_size-1)
  cdef int dim_y_unpad = dim_y - (patch_size-1)
  cdef int dim_z_unpad = dim_z - (patch_size-1)
  lp_np = np.zeros(((theta_range - 1)*phi_range, patch_size), dtype = DTYPE)
  image_enhanced_np = np.zeros((dim_x_unpad, dim_y_unpad, dim_z_unpad), dtype = DTYPE)
  image_enhanced_t_np = np.zeros((theta_range - 1)*phi_range, dtype = DTYPE)
  voxel_line_profile_np = np.zeros(patch_size)
  lrel_np = np.zeros(patch_size)
  lnorm_np = np.zeros(patch_size)
  image_patch_np = np.zeros((patch_size,patch_size,patch_size), dtype = np.double)
  lp_norm_np = np.zeros(((theta_range - 1)*phi_range,patch_size), dtype = np.double)
  cdef double [:,:] lp = lp_np
  cdef double [:,:] lp_norm = lp_norm_np
  cdef double [:,:,:] image_enhanced = image_enhanced_np
  cdef double [:] image_enhanced_t = image_enhanced_t_np
  cdef double [:] voxel_line_profile = voxel_line_profile_np
  cdef double lmin, lp_min, lp_max, lp_range
  cdef double [:] lrel = lrel_np
  cdef double [:] lnorm = lnorm_np
  cdef double [:,:,:] image_patch = image_patch_np
  for theta in range(1, theta_range):
    for phi in range(phi_range):
      angle_index = (theta - 1)*phi_range + phi
      intervals[0] = int(np.round(increment*np.cos(phi*np.pi/phi_range)*np.sin(theta*np.pi/theta_range)))
      intervals[1] = int(np.round(increment*np.sin(phi*np.pi/phi_range)*np.sin(theta*np.pi/theta_range)))
      intervals[2] = int(np.round(increment*np.cos(theta*np.pi/theta_range)))
      max_interval = intervals[np.argmax(np.abs(intervals))]
      interval_signs = np.sign(intervals)
      line_n = int(2*np.abs(max_interval)+1)
      if line_n < patch_size:
        line_diff = int((patch_size - line_n)/2)
        for li in range(line_n):
          line_matrices[li+line_diff, 0, angle_index] = int(np.round(interval_signs[0]*li*(2*np.abs(intervals[0])+1)/line_n) + increment -  intervals[0])
          line_matrices[li+line_diff, 1, angle_index] = int(np.round(interval_signs[1]*li*(2*np.abs(intervals[1])+1)/line_n) + increment -  intervals[1])
          line_matrices[li+line_diff, 2, angle_index] = int(np.round(interval_signs[2]*li*(2*np.abs(intervals[2])+1)/line_n) + increment -  intervals[2])
        for li in range(line_diff):
          line_matrices[li, :, angle_index] = line_matrices[line_diff, :, angle_index]
        for li in range(line_diff):
          line_matrices[li+line_n+line_diff, :, angle_index] = line_matrices[line_n + line_diff - 1, :, angle_index]
      else:
        for li in range(line_n):
          line_matrices[li, 0, angle_index] = int(np.floor(interval_signs[0]*li*(2*intervals[0]+1)/line_n) + increment -  intervals[0])
          line_matrices[li, 1, angle_index] = int(np.floor(interval_signs[1]*li*(2*intervals[1]+1)/line_n) + increment -  intervals[1])
          line_matrices[li, 2, angle_index] = int(np.floor(interval_signs[2]*li*(2*intervals[2]+1)/line_n) + increment -  intervals[2])
  for i in range(dim_x_unpad):
    for j in range(dim_y_unpad):
      for k in range(dim_z_unpad):
        image_patch = image_padded[i:i+patch_size,j:j+patch_size,k:k+patch_size]
        for t in range((theta_range-1)*phi_range):
          for li in range(patch_size):
            vli = line_matrices[li,0,t]
            vlj = line_matrices[li,1,t]
            vlk = line_matrices[li,2,t]
            lp[t,li] = image_patch[vli, vlj, vlk]
          lp_min = min(lp[t,:])
          lp_max = max(lp[t,:])
          lp_range = lp_max - lp_min
          lp_range = max(lp_range, 1e-8)
          for li in range(patch_size):
            lp_norm[t,li] = (lp[t,li] - lp_min)/lp_range
          image_enhanced_t[t] = lp_norm[t,increment]
        pixel_avg = 0
        for t in range((theta_range-1)*phi_range):
          pixel_avg += image_enhanced_t[t]
        pixel_avg /= (theta_range-1)*phi_range
        pixel_uq = np.percentile(image_enhanced_t, 25)
        pixel_lq = np.percentile(image_enhanced_t, 75)
        image_enhanced[i,j,k] = pixel_avg*(pixel_uq - pixel_lq)/(pixel_uq + pixel_lq + 1e-8)
  return(image_enhanced_np)
