import Queue

def bfs(img, int rows, int cols):
  # q = Queue.Queue()
  q = []
  enqueued = {}
  tags = [[]]
  
  cdef int tag_count = 0
  cdef int x, y, child_x, child_y, i, j, color
  cdef float b_shade, n_shade

  for i in range(rows):
    for j in range(cols):
      first_pix  = (i, j)
      if not first_pix in enqueued:
        tags[tag_count] = []
        q.append(first_pix)
        enqueued[first_pix] = True

        while len(q) > 0:  
          [y,x] = q.pop(0)
          color = img[y, x]
          b_shade = compute_b_shade(color, 255.001)

          for pix in [(y,x-1), (y,x+1), (y-1,x), (y+1,x)]:
            child_y = pix[0]
            child_x = pix[1]
            if child_y >= 0 and child_y < rows and child_x >= 0 and child_x < cols:
              n_shade = compute_b_shade(img[child_y, child_x], 255.001)
              if are_neighbors(n_shade, b_shade):
                if not pix in enqueued:
                  q.append(pix)
                  enqueued[pix] = True
          tags[tag_count].append((y, x, color))
        tag_count += 1
        tags.append([])
  return tags

cdef float compute_b_shade(int color, float offset):
  return color * (-1) + offset

cdef float compute_n_shade(int color, float offset):
  return color * (-1) + offset

cdef int are_neighbors(float n_shade, float b_shade):
  if (n_shade/b_shade) < 3 and (n_shade/b_shade) > 0.33:
    return 1
  return 0