import numpy as np

class frame(object):
    def __init__(self, num_params = 5):
        self.params = [0] * num_params
    def accum_params(self, params1, attr = 'params'):
        " accumulate params with given params "
        self.__dict__[attr] = [p + p1 for p, p1 in zip(self.__dict__[attr], params1)]
        return self
class P(frame):
    def __init__(self, y, num_params = 5, x_start = 0, sign=-1):
        self.sign = sign
        self.boundaries = [x_start]
        self.y = y
        self.params = [0] * num_params
    def terminate(self, end_bound):
        self.boundaries += [end_bound, self.y]
        del self.y
        return self
    def localize(self, x0, y0):
        self.boundaries = [b - x0 for b in self.boundaries[:2]] + [b - y0 for b in self.boundaries[2:]]
        return self
    def L(self):
        return self.params[0]
    def x_start(self):
        return self.boundaries[0]
    def x_end(self):
        return self.boundaries[1]
class segment(P):
    extend_func = (min, max, min, max)
    def __init__(self, P, fork_ = []):
        self.sign = P.sign
        self.boundaries = list(P.boundaries)    # making sure not to copy reference
        self.params = list(P.params)
        self.orientation_params = [0, 0]        # xD and abs_xD
        self.ave_x = (P.L() - 1) // 2
        self.Py_ = [(P, 0)]
        self.roots = 0
        self.fork_ = fork_
    def accum_P(self, P):
        " merge new P into segment "
        new_ave_x = (P.L() - 1) // 2
        xd = new_ave_x - self.ave_x
        self.ave_x = new_ave_x
        self.accum_params([xd, abs(xd)], 'orientation_params')
        self.accum_params(P.params).extend_boundaries(P.boundaries[:2])   # extend x-boundaries
        self.Py_.append((P, xd))
        self.roots = 0

        return self
    def extend_boundaries(self, bounds):
        " extend boundaries with given boundaries "
        for i in range(len(bounds)):
            self.boundaries[i] = self.extend_func[i](self.boundaries[i], bounds[i])
        return self
    def terminate(self, end_bound):
        self.boundaries.append(end_bound)
        return self
    def y_start(self):
        return self.boundaries[2]
    def y_end(self):
        return self.boundaries[3]
class blob(segment):
    def __init__(self, segment):
        self.sign = segment.sign
        self.boundaries = list(segment.boundaries)  # making sure not to copy reference
        self.params = [0] * len(segment.params)
        self.orientation_params = [0, 0, 0]         # xD, abs_xD, Ly
        self.segment_ = []
        self.accum_segment(segment)
        self.open_segments = 1
    def accum_segment(self, segment):
        self.segment_.append(segment)
        segment.blob = self
        return self
    def term_segment(self, segment, y):
        self.accum_params(segment.params).extend_boundaries(segment.boundaries[:2])
        self.accum_params(segment.orientation_params + [len(segment.Py_)], 'orientation_params')
        segment.terminate(y)
        self.open_segments += segment.roots - 1
    def merge(self, other):
        if not other is self:
            self.accum_params(other.params).extend_boundaries(other.boundaries)
            self.accum_params(other.orientation_params, 'orientation_params')
            self.open_segments += other.open_segments
            for seg in other.segment_:
                self.accum_segment(seg)
        self.open_segments -= 1
        return self
    def localize(self, frame, copy_dert_map_ = 1):
        " get a slice of global dert_map, localize coordinate system "

        x_0, x_n, y_0, y_n = self.boundaries    # x_start, x_end, y_start, y_end in short

        if copy_dert_map_:  # receive a slice of global map:
            self.dert_map_ = []
            for global_dert_map in frame.dert_map_:
                self.dert_map_.append(global_dert_map[y_0:y_n, x_0:x_n])

        # localize inner structures:
        blob_map = np.zeros((y_n - y_0, x_n - x_0), dtype=bool)
        for seg in self.segment_:
            seg.localize(x_0, y_0)
            for P, xd in seg.Py_:
                P.localize(x_0, y_0)
                x_start, x_end, y = P.boundaries
                blob_map[y, x_start:x_end] = True

        self.blob_map = blob_map
        return self