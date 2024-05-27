

class Segments:
    def __init__(self,segments):
        self.segments = segments
        self.N = len(segments)
        self.segment_lengths = [len(s) for s in segments]
        # self.segment_indices = np.cumsum(self.segment_lengths)
        
    def get_segment(self,idx):
        return self.segments[idx]
    
    def get_segment_indices(self,idx):
        return self.segment_indices[idx]
    
    def get_segment_length(self,idx):
        return self.segment_lengths