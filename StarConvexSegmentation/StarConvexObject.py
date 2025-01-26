class StarConvexObject:
    def __init__(self, center_x, center_y, vertex_distances, objectness_score, class_label):
        self.center_x = center_x
        self.center_y = center_y
        self.vertex_distances = vertex_distances
        self.objectness_score = objectness_score
        self.class_label = class_label