def get_severity(conf):
    if conf > 0.8:
        return "high"
    if conf > 0.6:
        return "medium"
    return "low"


def calc_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)