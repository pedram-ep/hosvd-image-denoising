def generate_spiral_offsets(radius):
    offsets = []
    for r in range(1, radius + 1):
        ring = []
        for dy in range(-r, r + 1):
            ring.append((-r, dy))
            ring.append((r, dy))
        for dx in range(-r + 1, r):
            ring.append((dx, -r))
            ring.append((dx, r))
        offsets.append(ring)
    return offsets