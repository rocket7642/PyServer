import pygame
import csv
import sys

# -----------------------------
# Configuration
# -----------------------------
CELL_SIZE = 2
CSV_FILE = "grid.csv"

MIN_VAL = -1
MAX_VAL = 40

# -----------------------------
# CSV Loader
# -----------------------------
def load_csv(filename):
    grid = []
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            grid.append([float(cell) for cell in row])
    return grid

# -----------------------------
# Value → Color Mapping
# -----------------------------
def value_to_color(value):
    # Clamp value
    value = max(MIN_VAL, min(MAX_VAL, value))

    # Normalize to [-1, 1]
    t = (value - MIN_VAL) / (MAX_VAL - MIN_VAL)
    t = t * 2 - 1

    if t <= 0:
        # Red → Green
        # t in [-1, 0]
        factor = t + 1  # [0, 1]
        r = int(255 * (1 - factor))
        g = int(255 * factor)
        b = 0
    else:
        # Green → Blue
        # t in [0, 1]
        factor = t
        r = 0
        g = int(255 * (1 - factor))
        b = int(255 * factor)

    return (r, g, b)

# -----------------------------
# Main
# -----------------------------
def main():
    grid = load_csv(CSV_FILE)
    rows = len(grid)
    cols = len(grid[0])

    pygame.init()
    screen = pygame.display.set_mode(
        (cols * CELL_SIZE, rows * CELL_SIZE)
    )
    pygame.display.set_caption("CSV Value → Color Grid")

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))

        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                color = value_to_color(value)
                rect = pygame.Rect(
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
                pygame.draw.rect(screen, color, rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()