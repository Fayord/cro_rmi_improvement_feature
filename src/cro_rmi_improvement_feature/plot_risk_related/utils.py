import random


def find_equal_count_boundaries(numbers, n_sections):
    """
    Given a list of numbers, find the boundary values that split the list into n_sections
    such that each section has (as close as possible) the same number of elements.
    Returns a list of boundary values (length n_sections+1).
    """
    assert n_sections > 0, "n_sections must be greater than 0"
    if not numbers or n_sections < 1:
        return []

    sorted_numbers = sorted(numbers)
    total = len(sorted_numbers)
    boundaries = [sorted_numbers[0]]
    for i in range(1, n_sections):
        idx = int(round(i * total / n_sections))
        # Clamp index to valid range
        idx = min(idx, total - 1)
        boundaries.append(sorted_numbers[idx])
    boundaries.append(sorted_numbers[-1])
    # -1 for min and +1 for max
    # boundaries[0] -= 1
    boundaries[-1] += 1
    return boundaries


if __name__ == "__main__":
    numbers = random.sample(range(1, 201), 100)  # 100 unique numbers from 1 to 100
    numbers.sort()
    print(numbers)
    print("numbers", len(numbers))
    boundaries = find_equal_count_boundaries(numbers, 4)
    # display member of each section
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        section = [num for num in numbers if start <= num < end]

        print(f"Section  {i + 1}: {len(section)} {section}")
    print(boundaries)
