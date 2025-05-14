import random


def find_equal_count_boundaries(numbers: list[float], n_sections: int):
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


def find_proportional_count_boundaries(
    numbers: list[float], proportion_member_list: list[int]
):
    """
    Given a list of numbers and a list of proportions (as percentages summing to 100),
    find the boundary values that split the list into sections according to those proportions.
    Returns a list of boundary values (length len(proportion_member_list)+1).
    """
    assert sum(proportion_member_list) == 100, "Sum of proportions must be 100"
    if not numbers or not proportion_member_list:
        return []

    sorted_numbers = sorted(numbers)
    total = len(sorted_numbers)
    boundaries = [sorted_numbers[0]]
    cumulative = 0
    for proportion in proportion_member_list[:-1]:
        cumulative += proportion
        idx = int(round(cumulative * total / 100))
        idx = min(idx, total - 1)
        boundaries.append(sorted_numbers[idx])
    boundaries.append(sorted_numbers[-1])
    boundaries[-1] += 1
    return boundaries


def get_level_from_boundaries(boundaries: list[int], test_number: float):
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if start <= test_number < end:
            return i + 1
    raise ValueError(f"Number {test_number} is outside the range of boundaries.")


if __name__ == "__main__":
    numbers = random.sample(range(1, 201), 100)  # 100 unique numbers from 1 to 100
    numbers.sort()
    print(numbers)
    print("numbers", len(numbers))
    boundaries = find_proportional_count_boundaries(numbers, [40, 40, 10, 10])

    # display member of each section
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        section = [num for num in numbers if start <= num < end]

        print(f"Section  {i + 1}: {len(section)} {section}")
    test_number = 170
    level = get_level_from_boundaries(boundaries, test_number)
    print(level)
    print(boundaries)
