import numpy as np
from typing import Iterable, Set, Tuple, List
from itertools import combinations
from math import inf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Type definitions
Rect1D = Tuple[int, int, int]  # (start_time, end_time, bandwidth)
Rect2D = Tuple[int, int, int, int]  # (x1, x2, y1, y2)
Slots1D = Iterable[Rect1D]
Slots2D = Set[Rect2D]

# Tracking Bandwidth Usage
current_bandwidth_usage = {}  # {time_point: current_max_bandwidth_used}

# All the Todos are the things need to be updated, prob will add some while I write the document
def get_current_max_bandwidth(x1, x2):
    """Get the current maximum bandwidth used in the time range [x1, x2]"""
    max_used = 0
    for t in range(x1, x2):
        max_used = max(max_used, current_bandwidth_usage.get(t, 0))
    return max_used


def update_bandwidth_usage(x1, x2, y1, y2):
    """Update the bandwidth usage for the time range [x1, x2]"""
    for t in range(x1, x2):
        current_bandwidth_usage[t] = max(current_bandwidth_usage.get(t, 0), y2)


def get_next_slot(unavailable: Slots1D, total: Slots1D) -> List[Rect2D]:
    """
    Compute available regions by subtracting unavailable from total,
    resulting in available bandwidth rectangle regions
    # Example: Slot 2 is included by Slot 1, Slot 4 in included by Slot 3 ...
    #
    """
    if not total:
        return []
    total_region = next(iter(total))
    t_x1, t_x2, t_h = total_region

    available_regions = [(t_x1, t_x2, 0, t_h)]

    unavailable_sorted = sorted(unavailable, key=lambda u: (u[0], u[2]))

    result_regions = []
    current_y = 0

    for u_x1, u_x2, u_h in unavailable_sorted:
        # If the current unavailable slot is lower than previous slot, skip
        if u_h <= current_y:
            continue

        if u_x1 > t_x1:
            result_regions.append((t_x1, u_x1, current_y, t_h))

        if u_h > current_y:
            result_regions.append((t_x1, u_x1, current_y, u_h))

        current_y = u_h

    if current_y < t_h:
        result_regions.append((t_x1, t_x2, current_y, t_h))

    valid_regions = [
        (x1, x2, y1, y2) for x1, x2, y1, y2 in result_regions
        if x2 > x1 and y2 > y1
    ]

    # Sort by (y1, x1, -y2, x2) as specified
    return sorted(valid_regions, key=lambda r: (r[2], r[0], -r[3], r[1]))


def compute_slot_areas(slots: List[Rect2D]) -> List[int]:
    """Compute area of each slot for available list"""
    return [(x2 - x1) * (y2 - y1) for x1, x2, y1, y2 in slots]


def r_sorted_by_area(rules: List[Tuple[int, int]], reverse: bool = False) -> List[Tuple[int, int]]:
    """
    Generate [(size, r_id)] sorted by size
    :param rules: [(size, priority)]
    :param reverse: whether to sort in descending order
    :return: [(size, r_id)]
    """
    r_with_index = [(size, idx + 1) for idx, (size, _) in enumerate(rules)]
    return sorted(r_with_index, key=lambda x: x[0], reverse=reverse)


def ordinal(n: int) -> str:
    """Return ordinal string, e.g., 1->1st, 2->2nd"""
    return f"{n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'}"


def can_fit_all_remaining_requests(r_remaining, current_slot_area):
    r_sum = sum(area for area, _ in r_remaining)
    print(f"Current slot area: {current_slot_area}")
    print(f"Remaining requests sum: {sum(area for area, _ in r_remaining)}")
    print(f"Can fit all: {r_sum <= current_slot_area}")
    return r_sum <= current_slot_area


def allocate_requests_in_slot(slot_rect, requests):
    """Fixed allocation function that correctly handles bandwidth stacking"""
    allocation = []
    wasted = []
    x1, x2, y1, y2 = slot_rect
    width = x2 - x1

    # Get the current maximum used bandwidth in the time period
    current_max_used = get_current_max_bandwidth(x1, x2)

    # Start allocation from the max(current, y1)
    y_cursor = max(y1, current_max_used)

    for area, rid in requests:
        h = area / width
        allocation.append((x1, x2, y_cursor, y_cursor + h, rid))
        # Update bandwidth usage
        update_bandwidth_usage(x1, x2, y_cursor, y_cursor + h)
        y_cursor += h
    # Calculate unused area
    if y_cursor < y2:
        wasted.append((x1, x2, y_cursor, y2))

    return allocation, wasted


def allocate_requests_in_slot_best_under(slot_rect, requests, request_r):
    """
    Situation: When we are having best_under request set
    Compute the priority ratio (if needed), based on the ratio compute the height of each requests in the set
    Based on height, calculate width
    Need to fill out this slot's bandwidth
    """
    allocation = []
    wasted = []
    x1, x2, y1, y2 = slot_rect
    slot_width = x2 - x1

    if not requests:
        wasted.append(slot_rect)
        return allocation, wasted

    current_max_used = get_current_max_bandwidth(int(x1), int(x2))

    # get initial y and the max_height
    y_cursor = max(y1, current_max_used)
    available_height = y2 - y_cursor

    total_priority = sum(request_r[rid - 1][1] for area, rid in requests)

    for r_area, rid in requests:
        # Based on priority, get height
        priority = request_r[rid - 1][1]
        height_ratio = priority / total_priority if total_priority > 0 else 1.0 / len(requests)
        allocated_height = available_height * height_ratio

        if allocated_height > 0:
            actual_width = r_area / allocated_height
        else:
            actual_width = slot_width
            allocated_height = r_area / actual_width if actual_width > 0 else 0

        # Situation: while applying priority, the calculated width might larger than the slot width
        # Then use the actual_width to calculate the better height
        if actual_width > slot_width:
            actual_width = slot_width
            allocated_height = r_area / actual_width

        x_end = x1 + actual_width
        y_end = y_cursor + allocated_height

        allocation.append((x1, x_end, y_cursor, y_end, rid))
        update_bandwidth_usage(int(x1), int(x_end), y_cursor, y_end)

        y_cursor = y_end

    # Calculate the waste area (might useful while facing "exact size")
    max_width_used = max((alloc[1] - alloc[0] for alloc in allocation), default=0)
    if max_width_used < slot_width:
        wasted.append((x1 + max_width_used, x2, y1, y2))

    if y_cursor < y2:
        wasted.append((x1, x1 + max_width_used, y_cursor, y2))

    return allocation, wasted


def find_best_fit_groups(r_remaining, current_slot_area):
    """
    To find out the set of requests which is best fit of the current slot
    Two possible situation:
    1. the sum of the set of requests is less than the slot area
    2. ... slightly larger than the slot area

    This helps to determine the less waste situation
    """
    best_under, best_under_sum = (), 0
    best_over, best_over_sum = (), inf
    for n in range(1, len(r_remaining) + 1):
        for combo in combinations(r_remaining, n):
            s = sum(area for area, _ in combo)
            # best fit within capacity
            if s <= current_slot_area and s > best_under_sum:
                best_under, best_under_sum = combo, s
                if s == current_slot_area:
                    break
            # best fit over capacity
            elif s > current_slot_area and s < best_over_sum:
                best_over, best_over_sum = combo, s
        # find the perfect fit
        if best_under_sum == current_slot_area:
            break
    return best_under, best_under_sum, best_over, best_over_sum


def evaluate_blank2_area(best_over_sum, current_slot_area, current_slot_rect, next_slot_rect):
    """
    Determine the waste while using the second condition: Best over
    """
    x1, x2, y1, y2 = current_slot_rect
    width = x2 - x1
    # the height that over the current slot
    overflow_h = (best_over_sum - current_slot_area) / width
    #if not next_slot_rect:
    #    return inf

    # get the width of waste area
    next_x1, next_x2, _, _ = next_slot_rect
    return overflow_h * abs(next_x2 - x2)


def apply_overflow_to_next_slots(slot_rects, slot_area_list, i, delta_h):
    """
    The height over the current slot will make the next slot smaller
    delta_h: the height over the current slot
    """
    for j in (i + 1, i + 2):
        if j >= len(slot_rects):
            break
        x1, x2, y1, y2 = slot_rects[j]
        new_y2 = y2 - delta_h
        # print(f"new_y2:{new_y2}")
        if new_y2 <= y1:
            slot_rects.pop(j)
            slot_area_list.pop(j)
        else:
            slot_rects[j] = (x1, x2, y1, new_y2)
            slot_area_list[j] = (x2 - x1) * (new_y2 - y1)


def calculate_priority_bandwidth_ratio(original_prio, priority):
    """
    Higher priority (larger number) gets more bandwidth
    """
    # check if the request list have the same priority, if yes, with same width
    unique_priorities = sorted(set(priority), reverse=True)
    if len(unique_priorities) == 1:
        return 1.0

    return float(original_prio)

def has_bandwidth_conflict(x1, x2, y1, y2, unavailable_slots):
    """
    Check if the allocation will occur conflict for both:
    1. total area (will the height/ width extend the total area limit)
    2. unavailable slots
    """
    for ux1, ux2, uh in unavailable_slots:
        if not (x2 <= ux1 or x1 >= ux2):
            if not (y1 >= uh or y2 <= 0):
                return True
    return False


def allocate_requests_fill_bandwidth(slot_rect, requests, request_r: List[Tuple[int, int]], unavailable_slots):
    """
    Priority-based bandwidth allocation:
    If all remaining requests can be accommodated, allocate bandwidth to each request proportionally based on priority.
    When it causes conflict, revert to the simple method： allocate_requests_in_slot （Todo: need to be updated, this is temp way to deal with conflict situation）

    """
    allocation = []
    wasted = []
    x1, x2, y1, y2 = slot_rect

    # Get current max bandwidth usage
    max_used_bandwidth = get_current_max_bandwidth(int(x1), int(x2))
    effective_y1 = max(y1, max_used_bandwidth)
    effective_height = y2 - effective_y1

    # If this slot has no available height left, or there are no requests to allocate at all, return
    if effective_height <= 0 or not requests:
        if effective_height <= 0:
            wasted.append((x1, x2, y1, y2))
        return allocation, wasted

    # Check if this is the final allocation where all remaining requests fit
    total_size = sum(area for area, _ in requests)

    #  if there is only one left, then fit the rest of bandwidth
    if len(requests) == 1:
        area, rid = requests[0]
        height = effective_height
        width = area / height if height > 0 else 0

        slot_width = x2 - x1
        if width > slot_width:
            width = slot_width
            # If the calculated width exceeds the slot width, recalculate the height.
            height = area / width if width > 0 else 0

        x_end = x1 + width
        y_end = effective_y1 + height

        allocation.append((x1, x_end, effective_y1, y_end, rid))
        update_bandwidth_usage(int(x1), int(x_end), effective_y1, y_end)

        # If this happens, need to re-allocate the height (need to fill out bandwidth)
        if y_end < y2:
            wasted.append((x1, x2, y_end, y2))
            print("the height need to be re-calculated")

        return allocation, wasted

    # By priority, determine the area
    if len(requests) > 1 and total_size <= (effective_height * (x2 - x1)):
        # get the all priorities and calculate ratios
        all_priorities = [request_r[rid - 1][1] for area, rid in requests]
        request_data = []
        total_priority_sum = 0

        for area, rid in requests:
            original_priority = request_r[rid - 1][1]
            priority_ratio = calculate_priority_bandwidth_ratio(original_priority, all_priorities)
            total_priority_sum += priority_ratio

            request_data.append({
                'area': area,
                'rid': rid,
                'priority': original_priority,
                'ratio': priority_ratio
            })

        # calculate height and width for each request based on priority ratio
        slot_width = x2 - x1
        has_conflict = False

        for req_data in request_data:
            height_ratio = req_data['ratio'] / total_priority_sum
            #print(f"priority ratio = {height_ratio}")
            height = effective_height * height_ratio
            #print(f"height:{height}")

            width = req_data['area'] / height if height > 0 else 0
            #print(f"width:{width}")

            req_data['height'] = height
            req_data['width'] = width

        # checking for out-of-bounds conditions or conflicts with unavailable areas.
        y_cursor_check = effective_y1
        for req_data in request_data:
            width = req_data['width']
            height = req_data['height']
            x_end = x1 + width
            y_end = y_cursor_check + height

            if (unavailable_slots and has_bandwidth_conflict(x1, x_end, y_cursor_check, y_end, unavailable_slots) or
                    y_end > y2 or
                    width > slot_width):
                has_conflict = True
                break
            y_cursor_check = y_end

        if has_conflict:
            print("Conflict detected, falling back to simple area-based allocation")
            return allocate_requests_in_slot(slot_rect, requests)

        # if there is no conflict of total slot / unavailable slot, then use priority to determine height
        y_cursor = effective_y1
        for req_data in request_data:
            width = req_data['width']
            height = req_data['height']

            # width should <= slot width
            if width > slot_width:
                width = slot_width
                height = req_data['area'] / width if width > 0 else 0

            x_end = x1 + width
            y_end = y_cursor + height

            allocation.append((x1, x_end, y_cursor, y_end, req_data['rid']))
            update_bandwidth_usage(int(x1), int(x_end), y_cursor, y_end)

            print(f"Allocated request R{req_data['rid']}: bandwidth [{y_cursor:.2f}, {y_end:.2f}]")
            y_cursor = y_end

    else:
        return allocate_requests_in_slot(slot_rect, requests)

    return allocation, wasted

def out_of_range (remaining_r, original_r_list, current_slot_area, total_available_area):
    """
    params:
    remaining_r: requests that haven't been allocated' (area, r_id)
    original_r_list: original r list (input data) (size/area, priority)
    """
    # Deal with out_of_range situation
    if not remaining_r:
        return [], [], []
    # If there is no request allocated, then use total_available_area instead of the last slot area
    # Todo: need to be updated for total_available_area part, this may apply to "used" a few slots but there still some slots empty
    if len(remaining_r) == len(original_r_list) and total_available_area is not None:
        effective_slot_area = total_available_area
    else:
        effective_slot_area = current_slot_area
    # Check the best fit request group
    best_under, best_under_sum, _, _ = find_best_fit_groups(remaining_r, effective_slot_area)
    # mark all out of range if best_under_sum > current_slot area
    if not best_under or best_under_sum > effective_slot_area:
        out_of_range_requests = remaining_r[:]
        return [], out_of_range_requests

    # Check if there are other requests have the same size in the best_group
    best_group_sizes = [area for area, rid in best_under]
    same_size_candidates = {}
    for area, rid in remaining_r:
        if area in best_group_sizes:
            if area not in same_size_candidates:
                same_size_candidates[area] = []
            same_size_candidates[area].append((area, rid))

    # If yes: Compare priority
    final_best_group = []
    used_requests = set()
    for size in best_group_sizes:
        if size in same_size_candidates:
            # Find the larger priority
            candidates = same_size_candidates[size]
            best_candidate = None
            highest_priority = -1

            for area, rid in candidates:
                if (area, rid) not in used_requests:
                    priority = original_r_list[rid - 1][1]
                    if priority > highest_priority:
                        highest_priority = priority
                        best_candidate = (area, rid)

            if best_candidate:
                final_best_group.append(best_candidate)
                used_requests.add(best_candidate)

    # If no: Place the best group into this slot_area and specify which remaining requests were not included.
    allocated_set = set(final_best_group)
    out_of_range_requests = [(area, rid) for area, rid in remaining_r
                           if (area, rid) not in allocated_set]

    return final_best_group, out_of_range_requests

def find_r_slot_with_allocation(
        r_list: List[Tuple[int, int]],
        slot_area_list: List[int],
        slot_rects: List[Rect2D],
        unavailable_slots: Slots1D,
        request_r: List[Tuple[int, int]]
) -> Tuple[List[str], List[Tuple[int, int, float, float, int]], List[Rect2D], float, float]:
    # Reset global bandwidth usage
    global current_bandwidth_usage
    current_bandwidth_usage = {}

    # Initialize bandwidth usage with unavailable slots
    for x1, x2, h in unavailable_slots:
        for t in range(x1, x2):
            current_bandwidth_usage[t] = h

    result = []
    r_remaining = r_list[:]
    allocation = []
    wasted = []

    # Get the total area, available area
    total_r_area = sum(area for area, _ in r_remaining)
    even_slot_sum = sum(slot_area_list[i] for i in range(1, len(slot_area_list) - 1, 2)) if len(
        slot_area_list) > 1 else 0
    last_slot_area = slot_area_list[-1] if slot_area_list else 0
    total_available_area = even_slot_sum + last_slot_area

    slot_index = 1
    i = 0
    compare_mode = True

    while r_remaining and i < len(slot_area_list):
        current_slot_area = slot_area_list[i]
        current_slot_rect = slot_rects[i]

        # When it is the last slot:
        if i == len(slot_area_list) - 1:
            # All the remaining requests can be fitted: allocate bandwidth based on priority/ fill in all the bandwidth
            if can_fit_all_remaining_requests(r_remaining, current_slot_area):
                result.append(f"All remaining requests {r_remaining} fitted in the {ordinal(slot_index)} area")
                alloc, waste = allocate_requests_fill_bandwidth(current_slot_rect, r_remaining, request_r, unavailable_slots)
                allocation.extend(alloc)
                wasted.extend(waste)
                return result, allocation, wasted, total_available_area, total_r_area

            # If not: find best group, and mark the rest as out_of_range
            best_group, out_of_range_reqs = out_of_range(r_remaining, request_r, current_slot_area, total_available_area)
            #print(f"This is the best group while out of range: {best_group}")

            if best_group:
                alloc, waste = allocate_requests_in_slot_best_under(current_slot_rect, best_group, request_r)
                allocation.extend(alloc)
                wasted.extend(waste)

            return result, allocation, wasted, total_available_area, total_r_area


        if compare_mode:
            # check if all remaining requests fit in the odd slots
            if can_fit_all_remaining_requests(r_remaining, current_slot_area):
                result.append(f"All remaining requests {r_remaining} fitted in the {ordinal(slot_index)} area")

                alloc, waste = allocate_requests_fill_bandwidth(current_slot_rect, r_remaining, request_r, unavailable_slots)
                allocation.extend(alloc)
                wasted.extend(waste)
                return result, allocation, wasted, total_available_area, total_r_area
            # Move to next slot
            i += 1
            slot_index += 1
            compare_mode = False
            continue
        # The minimum request in the request list larger than current_area
        min_request_size = min(area for area, _ in r_remaining)

        if min_request_size > current_slot_area:
            #  current area / next slot width = max_height_extend
            next_slot_rect = slot_rects[i + 1]
            #print(f"next_slot_rect {next_slot_rect}")
            #print(f"current_slot_rect {current_slot_rect}")

            # next_slot[x2] - current_slot[x2]
            next_slot_width = next_slot_rect[1] - current_slot_rect[1]

            # the maximum height higher than current_slot height
            max_height_extend = current_slot_area / next_slot_width
            current_slot_width = current_slot_rect[1] - 0
            current_slot_height = current_slot_rect[3] - 0


            min_request_height_extend = (min_request_size / current_slot_width) - current_slot_height
            #print(f"minimum request size:{min_request_size}")
            #print(f"next slot width:{next_slot_width}")
            #print(f"max height extend:{max_height_extend}")
            #print(f"min height extend:{min_request_height_extend}")

            if min_request_height_extend > max_height_extend:
                # skip this slot (Todo: add this slot to the next even number slot)
                result.append(f"No requests fit in the {ordinal(slot_index)} area")
                i += 1
                slot_index += 1
                compare_mode = True
                continue
            else:
                #place the smallest request in the current slot
                min_request = min(r_remaining, key=lambda x: x[0])
                alloc, waste = allocate_requests_in_slot(current_slot_rect, [min_request])
                allocation.extend(alloc)
                print(min_request)
                wasted.extend(waste)
                r_remaining.remove(min_request)
                i += 1
                slot_index += 1
                compare_mode = True
                continue

        # find best fit for current slot
        # best_under: Does not exceed the current area and fits as closely as possible
        # best_over: A slightly larger set that exceeds the current area: overflowing into the next slot, will reducing the area of the next slot
        best_under, best_under_sum, best_over, best_over_sum = find_best_fit_groups(r_remaining, current_slot_area)
        print(f"best_over: {best_over}")
        print(f"best_under:{best_under}")

        # Calculate waste
        blank_1 = (current_slot_area - best_under_sum) if best_under else inf
        blank_2 = inf
        if best_over and i + 1 < len(slot_rects):
            blank_2 = evaluate_blank2_area(best_over_sum, current_slot_area, current_slot_rect, slot_rects[i + 1])

        if best_under and best_under_sum <= current_slot_area:
            best_fit_group = best_under
            result.append(f"Requests {list(best_fit_group)} fitted in the {ordinal(slot_index)} area")
            alloc, waste = allocate_requests_in_slot_best_under(current_slot_rect, best_fit_group, request_r)
            allocation.extend(alloc)
        elif best_over and blank_2 < current_slot_area:
            best_fit_group = best_over
            result.append(f"Requests {list(best_fit_group)} fitted in the {ordinal(slot_index)} area")
            alloc, waste = allocate_requests_in_slot(current_slot_rect, best_fit_group)
            allocation.extend(alloc)
            wasted.extend(waste)
        else:
            # Skip if no fitted set of requests
            result.append(f"No requests fit in the {ordinal(slot_index)} area")
            i += 1
            slot_index += 1
            compare_mode = True
            continue

        # Remove allocated requests from remaining list
        for val in best_fit_group:
            r_remaining.remove(val)

        if best_fit_group == best_over and i + 1 < len(slot_rects):
            x1, x2, _, _ = current_slot_rect
            delta_h = (best_over_sum - current_slot_area) / (x2 - x1)
            apply_overflow_to_next_slots(slot_rects, slot_area_list, i, delta_h)

        i += 1
        slot_index += 1
        compare_mode = True
    # all the remaining unallocated requests in the last slot
    if r_remaining:
        result.append(f"These requests fit in the last area: {r_remaining}")
        alloc, waste = allocate_requests_in_slot(slot_rects[-1], r_remaining)
        allocation.extend(alloc)
        wasted.extend(waste)

    return result, allocation, wasted, total_available_area, total_r_area


def visualize_integrated_schedule(request_r, unavailable_slots, total_slots, allocations, waste_rects):
    """
    visualize （mostly the same of the one in the warehouse.ipynb）
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    max_time = 0
    if unavailable_slots:
        max_time = max(max_time, max(end for _, end, _ in unavailable_slots))
    if allocations:
        max_time = max(max_time, max(x2 for x1, x2, _, _, _ in allocations))
    max_time = max(max_time, 30)  # Minimum time range

    total_bandwidth = next(iter(total_slots))[2] if total_slots else 100

    for x1, x2, h in total_slots:
        ax.add_patch(patches.Rectangle((x1, 0), x2 - x1, h,
                                       fill=False, edgecolor='black', linewidth=2))

    for x1, x2, h in unavailable_slots:
        ax.add_patch(patches.Rectangle((x1, 0), x2 - x1, h,
                                       color='red', alpha=0.6, label="Unavailable"))
        ax.text((x1 + x2) / 2, h / 2, f'Reserved\n{h} BW',
                ha='center', va='center', fontsize=8, color='white', weight='bold')

    for x1, x2, y1, y2, rid in allocations:
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       color='blue', alpha=0.7))

        size, priority = request_r[rid - 1]
        height = y2 - y1
        width = x2 - x1

        label = f'R{rid}\nSize: {size}\nPrio: {priority}\nBW: {height:.1f}'
        ax.text(x1 + width / 2, y1 + height / 2, label,
                ha='center', va='center', fontsize=8, color='white', weight='bold')

    info_x = max_time + 2
    info_y = total_bandwidth - 5
    dy = total_bandwidth / 15

    ax.text(info_x, info_y, "Requests (size, priority, height):", fontsize=10, weight='bold')
    info_y -= dy

    rid_to_height = {}
    for x1, x2, y1, y2, rid in allocations:
        height = y2 - y1
        rid_to_height[rid] = height

    for i, (size, priority) in enumerate(request_r, 1):
        height = f"{rid_to_height.get(i, 0):.1f}" if i in rid_to_height else "0"
        ax.text(info_x, info_y, f"R{i}: ({size}, {priority}, {height})", fontsize=9)
        info_y -= dy

    ax.set_xlim(0, max_time + 8)
    ax.set_ylim(0, total_bandwidth + 10)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Bandwidth', fontsize=12)
    ax.grid(True, alpha=0.3)

    handles = []
    if unavailable_slots:
        handles.append(patches.Patch(color='red', alpha=0.6, label='Unavailable/Reserved'))
    if allocations:
        handles.append(patches.Patch(color='blue', alpha=0.7, label='Allocated Requests'))
    if handles:
        ax.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.show()