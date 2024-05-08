# -*- coding: utf-8 -*-

nums, target = [2, 7, 11, 15], 9

def solution(nums: list, target: int) -> list:
    history = {}

    for i, num in enumerate(nums):
        diff = target - num
        if diff in history:
            return [i, history[diff]]
        history[num] = i
print(solution(nums, target))
