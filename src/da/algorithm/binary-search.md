---
title: 二分搜索
---


## 基础

### 在左边插入

> [Leetcode 35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/ "Leetcode 35. 搜索插入位置")

一、题目

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
你可以假设数组中无重复元素。

二、解析

在左边插入。

代码如下：

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        low, high = 0, len(nums)
        while low < high:
            mid = (low + high) // 2
            if nums[mid] < target:
                low = mid + 1
            else:
                high = mid
        return low
```

### 在右边插入

> [Leetcode 744. 寻找比目标字母大的最小字母](https://leetcode-cn.com/problems/find-smallest-letter-greater-than-target/ "Leetcode 744. 寻找比目标字母大的最小字母")

一、题目

给你一个排序后的字符列表 letters ，列表中只包含小写英文字母。另给出一个目标字母 target，请你寻找在这一有序列表里比目标字母大的最小字母。

在比较时，字母是依序循环出现的。举个例子：如果目标字母 target = 'z' 并且字符列表为 letters = \['a', 'b']，则答案返回 'a'

二、解析

在右边插入。

代码如下：

```python
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        low, high = 0, len(letters)
        while low < high:
            mid = (low + high) // 2
            if letters[mid] <= target:
                low = mid + 1
            else:
                high = mid
                
        if low == len(letters):
            return letters[0] 
        return letters[low]
```



## Leetcode 编程题

### 704. 二分查找

> [Leetcode 704. 二分查找](https://leetcode-cn.com/problems/binary-search/ "Leetcode 704. 二分查找")

一、题目

给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

二、解析

在左边插入。

代码如下：

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        low, high = 0, len(nums)
        while low < high:
            mid = (low + high) // 2
            if nums[mid] < target:
                low = mid + 1
            else:
                high = mid
        
        if low >= len(nums) or nums[low] != target:
            return -1
        
        return low
```

### 34. 在排序数组中查找元素的第一个和最后一个位置

> [Leetcode 34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/ "Leetcode 34. 在排序数组中查找元素的第一个和最后一个位置")

一、题目

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

你的算法时间复杂度必须是 O(logn) 级别。

如果数组中不存在目标值，返回 \[-1, -1]。

二、解析

对于整个数组，使用二分搜索寻找**在左边插入**`target`的位置，该位置为**第一个位置**。

-   向右遍历，找到**最后一个位置**，时间复杂度为`O(n)`，不符合要求，因为题目要求时间复杂度必须是`O(logn)`级别。
-   对第一个位置右边的数组，使用二分搜索寻找**在右边插入**`target`的位置，该位置减一为**最后一个位置**，时间复杂度为`O(lgn)`。

代码如下：

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1, -1]
        
        low, high = 0, len(nums)
        while low < high:
            mid = (low + high) // 2
            if nums[mid] < target:
                low = mid + 1
            else:
                high = mid
        
        if low >= len(nums) or nums[low] != target:
            return [-1, -1]

        low2, high2 = low + 1, len(nums)
        while low2 < high2:
            mid = (low2 + high2) // 2
            if nums[mid] <= target:
                low2 = mid + 1
            else:
                high2 = mid

        return [low, high2 - 1]
```

### 852. 山脉数组的峰顶索引

> [Leetcode 852. 山脉数组的峰顶索引](https://leetcode-cn.com/problems/peak-index-in-a-mountain-array/ "Leetcode 852. 山脉数组的峰顶索引")

一、题目

我们把符合下列属性的数组 A 称作山脉：

-   A.length >= 3
-   存在 0 < i < A.length - 1 使得A\[0] < A\[1] < ... A\[i-1] < A\[i] > A\[i+1] > ... > A\[A.length - 1]

给定一个确定为山脉的数组，返回任何满足 A\[0] < A\[1] < ... A\[i-1] < A\[i] > A\[i+1] > ... > A\[A.length - 1] 的 i 的值。

二、解析

代码如下：

```python
class Solution:
    def peakIndexInMountainArray(self, A: List[int]) -> int:
        low, high = 0, len(A)
        while low < high:
            mid = (low + high) // 2
            if A[mid-1] < A[mid] < A[mid+1]:
                low = mid
            elif A[mid-1] > A[mid] > A[mid+1]:
                high = mid
            else:
                return mid
```

### 153. 寻找旋转排序数组中的最小值

> [Leetcode 153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/ "Leetcode 153. 寻找旋转排序数组中的最小值")

一、题目

假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组`[0,1,2,4,5,6,7]`可能变为`[4,5,6,7,0,1,2]`)。
请找出其中最小的元素。
你可以假设数组中不存在重复元素。

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/solution/xun-zhao-xuan-zhuan-pai-lie-shu-zu-zhong-de-zui-xi/ "Leetcode官方题解")

代码如下：

```python
class Solution(object):
    def findMin(self, nums):
        if len(nums) == 1:
            return nums[0]

        left = 0
        right = len(nums) - 1
        if nums[left] < nums[right]:
            return nums[left]

        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            if nums[mid - 1] > nums[mid]:
                return nums[mid]
            if nums[mid] > nums[0]:
                left = mid + 1
            else:
                right = mid - 1
        return -1
```

### 33. 搜索旋转排序数组

> [搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/ "Leetcode 33. 搜索旋转排序数组")

一、题目

整数数组 `nums` 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

二、解析

代码如下：

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        
        left, right = 0, len(nums) - 1
        while left + 1 < right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > nums[left]:
                ## case1: numbers between left and mid are sorted
                if nums[left] <= target < nums[mid]:
                    right = mid
                else:
                    left = mid + 1
            else:
                ## case2: numbers between mid and right are sorted
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid
                    
        if nums[left] == target:
            return left
        elif nums[right] == target:
            return right
        return -1
```

### 540. 有序数组中的单一元素

> [Leetcode 540. 有序数组中的单一元素](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/ "Leetcode 540. 有序数组中的单一元素")

一、题目

给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。
注意: 您的方案应该在 O(log n)时间复杂度和 O(1)空间复杂度中运行。

二、解析

计算元素个数。代码如下：

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] == nums[mid - 1]:
                if (mid - left) % 2 == 0:
                    right = mid - 2
                else:
                    left = mid + 1
            elif nums[mid] == nums[mid + 1]:
                if (right - mid) % 2 == 0:
                    left = mid + 2
                else:
                    right = mid - 1
            else:
                return nums[mid]
                
        return nums[left]
```

### 69. x的平方根

> [Leetcode 69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/ "Leetcode 69. x 的平方根")

一、题目

实现 int sqrt(int x) 函数。

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

二、解析

二分搜索完，可能比正常结果大1，再判断一下即可。

代码如下：

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        low, high = 0, x
        while low < high:
            mid = (low + high) // 2
            if mid * mid < x:
                low = mid + 1
            else:
                high = mid
        
        if low ** 2 > x:
            return low - 1

        return low
```

给定精度限制：

```python
class Solution:
    def mySqrt(self, x: int, threshold: float) -> int:
        low, high = 0, x
        while low + threshold < high:
            mid = (low + high) / 2
            if mid * mid < x:
                low = mid
            else:
                high = mid
        return low
```

### 50. x的n次幂（快速幂）

> [Leetcode 50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/ "Leetcode 50. Pow(x, n)")

一、题目

实现`pow(x, n)`，即计算 x 的 n 次幂函数。

二、解析

时间复杂度要求$O(\lg{n})$。

代码如下：

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:        
        def quickMul(base, N):
            ans = 1.0
            while N > 0:
                if N & 1 == 1:
                    ans *= base
                base *= base
                N //= 2
            return ans
        
        return quickMul(x, n) if n >= 0 else 1.0 / quickMul(x, -n)
```

### 4. 寻找两个正序数据的中位数

> [4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

一、题目

给定两个大小为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。
请你找出这两个正序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
你可以假设 nums1 和 nums2 不会同时为空。

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/solution/xun-zhao-liang-ge-you-xu-shu-zu-de-zhong-wei-s-114/ "Leetcode官方题解")

如果不要求时间复杂度，要求找到两个有序数组的中位数，最直观的思路有以下两种：

-   把两个数组按序合并，寻找中位数，时间复杂度为O(m+n)，空间复杂度为O(m+n)。
-   不需要合并，用两个指针分别指向两个数组的下标为0的位置，每次比较并移动较小的指针，直到到达中位数的位置，此时时间复杂度为O(m+n)，空间复杂度为O(1)。

官方给出两种满足时间复杂度的方法，一是二分搜索，二是划分数组。

1）二分搜索。

代码如下：

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def getKthElement(k):         
            index1, index2 = 0, 0
            while True:
                ## 特殊情况
                if index1 == m:
                    return nums2[index2 + k - 1]
                if index2 == n:
                    return nums1[index1 + k - 1]
                if k == 1:
                    return min(nums1[index1], nums2[index2])

                ## 正常情况
                newIndex1 = min(index1 + k // 2 - 1, m - 1)
                newIndex2 = min(index2 + k // 2 - 1, n - 1)
                pivot1, pivot2 = nums1[newIndex1], nums2[newIndex2]
                if pivot1 <= pivot2:
                    k = k - (newIndex1 - index1 + 1)
                    index1 = newIndex1 + 1
                else:
                    k = k - (newIndex2 - index2 + 1)
                    index2 = newIndex2 + 1
        
        m, n = len(nums1), len(nums2)
        totalLength = m + n
        if totalLength % 2 == 1:
            return getKthElement((totalLength + 1) // 2)
        return (getKthElement(totalLength // 2) + getKthElement(totalLength // 2 + 1)) / 2
```
