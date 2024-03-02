---
title: 数组
order: 1
category: 数据结构
tag:
  - 数组
  - Leetcode
---

<!-- more -->

## 基础

数组是最简单的一种数据结构，它占据一块连续的内存并按照顺序存储数据。

在创建数组时，必须指定数组的容量大小，然后根据大小分配内存。由于数组的内存是连续的，因此可以根据下标以 O(1) 的时间复杂度读写任何元素，访问时间效率高。

数组的空间效率不是很高，可能会有空闲的区域没有得到利用。为了解决这个问题，大部分编程语言都提供了动态数组，比如 C++ 中的 vector，Java 中的 ArrayList，Python 中的 list 等。


## Leetcode 编程题

### 1. 两数之和

> [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

代码如下：

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        saved = {}
        for i, num in enumerate(nums):
            if target - num in saved:
                return [saved[target-num], i]
            else:
                saved[num] = i 
```



### 167. 两数之和 II - 输入有序数组

> [167. 两数之和 II - 输入有序数组](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/ "167. 两数之和 II - 输入有序数组")

一、题目

给你一个下标从 1 开始的整数数组 numbers ，该数组已按 非递减顺序排列  ，请你从数组中找出满足相加之和等于目标数 target 的两个数。如果设这两个数分别是 numbers\[index1] 和 numbers\[index2] ，则 1 <= index1 < index2 <= numbers.length 。

以长度为 2 的整数数组 \[index1, index2] 的形式返回这两个整数的下标 index1 和 index2。

你可以假设每个输入只对应唯一的答案 ，而且你 不可以 重复使用相同的元素。

你所设计的解决方案必须只使用常量级的额外空间。

示例 1：

```text
输入：numbers = [2,7,11,15], target = 9
输出：[1,2]
解释：2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。返回 [1, 2] 。
```

二、解析

使用双指针。代码如下：

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1
        while left < right:
            if numbers[left] + numbers[right] == target:
                return [left + 1, right + 1]
            elif numbers[left] + numbers[right] < target:
                left += 1
            else:
                right -= 1
        
        return []
```



### 15. 三数之和

> [15. 三数之和](https://leetcode-cn.com/problems/3sum/ "15. 三数之和")

一、题目

给你一个整数数组 nums ，判断是否存在三元组 \[nums\[i], nums\[j], nums\[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums\[i] + nums\[j] + nums\[k] == 0 。请

你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

**示例 1：**

```text
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。

```

二、解析

> 参考 [Leetcode-guanpengchn](https://leetcode-cn.com/problems/3sum/solution/hua-jie-suan-fa-15-san-shu-zhi-he-by-guanpengchn/ "Leetcode-guanpengchn")

排序，二分搜索。注意去重！

代码如下：

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if not nums or len(nums) < 3:
            return []
        
        res = []
        nums.sort()
        for i in range(len(nums) - 2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue  
            left = i + 1
            right = len(nums) - 1
            while left < right:
                temp = nums[i] + nums[left] + nums[right]
                if temp == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]: 
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif temp < 0:
                    left += 1
                else:
                    right -= 1
        return res
```



### 16. 最接近的三数之和

> https://leetcode.cn/problems/3sum-closest

一、题目

给你一个长度为 n 的整数数组 nums 和 一个目标值 target。请你从 nums 中选出三个整数，使它们的和与 target 最接近。

返回这三个数的和。

假定每组输入只存在恰好一个解。

示例 1：

```text
输入：nums = [-1,2,1,-4], target = 1
输出：2
解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
```

示例 2：

```text
输入：nums = [0,0,0], target = 1
输出：0

```

提示：

-   3 <= nums.length <= 1000
-   -1000 <= nums\[i] <= 1000
-   -104 <= target <= 104

二、解析

> [https://leetcode.cn/problems/3sum-closest/solution/hua-jie-suan-fa-16-zui-jie-jin-de-san-shu-zhi-he-b/](https://leetcode.cn/problems/3sum-closest/solution/hua-jie-suan-fa-16-zui-jie-jin-de-san-shu-zhi-he-b/ "https://leetcode.cn/problems/3sum-closest/solution/hua-jie-suan-fa-16-zui-jie-jin-de-san-shu-zhi-he-b/")<https://leetcode.cn/problems/3sum-closest/solution/zui-jie-jin-de-san-shu-zhi-he-by-leetcode-solution/>

Python代码如下：

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        best = 10**7
        
        ## 根据差值的绝对值来更新答案
        def update(cur):
            nonlocal best
            if abs(cur - target) < abs(best - target):
                best = cur
        
        ## 枚举 a
        for i in range(n):
            ## 保证和上一次枚举的元素不相等
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            ## 使用双指针枚举 b 和 c
            j, k = i + 1, n - 1
            while j < k:
                s = nums[i] + nums[j] + nums[k]
                ## 如果和为 target 直接返回答案
                if s == target:
                    return target
                update(s)
                if s > target:
                    ## 如果和大于 target，移动 c 对应的指针
                    k0 = k - 1
                    ## 移动到下一个不相等的元素
                    while j < k0 and nums[k0] == nums[k]:
                        k0 -= 1
                    k = k0
                else:
                    ## 如果和小于 target，移动 b 对应的指针
                    j0 = j + 1
                    ## 移动到下一个不相等的元素
                    while j0 < k and nums[j0] == nums[j]:
                        j0 += 1
                    j = j0

        return best

```

Java代码如下：

```java
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int ans = nums[0] + nums[1] + nums[2];
        for(int i=0;i<nums.length;i++) {
            int start = i+1, end = nums.length - 1;
            while(start < end) {
                int sum = nums[start] + nums[end] + nums[i];
                if(Math.abs(target - sum) < Math.abs(target - ans))
                    ans = sum;
                if(sum > target)
                    end--;
                else if(sum < target)
                    start++;
                else
                    return ans;
            }
        }
        return ans;
    }
}
```



### 53. 最大子数组和

> [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/ "53. 最大子序和")
> [剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/ "剑指 Offer 42. 连续子数组的最大和")

一、题目

给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组** 是数组中的一个连续部分。

二、解析

> 参考 [Python实现 《算法导论 第三版》中的算法 第4章 分治策略](https://blog.csdn.net/shengchaohua163/article/details/82810189 "Python实现 《算法导论 第三版》中的算法 第4章 分治策略")

题目要求子数组不为空，所以连续子数组的和可以为负。代码如下：

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = -float('inf')
        temp = 0
        for num in nums:
            temp += num
            if temp > max_sum:
                max_sum = temp
            if temp < 0:
                temp = 0
        return max_sum
```

如果子数组可以为空，一般规定空数组之和为0，所以连续子数组的和最小为0。代码如下：

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = 0
        temp = 0
        for num in nums:
            temp += num
            if temp > max_sum:
                max_sum = temp
            if temp < 0:
                temp = 0
        return max_sum
```



### 283. 移动零

> [283. 移动零](https://leetcode-cn.com/problems/move-zeroes/ "283. 移动零")

一、题目

给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
说明: 1. 必须在原数组上操作，不能拷贝额外的数组。2. 尽量减少操作次数。

二、解析

保存一个下标，用来标识0的位置。

代码如下：

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        zero_idx = -1
        for i in range(len(nums)):
            if nums[i] == 0:
                continue
            zero_idx += 1
            nums[i], nums[zero_idx] = nums[zero_idx], nums[i]
```



### 215. 数组中的第K个最大元素

> [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/ "215. 数组中的第K个最大元素")

一、题目

在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

二、解析

使用快排的分割方法，分割数组后计算右边的元素个数。

Python 代码如下：

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partition(A, left, right):
            pivot = A[left]
            while left < right:
                while left < right and A[right] >= pivot:
                    right -= 1
                A[left] = A[right]
                while left < right and A[left] < pivot:
                    left += 1
                A[right] = A[left]
            A[left] = pivot
            return left
        
        def helper(A, left, right, k):
            if left <= right:
                p = partition(A, left, right)
                count = right - p + 1
                if count > k:
                    return helper(A, p + 1, right, k)
                elif count < k:
                    return helper(A, left, p - 1, k - count)
                return A[p]
            
        return helper(nums, 0, len(nums)-1, k)
```

Golang 代码如下：

```go
func findKthLargest(nums []int, k int) int {
	return find(nums, 0, len(nums)-1, k)
}

func find(nums []int, left, right, k int) int {
	if left > right {
		return 0
	}

	p := partition(nums, left, right)
	count := right - p + 1
	if count > k {
		return find(nums, p+1, right, k)
	} else if count < k {
		return find(nums, left, p-1, k-count)
	}
	return nums[p]
}

func partition(nums []int, left, right int) int {
	pivot := nums[left]
	for left < right {
		for left < right && nums[right] >= pivot {
			right -= 1
		}
		nums[left] = nums[right]
		for left < right && nums[left] < pivot {
			left += 1
		}
		nums[right] = nums[left]
	}
	nums[left] = pivot
	return left
}
```



### 915. 分割数组

> [915. 分割数组](https://leetcode-cn.com/problems/partition-array-into-disjoint-intervals/ "915. 分割数组")

一、题目

给定一个数组 A，将其划分为两个不相交（没有公共元素）的连续子数组 left 和 right，使得：

-   left 中的每个元素都小于或等于 right 中的每个元素。
-   left 和 right 都是非空的。
-   left 要尽可能小。

在完成这样的分组后返回 left 的长度。可以保证存在这样的划分方法。

二、解析

记录左边数组的最大值和遍历过的所有元素的最大值。

代码如下：

```python
class Solution:
    def partitionDisjoint(self, A: List[int]) -> int:
        if not A:
            return 0

        left_max = A[0]
        cur_max = A[0]
        index = 0

        for i in range(1, len(A)):
            cur_max = max(cur_max, A[i])
            if A[i] < left_max:
                left_max = cur_max
                index = i

        return index + 1
```



### 665. 非递减数列

> [665. 非递减数列](https://leetcode-cn.com/problems/non-decreasing-array/ "665. 非递减数列")

一、题目

给你一个长度为 n 的整数数组 nums ，请你判断在 最多 改变 1 个元素的情况下，该数组能否变成一个非递减数列。

我们是这样定义一个非递减数列的： 对于数组中任意的 i (0 <= i <= n-2)，总满足 nums\[i] <= nums\[i + 1]。

二、解析

> 参考 [Leetcode-码不停题](https://leetcode-cn.com/problems/non-decreasing-array/comments/59727 "Leetcode-码不停题")

代码如下：

```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        if not nums or len(nums) <= 2:
            return True

        count = 0
        for i in range(1, len(nums)):
            if nums[i - 1] <= nums[i]:
                continue
            
            count += 1
            if count >= 2:
                return False

            if i >= 2 and nums[i - 2] > nums[i]:
                nums[i] = nums[ i - 1]
            else:
                nums[i - 1] = nums[i]
        
        return True
```



### 448. 找到所有数组中消失的数字

> [448. 找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/ "448. 找到所有数组中消失的数字")

一、题目

给定一个范围在  1 ≤ a\[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
找到所有在 \[1, n] 范围之间没有出现在数组中的数字。
您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。

二、解析

把数组下标和数组中的数字联系在一起：把下标为`i`的元素「标记」为负数，表示整数`i+1`在数组中存在。

代码如下：

```python
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for x in nums: 
            if nums[abs(x) - 1] > 0:
                nums[abs(x) - 1] *= -1
        res = []
        for i in range(len(nums)):
            if nums[i] > 0:
                res.append(i + 1)
        return res
```



### 442. 数组中重复的数据

> [442. 数组中重复的数据](https://leetcode-cn.com/problems/find-all-duplicates-in-an-array/ "442. 数组中重复的数据")

一、题目

给定一个整数数组 a，其中1 ≤ a\[i] ≤ n （n为数组长度）, 其中有些元素出现两次而其他元素出现一次。
找到所有出现两次的元素。
你可以不用到任何额外空间并在O(n)时间复杂度内解决这个问题吗？

二、解析

把数组下标和数组中的数字联系在一起：把下标为`i`的元素「标记」为负数，表示整数`i+1`在数组中存在。

代码如下：

```python
class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        res = []
        for x in nums:
            if nums[abs(x) - 1] > 0:
                nums[abs(x) - 1] *= -1
            else:
                res.append(abs(x))
        return res
```



### LCR 120. 寻找文件副本（数组中重复的数字）

> [LCR 120. 寻找文件副本](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

一、题目

找出数组中重复的数字。

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

示例1：

```纯文本
输入：[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 
```

二、解析

考察沟通能力。

1）时间优先

使用哈希表，代码略。此时时间复杂度为O(n)，空间复杂度O(n)。

2）空间优先

2.1）使用原地排序，代码略。此时时间复杂度最好为O(nlgn)，空间复杂度为O(1)。

2.2）把每个数字都加1，此时和 Leetcode-442 相同，代码略；

2.3）单独判断0是否重复，其他数字不用变化，并把0修改为一个比较大的数字。

代码如下：

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        n = len(nums)
        zero_flag = False
        for i in range(n):
            if nums[i] == 0:
                if zero_flag: 
                    return 0
                zero_flag = True
                nums[i] = n
        for i in range(n):
            idx = abs(nums[i])
            if idx >= n: 
                continue
            if nums[idx] < 0:
                return idx
            nums[idx] = -abs(nums[idx])
        return -1
```

4）原地置换，相当于排序。此时时间复杂度为O(n)，空间复杂度O(1)。

代码如下：

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            while nums[i] != i:
                if nums[i] == nums[nums[i]]:
                    return nums[i]
                temp = nums[i]
                nums[i], nums[temp] = nums[temp], nums[i]
                ## print(nums)
        return -1
```

5）空间优先，只允许使用O(1)的空间，不允许修改数组。查看 Leetcode-287。



### 287. 寻找重复数

> [https://leetcode.cn/problems/find-the-duplicate-number/](https://leetcode.cn/problems/find-the-duplicate-number/ "https://leetcode.cn/problems/find-the-duplicate-number/")

一、题目

给定一个包含 n + 1 个整数的数组 nums ，其数字都在 \[1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。

假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。

你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。

二、解析

> [https://leetcode.cn/problems/find-the-duplicate-number/solution/xun-zhao-zhong-fu-shu-by-leetcode-solution/](https://leetcode.cn/problems/find-the-duplicate-number/solution/xun-zhao-zhong-fu-shu-by-leetcode-solution/ "https://leetcode.cn/problems/find-the-duplicate-number/solution/xun-zhao-zhong-fu-shu-by-leetcode-solution/")

1）二分查找。

2）二进制。

3）快慢指针。

我们先设置慢指针 slow 和快指针 fast ，慢指针每次走一步，快指针每次走两步，根据「Floyd 判圈算法」两个指针在有环的情况下一定会相遇，此时我们再将 slow 放置起点 0，两个指针每次同时移动一步，相遇的点就是答案。

Java 代码如下：

```java
class Solution {
    public int findDuplicate(int[] nums) {
        int slow = 0, fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
}

```



### 41. 缺失的第一个正数

> [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/ "41. 缺失的第一个正数")

一、题目

给你一个未排序的整数数组，请你找出其中没有出现的最小的正整数。

提示：你的算法的时间复杂度应为O(n)，并且只能使用常数级别的额外空间。

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/first-missing-positive/solution/que-shi-de-di-yi-ge-zheng-shu-by-leetcode-solution/ "Leetcode官方题解")

最简单的方法是使用哈希表，但是不满足空间复杂度。

仔细思考，结果必然在`1`到`N+1`之间，因为数组最多包含`N`个整数。所以，可以把数组下标和数组中的数字联系在一起：把下标为`i`的元素「标记」为负数，表示整数`i+1`在数组中存在。

由于我们只在意`[1,N]`中的数，因此我们可以先对数组进行遍历，把不在`[1,N]`范围内的数修改成任意一个大于`N`的数（例如`N+1`）。

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            if nums[i] <= 0:
                nums[i] = n + 1
        
        for i in range(n):
            num = abs(nums[i])
            if num <= n:
                nums[num - 1] = -abs(nums[num - 1])
        
        for i in range(n):
            if nums[i] > 0:
                return i + 1
        
        return n + 1
```



### 560. 和为K的子数组

> [560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/ "560. 和为K的子数组")

一、题目

给定一个整数数组和一个整数 k，你需要找到该数组中和为 k 的连续的子数组的个数。

二、解析

1）暴力法，时间复杂度$O(n^2)$。代码略。

2）前缀和 + 哈希表优化，时间复杂度$O(n)$。

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        res = 0
        cumu_sum = 0  ## 前缀和，累计和
        seen = {0:1}
        for num in nums:
            cumu_sum += num
            if cumu_sum - k in seen:
                res += seen[cumu_sum - k]
            if cumu_sum not in seen:
                seen[cumu_sum] = 0
            seen[cumu_sum] += 1
        
        return res
```



### 523. 连续的子数组和是否为K

> [523. 连续的子数组和](https://leetcode-cn.com/problems/continuous-subarray-sum/ "523. 连续的子数组和")

一、题目

给定一个包含 非负数 的数组和一个目标 整数`k`，编写一个函数来判断该数组是否含有连续的子数组，其大小至少为 2，且总和为 k 的倍数，即总和为 n\*k，其中 n 也是一个整数。

示例 1：

```纯文本
输入：[23,2,4,6,7], k = 6
输出：True
解释：[2,4] 是一个大小为 2 的子数组，并且和为 6。
```

二、解析

1）暴力，时间复杂度`O(n2)`。

代码如下：

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        if len(nums) < 2:
            return False
        n = len(nums)
        for i in range(n - 1):
            cumu_sum = nums[i]
            for j in range(i + 1, n):
                cumu_sum += nums[j]
                if cumu_sum == 0:
                    return True
                if k != 0 and cumu_sum % k == 0:
                    return True
        return False
```

2）前缀和 + 哈希表

> [https://leetcode.cn/problems/continuous-subarray-sum/solution/lian-xu-de-zi-shu-zu-he-by-leetcode-solu-rdzi/](https://leetcode.cn/problems/continuous-subarray-sum/solution/lian-xu-de-zi-shu-zu-he-by-leetcode-solu-rdzi/ "https://leetcode.cn/problems/continuous-subarray-sum/solution/lian-xu-de-zi-shu-zu-he-by-leetcode-solu-rdzi/")

由于哈希表存储的是每个余数第一次出现的下标，因此当遇到重复的余数时，根据当前下标和哈希表中存储的下标计算得到的子数组长度。如果子数组的长度大于等于 2，即存在符合要求的子数组。

```go
func checkSubarraySum(nums []int, k int) bool {
    m := len(nums)
    if m < 2 {
        return false
    }
    mp := map[int]int{0: -1}
    remainder := 0
    for i, num := range nums {
        remainder = (remainder + num) % k
        if prevIndex, has := mp[remainder]; has {
            if i-prevIndex >= 2 {
                return true
            }
        } else {
            mp[remainder] = i
        }
    }
    return false
}
```



### 974. 和可被 K 整除的子数组

> [974. 和可被 K 整除的子数组](https://leetcode-cn.com/problems/subarray-sums-divisible-by-k/ "974. 和可被 K 整除的子数组")

一、题目

给定一个整数数组`A`，返回其中元素之和可被`K`整除的（连续、非空）子数组的数目。
示例：

```纯文本
输入：A = [4,5,0,-2,-3,1], K = 5
输出：7
解释：
有 7 个子数组满足其元素之和可被 K = 5 整除：
[4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]
```

二、解析

1、暴力，超时

```python
class Solution:
    def subarraysDivByK(self, A: List[int], K: int) -> int:
        res = 0
        n = len(A)
        for i in range(n):
            cumu_sum = 0
            for j in range(i, n):
                cumu_sum += A[j]
                if cumu_sum % K == 0:
                    res += 1
        return res
```

2、哈希表 + 前缀和

```python
class Solution:
    def subarraysDivByK(self, A: List[int], K: int) -> int:
        saved = {0:1}
        res = 0
        cumu_sum = 0
        for num in A:
            cumu_sum += num
            mod = cumu_sum % K
            same = saved.get(mod, 0)
            res += same
            saved[mod] = same + 1
        return res
```



### 169. 多数元素/数组中出现次数超过一半的数字

> [169. 多数元素](https://leetcode-cn.com/problems/majority-element/ "169. 多数元素")[剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/ "剑指 Offer 39. 数组中出现次数超过一半的数字")

一、题目

给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。

二、解析

记录元素个数。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        res = None
        count = 0
        for n in nums:
            if count == 0:
                res = n
                count = 1
            elif res == n:
                count += 1
            elif res != n:
                count -= 1
        return res
```



### 54. 螺旋矩阵

> [54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/ "54. 螺旋矩阵")[剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/ "剑指 Offer 29. 顺时针打印矩阵")

一、题目

给定一个包含 m x n 个元素的矩阵（m行,n列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。

二、解析

代码如下：

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []
            
        m, n = len(matrix), len(matrix[0])
        min_row, max_row = 0, m - 1
        min_col, max_col = 0, n - 1
        res = []
        while True:
            for j in range(min_col, max_col + 1):
                res.append(matrix[min_row][j])
            min_row += 1
            if min_row > max_row:
                break
            for i in range(min_row, max_row + 1):
                res.append(matrix[i][max_col])
            max_col -= 1
            if max_col < min_col:
                break
            for j in range(max_col, min_col - 1, -1):
                res.append(matrix[max_row][j])
            max_row -= 1
            if max_row < min_row:
                break
            for i in range(max_row, min_row - 1, -1):
                res.append(matrix[i][min_col])
            min_col += 1
            if min_col > max_col:
                break

        return res
```



### 59. 螺旋矩阵 II

> [59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/ "59. 螺旋矩阵 II")

一、题目

给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。

二、解析

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        res = [[0] * n for _ in range(n)]
        num = 1 
        min_row, max_row = 0, n - 1
        min_col, max_col = 0, n - 1
        while True:
            for j in range(min_col, max_col + 1):
                res[min_row][j] = num
                num += 1
            min_row += 1
            if min_row > max_row:
                break
            for i in range(min_row, max_row + 1):
                res[i][max_col] = num
                num += 1
            max_col -= 1
            if max_col < min_col:
                break
            for j in range(max_col, min_col - 1, -1):
                res[max_row][j] = num
                num += 1
            max_row -= 1
            if max_row < min_row:
                break
            for i in range(max_row, min_row - 1, -1):
                res[i][min_col] = num
                num += 1
            min_col += 1
            if min_col > max_col:
                break

        return res
```



### 845. 数组中的最长山脉

> [845. 数组中的最长山脉](https://leetcode-cn.com/problems/longest-mountain-in-array/ "845. 数组中的最长山脉")

一、题目

把符合下列属性的数组 arr 称为 山脉数组 ：

-   arr.length >= 3
-   存在下标 i（0 < i < arr.length - 1），满足
    -   arr\[0] < arr\[1] < ... < arr\[i - 1] < arr\[i]
    -   arr\[i] > arr\[i + 1] > ... > arr\[arr.length - 1]

给出一个整数数组 arr，返回最长山脉子数组的长度。如果不存在山脉子数组，返回 0 。

示例 1：

```text
输入：arr = [2,1,4,7,3,2,5]
输出：5
解释：最长的山脉子数组是 [1,4,7,3,2]，长度为 5。
```

示例 2：

```text
输入：arr = [2,2,2]
输出：0
解释：不存在山脉子数组。
```

二、解析

> 参考 [官方题解](https://leetcode-cn.com/problems/longest-mountain-in-array/solution/shu-zu-zhong-de-zui-chang-shan-mai-by-leetcode/ "官方题解")

代码如下：

```python
class Solution:
    def longestMountain(self, A: List[int]) -> int:
        res = 0
        left = 0
        N = len(A)
        while left < N:
            end = left
            if end + 1 < N and A[end] < A[end + 1]: 
                while end + 1 < N and A[end] < A[end + 1]:
                    end += 1
                if end + 1 < N and A[end] > A[end + 1]:
                    while end + 1 < N and A[end] > A[end + 1]:
                        end += 1
                    res = max(res, end - left + 1)
            left = max(left + 1, end)
        return res
```



### 209. 长度最小的子数组（子数组和大于K）

> [209. 长度最小的子数组（子数组和大于K）](https://leetcode.cn/problems/minimum-size-subarray-sum "https://leetcode.cn/problems/minimum-size-subarray-sum")

一、题目

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 \[numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

示例 1：

```text
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
```

示例 2：

```text
输入：target = 4, nums = [1,4,4]
输出：1
```

示例 3：

```text
输入：target = 11, nums = [1,1,1,1,1,1,1,1]
输出：0
```

二、解析

> [官方题解](https://leetcode.cn/problems/minimum-size-subarray-sum/solution/chang-du-zui-xiao-de-zi-shu-zu-by-leetcode-solutio/)

1）暴力法。

代码如下：

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        ans = n + 1
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += nums[j]
                if total >= s:
                    ans = min(ans, j - i + 1)
                    break
        
        return 0 if ans == n + 1 else ans
```

2）前缀和加二分查找

代码如下：

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        ans = n + 1
        sums = [0]
        for i in range(n):
            sums.append(sums[-1] + nums[i])
        
        for i in range(1, n + 1):
            target = s + sums[i - 1]
            bound = bisect.bisect_left(sums, target)
            if bound != len(sums):
                ans = min(ans, bound - (i - 1))
        
        return 0 if ans == n + 1 else ans
```

3）滑动窗口。

左右指针，右指针向右走，如果子数组的和大于目标值，然后左指针向右走，减去对应的元素。

代码如下：

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        ans = n + 1
        start, end = 0, 0
        total = 0
        while end < n:
            total += nums[end]
            while total >= s:
                ans = min(ans, end - start + 1)
                total -= nums[start]
                start += 1
            end += 1
        
        return 0 if ans == n + 1 else ans
```



### 128. 最长连续序列

> [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/ "128. 最长连续序列")

一、题目

给定一个未排序的整数数组，找出最长连续序列的长度。
要求算法的时间复杂度为 O(n)。

示例 1：

```text
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

示例 2：

```text
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
```

二、解析

使用哈希表！

代码如下：

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        res = 0
        num_set = set(nums)

        for num in nums:
            if num - 1 not in num_set:
                cur_num = num
                cur_count = 1
                while cur_num + 1 in num_set:
                    cur_num += 1
                    cur_count += 1
                res = max(res, cur_count)

        return res
```
