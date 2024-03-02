---
title: 动态规划
order: 3
---

<!-- more -->

## 经典编程题

### 背包问题

> [背包九讲——全篇详细理解与代码实现](https://blog.csdn.net/yandaoqiusheng/article/details/84782655/ "背包九讲——全篇详细理解与代码实现")

给定一个容量为 capacity 的背包和 N 种物品：

- 物品的重量（或体积）用一个长度为 N 的数组 weight 表示，第 i 种物品的重量和价值分别为 weights[i]。
- 物品的价值用一个长度为 N 的数组 values 表示，第 i 种物品的价值为 values[i]。

请问，在不超过背包容量的条件下，如何选择物品，使得价值最大。

#### 0/1背包 I

每种物品最多选择1个，也就是说，可以不选择或者选择一个。

1）使用二维数组。

代码如下：

```python
def zero_one_backpack(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    for i in range(n):  ## 对于每一个物品
        for j in range(1, capacity + 1):  ## 对于每一个容量
            if j < weights[i]:
                dp[i + 1][j] = dp[i][j]
            else:
                dp[i + 1][j] = max(dp[i][j], dp[i][j - weights[i]] + values[i])
    return dp[-1][-1]
```

2）使用一维数组。

代码如下：

```python
def zero_one_backpack(weights, values, capacity):
    n = len(weights)
    dp = [0 for _ in range(capacity + 1)]
    for i in range(n):
        for j in range(capacity, weights[i] - 1, -1):
            if i == 0:
                dp[j] = values[i]
            else:
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[-1]
```



#### 0/1背包 II

每种物品最多选择1个，也就是说，可以不选择或者选择一个。要求：背包恰好装满。

1）使用二维数组

代码如下：

```python
def zero_one_backpack_full(weights, values, capacity):
    n = len(weights)
    dp = [[0] + [-1 for _ in range(capacity)] for _ in range(n + 1)]
    for i in range(n):
        for j in range(1, capacity + 1):
            if j < weights[i]:
                dp[i + 1][j] = dp[i][j]
            elif dp[i][j - weights[i]] != -1:
                dp[i + 1][j] = max(dp[i][j], dp[i][j - weights[i]] + values[i])
    return dp[-1][-1]
```

2）使用一维数组

代码如下：

```python
def zero_one_backpack_full(weights, values, capacity):
    n = len(weights)
    dp = [-1 for _ in range(capacity + 1)]
    dp[0] = 0
    for i in range(n):
        for j in range(capacity, weights[i] - 1, -1):
            if dp[i][j - weights[i]] != -1:
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[-1]
```



#### 完全背包

每种物品可以选择多个。

1）使用一维数组

代码如下：

```python
def complete_backpack(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for j in range(capacity + 1):
            for k in range(j // weights[i] + 1):
                dp[j] = max(dp[j], dp[j - k * weights[i]] + k * values[i])
    return dp[-1]
```

2）使用一维数组

代码如下：

```python
def complete_backpack_2(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for j in range(weights[i], capacity + 1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[-1]
```




## Leetcode 编程题

### 1143. 最长公共子序列

> [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/ "1143. 最长公共子序列")

一、题目

给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。

一个字符串的子序列是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。

若这两个字符串没有公共子序列，则返回 0。

二、解析

经典动态规划。

代码如下：

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n1 = len(text1)
        n2 = len(text2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        for i in range(n1):
            for j in range(n2):
                if text1[i] == text2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[-1][-1]
```

输出最长公共子序列：

```python
def print_lcs(text1, text2, dp):
    '''dp为求text1和text2最长公共子序列的状态矩阵'''
    n1 = len(text1)
    n2 = len(text2)
    lcs = []
    i = n1
    j = n2
    while i != 0 and j != 0:
        while dp[i - 1][j] == dp[i][j]:  ## 循环1，可以和循环2交换
            i -= 1
        while dp[i][j - 1] == dp[i][j]:  ## 循环2
            j -= 1
        lcs.append(text2[j - 1])
        i -= 1
        j -= 1
    return "".join(lcs[::-1])
```



### 5. 最长回文子串

> [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/ "5. 最长回文子串")

一、题目

给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

二、解析

1）动态规划。用`dp[i][j]`表示 s 中第 i 个字符到第 j 个字符组成的子串是否是回文子串。

代码如下：

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        longest_len = 0
        longest_pali = ""
        n = len(s)
        dp = [[0 for _ in range(n)] for _ in range(n)]
        for length in range(1, n + 1):
            for i in range(0, n - length + 1):
                j = i + length - 1
                if length <= 2:
                    dp[i][j] = s[i] == s[j]
                else:
                    dp[i][j] = dp[i + 1][j - 1] & (s[i] == s[j])
                if dp[i][j] and length > longest_len:
                    longest_len = length
                    longest_pali = s[i:i + length]
        return longest_pali
```

2）中心扩展

代码如下：

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def findP(string, left, right):
            while left >= 0 and right < len(string) and string[left] == string[right]:
                left -= 1
                right += 1
            return right - left -1

        if not s:
            return ""
        
        start = end = 0
        res = 0
        for i in range(len(s)-1):
            len1 = findP(s, i, i)
            len2 = findP(s, i, i + 1)
            length = max(len1, len2)
            if length > res:
                res = length
                start = i - (length - 1) // 2
                end = i + length // 2
        return s[start:end + 1]
```



### 516. 最长回文子序列

> [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/ "516. 最长回文子序列")

一、题目

给定一个字符串 s，找到其中最长的回文子序列，并返回该序列的长度。可以假设 s 的最大长度为 1000 。

二、解析

1）动态规划。

使用最长回文子串的思路。用`dp[i][j]`表示s中第i个字符到第j个字符组成的子串的最长的回文序列的长度。

状态转移方程为：

-   当`i=j`，即当前子串的长度为1，`dp[i][j]=1`；
-   当`i<j`，即当前子串的长度大于等于2：
    -   如果`s[i]=s[j]`，则`dp[i][j]=dp[i+1][j-1]+2`；
    -   如果`s[i]!=s[j]`，则`dp[i][j]=max(dp[i+1][j],dp[i][j-1])`；

代码如下：

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0 for _ in range(n)] for _ in range(n)]
        for length in range(1, n + 1):
            for i in range(0, n - length + 1):
                if length == 1:
                    dp[i][i] = 1
                    continue
                j = i + length - 1
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return dp[0][n - 1]
```

2）从后向前遍历。

> 参考 [Leetcode-元仲辛](https://leetcode-cn.com/problems/longest-palindromic-subsequence/solution/dong-tai-gui-hua-si-yao-su-by-a380922457-3/ "Leetcode-元仲辛")

代码如下：

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return dp[0][n - 1]
```



### 300. 最长上升子序列

> [300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/ "300. 最长上升子序列")

一、题目

给定一个无序的整数数组，找到其中最长上升子序列的长度。

二、解析

> 参考 [Leetcode-Solution](https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/zui-chang-shang-sheng-zi-xu-lie-by-leetcode-soluti/ "Leetcode-Solution")

1）动态规划。

定义`dp[i]`为考虑数组前`i`个数字，以第`i`个数字结尾的最长上升子序列的长度，注意`nums[i]`必须被选取。

从小到大计算`dp`数组的值，在计算`dp[i]`之前，我们已经计算出`dp[0...i-1]`的值，则状态转移方程为：`dp[i] = max(dp[j]) + 1 0 <= j < i & num[j] < num[i]`。

最后，整个数组的最长上升子序列即所有`dp[i]`中的最大值。

代码如下：

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```

2）贪心 + 二分查找。

如果我们要使上升子序列尽可能的长，则我们需要让序列上升得尽可能慢，因此我们希望每次在上升子序列最后加上的那个数尽可能的小。

代码如下：

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        seen = []
        for num in nums:
            if not seen or num > seen[-1]:
                seen.append(num)
            else:
                left, right = 0, len(seen)
                while left < right:
                    mid = (left + right) // 2
                    if seen[mid] < num:
                        left = mid + 1
                    else:
                        right = mid
                seen[left] = num
        return len(seen)
```



### 413. 等差数列划分

> [413. 等差数列划分](https://leetcode-cn.com/problems/arithmetic-slices/ "413. 等差数列划分")

一、题目

如果一个数列至少有三个元素，并且任意两个相邻元素之差相同，则称该数列为等差数列。

函数要返回数组 A 中所有为等差数组的子数组个数。

**示例 1：**

```
输入：nums = [1,2,3,4]
输出：3
解释：nums 中有三个子等差数组：[1, 2, 3]、[2, 3, 4] 和 [1, 2, 3, 4] 自身。
```

二、解析

代码如下：

```python
class Solution:
    def numberOfArithmeticSlices(self, A: List[int]) -> int:
        n = len(A)
        dp = [0] * n
        res = 0
        for i in range(2, n):
            if A[i] - A[i - 1] == A[i - 1] - A[i - 2]:
                dp[i] = dp[i - 1] + 1
                res += dp[i]
            else:
                dp[i] = 0
        return res
```



### 446. 等差数列划分 II - 子序列

> [446. 等差数列划分 II - 子序列](https://leetcode-cn.com/problems/arithmetic-slices-ii-subsequence/ "446. 等差数列划分 II - 子序列")

一、题目

如果一个数列至少有三个元素，并且任意两个相邻元素之差相同，则称该数列为等差数列。

函数要返回数组 A 中所有等差子序列的个数。

输入包含 N 个整数。每个整数都在 -231 和 231-1 之间，另外 0 ≤ N ≤ 1000。保证输出小于 231-1。

示例：

```纯文本
输入：[2, 4, 6, 8, 10]
输出：7

解释：
所有的等差子序列为：
[2,4,6]
[4,6,8]
[6,8,10]
[2,4,6,8]
[4,6,8,10]
[2,4,6,8,10]
[2,6,10]
```

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/arithmetic-slices-ii-subsequence/solution/deng-chai-shu-lie-hua-fen-ii-zi-xu-lie-by-leetcode/ "Leetcode官方题解")

代码如下：

```python
class Solution:
    def numberOfArithmeticSlices(self, A: List[int]) -> int:
        n = len(A)
        res = 0
        counter = [{} for _ in range(n)]
        for i in range(n):
            for j in range(i):
                delta = A[i] - A[j]
                pre = counter[j].get(delta, 0)
                cur = counter[i].get(delta, 0)
                counter[i][delta] = pre + cur + 1
                res += pre
        return res
```



### 1218. 最长定差子序列

> [1218. 最长定差子序列](https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/ "1218. 最长定差子序列")

一、题目

给你一个整数数组`arr`和一个整数`difference`，请你找出`arr`中所有相邻元素之间的差等于给定`difference`的等差子序列，并返回其中最长的等差子序列的长度。

示例 1：

```纯文本
输入：arr = [1,2,3,4], difference = 1
输出：4
解释：最长的等差子序列是 [1,2,3,4]。
```

二、解析

> 参考 [Leetcode-太阳家的猫](https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/comments/158881 "Leetcode-太阳家的猫")

代码如下：

```python
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        dp = {}
        res = 0
        for a in arr:
            dp[a] = dp.get(a - difference, 0) + 1
            res = max(res, dp[a])
        return res
```



### 115. 不同的子序列

> [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/ "115. 不同的子序列")

一、题目

给定一个字符串 S 和一个字符串 T，计算在 S 的子序列中 T 出现的个数。

一个字符串的一个子序列是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）

题目数据保证答案符合 32 位带符号整数范围。

示例 1：

```纯文本
输入：S = "rabbbit", T = "rabbit"
输出：3
解释：

如下图所示, 有 3 种可以从 S 中得到 "rabbit" 的方案。
(上箭头符号 ^ 表示选取的字母)

rabbbit
^^^^ ^^
rabbbit
^^ ^^^^
rabbbit
^^^ ^^^
```

二、解析

> 参考 [Leetcode-powcai](https://leetcode-cn.com/problems/distinct-subsequences/solution/dong-tai-gui-hua-by-powcai-5/ "Leetcode-powcai")
> 参考 [Leetcode-云想衣裳花想容](https://leetcode-cn.com/problems/distinct-subsequences/solution/dong-tai-gui-hua-shou-dong-mo-ni-dong-tai-zhuan-yi/ "Leetcode-云想衣裳花想容")

用`dp[i][j]`表示对于字符串T中第`0`个字符到第`i`字符组成的子串出现在字符串S中第`0`个字符到第`j`字符组成的子串的子序列的次数。

转移方程为：

```纯文本
dp[i][j] = dp[i-1][j-1] + dp[i][j-1] if S[j] == T[i];
dp[i][j] = dp[i][j-1], if S[j] != T[i];
```

对于dp矩阵的第一行，T 为空，因为空集是所有字符串子集，所以第一行都是1。对于dp矩阵的第一列, S 为空，所以第一列都是0。

代码如下：

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        n1 = len(s)
        n2 = len(t)
        dp = [[0] * (n1 + 1) for _ in range(n2 + 1)]
        for j in range(n1 + 1):
            dp[0][j] = 1
        for i in range(1, n2 + 1):
            for j in range(1, n1 + 1):
                if t[i - 1] == s[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  + dp[i][j - 1]
                else:
                    dp[i][j] = dp[i][j - 1]
        return dp[-1][-1]
```



### 332. 零钱兑换

> [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/ "322. 零钱兑换")

一、题目

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回-1。

二、解析

代码如下：

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
                
        return dp[amount] if dp[amount] != float('inf') else -1
```



### 518. 零钱兑换 II

> [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/ "518. 零钱兑换 II")

一、题目

给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。

二、解析

代码如下：

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] += dp[x - coin]
        return dp[amount]
```



### 198. 打家劫舍

> [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/ "198. 打家劫舍")

一、题目

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。
给定一个代表每个房屋存放金额的非负整数数组，计算你**在不触动警报装置的情况下**，一夜之内能够偷窃到的最高金额。

二、解析

简单来说，在不偷取相邻房屋的情况下，如何收获最大？

用`dp[i]`表示前`i`间房屋能偷窃到的最高总金额，状态转移方程为`dp[i]=max(dp[i−2]+nums[i],dp[i−1])`。

1）空间复杂度O(n)。

代码如下：

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        elif len(nums) <= 2:
            return max(nums)
            
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

        return dp[-1]
```

2）空间复杂度O(1)

代码如下：

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        elif len(nums) <= 2:
            return max(nums)
            
        pre = nums[0]
        cur = max(nums[0], nums[1])
        for n in nums[2:]:
            pre, cur = cur, max(n + pre, cur)
            
        return cur
```



### 213. 打家劫舍II

> [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/ "213. 打家劫舍 II")

一、题目

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都**围成一圈**，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。
给定一个代表每个房屋存放金额的非负整数数组，计算你**在不触动警报装置的情况下**，能够偷窃到的最高金额。

二、解析

头尾两个房屋不能同时抢。要么不抢第一个，`nums[1:]`；要么不抢最后一个，`nums[:-1]`。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        def helper(nums):
            dp = [0] * len(nums)
            dp[0] = nums[0]
            dp[1] = max(nums[0], nums[1])
            for i in range(2, len(nums)):
                dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
            return dp[-1]

        if not nums:
            return 0
        elif len(nums) <= 2:
            return max(nums)

        return max(helper(nums[1:]), helper(nums[:-1]))
```



### 337. 打家劫舍III

> [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/ "337. 打家劫舍 III")

一、题目

在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 

如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

二、解析

> 参考 [Leetcode-reals](https://leetcode-cn.com/problems/house-robber-iii/solution/san-chong-fang-fa-jie-jue-shu-xing-dong-tai-gui-hu/ "Leetcode-reals")

1、递归，超时。

```python
class Solution:
    def rob(self, root: TreeNode) -> int:
        def helper(node):
            if not node:
                return 0

            money = node.val
            if node.left is not None:
                money += helper(node.left.left) + helper(node.left.right)

            if node.right is not None:
                money += helper(node.right.left) + helper(node.right.right)

            return max(money, helper(node.left) + helper(node.right))

        return helper(root)
```

2、带记忆的递归，通过。

```python
class Solution:
    def rob(self, root: TreeNode) -> int:
        def helper(node, memo):
            if not node:
                return 0
            if node in memo:
                return memo[node]

            money = node.val
            if node.left is not None:
                money += helper(node.left.left, memo) + helper(node.left.right, memo)
            if node.right is not None:
                money += helper(node.right.left, memo) + helper(node.right.right, memo)

            res = max(money, helper(node.left, memo) + helper(node.right, memo))
            memo[node] = res
            return res

        memo = {}
        return helper(root, memo)
```

3、终极解法，使用了二叉树的一种递归模板。

```python
class Solution:
    def rob(self, root: TreeNode) -> int:
        def helper(node):
            if not node:
                return 0, 0  ## max money if not rob cur node, max money if rob cur node
            
            left = helper(node.left)
            right = helper(node.right)
            not_rob_cur_node = max(left[0], left[1]) + max(right[0], right[1])
            rob_cur_node = left[0] + right[0] + node.val
            return not_rob_cur_node, rob_cur_node

        res = helper(root)
        return max(res)
```

### 121. 买卖股票的最佳时机

> [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/ "121. 买卖股票的最佳时机")

一、题目

给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。

注意：你不能在买入股票前卖出股票。

二、解析

1）前后两次遍历，略。

2）记录【今天之前买入的最小值】，【最大获利】。计算【今天之前最小值买入，今天价格卖出的获利】，和【最大获利】进行比较。比较【今天价格】和【今天之前买入的最小值】，如果前者小于后者，则更新后者。

代码如下：

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices or len(prices) <= 1:
            return 0
        
        min_price = prices[0]
        max_profit = 0
        for price in prices[1:]:
            if price < min_price:
                min_price = price
            max_profit = max(max_profit, price - min_price)
        
        return max_profit
```

### 122. 买卖股票的最佳时机 II

> [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/ "122. 买卖股票的最佳时机 II")

一、题目

给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

二、解析

可以同一天买入和卖出，先卖出，再买入。

1）直接方法。记录一个最小价格，后面的价格大于最小价格，就可以卖出获利。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices or len(prices) <= 1:
            return 0
        
        min_price = prices[0]
        max_profit = 0
        for price in prices[1:]:
            if min_price < price:
                max_profit += price - min_price
            min_price = price
        return max_profit
```

2）比较两个相邻的元素

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                max_profit += prices[i] - prices[i - 1]
        return max_profit
```

### 70. 爬楼梯

> [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/ "70. 爬楼梯")

一、题目

假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

注意：给定 n 是一个正整数。

二、解析

状态转移方程为`dp[i] = dp[i - 1] + dp[i - 2], i >= 2`。

1）空间复杂度O(n)

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 1:
            return n
        
        dp = [0] * n
        dp[0] = 1
        dp[1] = 2
        for i in range(2, n):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n - 1]
```

2）空间复杂度O(1)

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 1:
            return n
        
        pre, cur = 1, 2
        for i in range(2, n):
            pre, cur = cur, pre + cur
        return cur
```



### 746. 爬楼梯II - 使用最小花费爬楼梯

> [746. 使用最小花费爬楼梯](https://leetcode-cn.com/problems/min-cost-climbing-stairs/ "746. 使用最小花费爬楼梯")

一、题目

数组的每个索引作为一个阶梯，第i个阶梯对应着一个非负数的体力花费值cost\[i]（索引从0开始）。

每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。

您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。

二、解析

要走到顶楼，所有台阶都可能走。转移方程为：`dp[i] = min(dp[i-1], dp[i-2]) + cost[i], i>=2`。

最后选择最后一个台阶和倒数第二个台阶的较小值。

1）空间复杂度O(n)

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        dp = [0] * n
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2, n):
            dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]
        return min(dp[n - 2], dp[n - 1])
```

2）空间复杂度O(1)

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        pre, cur = cost[0], cost[1]
        for c in cost[2:]:
            pre, cur = cur, min(pre, cur) + c
        return min(pre, cur)
```

### 139. 单词拆分

> [139. 单词拆分](https://leetcode.cn/problems/word-break/)

一、题目

给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

示例 1：

```text
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
```

示例 2：

```text
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
注意，你可以重复使用字典中的单词。
```

示例 3：

```text
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

二、解析

> [https://leetcode.cn/problems/word-break/solution/dan-ci-chai-fen-by-leetcode-solution/](https://leetcode.cn/problems/word-break/solution/dan-ci-chai-fen-by-leetcode-solution/ "https://leetcode.cn/problems/word-break/solution/dan-ci-chai-fen-by-leetcode-solution/")

1）动态规划。

我们定义 dp\[i] 表示字符串 s 前 i 个字符组成的字符串 s\[0..i−1] 是否能被空格拆分成若干个字典中出现的单词。

从前往后计算考虑转移方程，每次转移的时候我们需要枚举包含位置 i−1 的最后一个单词，看它是否出现在字典中以及除去这部分的字符串是否合法即可。

转移方程为：dp\[i]=dp\[j] && check(s\[j..i−1])。

对于边界条件，我们定义 dp\[0]=true 表示空串且合法。

代码如下：

```go
func wordBreak(s string, wordDict []string) bool {
    wordDictSet := make(map[string]bool)
    for _, w := range wordDict {
        wordDictSet[w] = true
    }
    dp := make([]bool, len(s) + 1)
    dp[0] = true
    for i := 1; i <= len(s); i++ {
        for j := 0; j < i; j++ {
            if dp[j] && wordDictSet[s[j:i]] {
                dp[i] = true
                break
            }
        }
    }
    return dp[len(s)]
}
```

2）使用字典树 Trie 树。

TODO



### 140. 单词拆分II

> [140. 单词拆分 II](https://leetcode.cn/problems/word-break-ii/)

一、题目

给定一个字符串 s 和一个字符串字典 wordDict ，在字符串 s 中增加空格来构建一个句子，使得句子中所有的单词都在词典中。以任意顺序 返回所有这些可能的句子。

注意：词典中的同一个单词可能在分段中被重复使用多次。

示例 1：

```text
输入:s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
输出:["cats and dog","cat sand dog"]
```

示例 2：

```text
输入:s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]
输出:["pine apple pen apple","pineapple pen apple","pine applepen apple"]
解释: 注意你可以重复使用字典中的单词。
```

示例 3：

```text
输入:s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
输出:[]
```

二、解析

> [https://leetcode.cn/problems/word-break-ii/solution/dan-ci-chai-fen-ii-by-leetcode-solution/](https://leetcode.cn/problems/word-break-ii/solution/dan-ci-chai-fen-ii-by-leetcode-solution/ "https://leetcode.cn/problems/word-break-ii/solution/dan-ci-chai-fen-ii-by-leetcode-solution/")

记忆化搜索。

第 139 题可以使用动态规划的方法判断是否可以拆分，因此这道题也可以使用动态规划的思想。但是这道题如果使用自底向上的动态规划的方法进行拆分，则无法事先判断拆分的可行性，在不能拆分的情况下会超时。

对于字符串 s，如果某个前缀是单词列表中的单词，则拆分出该单词，然后对 s 的剩余部分继续拆分。如果可以将整个字符串 s 拆分成单词列表中的单词，则得到一个句子。在对 s 的剩余部分拆分得到一个句子之后，将拆分出的第一个单词（即 s 的前缀）添加到句子的头部，即可得到一个完整的句子。上述过程可以通过回溯实现。

假设字符串 s 的长度为 n，回溯的时间复杂度在最坏情况下高达 O(n^n)。时间复杂度高的原因是存在大量重复计算，可以通过记忆化的方式降低时间复杂度。

具体做法是，使用哈希表存储字符串 s 的每个下标和从该下标开始的部分可以组成的句子列表，在回溯过程中如果遇到已经访问过的下标，则可以直接从哈希表得到结果，而不需要重复计算。如果到某个下标发现无法匹配，则哈希表中该下标对应的是空列表，因此可以对不能拆分的情况进行剪枝优化。

还有一个可优化之处为使用哈希集合存储单词列表中的单词，这样在判断一个字符串是否是单词列表中的单词时只需要判断该字符串是否在哈希集合中即可，而不再需要遍历单词列表。

代码如下：

```go
func wordBreak(s string, wordDict []string) (sentences []string) {
    wordSet := map[string]struct{}{}
    for _, w := range wordDict {
        wordSet[w] = struct{}{}
    }

    n := len(s)
    dp := make([][][]string, n)
    var backtrack func(index int) [][]string
    backtrack = func(index int) [][]string {
        if dp[index] != nil {
            return dp[index]
        }
        wordsList := [][]string{}
        for i := index + 1; i < n; i++ {
            word := s[index:i]
            if _, has := wordSet[word]; has {
                for _, nextWords := range backtrack(i) {
                    wordsList = append(wordsList, append([]string{word}, nextWords...))
                }
            }
        }
        word := s[index:]
        if _, has := wordSet[word]; has {
            wordsList = append(wordsList, []string{word})
        }
        dp[index] = wordsList
        return wordsList
    }
    for _, words := range backtrack(0) {
        sentences = append(sentences, strings.Join(words, " "))
    }
    return
}
```


### 42. 接雨水

> [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

一、题目

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

示例 1：

![image_iYYDCFB97Z](https://raw.githubusercontent.com/shengchaohua/my-images/main/images/202311251618036.png)

```text
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。
```

示例 2：

```text
输入：height = [4,2,0,3,2,5]
输出：9
```

二、解析

> [https://leetcode.cn/problems/trapping-rain-water/solution/jie-yu-shui-by-leetcode-solution-tuvc/](https://leetcode.cn/problems/trapping-rain-water/solution/jie-yu-shui-by-leetcode-solution-tuvc/ "https://leetcode.cn/problems/trapping-rain-water/solution/jie-yu-shui-by-leetcode-solution-tuvc/")

可以使用动态规划、单调栈或双指针。

1）动态规划。

TODO



### 64. 最小路径和

> [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/)

一、题目

给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

示例 1：

![image_AEoRkvcfdI](https://raw.githubusercontent.com/shengchaohua/my-images/main/images/202311251618563.png)

```text
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
```

**示例 2：**

```text
输入：grid = [[1,2,3],[4,5,6]]
输出：12
```

二、解析

动态规划。

由于路径的方向只能是向下或向右，因此网格的第一行的每个元素只能从左上角元素开始向右移动到达，网格的第一列的每个元素只能从左上角元素开始向下移动到达，此时的路径是唯一的，因此每个元素对应的最小路径和即为对应的路径上的数字总和。

对于不在第一行和第一列的元素，可以从其上方相邻元素向下移动一步到达，或者从其左方相邻元素向右移动一步到达，元素对应的最小路径和等于其上方相邻元素与其左方相邻元素两者对应的最小路径和中的最小值加上当前元素的值。由于每个元素对应的最小路径和与其相邻元素对应的最小路径和有关，因此可以使用动态规划求解。

代码如下：

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        
        rows, columns = len(grid), len(grid[0])
        dp = [[0] * columns for _ in range(rows)]
        dp[0][0] = grid[0][0]
        for i in range(1, rows):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, columns):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        for i in range(1, rows):
            for j in range(1, columns):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        
        return dp[rows - 1][columns - 1]

```



### 329. 矩阵中的最长递增路径

> [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/ "329. 矩阵中的最长递增路径")

一、题目

给定一个整数矩阵，找出最长递增路径的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。你不能在对角线方向上移动或移动到边界外（即不允许环绕）。

二、解析

> 参考 [Leetcode-gyx2110](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/comments/96976 "Leetcode-gyx2110")

动态规划，从小到大进行访问。

代码如下：

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        m, n = len(matrix), len(matrix[0])
        nums = [[matrix[i][j], i, j] for i in range(m) for j in range(n)]
        nums.sort(key=lambda x: x[0])
        
        dp = [[1 for _ in range(n)] for _ in range(m)]
        for num, i, j in nums:
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                neigh_x = i + dx
                neigh_y = j + dy
                if 0 <= neigh_x < m and 0 <= neigh_y < n:
                    if matrix[i][j] > matrix[neigh_x][neigh_y]:
                        dp[i][j] = max(dp[i][j], dp[neigh_x][neigh_y] + 1)
        return max(dp[i][j] for i in range(m) for j in range(n))
```



### 562. 矩阵中最长的连续1线段

> [562. 矩阵中最长的连续1线段](https://leetcode-cn.com/problems/longest-line-of-consecutive-one-in-matrix/ "562. 矩阵中最长的连续1线段")

一、题目

给定一个01矩阵 M，找到矩阵中最长的连续1线段。这条线段可以是水平的、垂直的、对角线的或者反对角线的。

示例:

```纯文本
输入:
[[0,1,1,0],
 [0,1,1,0],
 [0,0,0,1]]
输出: 3
提示: 给定矩阵中的元素数量不会超过 10,000。
```

二、解析

> 参考 [562. 矩阵中最长的连续1线段（DP）](https://blog.csdn.net/qq_21201267/article/details/107205044 "562. 矩阵中最长的连续1线段（DP）")

代码如下：

```python
class Solution:
    def longestLine(M):
        if not M or not M[0]:
            return 0
        maxlen = 0
        m = len(M)
        n = len(M[0])
        dp_hori = [[0 for _ in range(n)] for _ in range(m)]
        dp_veti = [[0 for _ in range(n)] for _ in range(m)]
        dp_diag = [[0 for _ in range(n)] for _ in range(m)]
        dp_anti_diag = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if M[i][j] == 1:
                    dp_hori[i][j] = dp_hori[i - 1][j] + 1 if i - 1 >= 0 else 1
              dp_veti[i][j] = dp_veti[i][j - 1] + 1 if j - 1 >= 0 else 1
              dp_diag[i][j] = dp_diag[i - 1][j + 1] + 1  if i - 1 >= 0 and j + 1 < n else 1
              dp_anti_diag[i][j] = dp_anti_diag[i - 1][j - 1] + 1 if i - 1 >= 0 and j - 1 >= 0 else 1
              maxlen = max(maxlen, dp_hori[i][j], dp_veti[i][j], 
                           dp_diag[i][j], dp_anti_diag[i][j])
      return maxlen
```



### 221. 最大正方形

> [221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/ "221. 最大正方形")

一、题目

在一个由 0 和 1 组成的二维矩阵内，找到只包含 1 的最大正方形，并返回其面积。

二、解析

代码如下：

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0
        
        max_side = 0
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    max_side = max(max_side, dp[i][j])
        return max_side * max_side
```



### 1277. 统计全为 1 的正方形子矩阵

> [1277. 统计全为 1 的正方形子矩阵](https://leetcode-cn.com/problems/count-square-submatrices-with-all-ones/ "1277. 统计全为 1 的正方形子矩阵")

一、题目

给你一个 m \* n 的矩阵，矩阵中的元素不是 0 就是 1，请你统计并返回其中完全由 1 组成的 正方形 子矩阵的个数。

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/count-square-submatrices-with-all-ones/solution/tong-ji-quan-wei-1-de-zheng-fang-xing-zi-ju-zhen-2/ "Leetcode官方题解")

代码如下：

```python
class Solution:
    def countSquares(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        res = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 1:
                    if i == 0 or j == 0:
                        dp[i][j] = matrix[i][j]
                    else:
                        dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
                    res += dp[i][j]
        return res
```



### 85. 最大矩形

> [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/ "85. 最大矩形")

一、题目

给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

**示例 1：**

![image_kEr1lPIjle](https://raw.githubusercontent.com/shengchaohua/my-images/main/images/202311251618048.png)

```text
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：6
解释：最大矩形如上图所示。
```

二、解析

> <https://leetcode.cn/problems/maximal-rectangle/solution/zui-da-ju-xing-by-leetcode-solution-bjlu/>

单调栈。

代码如下：

```go
func maximalRectangle(matrix [][]byte) (ans int) {
    if len(matrix) == 0 {
        return
    }
    m, n := len(matrix), len(matrix[0])
    left := make([][]int, m)
    for i, row := range matrix {
        left[i] = make([]int, n)
        for j, v := range row {
            if v == '0' {
                continue
            }
            if j == 0 {
                left[i][j] = 1
            } else {
                left[i][j] = left[i][j-1] + 1
            }
        }
    }
    for j := 0; j < n; j++ { // 对于每一列，使用基于柱状图的方法
        up := make([]int, m)
        down := make([]int, m)
        stk := []int{}
        for i, l := range left {
            for len(stk) > 0 && left[stk[len(stk)-1]][j] >= l[j] {
                stk = stk[:len(stk)-1]
            }
            up[i] = -1
            if len(stk) > 0 {
                up[i] = stk[len(stk)-1]
            }
            stk = append(stk, i)
        }
        stk = nil
        for i := m - 1; i >= 0; i-- {
            for len(stk) > 0 && left[stk[len(stk)-1]][j] >= left[i][j] {
                stk = stk[:len(stk)-1]
            }
            down[i] = m
            if len(stk) > 0 {
                down[i] = stk[len(stk)-1]
            }
            stk = append(stk, i)
        }
        for i, l := range left {
            height := down[i] - up[i] - 1
            area := height * l[j]
            ans = max(ans, area)
        }
    }
    return
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

```



### 72. 编辑距离

> [https://leetcode.cn/problems/edit-distance/](https://leetcode.cn/problems/edit-distance/ "https://leetcode.cn/problems/edit-distance/")

一、题目

给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符

示例 1：

```text
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

示例 2：

```text
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

二、解析

动态规划。

题目给定了两个单词，设为 A 和 B，这样我们就能够六种操作方法。

但我们可以发现，如果我们有单词 A 和单词 B：

-   对单词 A 删除一个字符和对单词 B 插入一个字符是等价的。例如当单词 A 为 doge，单词 B 为 dog 时，我们既可以删除单词 A 的最后一个字符 e，得到相同的 dog，也可以在单词 B 末尾添加一个字符 e，得到相同的 doge；
-   同理，对单词 B 删除一个字符和对单词 A 插入一个字符也是等价的；
-   对单词 A 替换一个字符和对单词 B 替换一个字符是等价的。例如当单词 A 为 bat，单词 B 为 cat 时，我们修改单词 A 的第一个字母 b -> c，和修改单词 B 的第一个字母 c -> b 是等价的。

这样以来，本质不同的操作实际上只有三种：

-   单词 A 中插入一个字符；
-   在单词 B 中插入一个字符；
-   修改单词 A 的一个字符。

这样以来，我们就可以把原问题转化为规模较小的子问题。我们用 A = horse，B = ros 作为例子，来看一看是如何把这个问题转化为规模较小的若干子问题的。

-   单词 A 中插入一个字符：如果我们知道 horse 到 ro 的编辑距离为 a，那么显然 horse 到 ros 的编辑距离不会超过 a + 1。这是因为我们可以在 a 次操作后将 horse 和 ro 变为相同的字符串，只需要额外的 1 次操作，在单词 A 的末尾添加字符 s，就能在 a + 1 次操作后将 horse 和 ro 变为相同的字符串；
-   在单词 B 中插入一个字符：如果我们知道 hors 到 ros 的编辑距离为 b，那么显然 horse 到 ros 的编辑距离不会超过 b + 1，原因同上；
-   修改单词 A 的一个字符：如果我们知道 hors 到 ro 的编辑距离为 c，那么显然 horse 到 ros 的编辑距离不会超过 c + 1，原因同上。

那么从 horse 变成 ros 的编辑距离应该为 min(a + 1, b + 1, c + 1)。

注意：为什么我们总是在单词 A 和 B 的末尾插入或者修改字符，能不能在其它的地方进行操作呢？答案是可以的，但是我们知道，操作的顺序是不影响最终的结果的。例如对于单词 cat，我们希望在 c 和 a 之间添加字符 d 并且将字符 t 修改为字符 b，那么这两个操作无论为什么顺序，都会得到最终的结果 cdab。

你可能觉得 horse 到 ro 这个问题也很难解决。但是没关系，我们可以继续用上面的方法拆分这个问题，对于这个问题拆分出来的所有子问题，我们也可以继续拆分，直到：

-   字符串 A 为空，如从空字符串转换到 ro，显然编辑距离为字符串 B 的长度，这里是 2；
-   字符串 B 为空，如从 horse 转换到空字符串，显然编辑距离为字符串 A 的长度，这里是 5。

因此，我们就可以使用动态规划来解决这个问题了。我们用 D\[i]\[j] 表示 A 的前 i 个字母和 B 的前 j 个字母之间的编辑距离。

代码如下：

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)
        
        ## 有一个字符串为空串
        if n * m == 0:
            return n + m
        
        ## DP 数组
        D = [[0] * (m + 1) for _ in range(n + 1)]
        
        ## 边界状态初始化
        for i in range(n + 1):
            D[i][0] = i
        for j in range(m + 1):
            D[0][j] = j
        
        ## 计算所有 DP 值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = D[i - 1][j] + 1
                down = D[i][j - 1] + 1
                left_down = D[i - 1][j - 1] 
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                D[i][j] = min(left, down, left_down)
        
        return D[n][m]
```



### 91. 解码方法

> [91. 解码方法](https://leetcode-cn.com/problems/decode-ways/ "91. 解码方法")

一、题目

一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：

```纯文本
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
```

要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：

-   "AAJF" ，将消息分组为 (1 1 10 6)
-   "KJF" ，将消息分组为 (11 10 6)

注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。

给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。

题目数据保证答案肯定是一个 32 位 的整数。

示例 1：

```text
输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
```

示例 2：

```text
输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
```

示例 3：

```text
输入：s = "06"
输出：0
解释："06" 无法映射到 "F" ，因为存在前导零（"6" 和 "06" 并不等价）。
```

二、解析

用`dp[i]`表示字符串第`0`个字符到第`i`个字符组成的子串的解码的总数，`dp[0]`和`dp[1]`可以直接求出。状态转移方程参考代码。

代码如下：

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == "0":
            return 0
        if len(s) == 1:
            return 1
        
        n = len(s)
        dp = [0] * n
        dp[0] = 1
        num = int(s[0:2])
        if num == 10 or num == 20:
            dp[1] = 1
        elif 11 <= num <= 19 or 21 <= num <= 26:
            dp[1] = 2
        elif s[1] == "0" :
            dp[1] = 0
        else:
            dp[1] = 1

        for i in range(2, n):
            if s[i] != "0":
                dp[i] = dp[i - 1]
            num = int(s[i - 1:i + 1])
            if 10 <= num <= 26:
                dp[i] += dp[i - 2]

        return dp[n - 1]
```

### LCR 46. 把数字翻译成字符串

> [剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/ "剑指 Offer 46. 把数字翻译成字符串")

一、题目

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

二、解析

用`dp[i]`表示表示字符串第`0`个字符到第`i`个字符组成的子串的翻译的总数，`dp[0]`和`dp[1]`可以直接求出。状态转移方程参考代码。

代码如下：

```python
class Solution:
    def translateNum(self, num: int) -> int:
        if 0 <= num < 10:
            return 1
        str_num = str(num)
        n = len(str_num)
        dp = [1] * n
        if 10 <= int(str_num[:2]) <= 25:
            dp[1] = 2
        for i in range(2, n):
            if 10 <= int(str_num[i - 1:i + 1]) <= 25:
                dp[i] = dp[i - 2] + dp[i - 1]
            else:
                dp[i] = dp[i - 1]
        return dp[n - 1]
```



### 357. 计算各个位数不同的数字个数

> [357. 计算各个位数不同的数字个数](https://leetcode-cn.com/problems/count-numbers-with-unique-digits/ "357. 计算各个位数不同的数字个数")

一、题目

给定一个非负整数 n，计算各位数字都不同的数字 x 的个数，其中 0 ≤ x < 10n 。

示例:

```纯文本
输入: 2
输出: 91 
解释: 答案应为除去 11,22,33,44,55,66,77,88,99 外，在 [0,100) 区间内的所有数字。
```

二、解析

代码如下：

```python
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        if n == 0:
            return 1
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 10
        for i in range(2, n + 1):
            dp[i] = (dp[i - 1] - dp[i - 2]) * (10 - (i - 1)) + dp[i-1]
        return dp[n]
```



### 96. 不同的二叉搜索树

> [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/ "96. 不同的二叉搜索树")

一、题目

给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？

二、解析

代码如下：

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
        return dp[n]
```



## 其他

### 圆环回原点

> [https://mp.weixin.qq.com/s/NZPaFsFrTybO3K3s7p7EVg](https://mp.weixin.qq.com/s/NZPaFsFrTybO3K3s7p7EVg "https://mp.weixin.qq.com/s/NZPaFsFrTybO3K3s7p7EVg")

圆环上有10个点，编号为0\~9。从0点出发，每次可以逆时针和顺时针走一步，问走n步回到0点共有多少种走法。

示例：

```python
输入: 2
输出: 2
解释：有2种方案。分别是0->1->0和0->9->0
```

二、解析

动态规划。

如果你之前做过leetcode的70题爬楼梯，则应该比较容易理解：走n步到0的方案数=走n-1步到1的方案数+走n-1步到9的方案数。

因此，若设dp\[i]\[j]为从0点出发走i步到j点的方案数，则递推式为：

![](https://mmbiz.qpic.cn/mmbiz_png/oD5ruyVxxVGRJ4bSda4dThHBeSbNib3NpjEWPqmIgHluopXk7FBTby4zWaLlggUwIicicCaPHz4ISHSrWGZuibUhxQ/640?wx_fmt=png\&wxfrom=5\&wx_lazy=1\&wx_co=1)

ps:公式之所以取余是因为j-1或j+1可能会超过圆环0\~9的范围。

代码如下：

```python
class Solution:
    def backToOrigin(self,n):
        #点的个数为10
        length = 10
        dp = [[0 for i in range(length)] for j in range(n+1)]
        dp[0][0] = 1
        for i in range(1,n+1):
            for j in range(length):
                ## dp[i][j]表示从0出发，走i步到j的方案数
                dp[i][j] = dp[i-1][(j-1+length)%length] + dp[i-1][(j+1)%length]
        return dp[n][0]
```

[https://leetcode.cn/problems/edit-distance/](https://leetcode.cn/problems/edit-distance/ "https://leetcode.cn/problems/edit-distance/")
