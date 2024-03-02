---
title: 贪心
order: 4
---



## Leetcode 编程题

### 55. 跳跃游戏

> [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/ "55. 跳跃游戏")

一、题目

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个位置。

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/jump-game/solution/tiao-yue-you-xi-by-leetcode-solution/ "Leetcode官方题解")

代码如下：

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        rightmost = 0
        for i in range(n):
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= n - 1:
                    return True
        return False
```



### 45. 跳跃游戏 II

> [45. 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/ "45. 跳跃游戏 II")

一、题目

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

说明: 假设你总是可以到达数组的最后一个位置。

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/jump-game-ii/solution/tiao-yue-you-xi-ii-by-leetcode-solution/ "Leetcode官方题解")

1）反向查找出发位置，时间复杂度$O(n^2)$。

Python 实现会超时，所以给出了 Java 实现。

```java
class Solution {
    public int jump(int[] nums) {
        int position = nums.length - 1;
        int steps = 0;
        while (position > 0) {
            for (int i = 0; i < position; i++) {
                if (i + nums[i] >= position) {
                    position = i;
                    steps++;
                    break;
                }
            }
        }
        return steps;
    }
}
```

2）正向查找可到达的最大位置，时间复杂度$O(n)$。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        max_pos = 0
        cur_end = 0
        steps = 0
        for i in range(n - 1):
            if max_pos >= i:
                max_pos = max(max_pos, i + nums[i])
                if i == cur_end:
                    cur_end = max_pos
                    steps += 1
        return steps
```



### 991. 坏了的计算器

> [991. 坏了的计算器](https://leetcode-cn.com/problems/broken-calculator/ "991. 坏了的计算器")

一、题目

在显示着数字的坏计算器上，我们可以执行以下两种操作：

-   双倍（Double）：将显示屏上的数字乘 2；
-   递减（Decrement）：将显示屏上的数字减 1 。

最初，计算器显示数字 X。
返回显示数字 Y 所需的最小操作数。

二、解析

逆向思维。从大的数字到小的数字比较简单，偶数就除以2，奇数就加1。

代码如下：

```python
class Solution:
    def brokenCalc(self, X: 'int', Y: 'int') -> 'int':
        res = 0
        while Y > X:
            res += 1
            if Y % 2 == 1:
                Y += 1
            else:
                Y //= 2

        return res + X - Y
```



### 452. 用最少数量的箭引爆气球

> [452. 用最少数量的箭引爆气球](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/)

一、题目

有一些球形气球贴在一堵用 XY 平面表示的墙面上。墙面上的气球记录在整数数组 points ，其中points\[i] = \[xstart, xend] 表示水平直径在 xstart 和 xend之间的气球。你不知道气球的确切 y 坐标。

一支弓箭可以沿着 x 轴从不同点 完全垂直 地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被 引爆 。可以射出的弓箭的数量 没有限制 。 弓箭一旦被射出之后，可以无限地前进。

给你一个数组 points ，返回引爆所有气球所必须射出的 最小 弓箭数 。

示例1：

```text
输入：points = [[10,16],[2,8],[1,6],[7,12]]
输出：2
解释：气球可以用2支箭来爆破:
-在x = 6处射出箭，击破气球[2,8]和[1,6]。
-在x = 11处发射箭，击破气球[10,16]和[7,12]。
```

**示例 2：**

```text
输入：points = [[1,2],[3,4],[5,6],[7,8]]
输出：4
解释：每个气球需要射出一支箭，总共需要4支箭。
```

二、解析

> [官方题解](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/solutions/494515/yong-zui-shao-shu-liang-de-jian-yin-bao-qi-qiu-1-2/)

排序加贪心。

代码如下：

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points:
            return 0
        
        points.sort(key=lambda balloon: balloon[1])
        pos = points[0][1]
        ans = 1
        for balloon in points:
            if balloon[0] > pos:
                pos = balloon[1]
                ans += 1
        
        return ans

```



### 135. 分发糖果

> [135. 分发糖果](https://leetcode.cn/problems/candy/)

一、题目

n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。

你需要按照以下要求，给这些孩子分发糖果：

-   每个孩子至少分配到 1 个糖果。
-   相邻两个孩子评分更高的孩子会获得更多的糖果。

请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。

示例 1：

```text
输入：ratings = [1,0,2]
输出：5
解释：你可以分别给第一个、第二个、第三个孩子分发 2、1、2 颗糖果。
```

示例 2：

```text
输入：ratings = [1,2,2]
输出：4
解释：你可以分别给第一个、第二个、第三个孩子分发 1、2、1 颗糖果。
  第三个孩子只得到 1 颗糖果，这满足题面中的两个条件。
```

提示：

-   n == ratings.length
-   1 <= n <= 2 \* 104
-   0 <= ratings\[i] <= 2 \* 104

二、解析

> [https://leetcode.cn/problems/candy/solution/fen-fa-tang-guo-by-leetcode-solution-f01p/](https://leetcode.cn/problems/candy/solution/fen-fa-tang-guo-by-leetcode-solution-f01p/ "https://leetcode.cn/problems/candy/solution/fen-fa-tang-guo-by-leetcode-solution-f01p/")

1）方法一，两次遍历。

我们可以将「相邻的孩子中，评分高的孩子必须获得更多的糖果」这句话拆分为两个规则，分别处理。

左规则：当 ratings\[i−1]\<ratings\[i] 时，i 号学生的糖果数量将比 i−1 号孩子的糖果数量多。

右规则：当 ratings\[i]>ratings\[i+1] 时，i 号学生的糖果数量将比 i+1 号孩子的糖果数量多。

我们遍历该数组两次，处理出每一个学生分别满足左规则或右规则时，最少需要被分得的糖果数量。每个人最终分得的糖果数量即为这两个数量的最大值。

具体地，以左规则为例：我们从左到右遍历该数组，假设当前遍历到位置 i，如果有 ratings\[i−1]\<ratings\[i] 那么 i 号学生的糖果数量将比 i−1 号孩子的糖果数量多，我们令 left\[i]=left\[i−1]+1 即可，否则我们令 left\[i]=1。

在实际代码中，我们先计算出左规则 left 数组，在计算右规则的时候只需要用单个变量记录当前位置的右规则，同时计算答案即可。

代码如下：

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        left = [0] * n
        for i in range(n):
            if i > 0 and ratings[i] > ratings[i - 1]:
                left[i] = left[i - 1] + 1
            else:
                left[i] = 1
        
        right = ret = 0
        for i in range(n - 1, -1, -1):
            if i < n - 1 and ratings[i] > ratings[i + 1]:
                right += 1
            else:
                right = 1
            ret += max(left[i], right)
        
        return ret

```

2）方法二，常数空间遍历。

TODO。
