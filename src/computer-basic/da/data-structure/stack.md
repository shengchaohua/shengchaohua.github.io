---
title: 栈
order: 3
---


## 基础

栈是一种动态集合。栈的特点是先进后出，最先进入的元素最后被删除。


## Leetcode 编程题

### 232. 用栈实现队列

> [232. 用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks)

一、题目

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：

实现 MyQueue 类：

-   void push(int x) 将元素 x 推到队列的末尾
-   int pop() 从队列的开头移除并返回元素
-   int peek() 返回队列开头的元素
-   boolean empty() 如果队列为空，返回 true ；否则，返回 false

说明：

-   你 只能 使用标准的栈操作 —— 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
-   你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。

二、解析

代码如下：

```python
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.stack1.append(x)
        

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        if len(self.stack2) == 0:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        if len(self.stack2) == 0:
            return -1
        return self.stack2.pop()

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        if len(self.stack2) == 0:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        if len(self.stack2) == 0:
            return -1
        return self.stack2[-1]
        

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return not self.stack1 and not self.stack2
```



### 224. 基本计算器

> [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/ "224. 基本计算器")

一、题目

实现一个基本的计算器来计算一个简单的字符串表达式的值。

字符串表达式可以包含左括号`(`，右括号`)`，加号`+`，减号`-`，非负整数和空格。

二、解析

> [官方题解](https://leetcode.cn/problems/basic-calculator/solutions/646369/ji-ben-ji-suan-qi-by-leetcode-solution-jvir/)

代码如下：

```python
class Solution:
    def calculate(self, s: str) -> int:
        ops = [1]
        sign = 1

        ret = 0
        n = len(s)
        i = 0
        while i < n:
            if s[i] == ' ':
                i += 1
            elif s[i] == '+':
                sign = ops[-1]
                i += 1
            elif s[i] == '-':
                sign = -ops[-1]
                i += 1
            elif s[i] == '(':
                ops.append(sign)
                i += 1
            elif s[i] == ')':
                ops.pop()
                i += 1
            else:
                num = 0
                while i < n and s[i].isdigit():
                    num = num * 10 + ord(s[i]) - ord('0')
                    i += 1
                ret += num * sign
        return ret
```



### 946. 验证栈序列

> [946. 验证栈序列](https://leetcode-cn.com/problems/validate-stack-sequences/ "946. 验证栈序列")

一、题目

给定 pushed 和 popped 两个序列，每个序列中的 值都不重复，只有当它们可能是在最初空栈上进行的推入 push 和弹出 pop 操作序列的结果时，返回 true；否则，返回 false。

二、解析

使用栈模拟。

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        index = 0
        for num in pushed:
            stack.append(num)
            while stack and stack[-1] == popped[index]:
                stack.pop()
                index += 1
                
        return not stack
```



### 155. 最小栈

> [155. 最小栈](https://leetcode-cn.com/problems/min-stack/ "155. 最小栈")

二、解析

增加一个辅助栈，用来保存当前栈内的最小元素。

代码如下：

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.aux_stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        if len(self.aux_stack) == 0:
            self.aux_stack.append(x)
        else:
            cur_min = self.aux_stack[-1]
            self.aux_stack.append(min(cur_min, x))

    def pop(self) -> None:
        if self.stack:
            self.stack.pop()
            self.aux_stack.pop()

    def top(self) -> int:
        if self.stack:
            return self.stack[-1]
        return 0

    def min(self) -> int:
        if self.aux_stack:
            return self.aux_stack[-1]

```



### 1003. 检查替换后的词是否有效

> [1003. 检查替换后的词是否有效](https://leetcode-cn.com/problems/check-if-word-is-valid-after-substitutions/ "1003. 检查替换后的词是否有效")

一、题目

给你一个字符串 s ，请你判断它是否 有效 。
字符串 s 有效 需要满足：假设开始有一个空字符串 t = "" ，你可以执行 任意次 下述操作将 t 转换为 s ：

将字符串 "abc" 插入到 t 中的任意位置。形式上，t 变为 tleft + "abc" + tright，其中 t == tleft + tright 。注意，tleft 和 tright 可能为 空 。
如果字符串 s 有效，则返回 true；否则，返回 false。

二、解析

使用栈模拟。

代码如下：

```python
class Solution:
    def isValid(self, S: str) -> bool:
        stack = []
        for ch in S:
            if ch == 'a':
                stack.append(ch)
            elif ch == 'b':
                if stack and stack[-1] == 'a':
                    stack.append(ch)
                else:
                    return False
            elif ch == 'c':
                if stack and stack[-1] == 'b':
                    stack.pop()
                    stack.pop()
                else:
                    return False

        return len(stack) == 0
```



### 20. 有效的括号（括号匹配）

> [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/ "20. 有效的括号")

一、题目

给定一个只包括'('，')'，'{'，'}'，'\['，']'的字符串，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

二、解析

代码如下：

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for c in s:
            if c == '(' or c == '[' or c == '{':
                stack.append(c)
            else:
                if not stack:
                    return False             
                elif stack[-1] == '(' and c == ')' or stack[-1] == '[' and c == ']' or stack[-1] == '{' and c == '}':
                    stack.pop()
                else:
                    return False
        return len(stack) == 0
```



### 856. 括号的分数

> [856. 括号的分数](https://leetcode-cn.com/problems/score-of-parentheses/ "856. 括号的分数")

一、题目

给定一个平衡括号字符串 S，按下述规则计算该字符串的分数：

-   () 得 1 分。
-   AB 得 A + B 分，其中 A 和 B 是平衡括号字符串。
-   (A) 得 2 \* A 分，其中 A 是平衡括号字符串。

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/score-of-parentheses/solution/gua-hao-de-fen-shu-by-leetcode/ "Leetcode官方题解")

使用栈模拟，遇到左括号就添加一个0，遇到右括号就弹出一个元素，并修改最后一个元素。

代码如下：

```python
class Solution:
    def scoreOfParentheses(self, S):
        stack = [0]  ## The score of the current frame
        for x in S:
            if x == '(':
                stack.append(0)
            else:
                v = stack.pop()
                stack[-1] += max(2 * v, 1)

        return stack[0]
```



### 921. 使括号有效的最少添加

> [921. 使括号有效的最少添加](https://leetcode-cn.com/problems/minimum-add-to-make-parentheses-valid/ "921. 使括号有效的最少添加")

一、题目

给定一个由`(`和`)`括号组成的字符串S，我们需要添加最少的括号（ `(`或是`)`，可以在任何位置），以使得到的括号字符串有效。

二、解析

使用栈模拟。代码如下：

```python
class Solution:
    def minAddToMakeValid(self, S: str) -> int:
        stack = []
        num = 0
        for c in S:
            if c == '(':
                stack.append(c)
            elif c == ')' :
                if stack and stack[-1] == '(':
                    stack.pop()
                else:
                    num += 1
        
        return num + len(stack)
```



### 32. 最长有效括号

> [32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/ "32. 最长有效括号")

一、题目&#x20;

给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。

二、解析

> [官方题解](https://leetcode-cn.com/problems/longest-valid-parentheses/solution/zui-chang-you-xiao-gua-hao-by-leetcode-solution/ "Leetcode官方题解")

1）始终保持栈底元素为当前已经遍历过的元素中「最后一个没有被匹配的右括号的下标」；

2）栈初始化包含一个`-1`，栈里其他元素用于维护左括号的下标：

-   对于遇到的每个'('，将它的下标放入栈中；
-   对于遇到的每个')'，先弹出栈顶元素，表示匹配了栈顶元素，弹出栈之后：
    -   如果栈为空，说明当前的右括号为没有被匹配的右括号，我们将其下标放入栈中来更新我们之前提到的「最后一个没有被匹配的右括号的下标」
    -   如果栈不为空，当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度」

代码如下：

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        res = 0
        stack = [-1]
        for i, ch in enumerate(s):
            if ch == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    res = max(res, i - stack[-1])
        return res
```



### 739. 每日温度

> [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/ "739. 每日温度")

一、题目

请根据每日`气温`列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。

例如，给定一个列表 temperatures = \[73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 \[1, 1, 4, 2, 1, 1, 0, 0]。

二、解析

使用栈模拟。栈存放了下标。

代码如下：

```python
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        res = [0] * len(T)
        stack = []
        for i, temper in enumerate(T):
            while stack and T[stack[-1]] < temper:
                idx = stack.pop()
                res[idx] = i - idx
            stack.append(i)

        return res
```



### 496. 下一个更大元素 I

> [496. 下一个更大元素 I](https://leetcode-cn.com/problems/next-greater-element-i/ "496. 下一个更大元素 I")

一、题目

给定两个**没有重复元素**的数组`nums1`和`nums2`，其中`nums1`是`nums2`的子集。找到`nums1`中每个元素在`nums2`中的下一个比其大的值。

示例 1：

```纯文本
输入：nums1 = [4,1,2], nums2 = [1,3,4,2].
输出：[-1,3,-1]
解释：nums1 中每个值的下一个更大元素如下所述：
- 4 ，用加粗斜体标识，nums2 = [1,3,4,2]。不存在下一个更大元素，所以答案是 -1 。
- 1 ，用加粗斜体标识，nums2 = [1,3,4,2]。下一个更大元素是 3 。
- 2 ，用加粗斜体标识，nums2 = [1,3,4,2]。不存在下一个更大元素，所以答案是 -1 。
```

示例 2：

```纯文本
输入：nums1 = [2,4], nums2 = [1,2,3,4].
输出：[3,-1]
解释：nums1 中每个值的下一个更大元素如下所述：
- 2 ，用加粗斜体标识，nums2 = [1,2,3,4]。下一个更大元素是 3 。
- 4 ，用加粗斜体标识，nums2 = [1,2,3,4]。不存在下一个更大元素，所以答案是 -1 。
```

提示：

-   1 <= nums1.length <= nums2.length <= 1000
-   0 <= nums1\[i], nums2\[i] <= 104
-   nums1和nums2中所有整数 互不相同
-   nums1 中的所有整数同样出现在 nums2 中

二、解析

先只考虑第二个数组即可。

代码如下：

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        seen = {}
        for num in nums2:
            while stack and stack[-1] < num:
                temp = stack.pop()
                seen[temp] = num
            stack.append(num)
        for num in stack:
            seen[num] = - 1

        return [seen[num] for num in nums1]
```



### 503. 下一个更大元素 II

> [503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/ "503. 下一个更大元素 II")

一、题目

给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。数字 x 的下一个更大的元素是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1。

**示例 1:**

```
输入: nums = [1,2,1]
输出: [2,-1,2]
解释: 第一个 1 的下一个更大的数是 2；
数字 2 找不到下一个更大的数； 
第二个 1 的下一个最大的数需要循环搜索，结果也是 2。
```

二、解析

遍历两次。

代码如下：

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        stack = []
        seen = {}
        double_range = list(range(len(nums))) * 2
        for i in double_range:
            while stack and nums[stack[-1]] < nums[i]:
                idx = stack.pop()
                if idx not in seen:
                    seen[idx] = nums[i]
            stack.append(i)

        for i in stack:
            if i not in seen:
                seen[i] = -1

        return [seen[i] for i in range(len(nums))]
```
