---
title: 字符串
order: 6
---

<!-- more -->

## 基础

字符串是指一个或多个字符的序列。

### 字符和数字转换

Python：

```python
a = "1"
aInt = ord(a) - ord("0") # ord returns the ascii code of a char
print(ord(a), ord("0"), aInt) # 49 48 1

b = 1
bStr = str(b) # bStr = "1"
print(b, bStr) # 1 "1"
```

Golang：

```go
a := '1'
aInt := rune(a) - rune('0') // rune returns the unicode of a char
fmt.Println(rune(a), rune('0'), aInt) // 49 48 1

b := 1
bStr := rune(b) + rune('0')
fmt.Println(b, bStr) // 1 49
```




## Leetcode 编程题

### 28. 实现 strStr() - 字符串搜索

> [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/ "28. 实现 strStr()")

一、题目

实现 strStr() 函数。

给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回 -1。

二、解析

1）暴力搜索

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle:
            return 0
        if not haystack:
            return -1
        
        for i in range(len(haystack) - len(needle) + 1):
            for j in range(len(needle)):
                if haystack[i + j] != needle[j]:
                    break
            else:
                return i
        
        return -1
```

2）KMP

> 参考 [小白之KMP算法详解及python实现](https://blog.csdn.net/weixin_39561100/article/details/80822208 "小白之KMP算法详解及python实现")

对于一个长度为`N`的字符串`S`：

-   前缀为第`0`个字符`S[0]`到第`j`个字符`S[j]`组成的子串，其中`0<=j<=N-1`。
-   后缀为第`j`个字符`S[j]`到第`N-1`个字符`S[N-1]`组成的子串，其中`0<=j<=N-1`。
-   真前（后）缀表示不包括字符串本身的前（后）缀。

对于字符串`abcab`：

-   前缀包括`a,ab,abc,abca,abcab`，后缀包括`abcab,bcab,cab,ab,b`。
-   真前缀包括`a,ab,abc,abca`，真后缀包括`bcab,cab,ab,b`。

对于一个长度为`N`的字符串`S`，它的 next 指针数组的长度与字符串长度相等，其中第`i`个元素表示【该字符串第`0`个字符`S[0]`到第`i`个字符`S[i]`组成的子串】的相同真前后缀的最大长度。

对于字符串`abcab`，next 指针数组为`[0,0,0,1,2]`。

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        def gen_next(string):
            N = len(string)
            next_pointer = [0] * N
            left = 0
            cur = 1
            while cur < N:
                if string[cur] == string[left]:
                    next_pointer[cur] = left + 1
                    left += 1
                    cur += 1
                elif left == 0:
                    next_pointer[cur] = 0
                    cur += 1
                else: ## left != 0
                    left = next_pointer[left - 1]
            return next_pointer

        nextp = gen_next(needle)
        m = len(haystack)
        n = len(needle)
        i, j = 0, 0
        while i < m and j < n:
            if haystack[i] == needle[j]:
                i += 1
                j += 1
            elif j == 0:
                i += 1
            else:
                j = nextp[j - 1]
        if j == n:
            return i - j
        return -1
```



### 8. 字符串转换整数 (atoi)

> [8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/ "8. 字符串转换整数 (atoi)")

一、题目

请你来实现一个 atoi 函数，使其能将字符串转换成整数。

二、解析

不要使用`strip`函数。

代码如下：

```python
class Solution:
    def myAtoi(self, string: str) -> int:
        if not string or string[0].isalpha():
            return 0
        
        begin = 0
        res = 0
        sign = 1
        while begin < len(string) and string[begin] == " ":
            begin += 1
        
        if begin >= len(string):
            return 0
        
        if string[begin] == "+":
            begin += 1
        elif string[begin] == "-":
            sign = -1
            begin += 1
        
        while begin < len(string) and string[begin].isdigit():
            res = res * 10 + int(string[begin])
            begin += 1
        
        res = res * sign
        if res > 2 ** 31 - 1:
            return 2 ** 31 - 1
        elif res < -2 ** 31:
            return -2 ** 31
        return res
```



### 3. 无重复字符的最长子串

> [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/ "3. 无重复字符的最长子串")

一、题目

给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

**示例 1:**

```纯文本
输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

示例 2:

```text
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

示例 3:

```text
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

**提示：**

-   `0 <= s.length <= 5 * 104`
-   `s` 由英文字母、数字、符号和空格组成

二、解析

> 参考 [Leetcode-powcai](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/solution/hua-dong-chuang-kou-by-powcai/ "Leetcode-powcai")

滑动窗口。使用左右两个指针，右指针每次往右走，如果满足条件（或不满足条件），左指针就一直收敛。 

代码如下：

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s: 
            return 0
        
        res = 0
        left = 0
        seen = set()
        for right in range(len(s)):
            while s[right] in seen:
                seen.remove(s[left])
                left += 1
            seen.add(s[right])
            res = max(res, right - left + 1)
        return res
```



### 1044. 最长重复子串

> [1044. 最长重复子串](https://leetcode.cn/problems/longest-duplicate-substring/)

一、题目

给你一个字符串 s ，考虑其所有 重复子串 ：即 s 的（连续）子串，在 s 中出现 2 次或更多次。这些出现之间可能存在重叠。

返回 任意一个 可能具有最长长度的重复子串。如果 s 不含重复子串，那么答案为 "" 。

示例 1：

```纯文本
输入：s = "banana"
输出："ana"
```

示例 2：

```text
输入：s = "abcd"
输出：""
```

提示：

-   2 <= s.length <= 3 \* 104
-   s 由小写英文字母组成

二、解析

1）暴力法。

> [[Python] 今天这题太难了我真不会](https://leetcode.cn/problems/longest-duplicate-substring/solutions/1172467/python-jin-tian-zhe-ti-tai-nan-liao-wo-z-6wbs/)

代码如下：

```python
class Solution:
    def longestDupSubstring(self, s: str) -> str:
        ans = ""
        for i in range(len(s)):
            length = len(ans) ## at least length
            while s[i:i+length+1] in s[i+1:]:
                ans = s[i:i+length + 1]
                length = len(ans)
        return ans
```

2）二分法 + Rabin-Karp 字符串编码

> [官方题解](https://leetcode.cn/problems/longest-duplicate-substring/solution/zui-chang-zhong-fu-zi-chuan-by-leetcode-0i9rd/)

二分法用来从判断合适的长度。举个例子，如果存在长度为 5 的子串为最长重复子串，那么我们就考虑长度大于 5 的子串是否有可能，而不需要考虑长度小于 5 的子串了。

Rabin-Karp 字符串编码用来对子串进行编码，用来判断是否有长度为 L 的重复子串。

代码如下：

```python
class Solution:
    def longestDupSubstring(self, s: str) -> str:
        ## 生成两个进制
        a1, a2 = random.randint(26, 100), random.randint(26, 100)
        ## 生成两个模
        mod1, mod2 = random.randint(10**9+7, 2**31-1), random.randint(10**9+7, 2**31-1)
        n = len(s)
        ## 先对所有字符进行编码
        arr = [ord(c)-ord('a') for c in s]
        ## 二分查找的范围是[1, n-1]
        l, r = 0, n
        length, start = 0, -1
        while l < r:
            m = l + (r - l) // 2
            idx = self.check(arr, m, a1, a2, mod1, mod2)
            ## 有重复子串，移动左边界
            if idx != -1:
                l = m + 1
                length = m
                start = idx
            ## 无重复子串，移动右边界
            else:
                r = m
        return s[start:start+length] if start != -1 else ""

    def check(self, arr, m, a1, a2, mod1, mod2):
        n = len(arr)
        aL1, aL2 = pow(a1, m) % mod1, pow(a2, m) % mod2
        h1, h2 = 0, 0
        for i in range(m):
            h1 = (h1 * a1 + arr[i]) % mod1
            h2 = (h2 * a2 + arr[i]) % mod2
        ## 存储一个编码组合是否出现过
        seen = {(h1, h2)}
        for start in range(1, n - m + 1):
            h1 = (h1 * a1 - arr[start - 1] * aL1 + arr[start + m - 1]) % mod1
            h2 = (h2 * a2 - arr[start - 1] * aL2 + arr[start + m - 1]) % mod2
            ## 如果重复，则返回重复串的起点
            if (h1, h2) in seen:
                return start
            seen.add((h1, h2))
        ## 没有重复，则返回-1
        return -1
```

复杂度分析：

-   时间复杂度：`O(nlogn)`
-   空间复杂度：`O(n)`



### 76. 最小覆盖子串

> [76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)

一、题目

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

注意：

-   对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
-   如果 s 中存在这样的子串，我们保证它是唯一的答案。

示例 1：

```text
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
```

示例 2：

```text
输入：s = "a", t = "a"
输出："a"
解释：整个字符串 s 是最小覆盖子串。
```

示例 3:

```text
输入: s = "a", t = "aa"
输出: ""
解释: t 中两个字符 'a' 均应包含在 s 的子串中，
因此没有符合条件的子字符串，返回空字符串。
```

二、解析

> [官方题解](https://leetcode.cn/problems/minimum-window-substring/solution/zui-xiao-fu-gai-zi-chuan-by-leetcode-solution/)

滑动窗口。使用左右两个指针，右指针每次往右走，如果满足条件（或不满足条件），左指针就一直收敛。

在滑动窗口类型的问题中都会有两个指针，一个用于「延伸」现有窗口的 right 指针，和一个用于「收缩」窗口的 left 指针。在任意时刻，只有一个指针运动，而另一个保持静止。我们在 s 上滑动窗口，通过移动 right 指针不断扩张窗口。当窗口包含全部所需的字符后，如果能收缩，我们就收缩窗口直到得到最小窗口。

Python 代码如下：

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        freq = {}
        for c in t:
            if c not in freq:
                freq[c] = 0
            freq[c] += 1

        sLen = len(s)
        maxLen = float("inf")
        res_left = res_right = -1
        counts = {}

        def check():
            for k in freq:
                if k not in counts or counts[k] < freq[k]:
                    return False
            return True

        left, right = 0, 0
        while right < sLen:
            right_char = s[right]
            if right_char in freq and freq[right_char] > 0:
                if right_char not in counts:
                    counts[right_char] = 0
                counts[right_char] += 1
            while check() and left <= right:
                if right - left + 1 < maxLen:
                    maxLen = right - left + 1
                    res_left, res_right = left, left + maxLen
                if s[left] in freq:
                    counts[s[left]] -= 1
                left += 1
            right += 1
        return s[res_left:res_right]
```

Golang 代码如下：

```go
func minWindow(s string, t string) string {
    ori, cnt := map[byte]int{}, map[byte]int{}
    for i := 0; i < len(t); i++ {
        ori[t[i]]++
    }

    sLen := len(s)
    len := math.MaxInt32
    ansL, ansR := -1, -1

    check := func() bool {
        for k, v := range ori {
            if cnt[k] < v {
                return false
            }
        }
        return true
    }
    for l, r := 0, 0; r < sLen; r++ {
        if ori[s[r]] > 0 {
            cnt[s[r]]++
        }
        for check() && l <= r {
            if (r - l + 1 < len) {
                len = r - l + 1
                ansL, ansR = l, l + len
            }
            if _, ok := ori[s[l]]; ok {
                cnt[s[l]] -= 1
            }
            l++
        }
    }
    if ansL == -1 {
        return ""
    }
    return s[ansL:ansR]
}
```



### 316. 去除重复字母

> [316. 去除重复字母](https://leetcode-cn.com/problems/remove-duplicate-letters/ "316. 去除重复字母")

一、题目

给你一个仅包含小写字母的字符串，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证返回结果的字典序最小（要求不能打乱其他字符的相对位置）。

示例 1:

```纯文本
输入: "bcabc"
输出: "abc"
```

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/remove-duplicate-letters/solution/qu-chu-zhong-fu-zi-mu-by-leetcode/ "Leetcode官方题解")

使用栈和哈希表。 栈里面保存中间结果，哈希表保存每个字母最后一次出现的位置。

代码如下：

```python
class Solution:
    def removeDuplicateLetters(self, s) -> int:
        stack = []
        seen = set()
        last_occurrence = {c: i for i, c in enumerate(s)}
        for i, c in enumerate(s):
            if c not in seen:
                while stack and c < stack[-1] and i < last_occurrence[stack[-1]]:
                    seen.discard(stack.pop())
                seen.add(c)
                stack.append(c)
        return ''.join(stack)
```



### 1081. 不同字符的最小子序列

> [1081. 不同字符的最小子序列](https://leetcode-cn.com/problems/smallest-subsequence-of-distinct-characters/ "1081. 不同字符的最小子序列")

和【316. 去除重复字母】相同。



### 12. 整数转罗马数字

> [12. 整数转罗马数字](https://leetcode-cn.com/problems/integer-to-roman/ "12. 整数转罗马数字")

一、题目

给定一个整数，将其转为罗马数字。输入确保在 1 到 3999 的范围内。

二、解析

代码如下：

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        roman_numbers = {1:'I', 4:'IV', 5:'V', 9:'IX',
                         10:'X', 40: 'XL', 50:'L', 90:'XC', 
                         100:'C', 400:'CD', 500:'D', 900:'CM', 
                         1000:'M'}
        res = []
        for d in sorted(roman_numbers.keys(), reverse=True):
            (r, num) = divmod(num, d)
            if r == 0:
                continue
            res.append(roman_numbers[d]*r)
        return ''.join(res)
```



### 13. 罗马数字转整数

> [13. 罗马数字转整数](https://leetcode-cn.com/problems/roman-to-integer/ "13. 罗马数字转整数")

一、题目

罗马数字包含以下七种字符:`I`，`V`，`X`，`L`，`C`，`D`和`M`。
给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

二、解析

代码如下：

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        res = 0
        for i in range(len(s)):
            if i < len(s) - 1 and roman_map[s[i]] < roman_map[s[i + 1]]:
                res -= roman_map[s[i]]
            else:
                res += roman_map[s[i]]    
        return res
```



### 556. 下一个更大元素 III

> [556. 下一个更大元素 III](https://leetcode-cn.com/problems/next-greater-element-iii/ "556. 下一个更大元素 III")

一、题目

给定一个32位正整数 n，你需要找到最小的 32 位整数，其与 n 中存在的位数完全相同，并且其值大于n。如果不存在这样的32位整数，则返回-1。

二、解析

把数字转成字符数组，按字母表的顺序找到可以交换的位置。

代码如下：

```python
class Solution:
    def nextGreaterElement(self, n: int) -> int:
        chars = list(str(n))
        length = len(chars)
        right = length - 2
        while right >= 0 and chars[right] >= chars[right + 1]:
            right -= 1
        if right < 0:
            return -1
        ## 在left右边，找到最后一个大于chars[left]的数字
        left = right
        right = right + 1
        while right < length and chars[right] > chars[left]:
            right += 1
        chars[left], chars[right - 1] = chars[right - 1], chars[left]
        ## left右边的数组进行翻转
        begin = left + 1
        end = length - 1
        while begin < end:
            chars[begin], chars[end] = chars[end], chars[begin]
            begin += 1
            end -= 1
        ## 计算
        res = 0
        for ch in chars:
            res = res * 10 + int(ch)
        return res if res < 2 ** 31 - 1 else -1
```



### 556-1. 上一个更小元素

> 注意：Leetcode 没有这道题。

一、题目

给定一个32位正整数 n，你需要找到最大的32位整数，其与 n 中存在的位数完全相同，并且其值小于n。如果不存在这样的32位整数，则返回-1。

二、解析

把数字转成字符数组，按字母表的顺序找到可以交换的位置。

```python
class Solution:
    def lastSmallerElement(self, n: int) -> int:
        chars = list(str(n))
        length = len(chars)
        right = length - 2
        while right >= 0 and chars[right] <= chars[right + 1]:
            right -= 1
        if right < 0:
            return -1
        ## 在left右边，找到最后一个小于chars[left]的数字
        left = right
        right = right + 1
        while right < length and chars[right] < chars[left]:
            right += 1
        chars[left], chars[right - 1] = chars[right - 1], chars[left]
        ## 右边的数组进行翻转
        begin = left + 1
        end = length - 1
        while begin < end:
            chars[begin], chars[end] = chars[end], chars[begin]
            begin += 1
            end -= 1
        if chars[0] == "0":
            return -1
        res = 0
        for ch in chars:
            res = res * 10 + int(ch)
        return res
```



### 224. 基本计算器

> [224. 基本计算器](https://leetcode.cn/problems/basic-calculator/)

一、题目

给你一个字符串表达式 `s` ，请你实现一个基本计算器来计算并返回它的值。

注意:不允许使用任何将字符串作为数学表达式计算的内置函数，比如 `eval()` 。

示例 1：

```text
输入：s = "1 + 1"
输出：2
```

示例 2：

```text
输入：s = " 2-1 + 2 "
输出：3
```

示例 3：

```text
输入：s = "(1+(4+5+2)-3)+(6+8)"
输出：23
```

二、解析

> [官方题解](https://leetcode.cn/problems/basic-calculator/solution/ji-ben-ji-suan-qi-by-leetcode-solution-jvir/ "https://leetcode.cn/problems/basic-calculator/solution/ji-ben-ji-suan-qi-by-leetcode-solution-jvir/")

括号展开 + 栈。

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



### 227. 基本计算器 II

> [227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/ "227. 基本计算器 II")

一、题目

给你一个字符串表达式 s ，请你实现一个基本计算器来计算并返回它的值。

整数除法仅保留整数部分。

你可以假设给定的表达式总是有效的。所有中间结果将在 \[-231, 231 - 1] 的范围内。

注意：不允许使用任何将字符串作为数学表达式计算的内置函数，比如 eval() 。

示例 1：

```text
输入：s = "3+2*2"
输出：7
```

示例 2：

```text
输入：s = " 3/2 "
输出：1
```

示例 3：

```text
输入：s = " 3+5 / 2 "
输出：5
```

二、解析

代码如下：

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        sign = '+'
        digit = 0
        nums = []
        s = s.strip()
        for i in range(len(s)):
            if s[i] == ' ': 
                continue
            if s[i].isdigit(): 
                digit = digit * 10 + int(s[i])
            if not s[i].isdigit() or i == len(s) - 1:
                if sign == '+':
                    nums.append(digit)
                elif sign == '-':
                    nums.append(-digit)
                elif sign == '*':
                    tmp = nums[-1] * digit
                    nums[-1] = tmp
                elif sign == '/':
                    ## print(nums, digit)
                    tmp = abs(nums[-1]) // digit
                    if nums[-1] < 0:
                        tmp = -tmp
                    nums[-1] = tmp
                digit = 0
                sign = s[i]
                
        return sum(nums)
```



### 394. 字符串解码

> [394. 字符串解码](https://leetcode.cn/problems/decode-string/)

一、题目

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: k\[encoded\_string]，表示其中方括号内部的 encoded\_string 正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2\[4] 的输入。

示例 1：

```text
输入：s = "3[a]2[bc]"
输出："aaabcbc"
```

示例 2：

```text
输入：s = "3[a2[c]]"
输出："accaccacc"
```

示例 3：

```text
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"
```

示例 4：

```text
输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"
```

提示：

-   0 <= s.length <= 30
-   s 由小写英文字母、数字和方括号 '\[]' 组成
-   s 保证是一个 有效 的输入。
-   s 中所有整数的取值范围为 \[1, 300]&#x20;

二、解析

> [https://leetcode.cn/problems/decode-string/solution/decode-string-fu-zhu-zhan-fa-di-gui-fa-by-jyd/](https://leetcode.cn/problems/decode-string/solution/decode-string-fu-zhu-zhan-fa-di-gui-fa-by-jyd/ "https://leetcode.cn/problems/decode-string/solution/decode-string-fu-zhu-zhan-fa-di-gui-fa-by-jyd/")

1）使用栈。

构建辅助栈 stack， 遍历字符串 s 中每个字符 c；

-   当 c 为数字时，将数字字符转化为数字 multi，用于后续倍数计算；
-   当 c 为字母时，在 res 尾部添加 c；
-   当 c 为 \[ 时，将当前 multi 和 res 入栈，并分别置空置 0：
    -   记录此 \[ 前的临时结果 res 至栈，用于发现对应 ] 后的拼接操作；
    -   记录此 \[ 前的倍数 multi 至栈，用于发现对应 ] 后，获取 multi × \[...] 字符串。
    -   进入到新 \[ 后，res 和 multi 重新记录。
-   当 c 为 ] 时，stack 出栈，拼接字符串 res = last\_res + cur\_multi \* res，其中:
    -   last\_res是上个 \[ 到当前 \[ 的字符串，例如 "3\[a2\[c]]" 中的 a；
    -   cur\_multi是当前 \[ 到 ] 内字符串的重复倍数，例如 "3\[a2\[c]]" 中的 2。

代码如下：

```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack, res, multi = [], "", 0
        for c in s:
            if c == '[':
                stack.append([multi, res])
                res, multi = "", 0
            elif c == ']':
                cur_multi, last_res = stack.pop()
                res = last_res + cur_multi * res
            elif '0' <= c <= '9':
                multi = multi * 10 + int(c)            
            else:
                res += c
        return res
```

2）递归。

代码如下：

```python
class Solution:
    def decodeString(self, s: str) -> str:
        def dfs(s, i):
            res, multi = "", 0
            while i < len(s):
                if '0' <= s[i] <= '9':
                    multi = multi * 10 + int(s[i])
                elif s[i] == '[':
                    i, tmp = dfs(s, i + 1)
                    res += multi * tmp
                    multi = 0
                elif s[i] == ']':
                    return i, res
                else:
                    res += s[i]
                i += 1
            return res
        return dfs(s,0)
```



### 443. 压缩字符串

> [443. 压缩字符串](https://leetcode-cn.com/problems/string-compression/ "443. 压缩字符串")

一、题目

给你一个字符数组 chars ，请使用下述算法压缩：

从一个空字符串 s 开始。对于 chars 中的每组 连续重复字符 ：

- 如果这一组长度为 1 ，则将字符追加到 s 中。
- 否则，需要向 s 追加字符，后跟这一组的长度。

压缩后得到的字符串 s 不应该直接返回 ，需要转储到字符数组 chars 中。需要注意的是，如果组长度为 10 或 10 以上，则在 chars 数组中会被拆分为多个字符。

请在 修改完输入数组后 ，返回该数组的新长度。

你必须设计并实现一个只使用常量额外空间的算法来解决此问题。

示例1：

```纯文本
输入：chars = ["a","a","b","b","c","c","c"]
输出：返回 6 ，输入数组的前 6 个字符应该是：["a","2","b","2","c","3"]
解释："aa" 被 "a2" 替代。"bb" 被 "b2" 替代。"ccc" 被 "c3" 替代。
```

示例2：

```纯文本
输入：chars = ["a"]
输出：返回 1 ，输入数组的前 1 个字符应该是：["a"]
解释：唯一的组是“a”，它保持未压缩，因为它是一个字符。
```

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/string-compression/solution/ya-suo-zi-fu-chuan-by-leetcode/ "Leetcode官方题解")

双指针。我们可以使用双指针分别标志我们在字符串中读和写的位置。每次当读指针 read 移动到某一段连续相同子串的最右侧，我们就在写指针 write 处依次写入该子串对应的字符和子串长度即可

代码如下：

```python
class Solution(object):
    def compress(self, chars):
        anchor = write = 0
        for read, c in enumerate(chars):
            if read + 1 == len(chars) or chars[read + 1] != c:
                chars[write] = chars[anchor]
                write += 1
                if read > anchor:
                    for digit in str(read - anchor + 1):
                        chars[write] = digit
                        write += 1
                anchor = read + 1
        return write
```



### 6. Z 字形变换

> [6. Z 字形变换](https://leetcode-cn.com/problems/zigzag-conversion/ "6. Z 字形变换")

一、题目

将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。

将一个给定字符串 `s` 根据给定的行数 `numRows` ，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 `"PAYPALISHIRING"` 行数为 `3` 时，排列如下：

```
P   A   H   N
A P L S I I G
Y   I   R
```

之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如：`"PAHNAPLSIIGYIR"`。

二、解析

设置一个flag变量。

代码如下：

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        
        res = [[] for _ in range(numRows)]
        row = 0
        flag = 1
        for c in s:
            res[row].append(c)
            if row == 0:
                flag = 1
            elif row == numRows - 1:
                flag = -1
            row += flag 
        return ''.join(''.join(r) for r in res)
```



### 696. 计数二进制子串

> [696. 计数二进制子串](https://leetcode-cn.com/problems/count-binary-substrings/ "696. 计数二进制子串")

一、题目

给定一个字符串`s`，计算具有相同数量0和1的非空(连续)子字符串的数量，并且这些子字符串中的所有0和所有1都是组合在一起的。
重复出现的子串要计算它们出现的次数。

示例 1：

```text
输入：s = "00110011"
输出：6
解释：6 个子串满足具有相同数量的连续 1 和 0 ："0011"、"01"、"1100"、"10"、"0011" 和 "01" 。
注意，一些重复出现的子串（不同位置）要统计它们出现的次数。
另外，"00110011" 不是有效的子串，因为所有的 0（还有 1 ）没有组合在一起。
```

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/count-binary-substrings/solution/ji-shu-er-jin-zhi-zi-chuan-by-leetcode/ "Leetcode官方题解")

1）中心扩展法

```python
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        count = 0
        for i in range(len(s) - 1):
            if s[i] != s[i + 1]:
                count += 1
                for j in range(1, min(i + 1, len(s) - i - 1)):
                    if not (s[i - j] == s[i] and s[i + 1] == s[i + 1 + j]):
                        break
                    count += 1                    
        return count
```

2）线性扫描

```python
class Solution(object):
    def countBinarySubstrings(self, s):
        ans, prev, cur = 0, 0, 1
        for i in range(1, len(s)):
            if s[i-1] != s[i]:
                ans += min(prev, cur)
                prev, cur = cur, 1
            else:
                cur += 1
        return ans + min(prev, cur)
```



### 1071. 字符串的最大公因子

> [1071. 字符串的最大公因子](https://leetcode-cn.com/problems/greatest-common-divisor-of-strings/ "1071. 字符串的最大公因子")

一、题目

对于字符串`S`和`T`，只有在`S = T + ... + T`（`T`与自身连接 1 次或多次）时，我们才认定“`T` 能除尽 `S`”。
返回最长字符串`X`，要求满足`X`能除尽`str1`且`X`能除尽`str2`。

二、解析

代码如下：

```python
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        def gcd(a, b):
            if b == 0:
                return a
            return gcd(b, a % b)
        
        if str1 + str2 != str2 + str1:
            return ''
        return str1[:gcd(len(str1), len(str2))]
```



### 415. 字符串相加

> [415. 字符串相加](https://leetcode-cn.com/problems/add-strings/ "415. 字符串相加")

一、题目

给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和。

二、解析

代码如下：

```python
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        len1 = len(num1)
        len2 = len(num2)
        i = len1 - 1
        j = len2 - 1
        res = []
        carry = 0
        while i >= 0 or j >= 0 or carry != 0:
            d1 = int(num1[i]) if i >= 0 else 0
            d2 = int(num2[j]) if j >= 0 else 0
            temp = d1 + d2 + carry
            carry, temp = divmod(temp, 10)
            res.append(str(temp))
            i -= 1
            j -= 1
        return ''.join(res[::-1])
```



### 43. 字符串相乘

> [43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/ "43. 字符串相乘")

一、题目

给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。

二、解析

> 参考 [官方题解](https://leetcode-cn.com/problems/multiply-strings/solution/zi-fu-chuan-xiang-cheng-by-leetcode-solution/ "官方题解")

代码如下：

```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0":
            return "0"
        
        m, n = len(num1), len(num2)
        ansArr = [0] * (m + n)
        for i in range(m - 1, -1, -1):
            x = int(num1[i])
            for j in range(n - 1, -1, -1):
                ansArr[i + j + 1] += x * int(num2[j])
        
        for i in range(m + n - 1, 0, -1):
            ansArr[i - 1] += ansArr[i] // 10
            ansArr[i] %= 10
        
        index = 1 if ansArr[0] == 0 else 0
        return "".join(str(x) for x in ansArr[index:])
```



### 424. 替换后的最长重复字符

> [424. 替换后的最长重复字符](https://leetcode-cn.com/problems/longest-repeating-character-replacement/ "424. 替换后的最长重复字符")

一、题目

给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，总共可最多替换 *k* 次。在执行上述操作后，找到包含重复字母的最长子串的长度。

**示例 1：**

```纯文本
输入：s = "ABAB", k = 2
输出：4
解释：用两个'A'替换为两个'B',反之亦然。
```

二、解析

> [官方题解](https://leetcode.cn/problems/longest-repeating-character-replacement/solutions/)

使用双指针。使用左右两个指针，右指针每次往右走，如果满足条件（或不满足条件），左指针只收敛一次。之前遇到过左指针一直收缩，参考【3. 无重复字符的最长子串】。

通过枚举字符串中的每一个位置作为右端点，然后找到其最远的左端点的位置，满足该区间内除了出现次数最多的那一类字符之外，剩余的字符（即非最长重复字符）数量不超过 k 个。

这样我们可以想到使用双指针维护这些区间，每次右指针右移，如果区间仍然满足条件，那么左指针不移动，否则左指针至多右移一格，保证区间长度不减小。

另外，每次区间右移，我们更新右移位置的字符出现的次数，然后尝试用它更新重复字符出现次数的历史最大值，最后我们使用该最大值计算出区间内非最长重复字符的数量，以此判断左指针是否需要右移即可。

Python 代码如下：

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        nums = {c: 0 for c in s}
        n = len(s)
        maxn = 0 # 窗口中曾出现某字母的最大次数
        left = right = 0

        while right < n:
            nums[s[right]] += 1
            maxn = max(maxn, nums[s[right]])
            if right - left + 1 - maxn > k:
                nums[s[left]] -= 1
                left += 1
            right += 1
        return right - left
```



## 其他编程题

### 汉字读法的整数转为阿拉伯数字

一、题目

示例：

```纯文本
五千四百零三万一千二百
54031200
```

二、解析

不完全正确。

代码如下：

```python
def chinese2int(string):
    """string表示的数字小于一亿"""
    digits_map = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, 
                  "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    unit_map = {"千": 1000, "百": 100, "十": 10, "零": 0}
    if len(string) == 1:
        return digits_map[string[0]]
    num = 0
    i = 0
    N = len(string)
    while i < N:
        if i < N and string[i] == "万":
            num *= 10000
            i += 1
        if i == N - 1:
            num += digits_map[string[i]]
            i += 1
            break
        end = i
        while end < N and string[end] not in unit_map:
            end += 1
        if i == end:
            if string[end] == "零":
                end += 1
                temp = digits_map[string[end]]
                end += 1
                if end < N and string[end] == "十":
                    temp *= 10
                    end += 1
                num += temp
            elif string[end] == "十":
                num += 10
                end += 1
                num += digits_map[string[end]]
                end += 1
        elif end < N:
            num += digits_map[string[i]] * unit_map[string[end]]
            end += 1
        i = end
    return num


if __name__ == "__main__":
    test_cases = [["零", 0],
                  ["一", 1],
                  ["十", 10],
                  ["十一", 11],
                  ["二十", 20],
                  ["一百", 100],
                  ["一百零一", 101],
                  ["一百一十一", 111],
                  ["一千", 1000],
                  ["一千零一", 1001],
                  ["五千零二十万一千二百零五", 50201205]]
    for string, ans in test_cases:
        res = chinese2int(string)
        print(string, ans, res)
```

